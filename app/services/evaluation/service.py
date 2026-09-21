import time
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from app.core.exceptions import ApplicationError
from app.db.models import (
    EvaluationCaseResult,
    EvaluationMode,
    EvaluationRun,
    EvaluationRunStatus,
)
from app.providers.embeddings.base import EmbeddingProvider
from app.providers.generation import GenerationProvider
from app.providers.generation.prompt import PROMPT_VERSION
from app.repositories.evaluations import EvaluationRepository
from app.schemas.evaluation import (
    EvaluationCaseListResponse,
    EvaluationCaseResponse,
    EvaluationRunResponse,
    GoldenCase,
)
from app.schemas.query import GroundedQueryResponse, QueryRunResponse
from app.schemas.retrieval import RetrievalSearchResponse
from app.services.evaluation.datasets import DatasetRegistry
from app.services.evaluation.fixtures import EvaluationFixturePreparerProtocol
from app.services.evaluation.metrics import (
    aggregate_case_metrics,
    citation_source_metrics,
    match_facts,
    retrieval_metrics,
)
from app.services.generation import GroundedAnswerServiceProtocol
from app.services.retrieval import RetrievalServiceProtocol


class EvaluationServiceProtocol(Protocol):
    async def run(
        self,
        dataset_name: str,
        mode: EvaluationMode,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> EvaluationRunResponse: ...

    async def get_run(self, run_id: uuid.UUID) -> EvaluationRunResponse: ...

    async def list_cases(self, run_id: uuid.UUID) -> EvaluationCaseListResponse: ...


class EvaluationService:
    def __init__(
        self,
        *,
        registry: DatasetRegistry,
        fixtures: EvaluationFixturePreparerProtocol,
        retrieval: RetrievalServiceProtocol,
        grounded_answers: GroundedAnswerServiceProtocol,
        embeddings: EmbeddingProvider,
        generation: GenerationProvider,
        repository: EvaluationRepository,
        default_top_k: int,
        default_min_score: float,
    ) -> None:
        self.registry = registry
        self.fixtures = fixtures
        self.retrieval = retrieval
        self.grounded_answers = grounded_answers
        self.embeddings = embeddings
        self.generation = generation
        self.repository = repository
        self.default_top_k = default_top_k
        self.default_min_score = default_min_score

    async def run(
        self,
        dataset_name: str,
        mode: EvaluationMode,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> EvaluationRunResponse:
        dataset = self.registry.load(dataset_name)
        effective_top_k = self.default_top_k if top_k is None else top_k
        effective_min_score = self.default_min_score if min_score is None else min_score
        mapping = await self.fixtures.prepare(dataset)
        allowed_document_ids = {uuid.UUID(document_id) for document_id in mapping}
        started = datetime.now(UTC)
        timer = time.perf_counter()
        run = EvaluationRun(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            mode=mode,
            status=EvaluationRunStatus.RUNNING,
            retrieval_top_k=effective_top_k,
            retrieval_min_score=effective_min_score,
            embedding_provider=self.embeddings.name,
            embedding_model=self.embeddings.model,
            embedding_version=self.embeddings.version,
            generation_provider=(
                self.generation.name if mode == EvaluationMode.END_TO_END else None
            ),
            generation_model=(self.generation.model if mode == EvaluationMode.END_TO_END else None),
            prompt_version=(PROMPT_VERSION if mode == EvaluationMode.END_TO_END else None),
            case_count=len(dataset.cases),
            aggregate_metrics={},
            started_at=started,
        )
        await self.repository.add_run_and_commit(run)
        payloads: list[dict[str, Any]] = []
        models: list[EvaluationCaseResult] = []
        try:
            for case in dataset.cases:
                payload = await self._evaluate_case(
                    case,
                    mode,
                    mapping,
                    allowed_document_ids,
                    effective_top_k,
                    effective_min_score,
                )
                payloads.append(payload)
                model_payload = {
                    key: value for key, value in payload.items() if key != "matching_citations"
                }
                models.append(EvaluationCaseResult(evaluation_run_id=run.id, **model_payload))
            run.aggregate_metrics = aggregate_case_metrics(
                payloads,
                top_k=effective_top_k,
                include_generation=mode == EvaluationMode.END_TO_END,
            )
            run.status = EvaluationRunStatus.COMPLETED
            run.completed_at = datetime.now(UTC)
            run.duration_ms = self._elapsed_ms(timer)
            await self.repository.complete_run(run, models)
        except Exception as exc:
            await self.repository.rollback()
            run.status = EvaluationRunStatus.FAILED
            run.completed_at = datetime.now(UTC)
            run.duration_ms = self._elapsed_ms(timer)
            run.error_category = "evaluation_execution_failed"
            run.error_message = "The evaluation run did not complete."
            await self.repository.add_run_and_commit(run)
            raise ApplicationError(
                status_code=500,
                code="evaluation_execution_failed",
                message="The evaluation run did not complete.",
                details={"run_id": str(run.id)},
            ) from exc
        return self._run_response(run)

    async def _evaluate_case(
        self,
        case: GoldenCase,
        mode: EvaluationMode,
        document_mapping: dict[str, str],
        allowed_document_ids: set[uuid.UUID],
        top_k: int,
        min_score: float,
    ) -> dict[str, Any]:
        if mode == EvaluationMode.RETRIEVAL_ONLY:
            retrieval = await self.retrieval.retrieve(
                case.question, top_k=top_k, min_score=min_score, document_ids=allowed_document_ids
            )
            return self._retrieval_case(case, retrieval, document_mapping)
        try:
            answer = await self.grounded_answers.query(
                case.question,
                top_k=top_k,
                min_score=min_score,
                document_ids=allowed_document_ids,
            )
            persisted = await self.grounded_answers.get_run(answer.run_id)
            return self._answer_case(case, answer, persisted, document_mapping)
        except ApplicationError as exc:
            return self._error_case(case, exc)

    @staticmethod
    def _retrieval_case(
        case: GoldenCase,
        retrieval: RetrievalSearchResponse,
        mapping: dict[str, str],
    ) -> dict[str, Any]:
        sources = [
            mapping.get(str(result.document_id), str(result.document_id))
            for result in retrieval.results
        ]
        hit, recall, reciprocal_rank = retrieval_metrics(case.expected_sources, sources)
        return {
            "case_id": case.id,
            "question": case.question,
            "answerable": case.answerable,
            "expected_sources": case.expected_sources,
            "retrieved_sources": sources,
            "expected_facts": [fact.model_dump() for fact in case.expected_facts],
            "answer": None,
            "abstained": None,
            "facts_matched": None,
            "facts_expected": len(case.expected_facts),
            "fact_coverage": None,
            "source_hit": hit,
            "source_recall": recall,
            "reciprocal_rank": reciprocal_rank,
            "citations_returned": 0,
            "citations_valid": 0,
            "invalid_citation_count": 0,
            "citation_validity_rate": None,
            "citation_source_precision": None,
            "citation_source_recall": None,
            "matching_citations": 0,
            "retrieval_ms": retrieval.diagnostics.total_ms,
            "generation_ms": None,
            "total_ms": retrieval.diagnostics.total_ms,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "query_run_id": None,
            "error_category": None,
        }

    @staticmethod
    def _answer_case(
        case: GoldenCase,
        answer: GroundedQueryResponse,
        persisted: QueryRunResponse,
        mapping: dict[str, str],
    ) -> dict[str, Any]:
        retrieved = [
            mapping.get(str(item["document_id"]), str(item["document_id"]))
            for item in persisted.retrieval_snapshot
        ]
        cited = [
            mapping.get(str(item.document_id), str(item.document_id)) for item in answer.citations
        ]
        hit, recall, reciprocal_rank = retrieval_metrics(case.expected_sources, retrieved)
        matched, expected, coverage = match_facts(answer.answer, case.expected_facts)
        citation_precision, citation_recall = citation_source_metrics(case.expected_sources, cited)
        matching_citations = len(set(case.expected_sources).intersection(cited))
        citation_count = len(answer.citations)
        return {
            "case_id": case.id,
            "question": case.question,
            "answerable": case.answerable,
            "expected_sources": case.expected_sources,
            "retrieved_sources": retrieved,
            "expected_facts": [fact.model_dump() for fact in case.expected_facts],
            "answer": answer.answer,
            "abstained": answer.abstained,
            "facts_matched": matched,
            "facts_expected": expected,
            "fact_coverage": coverage,
            "source_hit": hit,
            "source_recall": recall,
            "reciprocal_rank": reciprocal_rank,
            "citations_returned": citation_count,
            "citations_valid": citation_count,
            "invalid_citation_count": 0,
            "citation_validity_rate": 1.0 if citation_count else None,
            "citation_source_precision": citation_precision,
            "citation_source_recall": citation_recall,
            "matching_citations": matching_citations,
            "retrieval_ms": answer.diagnostics.retrieval_ms,
            "generation_ms": answer.diagnostics.generation_ms,
            "total_ms": answer.diagnostics.total_ms,
            "input_tokens": answer.diagnostics.input_tokens,
            "output_tokens": answer.diagnostics.output_tokens,
            "total_tokens": answer.diagnostics.total_tokens,
            "query_run_id": answer.run_id,
            "error_category": None,
        }

    @staticmethod
    def _error_case(case: GoldenCase, exc: ApplicationError) -> dict[str, Any]:
        return {
            "case_id": case.id,
            "question": case.question,
            "answerable": case.answerable,
            "expected_sources": case.expected_sources,
            "retrieved_sources": [],
            "expected_facts": [fact.model_dump() for fact in case.expected_facts],
            "answer": None,
            "abstained": None,
            "facts_matched": 0 if case.expected_facts else None,
            "facts_expected": len(case.expected_facts),
            "fact_coverage": 0.0 if case.expected_facts else None,
            "source_hit": False,
            "source_recall": 0.0,
            "reciprocal_rank": 0.0,
            "citations_returned": 0,
            "citations_valid": 0,
            "invalid_citation_count": int(exc.code == "citation_integrity_failure"),
            "citation_validity_rate": None,
            "citation_source_precision": None,
            "citation_source_recall": 0.0 if case.expected_sources else None,
            "matching_citations": 0,
            "retrieval_ms": 0.0,
            "generation_ms": None,
            "total_ms": 0.0,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "query_run_id": _run_id_from_error(exc),
            "error_category": exc.code,
        }

    async def get_run(self, run_id: uuid.UUID) -> EvaluationRunResponse:
        run = await self.repository.get_run(run_id)
        if run is None:
            raise ApplicationError(
                status_code=404,
                code="evaluation_run_not_found",
                message="The requested evaluation run does not exist.",
                details={"run_id": str(run_id)},
            )
        return self._run_response(run)

    async def list_cases(self, run_id: uuid.UUID) -> EvaluationCaseListResponse:
        if await self.repository.get_run(run_id) is None:
            raise ApplicationError(
                status_code=404,
                code="evaluation_run_not_found",
                message="The requested evaluation run does not exist.",
            )
        cases = await self.repository.list_cases(run_id)
        return EvaluationCaseListResponse(
            run_id=run_id,
            items=[self._case_response(case) for case in cases],
        )

    @staticmethod
    def _run_response(run: EvaluationRun) -> EvaluationRunResponse:
        return EvaluationRunResponse(
            run_id=run.id,
            dataset_name=run.dataset_name,
            dataset_version=run.dataset_version,
            mode=run.mode,
            status=run.status,
            retrieval_top_k=run.retrieval_top_k,
            retrieval_min_score=run.retrieval_min_score,
            embedding_provider=run.embedding_provider,
            embedding_model=run.embedding_model,
            embedding_version=run.embedding_version,
            generation_provider=run.generation_provider,
            generation_model=run.generation_model,
            prompt_version=run.prompt_version,
            case_count=run.case_count,
            aggregate_metrics=run.aggregate_metrics,
            started_at=run.started_at,
            completed_at=run.completed_at,
            duration_ms=run.duration_ms,
            error_category=run.error_category,
            error_message=run.error_message,
            created_at=run.created_at,
        )

    @staticmethod
    def _case_response(case: EvaluationCaseResult) -> EvaluationCaseResponse:
        return EvaluationCaseResponse(
            case_id=case.case_id,
            question=case.question,
            answerable=case.answerable,
            expected_sources=case.expected_sources,
            retrieved_sources=case.retrieved_sources,
            answer=case.answer,
            abstained=case.abstained,
            fact_coverage=case.fact_coverage,
            source_hit=case.source_hit,
            source_recall=case.source_recall,
            reciprocal_rank=case.reciprocal_rank,
            citation_validity_rate=case.citation_validity_rate,
            citation_source_precision=case.citation_source_precision,
            citation_source_recall=case.citation_source_recall,
            retrieval_ms=case.retrieval_ms,
            generation_ms=case.generation_ms,
            total_ms=case.total_ms,
            query_run_id=case.query_run_id,
            error_category=case.error_category,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)


def _run_id_from_error(exc: ApplicationError) -> uuid.UUID | None:
    value = exc.details.get("run_id")
    try:
        return uuid.UUID(value) if isinstance(value, str) else None
    except ValueError:
        return None
