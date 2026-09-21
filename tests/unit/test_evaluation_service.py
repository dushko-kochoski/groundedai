import uuid
from datetime import UTC, datetime
from typing import Any, cast

from app.db.models import EvaluationMode, EvaluationRun, EvaluationRunStatus
from app.schemas.evaluation import (
    EvaluationDocumentSpec,
    ExpectedFact,
    GoldenCase,
    GoldenDataset,
)
from app.schemas.query import (
    GroundedQueryDiagnostics,
    GroundedQueryResponse,
    QueryRunResponse,
    ValidatedCitation,
)
from app.schemas.retrieval import (
    RetrievalDiagnostics,
    RetrievalResult,
    RetrievalSearchResponse,
)
from app.services.evaluation.service import EvaluationService


class FakeRegistry:
    def __init__(self, dataset: GoldenDataset) -> None:
        self.dataset = dataset

    def load(self, name: str) -> GoldenDataset:
        assert name == self.dataset.name
        return self.dataset


class FakeFixtures:
    def __init__(self, document_id: uuid.UUID) -> None:
        self.document_id = document_id

    async def prepare(self, dataset: GoldenDataset) -> dict[str, str]:
        return {str(self.document_id): dataset.documents[0].id}


class FakeEmbedding:
    name = "deterministic-feature-hashing"
    model = "feature-hashing-v1"
    version = "1"
    dimension = 1536

    async def embed_text(self, text: str) -> list[float]:
        return [0.0] * self.dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]


class FakeGeneration:
    name = "deterministic"
    model = "extractive-v1"


class FakeRetrieval:
    def __init__(self, response: RetrievalSearchResponse) -> None:
        self.response = response
        self.calls = 0

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        document_ids: set[uuid.UUID] | None = None,
    ) -> RetrievalSearchResponse:
        self.calls += 1
        return self.response.model_copy(update={"query": query})


class FakeGrounded:
    def __init__(
        self,
        response: GroundedQueryResponse,
        persisted: QueryRunResponse,
    ) -> None:
        self.response = response
        self.persisted = persisted
        self.calls = 0

    async def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        document_ids: set[uuid.UUID] | None = None,
    ) -> GroundedQueryResponse:
        self.calls += 1
        return self.response.model_copy(update={"question": question})

    async def get_run(self, run_id: uuid.UUID) -> QueryRunResponse:
        assert run_id == self.response.run_id
        return self.persisted


class FakeRepository:
    def __init__(self) -> None:
        self.run: EvaluationRun | None = None
        self.cases: list[object] = []

    async def add_run_and_commit(self, run: EvaluationRun) -> None:
        if run.id is None:
            run.id = uuid.uuid4()
        if run.created_at is None:
            run.created_at = datetime.now(UTC)
        self.run = run

    async def complete_run(self, run: EvaluationRun, cases: list[object]) -> None:
        self.run = run
        self.cases = cases

    async def get_run(self, run_id: uuid.UUID) -> EvaluationRun | None:
        return self.run if self.run is not None and self.run.id == run_id else None

    async def list_cases(self, run_id: uuid.UUID) -> list[object]:
        return self.cases

    async def rollback(self) -> None:
        return None


def _dataset(*cases: GoldenCase) -> GoldenDataset:
    return GoldenDataset(
        name="tiny",
        version="1",
        documents=[EvaluationDocumentSpec(id="policy", filename="policy.txt")],
        cases=list(cases),
    )


def _case(*, answerable: bool = True) -> GoldenCase:
    return GoldenCase(
        id="case-1",
        question="What is the policy?" if answerable else "Who founded the company?",
        answerable=answerable,
        expected_facts=[ExpectedFact(id="fact", acceptable_phrases=["two days each week"])]
        if answerable
        else [],
        expected_sources=["policy"] if answerable else [],
    )


def _retrieval(document_id: uuid.UUID, *, results: bool = True) -> RetrievalSearchResponse:
    items = (
        [
            RetrievalResult(
                chunk_id=uuid.uuid4(),
                document_id=document_id,
                filename="policy.txt",
                chunk_index=0,
                text="Employees may work remotely two days each week.",
                score=0.8,
            )
        ]
        if results
        else []
    )
    return RetrievalSearchResponse(
        query="question",
        results=items,
        diagnostics=RetrievalDiagnostics(
            provider="deterministic-feature-hashing",
            model="feature-hashing-v1",
            dimension=1536,
            version="1",
            requested_top_k=5,
            database_candidates_considered=len(items),
            returned_count=len(items),
            min_score=0.15,
            max_returned_chars=6000,
            returned_chars=sum(len(item.text) for item in items),
            embedding_ms=1.0,
            search_ms=1.0,
            total_ms=2.0,
        ),
    )


def _grounded(document_id: uuid.UUID, *, abstained: bool = False) -> FakeGrounded:
    run_id = uuid.uuid4()
    diagnostics = GroundedQueryDiagnostics(
        retrieved_count=0 if abstained else 1,
        retrieval_ms=2.0,
        generation_ms=0.0 if abstained else 1.0,
        total_ms=2.0 if abstained else 3.0,
        provider="deterministic",
        model="extractive-v1",
        prompt_version="grounded_answer_v1",
    )
    citations = (
        []
        if abstained
        else [
            ValidatedCitation(
                source_id="S1",
                document_id=document_id,
                chunk_id=uuid.uuid4(),
                filename="policy.txt",
                chunk_index=0,
                page_number=None,
                section=None,
                score=0.8,
            )
        ]
    )
    response = GroundedQueryResponse(
        run_id=run_id,
        question="question",
        answer=None if abstained else "Employees may work remotely two days each week.",
        abstained=abstained,
        abstention_reason="No evidence." if abstained else None,
        citations=citations,
        diagnostics=diagnostics,
    )
    persisted = QueryRunResponse(
        run_id=run_id,
        status="completed",
        question="question",
        answer=response.answer,
        abstained=abstained,
        abstention_reason=response.abstention_reason,
        citations=[],
        retrieval_snapshot=[] if abstained else [{"document_id": str(document_id)}],
        diagnostics=diagnostics,
        error_code=None,
        error_message=None,
        created_at=datetime.now(UTC),
    )
    return FakeGrounded(response, persisted)


def _service(
    dataset: GoldenDataset,
    document_id: uuid.UUID,
    retrieval: FakeRetrieval,
    grounded: FakeGrounded,
) -> tuple[EvaluationService, FakeRepository]:
    repository = FakeRepository()
    return (
        EvaluationService(
            registry=FakeRegistry(dataset),  # type: ignore[arg-type]
            fixtures=FakeFixtures(document_id),
            retrieval=retrieval,
            grounded_answers=grounded,
            embeddings=FakeEmbedding(),
            generation=FakeGeneration(),  # type: ignore[arg-type]
            repository=repository,  # type: ignore[arg-type]
            default_top_k=5,
            default_min_score=0.15,
        ),
        repository,
    )


async def test_retrieval_only_evaluates_without_generation() -> None:
    document_id = uuid.uuid4()
    retrieval = FakeRetrieval(_retrieval(document_id))
    grounded = _grounded(document_id)
    service, repository = _service(_dataset(_case()), document_id, retrieval, grounded)

    result = await service.run("tiny", EvaluationMode.RETRIEVAL_ONLY)
    metrics = cast(dict[str, Any], result.aggregate_metrics)

    assert result.status == EvaluationRunStatus.COMPLETED
    assert metrics["retrieval"]["hit_at_k"] == 1.0
    assert retrieval.calls == 1
    assert grounded.calls == 0
    assert len(repository.cases) == 1


async def test_end_to_end_measures_facts_citations_and_answer() -> None:
    document_id = uuid.uuid4()
    retrieval = FakeRetrieval(_retrieval(document_id))
    grounded = _grounded(document_id)
    service, _ = _service(_dataset(_case()), document_id, retrieval, grounded)

    result = await service.run("tiny", EvaluationMode.END_TO_END)
    metrics = cast(dict[str, Any], result.aggregate_metrics)

    assert metrics["answers"]["mean_fact_coverage"] == 1.0
    assert metrics["citations"]["source_citation_precision"] == 1.0
    assert metrics["abstention"]["correct_answers"] == 1
    assert retrieval.calls == 0
    assert grounded.calls == 1


async def test_end_to_end_records_correct_abstention() -> None:
    document_id = uuid.uuid4()
    retrieval = FakeRetrieval(_retrieval(document_id, results=False))
    grounded = _grounded(document_id, abstained=True)
    service, _ = _service(_dataset(_case(answerable=False)), document_id, retrieval, grounded)

    result = await service.run("tiny", EvaluationMode.END_TO_END)
    metrics = cast(dict[str, Any], result.aggregate_metrics)

    assert metrics["abstention"]["correct_abstentions"] == 1
    assert metrics["abstention"]["false_answer_rate"] == 0.0
