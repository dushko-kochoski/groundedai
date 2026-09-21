import time
import uuid
from typing import Protocol

from pydantic import ValidationError

from app.core.exceptions import ApplicationError
from app.db.models import QueryRun, QueryRunStatus
from app.providers.generation import (
    GenerationProvider,
    GenerationProviderError,
    GenerationResult,
    GroundedModelOutput,
)
from app.providers.generation.prompt import GroundedPrompt, build_grounded_prompt
from app.schemas.query import (
    GroundedQueryDiagnostics,
    GroundedQueryResponse,
    QueryRunResponse,
    ValidatedCitation,
)
from app.schemas.retrieval import RetrievalResult, RetrievalSearchResponse
from app.services.retrieval import RetrievalServiceProtocol

PRE_RETRIEVAL_ABSTENTION = "No sufficiently relevant evidence was found."


class QueryRunRepositoryProtocol(Protocol):
    async def add_and_commit(self, run: QueryRun) -> None: ...

    async def get(self, run_id: uuid.UUID) -> QueryRun | None: ...

    async def rollback(self) -> None: ...


class GroundedAnswerServiceProtocol(Protocol):
    async def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        document_ids: set[uuid.UUID] | None = None,
    ) -> GroundedQueryResponse: ...

    async def get_run(self, run_id: uuid.UUID) -> QueryRunResponse: ...


class CitationIntegrityError(ValueError):
    pass


class GroundedAnswerService:
    def __init__(
        self,
        *,
        retrieval: RetrievalServiceProtocol,
        generation: GenerationProvider,
        runs: QueryRunRepositoryProtocol,
    ) -> None:
        self.retrieval = retrieval
        self.generation = generation
        self.runs = runs

    async def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        document_ids: set[uuid.UUID] | None = None,
    ) -> GroundedQueryResponse:
        started = time.perf_counter()
        retrieval = await self.retrieval.retrieve(
            question,
            top_k=top_k,
            min_score=min_score,
            document_ids=document_ids,
        )
        prompt = build_grounded_prompt(retrieval.query, retrieval.results)
        retrieval_snapshot = self._retrieval_snapshot(prompt, retrieval.results)

        if not retrieval.results:
            return await self._persist_completed(
                retrieval=retrieval,
                prompt=prompt,
                answer=None,
                abstained=True,
                abstention_reason=PRE_RETRIEVAL_ABSTENTION,
                citations=[],
                retrieval_snapshot=retrieval_snapshot,
                generation_ms=0.0,
                total_ms=self._elapsed_ms(started),
                generation_result=None,
            )

        generation_started = time.perf_counter()
        try:
            generation_result = await self.generation.generate_grounded_answer(prompt)
            output = GroundedModelOutput.model_validate(generation_result.output.model_dump())
            generation_ms = self._elapsed_ms(generation_started)
            citations = (
                [] if output.abstained else self._resolve_citations(output, retrieval.results)
            )
        except GenerationProviderError as exc:
            generation_ms = self._elapsed_ms(generation_started)
            run = await self._persist_failure(
                retrieval,
                prompt,
                retrieval_snapshot,
                generation_ms,
                self._elapsed_ms(started),
                exc.category,
                str(exc),
            )
            status_code = 504 if exc.category == "provider_timeout" else 502
            raise ApplicationError(
                status_code=status_code,
                code=exc.category,
                message=str(exc),
                details={"run_id": str(run.id)},
            ) from exc
        except (ValidationError, CitationIntegrityError) as exc:
            generation_ms = self._elapsed_ms(generation_started)
            category = (
                "citation_integrity_failure"
                if isinstance(exc, CitationIntegrityError)
                else "malformed_provider_output"
            )
            message = (
                "The generated citations could not be validated."
                if isinstance(exc, CitationIntegrityError)
                else "The generation provider returned an invalid structured answer."
            )
            run = await self._persist_failure(
                retrieval,
                prompt,
                retrieval_snapshot,
                generation_ms,
                self._elapsed_ms(started),
                category,
                message,
            )
            raise ApplicationError(
                status_code=502,
                code=category,
                message=message,
                details={"run_id": str(run.id)},
            ) from exc

        return await self._persist_completed(
            retrieval=retrieval,
            prompt=prompt,
            answer=output.answer,
            abstained=output.abstained,
            abstention_reason=output.abstention_reason,
            citations=citations,
            retrieval_snapshot=retrieval_snapshot,
            generation_ms=generation_ms,
            total_ms=self._elapsed_ms(started),
            generation_result=generation_result,
        )

    async def get_run(self, run_id: uuid.UUID) -> QueryRunResponse:
        run = await self.runs.get(run_id)
        if run is None:
            raise ApplicationError(
                status_code=404,
                code="query_run_not_found",
                message="The requested query run does not exist.",
                details={"run_id": str(run_id)},
            )
        return self._run_response(run)

    @staticmethod
    def _resolve_citations(
        output: GroundedModelOutput, results: list[RetrievalResult]
    ) -> list[ValidatedCitation]:
        alias_map = {f"S{index}": result for index, result in enumerate(results, start=1)}
        citations: list[ValidatedCitation] = []
        seen: set[str] = set()
        for alias in output.citations:
            if alias in seen:
                continue
            result = alias_map.get(alias)
            if result is None:
                raise CitationIntegrityError(f"unknown source alias: {alias}")
            seen.add(alias)
            citations.append(
                ValidatedCitation(
                    source_id=alias,
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    filename=result.filename,
                    chunk_index=result.chunk_index,
                    page_number=result.page_number,
                    section=result.section,
                    score=result.score,
                )
            )
        if not citations:
            raise CitationIntegrityError("a grounded answer requires at least one citation")
        return citations

    @staticmethod
    def _retrieval_snapshot(
        prompt: GroundedPrompt, results: list[RetrievalResult]
    ) -> list[dict[str, object]]:
        return [
            {
                "source_id": source.alias,
                "chunk_id": source.chunk_id,
                "document_id": source.document_id,
                "filename": source.filename,
                "chunk_index": result.chunk_index,
                "page_number": source.page_number,
                "section": source.section,
                "score": result.score,
            }
            for source, result in zip(prompt.sources, results, strict=True)
        ]

    async def _persist_completed(
        self,
        *,
        retrieval: RetrievalSearchResponse,
        prompt: GroundedPrompt,
        answer: str | None,
        abstained: bool,
        abstention_reason: str | None,
        citations: list[ValidatedCitation],
        retrieval_snapshot: list[dict[str, object]],
        generation_ms: float,
        total_ms: float,
        generation_result: GenerationResult | None,
    ) -> GroundedQueryResponse:
        usage = generation_result.usage if generation_result is not None else None
        run = QueryRun(
            status=QueryRunStatus.COMPLETED,
            question=retrieval.query,
            answer=answer,
            abstained=abstained,
            abstention_reason=abstention_reason,
            generation_provider=self.generation.name,
            generation_model=self.generation.model,
            prompt_version=prompt.version,
            retrieval_top_k=retrieval.diagnostics.requested_top_k,
            retrieval_min_score=retrieval.diagnostics.min_score,
            retrieval_snapshot=retrieval_snapshot,
            citation_snapshot=[citation.model_dump(mode="json") for citation in citations],
            retrieval_ms=retrieval.diagnostics.total_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )
        await self.runs.add_and_commit(run)
        return self._completed_response(run, citations)

    async def _persist_failure(
        self,
        retrieval: RetrievalSearchResponse,
        prompt: GroundedPrompt,
        retrieval_snapshot: list[dict[str, object]],
        generation_ms: float,
        total_ms: float,
        error_code: str,
        error_message: str,
    ) -> QueryRun:
        run = QueryRun(
            status=QueryRunStatus.FAILED,
            question=retrieval.query,
            answer=None,
            abstained=None,
            abstention_reason=None,
            generation_provider=self.generation.name,
            generation_model=self.generation.model,
            prompt_version=prompt.version,
            retrieval_top_k=retrieval.diagnostics.requested_top_k,
            retrieval_min_score=retrieval.diagnostics.min_score,
            retrieval_snapshot=retrieval_snapshot,
            citation_snapshot=[],
            retrieval_ms=retrieval.diagnostics.total_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            error_code=error_code,
            error_message=error_message,
        )
        await self.runs.add_and_commit(run)
        return run

    @staticmethod
    def _completed_response(
        run: QueryRun, citations: list[ValidatedCitation]
    ) -> GroundedQueryResponse:
        return GroundedQueryResponse(
            run_id=run.id,
            question=run.question,
            answer=run.answer,
            abstained=bool(run.abstained),
            abstention_reason=run.abstention_reason,
            citations=citations,
            diagnostics=GroundedAnswerService._diagnostics(run),
        )

    @staticmethod
    def _run_response(run: QueryRun) -> QueryRunResponse:
        return QueryRunResponse(
            run_id=run.id,
            status=run.status,
            question=run.question,
            answer=run.answer,
            abstained=run.abstained,
            abstention_reason=run.abstention_reason,
            citations=run.citation_snapshot,
            retrieval_snapshot=run.retrieval_snapshot,
            diagnostics=GroundedAnswerService._diagnostics(run),
            error_code=run.error_code,
            error_message=run.error_message,
            created_at=run.created_at,
        )

    @staticmethod
    def _diagnostics(run: QueryRun) -> GroundedQueryDiagnostics:
        return GroundedQueryDiagnostics(
            retrieved_count=len(run.retrieval_snapshot),
            retrieval_ms=run.retrieval_ms,
            generation_ms=run.generation_ms,
            total_ms=run.total_ms,
            provider=run.generation_provider,
            model=run.generation_model,
            prompt_version=run.prompt_version,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            total_tokens=run.total_tokens,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)
