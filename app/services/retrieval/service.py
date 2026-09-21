import logging
import math
import time
import uuid
from typing import Protocol

from app.core.exceptions import ApplicationError
from app.providers.embeddings.base import EmbeddingProvider
from app.repositories.retrieval import RetrievalCandidate
from app.schemas.retrieval import (
    RetrievalDiagnostics,
    RetrievalResult,
    RetrievalSearchResponse,
)

logger = logging.getLogger(__name__)


class RetrievalRepositoryProtocol(Protocol):
    async def search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        embedding_version: str,
        document_ids: set[uuid.UUID] | None = None,
    ) -> list[RetrievalCandidate]: ...


class RetrievalServiceProtocol(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        document_ids: set[uuid.UUID] | None = None,
    ) -> RetrievalSearchResponse: ...


def cosine_distance_to_score(distance: float) -> float:
    """Convert pgvector cosine distance to a higher-is-more-relevant score."""
    if not math.isfinite(distance):
        raise ValueError("cosine distance must be finite")
    return max(-1.0, min(1.0, 1.0 - distance))


class RetrievalService:
    def __init__(
        self,
        *,
        repository: RetrievalRepositoryProtocol,
        embeddings: EmbeddingProvider,
        default_top_k: int,
        max_top_k: int,
        default_min_score: float,
        max_query_chars: int,
        max_returned_chars: int,
    ) -> None:
        self.repository = repository
        self.embeddings = embeddings
        self.default_top_k = default_top_k
        self.max_top_k = max_top_k
        self.default_min_score = default_min_score
        self.max_query_chars = max_query_chars
        self.max_returned_chars = max_returned_chars

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        document_ids: set[uuid.UUID] | None = None,
    ) -> RetrievalSearchResponse:
        total_started = time.perf_counter()
        normalized_query = self._validate_query(query)
        effective_top_k = self.default_top_k if top_k is None else top_k
        effective_min_score = self.default_min_score if min_score is None else min_score
        self._validate_options(effective_top_k, effective_min_score)

        embedding_started = time.perf_counter()
        query_embedding = await self.embeddings.embed_text(normalized_query)
        embedding_ms = self._elapsed_ms(embedding_started)

        search_started = time.perf_counter()
        candidates = await self.repository.search(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            embedding_provider=self.embeddings.name,
            embedding_model=self.embeddings.model,
            embedding_dimension=self.embeddings.dimension,
            embedding_version=self.embeddings.version,
            document_ids=document_ids,
        )
        search_ms = self._elapsed_ms(search_started)

        scored = sorted(
            (
                (candidate, cosine_distance_to_score(candidate.cosine_distance))
                for candidate in candidates
            ),
            key=lambda item: (-item[1], str(item[0].chunk_id)),
        )
        results = self._filter_and_bound(scored, effective_min_score)
        total_ms = self._elapsed_ms(total_started)
        returned_chars = sum(len(result.text) for result in results)
        diagnostics = RetrievalDiagnostics(
            provider=self.embeddings.name,
            model=self.embeddings.model,
            dimension=self.embeddings.dimension,
            version=self.embeddings.version,
            requested_top_k=effective_top_k,
            database_candidates_considered=len(candidates),
            returned_count=len(results),
            min_score=effective_min_score,
            max_returned_chars=self.max_returned_chars,
            returned_chars=returned_chars,
            embedding_ms=embedding_ms,
            search_ms=search_ms,
            total_ms=total_ms,
        )
        logger.info(
            "retrieval_completed",
            extra={
                "provider": diagnostics.provider,
                "query_chars": len(normalized_query),
                "top_k": effective_top_k,
                "min_score": effective_min_score,
                "database_candidates_considered": len(candidates),
                "returned_count": len(results),
                "returned_chars": returned_chars,
                "embedding_ms": embedding_ms,
                "search_ms": search_ms,
                "total_ms": total_ms,
            },
        )
        return RetrievalSearchResponse(
            query=normalized_query,
            results=results,
            diagnostics=diagnostics,
        )

    def _validate_query(self, query: str) -> str:
        if not query or not query.strip():
            raise ApplicationError(
                status_code=400,
                code="invalid_query",
                message="The retrieval query must not be empty.",
            )
        if len(query) > self.max_query_chars:
            raise ApplicationError(
                status_code=400,
                code="query_too_long",
                message="The retrieval query exceeds the configured character limit.",
                details={"max_query_chars": self.max_query_chars},
            )
        return " ".join(query.split())

    def _validate_options(self, top_k: int, min_score: float) -> None:
        if top_k < 1 or top_k > self.max_top_k:
            raise ApplicationError(
                status_code=400,
                code="invalid_top_k",
                message="top_k is outside the configured bounds.",
                details={"minimum": 1, "maximum": self.max_top_k},
            )
        if not math.isfinite(min_score) or not -1.0 <= min_score <= 1.0:
            raise ApplicationError(
                status_code=400,
                code="invalid_min_score",
                message="min_score must be between -1.0 and 1.0.",
            )

    def _filter_and_bound(
        self,
        scored: list[tuple[RetrievalCandidate, float]],
        min_score: float,
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        used_chars = 0
        for candidate, score in scored:
            if score < min_score:
                continue
            if used_chars + len(candidate.text) > self.max_returned_chars:
                break
            results.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    filename=candidate.filename,
                    chunk_index=candidate.chunk_index,
                    text=candidate.text,
                    score=score,
                    page_number=self._page_number(candidate.metadata),
                    section=self._section(candidate.metadata),
                )
            )
            used_chars += len(candidate.text)
        return results

    @staticmethod
    def _page_number(metadata: dict[str, object]) -> int | None:
        value = metadata.get("page_number")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _section(metadata: dict[str, object]) -> str | None:
        value = metadata.get("section")
        return value if isinstance(value, str) else None

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)
