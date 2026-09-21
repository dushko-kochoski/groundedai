import uuid

import pytest

from app.core.exceptions import ApplicationError
from app.providers.embeddings import DeterministicEmbeddingProvider
from app.repositories.retrieval import RetrievalCandidate
from app.services.retrieval.service import RetrievalService, cosine_distance_to_score


class FakeRetrievalRepository:
    def __init__(self, candidates: list[RetrievalCandidate]) -> None:
        self.candidates = candidates
        self.last_top_k: int | None = None

    async def search(
        self, *, query_embedding: list[float], top_k: int
    ) -> list[RetrievalCandidate]:
        assert len(query_embedding) == 256
        self.last_top_k = top_k
        return self.candidates[:top_k]


def candidate(*, distance: float, text: str, index: int = 0) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=f"document-{index}.txt",
        chunk_index=index,
        text=text,
        metadata={"page_number": None, "section": "Policy"},
        cosine_distance=distance,
    )


def service(
    repository: FakeRetrievalRepository,
    *,
    max_query_chars: int = 100,
    max_returned_chars: int = 1000,
) -> RetrievalService:
    return RetrievalService(
        repository=repository,
        embeddings=DeterministicEmbeddingProvider(256),
        default_top_k=5,
        max_top_k=10,
        default_min_score=0.15,
        max_query_chars=max_query_chars,
        max_returned_chars=max_returned_chars,
    )


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(0.0, 1.0), (0.25, 0.75), (1.0, 0.0), (2.0, -1.0), (-0.0001, 1.0)],
)
def test_cosine_distance_to_score(distance: float, expected: float) -> None:
    assert cosine_distance_to_score(distance) == expected


def test_cosine_distance_to_score_rejects_non_finite_value() -> None:
    with pytest.raises(ValueError):
        cosine_distance_to_score(float("nan"))


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
async def test_empty_query_is_rejected(query: str) -> None:
    with pytest.raises(ApplicationError) as error:
        await service(FakeRetrievalRepository([])).retrieve(query)

    assert error.value.code == "invalid_query"


async def test_oversized_query_is_rejected_without_truncation() -> None:
    with pytest.raises(ApplicationError) as error:
        await service(FakeRetrievalRepository([]), max_query_chars=5).retrieve("123456")

    assert error.value.code == "query_too_long"


async def test_defaults_normalization_ordering_and_metadata() -> None:
    repository = FakeRetrievalRepository(
        [
            candidate(distance=0.8, text="weak", index=2),
            candidate(distance=0.1, text="strong", index=0),
            candidate(distance=0.4, text="medium", index=1),
        ]
    )

    response = await service(repository).retrieve("  refund\n purchase  ")

    assert response.query == "refund purchase"
    assert repository.last_top_k == 5
    assert [result.text for result in response.results] == ["strong", "medium", "weak"]
    assert [result.score for result in response.results] == sorted(
        [result.score for result in response.results], reverse=True
    )
    assert response.results[0].section == "Policy"
    assert response.diagnostics.database_candidates_considered == 3
    assert response.diagnostics.provider == "deterministic-feature-hashing"


async def test_top_k_and_min_score_are_validated() -> None:
    retrieval = service(FakeRetrievalRepository([]))

    with pytest.raises(ApplicationError, match="top_k") as top_k_error:
        await retrieval.retrieve("query", top_k=11)
    with pytest.raises(ApplicationError, match="min_score") as score_error:
        await retrieval.retrieve("query", min_score=1.1)

    assert top_k_error.value.code == "invalid_top_k"
    assert score_error.value.code == "invalid_min_score"


async def test_threshold_filters_results_and_no_match_is_successful() -> None:
    repository = FakeRetrievalRepository(
        [candidate(distance=0.1, text="strong"), candidate(distance=0.7, text="weak")]
    )

    response = await service(repository).retrieve("refund", min_score=0.5)
    no_match = await service(repository).retrieve("refund", min_score=0.95)

    assert [result.text for result in response.results] == ["strong"]
    assert no_match.results == []
    assert no_match.diagnostics.returned_count == 0


async def test_character_budget_keeps_only_complete_relevance_prefix() -> None:
    repository = FakeRetrievalRepository(
        [candidate(distance=0.1, text="123456"), candidate(distance=0.2, text="abcdef")]
    )

    response = await service(repository, max_returned_chars=10).retrieve(
        "query", min_score=-1.0
    )

    assert [result.text for result in response.results] == ["123456"]
    assert response.diagnostics.returned_chars == 6


async def test_retrieval_results_are_deterministic() -> None:
    candidates = [candidate(distance=0.2, text="same input")]
    retrieval = service(FakeRetrievalRepository(candidates))

    first = await retrieval.retrieve("repeatable query")
    second = await retrieval.retrieve("repeatable query")

    assert first.results == second.results

