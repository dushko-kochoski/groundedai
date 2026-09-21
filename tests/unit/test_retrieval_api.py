import uuid
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_retrieval_service
from app.main import create_app
from app.providers.embeddings import DeterministicEmbeddingProvider
from app.repositories.retrieval import RetrievalCandidate
from app.services.retrieval import RetrievalService


class ApiRetrievalRepository:
    def __init__(self) -> None:
        self.candidates = [
            self._candidate("refund evidence", 0.1, 0),
            self._candidate("backup evidence", 0.5, 1),
            self._candidate("weak evidence", 0.9, 2),
        ]

    async def search(
        self, *, query_embedding: list[float], top_k: int
    ) -> list[RetrievalCandidate]:
        assert query_embedding
        return self.candidates[:top_k]

    @staticmethod
    def _candidate(text: str, distance: float, index: int) -> RetrievalCandidate:
        return RetrievalCandidate(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename=f"source-{index}.txt",
            chunk_index=index,
            text=text,
            metadata={},
            cosine_distance=distance,
        )


@pytest.fixture
async def retrieval_client() -> AsyncIterator[AsyncClient]:
    app: FastAPI = create_app()
    service = RetrievalService(
        repository=ApiRetrievalRepository(),
        embeddings=DeterministicEmbeddingProvider(256),
        default_top_k=3,
        max_top_k=5,
        default_min_score=0.0,
        max_query_chars=20,
        max_returned_chars=1000,
    )
    app.dependency_overrides[get_retrieval_service] = lambda: service
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_valid_query_returns_typed_ordered_results(retrieval_client: AsyncClient) -> None:
    response = await retrieval_client.post(
        "/api/v1/retrieval/search", json={"query": " refund   policy "}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "refund policy"
    assert [item["score"] for item in body["results"]] == sorted(
        [item["score"] for item in body["results"]], reverse=True
    )
    assert "embedding" not in body["results"][0]
    assert "storage_key" not in body["results"][0]


@pytest.mark.parametrize("query", ["", "   "])
async def test_empty_query_returns_structured_error(
    retrieval_client: AsyncClient, query: str
) -> None:
    response = await retrieval_client.post("/api/v1/retrieval/search", json={"query": query})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_query"


async def test_oversized_query_returns_structured_error(retrieval_client: AsyncClient) -> None:
    response = await retrieval_client.post(
        "/api/v1/retrieval/search", json={"query": "x" * 21}
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "query_too_long"


async def test_custom_top_k_and_configured_maximum(retrieval_client: AsyncClient) -> None:
    limited = await retrieval_client.post(
        "/api/v1/retrieval/search",
        json={"query": "refund", "top_k": 1, "min_score": -1.0},
    )
    excessive = await retrieval_client.post(
        "/api/v1/retrieval/search", json={"query": "refund", "top_k": 6}
    )

    assert limited.status_code == 200
    assert len(limited.json()["results"]) == 1
    assert excessive.status_code == 400
    assert excessive.json()["error"]["code"] == "invalid_top_k"


async def test_min_score_filters_and_can_return_no_results(
    retrieval_client: AsyncClient,
) -> None:
    filtered = await retrieval_client.post(
        "/api/v1/retrieval/search", json={"query": "refund", "min_score": 0.6}
    )
    empty = await retrieval_client.post(
        "/api/v1/retrieval/search", json={"query": "refund", "min_score": 0.95}
    )

    assert [item["text"] for item in filtered.json()["results"]] == ["refund evidence"]
    assert empty.status_code == 200
    assert empty.json()["results"] == []
