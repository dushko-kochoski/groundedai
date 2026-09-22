import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_grounded_answer_service
from app.core.exceptions import ApplicationError
from app.db.models import QueryRunStatus
from app.main import create_app
from app.schemas.query import (
    GroundedQueryDiagnostics,
    GroundedQueryResponse,
    QueryRunResponse,
    ValidatedCitation,
)


class FakeGroundedService:
    def __init__(self) -> None:
        self.run_id = uuid.uuid4()
        self.document_id = uuid.uuid4()
        self.chunk_id = uuid.uuid4()

    async def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> GroundedQueryResponse:
        if not question.strip():
            raise ApplicationError(status_code=400, code="invalid_query", message="Empty.")
        if question in {"provider failure", "invalid output"}:
            code = (
                "provider_unavailable"
                if question == "provider failure"
                else "malformed_provider_output"
            )
            raise ApplicationError(
                status_code=502,
                code=code,
                message="Unavailable.",
                details={"run_id": str(self.run_id)},
            )
        abstained = question in {"absent", "provider abstains"}
        return GroundedQueryResponse(
            run_id=self.run_id,
            question=question,
            answer=None if abstained else "Remote work is allowed twice weekly.",
            abstained=abstained,
            abstention_reason="No evidence." if abstained else None,
            citations=[]
            if abstained
            else [
                ValidatedCitation(
                    source_id="S1",
                    document_id=self.document_id,
                    chunk_id=self.chunk_id,
                    filename="handbook.pdf",
                    chunk_index=0,
                    page_number=4,
                    section=None,
                    score=0.8,
                )
            ],
            diagnostics=self.diagnostics(retrieved_count=0 if question == "absent" else 1),
        )

    async def get_run(self, run_id: uuid.UUID) -> QueryRunResponse:
        if run_id != self.run_id:
            raise ApplicationError(
                status_code=404,
                code="query_run_not_found",
                message="Missing.",
            )
        return QueryRunResponse(
            run_id=self.run_id,
            status=QueryRunStatus.COMPLETED,
            question="What is the policy?",
            answer="Remote work is allowed twice weekly.",
            abstained=False,
            abstention_reason=None,
            citations=[{"source_id": "S1", "chunk_id": str(self.chunk_id)}],
            retrieval_snapshot=[{"source_id": "S1", "chunk_id": str(self.chunk_id)}],
            diagnostics=self.diagnostics(1),
            error_code=None,
            error_message=None,
            created_at=datetime.now(UTC),
        )

    @staticmethod
    def diagnostics(retrieved_count: int) -> GroundedQueryDiagnostics:
        return GroundedQueryDiagnostics(
            retrieved_count=retrieved_count,
            retrieval_ms=1,
            generation_ms=2,
            total_ms=3,
            provider="deterministic",
            model="extractive-v1",
            prompt_version="grounded_answer_v1",
            evidence_policy_version="evidence_policy_v1",
            evidence_min_score=0.26,
            top_evidence_score=0.8 if retrieved_count else None,
            evidence_sufficient=bool(retrieved_count),
            abstention_category="no_retrieval_evidence" if not retrieved_count else None,
        )


@pytest.fixture
async def query_client() -> AsyncIterator[tuple[AsyncClient, FakeGroundedService]]:
    app: FastAPI = create_app()
    service = FakeGroundedService()
    app.dependency_overrides[get_grounded_answer_service] = lambda: service
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, service


async def test_supported_query_and_query_run_are_typed(
    query_client: tuple[AsyncClient, FakeGroundedService],
) -> None:
    client, service = query_client

    response = await client.post("/api/v1/query", json={"question": "What is the policy?"})
    run = await client.get(f"/api/v1/query-runs/{service.run_id}")

    assert response.status_code == 200
    assert response.json()["citations"][0]["source_id"] == "S1"
    assert response.json()["diagnostics"]["prompt_version"] == "grounded_answer_v1"
    assert run.status_code == 200
    assert run.json()["status"] == "completed"


@pytest.mark.parametrize("question", ["absent", "provider abstains"])
async def test_abstention_responses(
    query_client: tuple[AsyncClient, FakeGroundedService], question: str
) -> None:
    client, _ = query_client
    response = await client.post("/api/v1/query", json={"question": question})

    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["citations"] == []


@pytest.mark.parametrize(
    ("question", "status", "code"),
    [
        ("   ", 400, "invalid_query"),
        ("provider failure", 502, "provider_unavailable"),
        ("invalid output", 502, "malformed_provider_output"),
    ],
)
async def test_query_errors_are_structured(
    query_client: tuple[AsyncClient, FakeGroundedService],
    question: str,
    status: int,
    code: str,
) -> None:
    client, _ = query_client
    response = await client.post("/api/v1/query", json={"question": question})

    assert response.status_code == status
    assert response.json()["error"]["code"] == code


async def test_unknown_run_returns_404(
    query_client: tuple[AsyncClient, FakeGroundedService],
) -> None:
    client, _ = query_client
    response = await client.get(f"/api/v1/query-runs/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "query_run_not_found"


async def test_missing_question_is_rejected_by_request_schema(
    query_client: tuple[AsyncClient, FakeGroundedService],
) -> None:
    client, _ = query_client
    response = await client.post("/api/v1/query", json={})

    assert response.status_code == 422
