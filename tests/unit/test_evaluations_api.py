import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_evaluation_service
from app.core.exceptions import ApplicationError
from app.db.models import EvaluationMode, EvaluationRunStatus
from app.main import create_app
from app.schemas.evaluation import (
    EvaluationCaseListResponse,
    EvaluationRunResponse,
)


class FakeEvaluationService:
    def __init__(self) -> None:
        self.run_id = uuid.uuid4()

    async def run(
        self,
        dataset_name: str,
        mode: EvaluationMode,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        evidence_min_score: float | None = None,
    ) -> EvaluationRunResponse:
        if dataset_name != "golden_v1":
            raise ApplicationError(
                status_code=404,
                code="evaluation_dataset_not_found",
                message="Missing.",
            )
        return self.response(mode, top_k or 5, min_score if min_score is not None else 0.15)

    async def get_run(self, run_id: uuid.UUID) -> EvaluationRunResponse:
        if run_id != self.run_id:
            raise ApplicationError(
                status_code=404, code="evaluation_run_not_found", message="Missing."
            )
        return self.response(EvaluationMode.END_TO_END, 5, 0.15)

    async def list_cases(self, run_id: uuid.UUID) -> EvaluationCaseListResponse:
        if run_id != self.run_id:
            raise ApplicationError(
                status_code=404, code="evaluation_run_not_found", message="Missing."
            )
        return EvaluationCaseListResponse(run_id=run_id, items=[])

    def response(self, mode: EvaluationMode, top_k: int, min_score: float) -> EvaluationRunResponse:
        now = datetime.now(UTC)
        return EvaluationRunResponse(
            run_id=self.run_id,
            dataset_name="golden_v1",
            dataset_version="1.0.0",
            mode=mode,
            status=EvaluationRunStatus.COMPLETED,
            retrieval_top_k=top_k,
            retrieval_min_score=min_score,
            embedding_provider="deterministic-feature-hashing",
            embedding_model="feature-hashing-v1",
            embedding_version="1",
            generation_provider="deterministic" if mode == EvaluationMode.END_TO_END else None,
            generation_model="extractive-v1" if mode == EvaluationMode.END_TO_END else None,
            prompt_version="grounded_answer_v1" if mode == EvaluationMode.END_TO_END else None,
            evidence_policy_version="evidence_policy_v1",
            evidence_min_score=0.26,
            case_count=30,
            aggregate_metrics={"retrieval": {"hit_at_k": 1.0}},
            started_at=now,
            completed_at=now,
            duration_ms=10.0,
            error_category=None,
            error_message=None,
            created_at=now,
        )


@pytest.fixture
async def evaluation_client() -> AsyncIterator[tuple[AsyncClient, FakeEvaluationService]]:
    app: FastAPI = create_app()
    service = FakeEvaluationService()
    app.dependency_overrides[get_evaluation_service] = lambda: service
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, service


async def test_run_and_fetch_evaluation(
    evaluation_client: tuple[AsyncClient, FakeEvaluationService],
) -> None:
    client, service = evaluation_client
    created = await client.post(
        "/api/v1/evaluations/run",
        json={"dataset": "golden_v1", "mode": "retrieval_only", "top_k": 3},
    )
    fetched = await client.get(f"/api/v1/evaluations/runs/{service.run_id}")
    cases = await client.get(f"/api/v1/evaluations/runs/{service.run_id}/cases")

    assert created.status_code == 200
    assert created.json()["mode"] == "retrieval_only"
    assert created.json()["retrieval_top_k"] == 3
    assert fetched.status_code == 200
    assert cases.status_code == 200 and cases.json()["items"] == []


async def test_unknown_dataset_and_run_are_structured_errors(
    evaluation_client: tuple[AsyncClient, FakeEvaluationService],
) -> None:
    client, _ = evaluation_client
    unknown_dataset = await client.post(
        "/api/v1/evaluations/run", json={"dataset": "unknown", "mode": "end_to_end"}
    )
    unknown_run = await client.get(f"/api/v1/evaluations/runs/{uuid.uuid4()}")

    assert unknown_dataset.status_code == 404
    assert unknown_dataset.json()["error"]["code"] == "evaluation_dataset_not_found"
    assert unknown_run.status_code == 404


async def test_invalid_mode_is_rejected_by_schema(
    evaluation_client: tuple[AsyncClient, FakeEvaluationService],
) -> None:
    client, _ = evaluation_client
    response = await client.post(
        "/api/v1/evaluations/run", json={"dataset": "golden_v1", "mode": "invalid"}
    )

    assert response.status_code == 422
