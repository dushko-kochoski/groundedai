import uuid
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_evaluation_service
from app.schemas.documents import ErrorResponse
from app.schemas.evaluation import (
    EvaluationCaseListResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
)
from app.services.evaluation import EvaluationServiceProtocol

router = APIRouter(prefix="/api/v1/evaluations", tags=["evaluations"])


@router.post(
    "/run",
    response_model=EvaluationRunResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def run_evaluation(
    request: EvaluationRunRequest,
    service: Annotated[EvaluationServiceProtocol, Depends(get_evaluation_service)],
) -> EvaluationRunResponse:
    return await service.run(
        request.dataset,
        request.mode,
        top_k=request.top_k,
        min_score=request.min_score,
        evidence_min_score=request.evidence_min_score,
    )


@router.get(
    "/runs/{run_id}",
    response_model=EvaluationRunResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_evaluation_run(
    run_id: uuid.UUID,
    service: Annotated[EvaluationServiceProtocol, Depends(get_evaluation_service)],
) -> EvaluationRunResponse:
    return await service.get_run(run_id)


@router.get(
    "/runs/{run_id}/cases",
    response_model=EvaluationCaseListResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_evaluation_cases(
    run_id: uuid.UUID,
    service: Annotated[EvaluationServiceProtocol, Depends(get_evaluation_service)],
) -> EvaluationCaseListResponse:
    return await service.list_cases(run_id)
