import uuid
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_grounded_answer_service
from app.schemas.documents import ErrorResponse
from app.schemas.query import GroundedQueryRequest, GroundedQueryResponse, QueryRunResponse
from app.services.generation import GroundedAnswerServiceProtocol

router = APIRouter(prefix="/api/v1", tags=["grounded-query"])


@router.post(
    "/query",
    response_model=GroundedQueryResponse,
    responses={
        400: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def query_knowledge_base(
    request: GroundedQueryRequest,
    service: Annotated[GroundedAnswerServiceProtocol, Depends(get_grounded_answer_service)],
) -> GroundedQueryResponse:
    return await service.query(
        request.question,
        top_k=request.top_k,
        min_score=request.min_score,
    )


@router.get(
    "/query-runs/{run_id}",
    response_model=QueryRunResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_query_run(
    run_id: uuid.UUID,
    service: Annotated[GroundedAnswerServiceProtocol, Depends(get_grounded_answer_service)],
) -> QueryRunResponse:
    return await service.get_run(run_id)
