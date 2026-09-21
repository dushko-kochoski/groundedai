from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from app.db.health import DatabaseReadiness, check_database_readiness
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report process liveness without relying on external services."""
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
async def ready(
    response: Response,
    readiness: Annotated[DatabaseReadiness, Depends(check_database_readiness)],
) -> ReadinessResponse:
    """Report whether the database and pgvector dependency are usable."""
    if not readiness.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if readiness.ready else "not_ready",
        database=readiness.database,
        pgvector=readiness.pgvector,
    )

