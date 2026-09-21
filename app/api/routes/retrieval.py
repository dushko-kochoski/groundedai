from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_retrieval_service
from app.schemas.documents import ErrorResponse
from app.schemas.retrieval import RetrievalSearchRequest, RetrievalSearchResponse
from app.services.retrieval import RetrievalServiceProtocol

router = APIRouter(prefix="/api/v1/retrieval", tags=["retrieval"])


@router.post(
    "/search",
    response_model=RetrievalSearchResponse,
    responses={400: {"model": ErrorResponse}},
)
async def search_documents(
    request: RetrievalSearchRequest,
    service: Annotated[RetrievalServiceProtocol, Depends(get_retrieval_service)],
) -> RetrievalSearchResponse:
    return await service.retrieve(
        request.query,
        top_k=request.top_k,
        min_score=request.min_score,
    )
