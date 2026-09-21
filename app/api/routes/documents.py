import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from app.api.dependencies import get_document_service
from app.schemas.documents import (
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    ErrorResponse,
)
from app.services.ingestion.service import DocumentServiceProtocol

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses=ERROR_RESPONSES,
)
async def upload_document(
    service: Annotated[DocumentServiceProtocol, Depends(get_document_service)],
    file: Annotated[UploadFile, File(description="UTF-8 TXT or Markdown document")],
) -> DocumentResponse:
    try:
        return await service.ingest(file)
    finally:
        await file.close()


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    service: Annotated[DocumentServiceProtocol, Depends(get_document_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    return await service.list_documents(limit=limit, offset=offset)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_document(
    document_id: uuid.UUID,
    service: Annotated[DocumentServiceProtocol, Depends(get_document_service)],
) -> DocumentResponse:
    return await service.get_document(document_id)


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}},
)
async def delete_document(
    document_id: uuid.UUID,
    service: Annotated[DocumentServiceProtocol, Depends(get_document_service)],
) -> DeleteResponse:
    await service.delete_document(document_id)
    return DeleteResponse(status="deleted", document_id=document_id)
