import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.db.models import DocumentStatus


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    content_type: str
    checksum: str
    status: DocumentStatus
    size_bytes: int | None
    chunk_count: int
    metadata: dict[str, Any]
    created_at: datetime
    processed_at: datetime | None


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    total: int = Field(ge=0)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any]


class ErrorResponse(BaseModel):
    error: ErrorDetail


class DeleteResponse(BaseModel):
    status: Literal["deleted"]
    document_id: uuid.UUID

