import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI, UploadFile
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_document_service
from app.core.exceptions import ApplicationError
from app.db.models import DocumentStatus
from app.main import create_app
from app.schemas.documents import DocumentListResponse, DocumentResponse


class FakeDocumentService:
    def __init__(self) -> None:
        self.document = make_document()
        self.deleted = False

    async def ingest(self, upload: UploadFile) -> DocumentResponse:
        content = await upload.read()
        if upload.filename == "unsupported.pdf":
            raise ApplicationError(
                status_code=415,
                code="unsupported_file_type",
                message="Only text is supported.",
            )
        if len(content) > 5:
            raise ApplicationError(
                status_code=413,
                code="upload_too_large",
                message="Too large.",
            )
        if content == b"dupe":
            raise ApplicationError(
                status_code=409,
                code="duplicate_document",
                message="Already exists.",
                details={"document_id": str(self.document.id)},
            )
        return self.document.model_copy(update={"filename": upload.filename or "unknown.txt"})

    async def list_documents(self, *, limit: int, offset: int) -> DocumentListResponse:
        items = [] if self.deleted else [self.document]
        return DocumentListResponse(items=items, limit=limit, offset=offset, total=len(items))

    async def get_document(self, document_id: uuid.UUID) -> DocumentResponse:
        if self.deleted or document_id != self.document.id:
            raise not_found(document_id)
        return self.document

    async def delete_document(self, document_id: uuid.UUID) -> None:
        if document_id != self.document.id:
            raise not_found(document_id)
        self.deleted = True


def make_document() -> DocumentResponse:
    now = datetime.now(UTC)
    return DocumentResponse(
        id=uuid.uuid4(),
        filename="notes.txt",
        content_type="text/plain",
        checksum="a" * 64,
        status=DocumentStatus.READY,
        size_bytes=4,
        chunk_count=1,
        metadata={},
        created_at=now,
        processed_at=now,
    )


def not_found(document_id: uuid.UUID) -> ApplicationError:
    return ApplicationError(
        status_code=404,
        code="document_not_found",
        message="Missing.",
        details={"document_id": str(document_id)},
    )


@pytest.fixture
async def documents_client() -> AsyncIterator[tuple[AsyncClient, FakeDocumentService]]:
    app: FastAPI = create_app()
    service = FakeDocumentService()
    app.dependency_overrides[get_document_service] = lambda: service
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, service


@pytest.mark.parametrize("filename", ["notes.txt", "README.md"])
async def test_upload_returns_201(
    documents_client: tuple[AsyncClient, FakeDocumentService], filename: str
) -> None:
    client, _ = documents_client

    response = await client.post(
        "/api/v1/documents", files={"file": (filename, b"text", "text/plain")}
    )

    assert response.status_code == 201
    assert response.json()["filename"] == filename


@pytest.mark.parametrize(
    ("filename", "content", "expected_status", "expected_code"),
    [
        ("unsupported.pdf", b"x", 415, "unsupported_file_type"),
        ("large.txt", b"123456", 413, "upload_too_large"),
        ("duplicate.txt", b"dupe", 409, "duplicate_document"),
    ],
)
async def test_upload_errors_are_structured(
    documents_client: tuple[AsyncClient, FakeDocumentService],
    filename: str,
    content: bytes,
    expected_status: int,
    expected_code: str,
) -> None:
    client, _ = documents_client

    response = await client.post(
        "/api/v1/documents", files={"file": (filename, content, "text/plain")}
    )

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code


async def test_list_detail_delete_and_missing_document(
    documents_client: tuple[AsyncClient, FakeDocumentService],
) -> None:
    client, service = documents_client

    listing = await client.get("/api/v1/documents?limit=10&offset=0")
    detail = await client.get(f"/api/v1/documents/{service.document.id}")
    deleted = await client.delete(f"/api/v1/documents/{service.document.id}")
    missing = await client.get(f"/api/v1/documents/{service.document.id}")

    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert detail.status_code == 200
    assert "embedding" not in detail.json()
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "document_not_found"
