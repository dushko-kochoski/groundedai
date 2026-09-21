import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.core.exceptions import ApplicationError
from app.db.models import Document
from app.repositories.documents import DocumentRecord
from app.services.ingestion.chunking import ParagraphChunker
from app.services.ingestion.parsers import DocumentParserRegistry
from app.services.ingestion.service import DocumentService
from app.services.storage import LocalDocumentStorage


class FakeRepository:
    async def get_by_checksum(self, checksum: str) -> Document | None:
        return None

    async def add(self, document: Document) -> None:
        return None

    def add_chunks(self, chunks: list[object]) -> None:
        return None

    async def rollback(self) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def get(self, document_id: object) -> DocumentRecord | None:
        return None


class WrongCountEmbeddingProvider:
    name = "test"
    model = "test-model"
    version = "1"
    dimension = 1536

    async def embed_text(self, text: str) -> list[float]:
        return [0.0] * self.dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return []


async def test_storage_is_cleaned_when_embedding_fails(tmp_path: Path) -> None:
    service = DocumentService(
        repository=FakeRepository(),  # type: ignore[arg-type]
        storage=LocalDocumentStorage(tmp_path),
        chunker=ParagraphChunker(target_chars=100, overlap_chars=10),
        embeddings=WrongCountEmbeddingProvider(),
        parsers=DocumentParserRegistry(),
        max_upload_bytes=1024,
    )
    upload = UploadFile(
        BytesIO(b"usable text"),
        filename="failure.txt",
        headers=Headers({"content-type": "text/plain"}),
    )

    with pytest.raises(ApplicationError, match="could not be processed"):
        await service.ingest(upload)

    assert await asyncio.to_thread(lambda: list(tmp_path.iterdir())) == []
