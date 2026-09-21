import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Protocol

from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import ApplicationError
from app.db.models import Document, DocumentChunk, DocumentStatus
from app.providers.embeddings.base import EmbeddingProvider
from app.repositories.documents import DocumentRecord, DocumentRepository
from app.schemas.documents import DocumentListResponse, DocumentResponse
from app.services.ingestion.chunking import ParagraphChunker
from app.services.ingestion.validation import (
    decode_and_normalize_text,
    read_upload_bounded,
    validate_upload_metadata,
)
from app.services.storage import LocalDocumentStorage

logger = logging.getLogger(__name__)


class DocumentServiceProtocol(Protocol):
    async def ingest(self, upload: UploadFile) -> DocumentResponse: ...

    async def list_documents(self, *, limit: int, offset: int) -> DocumentListResponse: ...

    async def get_document(self, document_id: uuid.UUID) -> DocumentResponse: ...

    async def delete_document(self, document_id: uuid.UUID) -> None: ...


class DocumentService:
    def __init__(
        self,
        *,
        repository: DocumentRepository,
        storage: LocalDocumentStorage,
        chunker: ParagraphChunker,
        embeddings: EmbeddingProvider,
        max_upload_bytes: int,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.chunker = chunker
        self.embeddings = embeddings
        self.max_upload_bytes = max_upload_bytes

    async def ingest(self, upload: UploadFile) -> DocumentResponse:
        metadata = validate_upload_metadata(upload.filename, upload.content_type)
        payload = await read_upload_bounded(upload, self.max_upload_bytes)

        existing = await self.repository.get_by_checksum(payload.checksum)
        if existing is not None:
            self._raise_duplicate(existing)

        text = decode_and_normalize_text(payload.content)
        chunks = self.chunker.chunk(text)
        document_id = uuid.uuid4()
        document = Document(
            id=document_id,
            filename=metadata.filename,
            content_type=metadata.content_type,
            checksum=payload.checksum,
            status=DocumentStatus.PROCESSING,
            size_bytes=len(payload.content),
            metadata_={"source_extension": metadata.extension},
        )
        storage_key: str | None = None

        try:
            await self.repository.add(document)
        except IntegrityError:
            await self.repository.rollback()
            duplicate = await self.repository.get_by_checksum(payload.checksum)
            if duplicate is not None:
                self._raise_duplicate(duplicate)
            raise

        try:
            storage_key = await asyncio.to_thread(
                self.storage.write,
                document_id,
                metadata.extension,
                payload.content,
            )
            document.storage_key = storage_key
            vectors = await self.embeddings.embed_batch([chunk.text for chunk in chunks])
            self.repository.add_chunks(
                [
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk.index,
                        text=chunk.text,
                        char_count=chunk.char_count,
                        embedding=vector,
                        metadata_={},
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ]
            )
            document.status = DocumentStatus.READY
            document.processed_at = datetime.now(UTC)
            await self.repository.commit()
        except Exception as exc:
            await self.repository.rollback()
            if storage_key is not None:
                try:
                    await asyncio.to_thread(self.storage.delete, storage_key)
                except OSError:
                    logger.exception(
                        "ingestion_file_cleanup_failed",
                        extra={"document_id": str(document_id), "storage_key": storage_key},
                    )
            raise ApplicationError(
                status_code=500,
                code="document_ingestion_failed",
                message="The document could not be processed safely.",
                details={"document_id": str(document_id)},
            ) from exc

        record = await self.repository.get(document_id)
        if record is None:
            raise RuntimeError("persisted document could not be reloaded")
        return self._to_response(record)

    async def list_documents(self, *, limit: int, offset: int) -> DocumentListResponse:
        records = await self.repository.list(limit=limit, offset=offset)
        return DocumentListResponse(
            items=[self._to_response(record) for record in records],
            limit=limit,
            offset=offset,
            total=await self.repository.count(),
        )

    async def get_document(self, document_id: uuid.UUID) -> DocumentResponse:
        record = await self.repository.get(document_id)
        if record is None:
            raise ApplicationError(
                status_code=404,
                code="document_not_found",
                message="The requested document does not exist.",
                details={"document_id": str(document_id)},
            )
        return self._to_response(record)

    async def delete_document(self, document_id: uuid.UUID) -> None:
        record = await self.repository.get(document_id)
        if record is None:
            raise ApplicationError(
                status_code=404,
                code="document_not_found",
                message="The requested document does not exist.",
                details={"document_id": str(document_id)},
            )

        await self.repository.delete(record.document)
        try:
            if record.document.storage_key is not None:
                await asyncio.to_thread(self.storage.delete, record.document.storage_key)
            await self.repository.commit()
        except Exception:
            await self.repository.rollback()
            raise

    @staticmethod
    def _raise_duplicate(document: Document) -> None:
        raise ApplicationError(
            status_code=409,
            code="duplicate_document",
            message="A document with the same content already exists.",
            details={"document_id": str(document.id), "checksum": document.checksum},
        )

    @staticmethod
    def _to_response(record: DocumentRecord) -> DocumentResponse:
        document = record.document
        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            content_type=document.content_type,
            checksum=document.checksum,
            status=document.status,
            size_bytes=document.size_bytes,
            chunk_count=record.chunk_count,
            metadata=document.metadata_,
            created_at=document.created_at,
            processed_at=document.processed_at,
        )
