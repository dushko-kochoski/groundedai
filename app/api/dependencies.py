from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.providers.embeddings import DeterministicEmbeddingProvider
from app.repositories.documents import DocumentRepository
from app.services.ingestion.chunking import ParagraphChunker
from app.services.ingestion.service import DocumentService, DocumentServiceProtocol
from app.services.storage import LocalDocumentStorage


def get_document_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentServiceProtocol:
    return DocumentService(
        repository=DocumentRepository(session),
        storage=LocalDocumentStorage(settings.upload_dir),
        chunker=ParagraphChunker(
            target_chars=settings.chunk_target_chars,
            overlap_chars=settings.chunk_overlap_chars,
        ),
        embeddings=DeterministicEmbeddingProvider(settings.embedding_dimension),
        max_upload_bytes=settings.max_upload_bytes,
    )

