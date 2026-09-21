from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.providers.embeddings import DeterministicEmbeddingProvider, OpenAIEmbeddingProvider
from app.providers.embeddings.base import EmbeddingProvider
from app.providers.generation import (
    DeterministicGenerationProvider,
    GenerationProvider,
    OpenAIGenerationProvider,
)
from app.repositories.documents import DocumentRepository
from app.repositories.query_runs import QueryRunRepository
from app.repositories.retrieval import RetrievalRepository
from app.services.generation import GroundedAnswerService, GroundedAnswerServiceProtocol
from app.services.ingestion.chunking import ParagraphChunker
from app.services.ingestion.parsers import DocumentParserRegistry
from app.services.ingestion.service import DocumentService, DocumentServiceProtocol
from app.services.retrieval import RetrievalService, RetrievalServiceProtocol
from app.services.storage import LocalDocumentStorage


def get_embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        if settings.embedding_api_key is None:
            raise RuntimeError("OpenAI embedding API key is not configured")
        return OpenAIEmbeddingProvider(
            api_key=settings.embedding_api_key.get_secret_value(),
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_retries=settings.embedding_max_retries,
            batch_size=settings.embedding_batch_size,
        )
    return DeterministicEmbeddingProvider(settings.embedding_dimension)


def get_generation_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> GenerationProvider:
    if settings.generation_provider == "openai":
        if settings.openai_api_key is None:
            raise RuntimeError("OpenAI generation API key is not configured")
        return OpenAIGenerationProvider(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.generation_model,
            timeout_seconds=settings.generation_timeout_seconds,
            max_retries=settings.generation_max_retries,
            max_output_tokens=settings.generation_max_output_tokens,
        )
    return DeterministicGenerationProvider()


def get_document_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    embeddings: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
) -> DocumentServiceProtocol:
    return DocumentService(
        repository=DocumentRepository(session),
        storage=LocalDocumentStorage(settings.upload_dir),
        chunker=ParagraphChunker(
            target_chars=settings.chunk_target_chars,
            overlap_chars=settings.chunk_overlap_chars,
        ),
        embeddings=embeddings,
        parsers=DocumentParserRegistry(),
        max_upload_bytes=settings.max_upload_bytes,
    )


def get_retrieval_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    embeddings: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
) -> RetrievalServiceProtocol:
    return RetrievalService(
        repository=RetrievalRepository(session),
        embeddings=embeddings,
        default_top_k=settings.retrieval_top_k,
        max_top_k=settings.retrieval_max_top_k,
        default_min_score=settings.retrieval_min_score,
        max_query_chars=settings.retrieval_max_query_chars,
        max_returned_chars=settings.retrieval_max_returned_chars,
    )


def get_grounded_answer_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    retrieval: Annotated[RetrievalServiceProtocol, Depends(get_retrieval_service)],
    generation: Annotated[GenerationProvider, Depends(get_generation_provider)],
) -> GroundedAnswerServiceProtocol:
    return GroundedAnswerService(
        retrieval=retrieval,
        generation=generation,
        runs=QueryRunRepository(session),
    )
