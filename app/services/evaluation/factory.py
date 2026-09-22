from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.providers.embeddings import DeterministicEmbeddingProvider, OpenAIEmbeddingProvider
from app.providers.embeddings.base import EmbeddingProvider
from app.providers.generation import (
    DeterministicGenerationProvider,
    GenerationProvider,
    OpenAIGenerationProvider,
)
from app.repositories.documents import DocumentRepository
from app.repositories.evaluations import EvaluationRepository
from app.repositories.query_runs import QueryRunRepository
from app.repositories.retrieval import RetrievalRepository
from app.services.evaluation.datasets import DatasetRegistry
from app.services.evaluation.fixtures import EvaluationFixturePreparer
from app.services.evaluation.service import EvaluationService
from app.services.generation import GroundedAnswerService
from app.services.generation.evidence import build_evidence_policy
from app.services.ingestion.chunking import ParagraphChunker
from app.services.ingestion.parsers import DocumentParserRegistry
from app.services.ingestion.service import DocumentService
from app.services.retrieval import RetrievalService
from app.services.storage import LocalDocumentStorage


def build_evaluation_service(session: AsyncSession, settings: Settings) -> EvaluationService:
    embeddings = _embeddings(settings)
    generation = _generation(settings)
    evidence_policy = build_evidence_policy(
        embedding_provider=embeddings.name,
        embedding_model=embeddings.model,
        embedding_version=embeddings.version,
        deterministic_minimum_score=settings.evidence_deterministic_min_score,
        openai_minimum_score=settings.evidence_openai_min_score,
    )
    documents_repository = DocumentRepository(session)
    document_service = DocumentService(
        repository=documents_repository,
        storage=LocalDocumentStorage(settings.upload_dir),
        chunker=ParagraphChunker(
            target_chars=settings.chunk_target_chars,
            overlap_chars=settings.chunk_overlap_chars,
        ),
        embeddings=embeddings,
        parsers=DocumentParserRegistry(),
        max_upload_bytes=settings.max_upload_bytes,
    )
    retrieval = RetrievalService(
        repository=RetrievalRepository(session),
        embeddings=embeddings,
        default_top_k=settings.retrieval_top_k,
        max_top_k=settings.retrieval_max_top_k,
        default_min_score=settings.retrieval_min_score,
        max_query_chars=settings.retrieval_max_query_chars,
        max_returned_chars=settings.retrieval_max_returned_chars,
    )
    grounded = GroundedAnswerService(
        retrieval=retrieval,
        generation=generation,
        runs=QueryRunRepository(session),
        evidence_policy=evidence_policy,
    )
    registry = DatasetRegistry(settings.evaluation_root)
    return EvaluationService(
        registry=registry,
        fixtures=EvaluationFixturePreparer(
            registry=registry,
            documents=documents_repository,
            ingestion=document_service,
        ),
        retrieval=retrieval,
        grounded_answers=grounded,
        embeddings=embeddings,
        generation=generation,
        repository=EvaluationRepository(session),
        default_top_k=settings.retrieval_top_k,
        default_min_score=settings.retrieval_min_score,
        evidence_policy_version=evidence_policy.version,
        default_evidence_min_score=evidence_policy.minimum_score,
    )


def _embeddings(settings: Settings) -> EmbeddingProvider:
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


def _generation(settings: Settings) -> GenerationProvider:
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
