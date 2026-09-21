import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk, DocumentStatus


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    text: str
    metadata: dict[str, Any]
    cosine_distance: float


class RetrievalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        embedding_version: str,
        document_ids: set[uuid.UUID] | None = None,
    ) -> list[RetrievalCandidate]:
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        statement = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                Document.filename,
                DocumentChunk.chunk_index,
                DocumentChunk.text,
                DocumentChunk.metadata_.label("chunk_metadata"),
                distance.label("cosine_distance"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                Document.status == DocumentStatus.READY,
                DocumentChunk.embedding_provider == embedding_provider,
                DocumentChunk.embedding_model == embedding_model,
                DocumentChunk.embedding_dimension == embedding_dimension,
                DocumentChunk.embedding_version == embedding_version,
            )
            .order_by(distance.asc(), DocumentChunk.id.asc())
            .limit(top_k)
        )
        if document_ids is not None:
            statement = statement.where(Document.id.in_(document_ids))
        else:
            statement = statement.where(
                ~Document.metadata_.contains({"evaluation_fixture": True})
            )
        rows = (await self.session.execute(statement)).all()
        return [
            RetrievalCandidate(
                chunk_id=row.id,
                document_id=row.document_id,
                filename=row.filename,
                chunk_index=row.chunk_index,
                text=row.text,
                metadata=row.chunk_metadata,
                cosine_distance=float(row.cosine_distance),
            )
            for row in rows
        ]
