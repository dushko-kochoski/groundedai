import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from app.db.models import Document, DocumentChunk


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document: Document
    chunk_count: int


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_checksum(self, checksum: str) -> Document | None:
        result = await self.session.execute(select(Document).where(Document.checksum == checksum))
        return result.scalar_one_or_none()

    async def add(self, document: Document) -> None:
        self.session.add(document)
        await self.session.flush()

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.session.add_all(chunks)

    async def get(self, document_id: uuid.UUID) -> DocumentRecord | None:
        statement = self._record_query().where(Document.id == document_id)
        row = (await self.session.execute(statement)).one_or_none()
        if row is None:
            return None
        return DocumentRecord(document=row[0], chunk_count=row[1])

    async def list(self, *, limit: int, offset: int) -> list[DocumentRecord]:
        statement = (
            self._record_query()
            .order_by(Document.created_at.desc(), Document.id.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = (await self.session.execute(statement)).all()
        return [DocumentRecord(document=row[0], chunk_count=row[1]) for row in rows]

    async def count(self) -> int:
        return int(await self.session.scalar(select(func.count(Document.id))) or 0)

    async def delete(self, document: Document) -> None:
        await self.session.delete(document)
        await self.session.flush()

    async def mark_evaluation_fixture(self, document: Document, fixture_id: str) -> None:
        document.metadata_ = {
            **document.metadata_,
            "evaluation_fixture": True,
            "evaluation_fixture_id": fixture_id,
        }
        await self.session.commit()
        await self.session.refresh(document)

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()

    @staticmethod
    def _record_query() -> Select[tuple[Document, int]]:
        chunk_count = (
            select(func.count(DocumentChunk.id))
            .where(DocumentChunk.document_id == Document.id)
            .correlate(Document)
            .scalar_subquery()
        )
        return select(Document, chunk_count.label("chunk_count"))
