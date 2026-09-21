import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import QueryRun


class QueryRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_and_commit(self, run: QueryRun) -> None:
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)

    async def get(self, run_id: uuid.UUID) -> QueryRun | None:
        result = await self.session.execute(select(QueryRun).where(QueryRun.id == run_id))
        return result.scalar_one_or_none()

    async def rollback(self) -> None:
        await self.session.rollback()
