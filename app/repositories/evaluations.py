import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import EvaluationCaseResult, EvaluationRun


class EvaluationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_run_and_commit(self, run: EvaluationRun) -> None:
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)

    async def complete_run(self, run: EvaluationRun, cases: list[EvaluationCaseResult]) -> None:
        self.session.add_all(cases)
        await self.session.commit()
        await self.session.refresh(run)

    async def get_run(self, run_id: uuid.UUID) -> EvaluationRun | None:
        result = await self.session.execute(select(EvaluationRun).where(EvaluationRun.id == run_id))
        return result.scalar_one_or_none()

    async def list_cases(self, run_id: uuid.UUID) -> list[EvaluationCaseResult]:
        result = await self.session.execute(
            select(EvaluationCaseResult)
            .where(EvaluationCaseResult.evaluation_run_id == run_id)
            .order_by(EvaluationCaseResult.case_id)
        )
        return list(result.scalars())

    async def rollback(self) -> None:
        await self.session.rollback()
