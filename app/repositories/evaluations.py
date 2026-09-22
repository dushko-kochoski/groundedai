import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import EvaluationCaseResult, EvaluationRun, QueryRun


@dataclass(frozen=True, slots=True)
class CalibrationCaseSnapshot:
    case_id: str
    answerable: bool
    expected_sources: list[str]
    retrieved_sources: list[str]
    facts_expected: int
    facts_matched: int | None
    abstained: bool | None
    citations_returned: int
    citations_valid: int
    invalid_citation_count: int
    citation_source_precision: float | None
    retrieval_ms: float
    generation_ms: float | None
    total_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error_category: str | None
    source_hit: bool
    source_recall: float
    reciprocal_rank: float
    retrieval_snapshot: list[dict[str, Any]]


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

    async def calibration_snapshots(
        self, run_id: uuid.UUID
    ) -> list[CalibrationCaseSnapshot]:
        result = await self.session.execute(
            select(EvaluationCaseResult, QueryRun.retrieval_snapshot)
            .join(QueryRun, QueryRun.id == EvaluationCaseResult.query_run_id, isouter=True)
            .where(EvaluationCaseResult.evaluation_run_id == run_id)
            .order_by(EvaluationCaseResult.case_id)
        )
        return [
            CalibrationCaseSnapshot(
                case_id=case.case_id,
                answerable=case.answerable,
                expected_sources=case.expected_sources,
                retrieved_sources=case.retrieved_sources,
                facts_expected=case.facts_expected,
                facts_matched=case.facts_matched,
                abstained=case.abstained,
                citations_returned=case.citations_returned,
                citations_valid=case.citations_valid,
                invalid_citation_count=case.invalid_citation_count,
                citation_source_precision=case.citation_source_precision,
                retrieval_ms=case.retrieval_ms,
                generation_ms=case.generation_ms,
                total_ms=case.total_ms,
                input_tokens=case.input_tokens,
                output_tokens=case.output_tokens,
                total_tokens=case.total_tokens,
                error_category=case.error_category,
                source_hit=case.source_hit,
                source_recall=case.source_recall,
                reciprocal_rank=case.reciprocal_rank,
                retrieval_snapshot=snapshot or [],
            )
            for case, snapshot in result.all()
        ]

    async def rollback(self) -> None:
        await self.session.rollback()
