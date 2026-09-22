import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class EvaluationMode(StrEnum):
    RETRIEVAL_ONLY = "retrieval_only"
    END_TO_END = "end_to_end"


class EvaluationRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (Index("ix_evaluation_runs_created_at", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_name: Mapped[str] = mapped_column(String(127), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[EvaluationMode] = mapped_column(
        Enum(
            EvaluationMode,
            name="evaluation_mode",
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
    )
    status: Mapped[EvaluationRunStatus] = mapped_column(
        Enum(
            EvaluationRunStatus,
            name="evaluation_run_status",
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
    )
    retrieval_top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_min_score: Mapped[float] = mapped_column(Float, nullable=False)
    embedding_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(127), nullable=False)
    embedding_version: Mapped[str] = mapped_column(String(64), nullable=False)
    generation_provider: Mapped[str | None] = mapped_column(String(64))
    generation_model: Mapped[str | None] = mapped_column(String(127))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    evidence_policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_min_score: Mapped[float | None] = mapped_column(Float)
    case_count: Mapped[int] = mapped_column(Integer, nullable=False)
    aggregate_metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    error_category: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    cases: Mapped[list["EvaluationCaseResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class EvaluationCaseResult(Base):
    __tablename__ = "evaluation_case_results"
    __table_args__ = (
        Index("ix_evaluation_case_results_run_id", "evaluation_run_id"),
        UniqueConstraint(
            "evaluation_run_id", "case_id", name="uq_evaluation_case_results_run_case"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    case_id: Mapped[str] = mapped_column(String(127), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answerable: Mapped[bool] = mapped_column(nullable=False)
    expected_sources: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    retrieved_sources: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    expected_facts: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    abstained: Mapped[bool | None] = mapped_column()
    facts_matched: Mapped[int | None] = mapped_column(Integer)
    facts_expected: Mapped[int] = mapped_column(Integer, nullable=False)
    fact_coverage: Mapped[float | None] = mapped_column(Float)
    source_hit: Mapped[bool] = mapped_column(nullable=False)
    source_recall: Mapped[float] = mapped_column(Float, nullable=False)
    reciprocal_rank: Mapped[float] = mapped_column(Float, nullable=False)
    citations_returned: Mapped[int] = mapped_column(Integer, nullable=False)
    citations_valid: Mapped[int] = mapped_column(Integer, nullable=False)
    invalid_citation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    citation_validity_rate: Mapped[float | None] = mapped_column(Float)
    citation_source_precision: Mapped[float | None] = mapped_column(Float)
    citation_source_recall: Mapped[float | None] = mapped_column(Float)
    retrieval_ms: Mapped[float] = mapped_column(Float, nullable=False)
    generation_ms: Mapped[float | None] = mapped_column(Float)
    total_ms: Mapped[float] = mapped_column(Float, nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    query_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("query_runs.id", ondelete="SET NULL")
    )
    error_category: Mapped[str | None] = mapped_column(String(64))
    run: Mapped[EvaluationRun] = relationship(back_populates="cases")
