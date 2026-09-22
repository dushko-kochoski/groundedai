import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Boolean, DateTime, Enum, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class QueryRunStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class QueryRun(Base):
    __tablename__ = "query_runs"
    __table_args__ = (Index("ix_query_runs_created_at", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[QueryRunStatus] = mapped_column(
        Enum(
            QueryRunStatus,
            name="query_run_status",
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    abstained: Mapped[bool | None] = mapped_column(Boolean)
    abstention_reason: Mapped[str | None] = mapped_column(Text)
    generation_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    generation_model: Mapped[str] = mapped_column(String(127), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_min_score: Mapped[float | None] = mapped_column(Float)
    top_evidence_score: Mapped[float | None] = mapped_column(Float)
    evidence_sufficient: Mapped[bool | None] = mapped_column(Boolean)
    abstention_category: Mapped[str | None] = mapped_column(String(64))
    retrieval_top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_min_score: Mapped[float] = mapped_column(Float, nullable=False)
    retrieval_snapshot: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    citation_snapshot: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    retrieval_ms: Mapped[float] = mapped_column(Float, nullable=False)
    generation_ms: Mapped[float] = mapped_column(Float, nullable=False)
    total_ms: Mapped[float] = mapped_column(Float, nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
