"""Persist evaluation runs and per-case results.

Revision ID: 20260921_0005
Revises: 20260921_0004
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260921_0005"
down_revision: str | None = "20260921_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    mode = postgresql.ENUM(
        "retrieval_only", "end_to_end", name="evaluation_mode", create_type=False
    )
    status = postgresql.ENUM(
        "running", "completed", "failed", name="evaluation_run_status", create_type=False
    )
    mode.create(op.get_bind(), checkfirst=True)
    status.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "evaluation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_name", sa.String(127), nullable=False),
        sa.Column("dataset_version", sa.String(64), nullable=False),
        sa.Column("mode", mode, nullable=False),
        sa.Column("status", status, nullable=False),
        sa.Column("retrieval_top_k", sa.Integer(), nullable=False),
        sa.Column("retrieval_min_score", sa.Float(), nullable=False),
        sa.Column("embedding_provider", sa.String(64), nullable=False),
        sa.Column("embedding_model", sa.String(127), nullable=False),
        sa.Column("embedding_version", sa.String(64), nullable=False),
        sa.Column("generation_provider", sa.String(64), nullable=True),
        sa.Column("generation_model", sa.String(127), nullable=True),
        sa.Column("prompt_version", sa.String(64), nullable=True),
        sa.Column("case_count", sa.Integer(), nullable=False),
        sa.Column("aggregate_metrics", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("error_category", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_runs")),
    )
    op.create_index("ix_evaluation_runs_created_at", "evaluation_runs", ["created_at"])
    op.create_table(
        "evaluation_case_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("evaluation_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("case_id", sa.String(127), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answerable", sa.Boolean(), nullable=False),
        sa.Column("expected_sources", postgresql.JSONB(), nullable=False),
        sa.Column("retrieved_sources", postgresql.JSONB(), nullable=False),
        sa.Column("expected_facts", postgresql.JSONB(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("abstained", sa.Boolean(), nullable=True),
        sa.Column("facts_matched", sa.Integer(), nullable=True),
        sa.Column("facts_expected", sa.Integer(), nullable=False),
        sa.Column("fact_coverage", sa.Float(), nullable=True),
        sa.Column("source_hit", sa.Boolean(), nullable=False),
        sa.Column("source_recall", sa.Float(), nullable=False),
        sa.Column("reciprocal_rank", sa.Float(), nullable=False),
        sa.Column("citations_returned", sa.Integer(), nullable=False),
        sa.Column("citations_valid", sa.Integer(), nullable=False),
        sa.Column("invalid_citation_count", sa.Integer(), nullable=False),
        sa.Column("citation_validity_rate", sa.Float(), nullable=True),
        sa.Column("citation_source_precision", sa.Float(), nullable=True),
        sa.Column("citation_source_recall", sa.Float(), nullable=True),
        sa.Column("retrieval_ms", sa.Float(), nullable=False),
        sa.Column("generation_ms", sa.Float(), nullable=True),
        sa.Column("total_ms", sa.Float(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("query_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("error_category", sa.String(64), nullable=True),
        sa.ForeignKeyConstraint(
            ["evaluation_run_id"], ["evaluation_runs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["query_run_id"], ["query_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_case_results")),
        sa.UniqueConstraint(
            "evaluation_run_id", "case_id", name="uq_evaluation_case_results_run_case"
        ),
    )
    op.create_index(
        "ix_evaluation_case_results_run_id",
        "evaluation_case_results",
        ["evaluation_run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_evaluation_case_results_run_id", table_name="evaluation_case_results")
    op.drop_table("evaluation_case_results")
    op.drop_index("ix_evaluation_runs_created_at", table_name="evaluation_runs")
    op.drop_table("evaluation_runs")
    postgresql.ENUM(name="evaluation_run_status").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="evaluation_mode").drop(op.get_bind(), checkfirst=True)
