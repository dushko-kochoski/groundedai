"""Persist grounded query runs.

Revision ID: 20260921_0004
Revises: 20260921_0003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260921_0004"
down_revision: str | None = "20260921_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    status = postgresql.ENUM("completed", "failed", name="query_run_status", create_type=False)
    status.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "query_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", status, nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("abstained", sa.Boolean(), nullable=True),
        sa.Column("abstention_reason", sa.Text(), nullable=True),
        sa.Column("generation_provider", sa.String(64), nullable=False),
        sa.Column("generation_model", sa.String(127), nullable=False),
        sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("retrieval_top_k", sa.Integer(), nullable=False),
        sa.Column("retrieval_min_score", sa.Float(), nullable=False),
        sa.Column("retrieval_snapshot", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column("citation_snapshot", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column("retrieval_ms", sa.Float(), nullable=False),
        sa.Column("generation_ms", sa.Float(), nullable=False),
        sa.Column("total_ms", sa.Float(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_query_runs")),
    )
    op.create_index("ix_query_runs_created_at", "query_runs", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_query_runs_created_at", table_name="query_runs")
    op.drop_table("query_runs")
    postgresql.ENUM(name="query_run_status").drop(op.get_bind(), checkfirst=True)
