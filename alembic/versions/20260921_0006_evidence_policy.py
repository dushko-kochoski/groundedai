"""Persist evidence sufficiency decisions and policy configuration.

Revision ID: 20260921_0006
Revises: 20260921_0005
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260921_0006"
down_revision: str | None = "20260921_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "query_runs",
        sa.Column(
            "evidence_policy_version",
            sa.String(64),
            server_default="retrieval_presence_v0",
            nullable=False,
        ),
    )
    op.add_column("query_runs", sa.Column("evidence_min_score", sa.Float(), nullable=True))
    op.add_column("query_runs", sa.Column("top_evidence_score", sa.Float(), nullable=True))
    op.add_column("query_runs", sa.Column("evidence_sufficient", sa.Boolean(), nullable=True))
    op.add_column("query_runs", sa.Column("abstention_category", sa.String(64), nullable=True))
    op.execute(
        """
        UPDATE query_runs
        SET evidence_min_score = retrieval_min_score,
            top_evidence_score = CASE
                WHEN jsonb_array_length(retrieval_snapshot) > 0
                THEN (retrieval_snapshot->0->>'score')::double precision
                ELSE NULL
            END,
            evidence_sufficient = jsonb_array_length(retrieval_snapshot) > 0,
            abstention_category = CASE
                WHEN status::text = 'completed' AND abstained = true
                     AND jsonb_array_length(retrieval_snapshot) = 0
                THEN 'no_retrieval_evidence'
                WHEN status::text = 'completed' AND abstained = true
                THEN 'provider_abstention'
                ELSE NULL
            END
        """
    )
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "evidence_policy_version",
            sa.String(64),
            server_default="retrieval_presence_v0",
            nullable=False,
        ),
    )
    op.add_column("evaluation_runs", sa.Column("evidence_min_score", sa.Float(), nullable=True))
    op.execute("UPDATE evaluation_runs SET evidence_min_score = retrieval_min_score")


def downgrade() -> None:
    op.drop_column("evaluation_runs", "evidence_min_score")
    op.drop_column("evaluation_runs", "evidence_policy_version")
    op.drop_column("query_runs", "abstention_category")
    op.drop_column("query_runs", "evidence_sufficient")
    op.drop_column("query_runs", "top_evidence_score")
    op.drop_column("query_runs", "evidence_min_score")
    op.drop_column("query_runs", "evidence_policy_version")
