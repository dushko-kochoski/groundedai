"""Adopt one 1536-dimensional embedding space and record its identity.

Revision ID: 20260921_0003
Revises: 20260921_0002
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "20260921_0003"
down_revision: str | None = "20260921_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _require_empty_documents(message: str) -> None:
    escaped = message.replace("'", "''")
    op.execute(
        sa.text(
            "DO $$ BEGIN "
            "IF EXISTS (SELECT 1 FROM documents) THEN "
            f"RAISE EXCEPTION '{escaped}'; "
            "END IF; END $$;"
        )
    )


def upgrade() -> None:
    _require_empty_documents(
        "Embedding schema changed to 1536 dimensions; delete existing documents through the API "
        "before migration, then re-ingest them"
    )
    op.alter_column(
        "document_chunks",
        "embedding",
        existing_type=Vector(256),
        type_=Vector(1536),
        postgresql_using="embedding::vector(1536)",
        existing_nullable=False,
    )
    op.add_column(
        "document_chunks", sa.Column("embedding_provider", sa.String(64), nullable=False)
    )
    op.add_column(
        "document_chunks", sa.Column("embedding_model", sa.String(127), nullable=False)
    )
    op.add_column(
        "document_chunks", sa.Column("embedding_dimension", sa.Integer(), nullable=False)
    )
    op.add_column(
        "document_chunks", sa.Column("embedding_version", sa.String(64), nullable=False)
    )


def downgrade() -> None:
    _require_empty_documents(
        "Embedding schema downgrade requires deleting all 1536-dimensional documents first"
    )
    op.drop_column("document_chunks", "embedding_version")
    op.drop_column("document_chunks", "embedding_dimension")
    op.drop_column("document_chunks", "embedding_model")
    op.drop_column("document_chunks", "embedding_provider")
    op.alter_column(
        "document_chunks",
        "embedding",
        existing_type=Vector(1536),
        type_=Vector(256),
        postgresql_using="embedding::vector(256)",
        existing_nullable=False,
    )
