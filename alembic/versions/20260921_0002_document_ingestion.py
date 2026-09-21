"""Add document ingestion storage fields and chunks.

Revision ID: 20260921_0002
Revises: 20260921_0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260921_0002"
down_revision: str | None = "20260921_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSION = 256


def upgrade() -> None:
    op.add_column("documents", sa.Column("storage_key", sa.String(length=255), nullable=True))
    op.add_column("documents", sa.Column("size_bytes", sa.BigInteger(), nullable=True))
    op.create_unique_constraint("uq_documents_storage_key", "documents", ["storage_key"])

    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_chunks")),
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name=op.f("uq_document_chunks_document_id"),
        ),
    )
    op.create_index(
        "ix_document_chunks_document_id", "document_chunks", ["document_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_constraint("uq_documents_storage_key", "documents", type_="unique")
    op.drop_column("documents", "size_bytes")
    op.drop_column("documents", "storage_key")
