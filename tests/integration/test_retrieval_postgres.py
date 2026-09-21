import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.db.models import Document, DocumentChunk, DocumentStatus
from app.db.session import get_db_session
from app.main import create_app
from app.providers.embeddings import DeterministicEmbeddingProvider

pytestmark = pytest.mark.integration


@pytest.fixture
async def retrieval_postgres_client(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, async_sessionmaker[AsyncSession]]]:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    engine = create_async_engine(database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(
        app_env="test",
        database_url=database_url,
        upload_dir=tmp_path / "uploads",
        max_upload_bytes=4096,
        chunk_target_chars=200,
        chunk_overlap_chars=20,
        embedding_dimension=256,
        retrieval_top_k=5,
        retrieval_max_top_k=10,
        retrieval_min_score=0.15,
        retrieval_max_query_chars=200,
        retrieval_max_returned_chars=1000,
    )

    async def test_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app: FastAPI = create_app()
    app.dependency_overrides[get_db_session] = test_session
    app.dependency_overrides[get_settings] = lambda: settings

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, session_factory

    async with engine.begin() as connection:
        await connection.execute(
            text("DELETE FROM documents WHERE filename LIKE 'retrieval-integration-%'")
        )
    await engine.dispose()


async def test_exact_pgvector_retrieval_ranking_thresholds_and_ready_filter(
    retrieval_postgres_client: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = retrieval_postgres_client
    refund_text = b"Refund requests are accepted within thirty days of purchase."
    backup_text = b"Database backups run every Sunday at midnight."
    query = "refund purchase thirty days"

    refund_upload = await client.post(
        "/api/v1/documents",
        files={"file": ("retrieval-integration-refund.txt", refund_text, "text/plain")},
    )
    backup_upload = await client.post(
        "/api/v1/documents",
        files={"file": ("retrieval-integration-backup.txt", backup_text, "text/plain")},
    )
    assert refund_upload.status_code == 201, refund_upload.text
    assert backup_upload.status_code == 201, backup_upload.text
    refund_id = uuid.UUID(refund_upload.json()["id"])
    backup_id = uuid.UUID(backup_upload.json()["id"])

    provider = DeterministicEmbeddingProvider(256)
    processing_document = Document(
        filename="retrieval-integration-processing.txt",
        content_type="text/plain",
        checksum="f" * 64,
        status=DocumentStatus.PROCESSING,
        size_bytes=len(query),
        metadata_={},
    )
    async with session_factory() as session:
        session.add(processing_document)
        await session.flush()
        session.add(
            DocumentChunk(
                document_id=processing_document.id,
                chunk_index=0,
                text=query,
                char_count=len(query),
                embedding=await provider.embed_text(query),
                metadata_={},
            )
        )
        await session.commit()
    processing_id = processing_document.id

    search = await client.post(
        "/api/v1/retrieval/search",
        json={"query": query, "top_k": 5, "min_score": -1.0},
    )
    assert search.status_code == 200, search.text
    body = search.json()
    results = body["results"]
    assert results[0]["document_id"] == str(refund_id)
    assert str(processing_id) not in {result["document_id"] for result in results}
    assert {result["document_id"] for result in results} == {str(refund_id), str(backup_id)}
    assert [result["score"] for result in results] == sorted(
        [result["score"] for result in results], reverse=True
    )
    assert body["diagnostics"]["database_candidates_considered"] == 2

    limited = await client.post(
        "/api/v1/retrieval/search",
        json={"query": query, "top_k": 1, "min_score": -1.0},
    )
    assert len(limited.json()["results"]) == 1
    assert limited.json()["results"][0]["document_id"] == str(refund_id)

    midpoint = (results[0]["score"] + results[1]["score"]) / 2
    thresholded = await client.post(
        "/api/v1/retrieval/search",
        json={"query": query, "top_k": 5, "min_score": midpoint},
    )
    assert [result["document_id"] for result in thresholded.json()["results"]] == [
        str(refund_id)
    ]

    no_match = await client.post(
        "/api/v1/retrieval/search",
        json={"query": "volcano tectonic magma", "top_k": 5, "min_score": 0.99},
    )
    assert no_match.status_code == 200
    assert no_match.json()["results"] == []

    async with session_factory() as session:
        vector_rows = (
            await session.execute(
                text(
                    "SELECT d.id, vector_dims(c.embedding) "
                    "FROM documents d JOIN document_chunks c ON c.document_id = d.id "
                    "WHERE d.id IN (:refund_id, :backup_id) ORDER BY d.id"
                ),
                {"refund_id": refund_id, "backup_id": backup_id},
            )
        ).all()
    assert len(vector_rows) == 2
    assert all(row[1] == 256 for row in vector_rows)

    assert (await client.delete(f"/api/v1/documents/{refund_id}")).status_code == 200
    assert (await client.delete(f"/api/v1/documents/{backup_id}")).status_code == 200
