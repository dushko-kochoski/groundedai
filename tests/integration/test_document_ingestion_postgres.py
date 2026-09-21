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
from app.db.session import get_db_session
from app.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture
async def postgres_client(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, Path]]:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    engine = create_async_engine(database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    upload_dir = tmp_path / "uploads"
    settings = Settings(
        app_env="test",
        database_url=database_url,
        upload_dir=upload_dir,
        max_upload_bytes=1024,
        chunk_target_chars=100,
        chunk_overlap_chars=20,
        embedding_dimension=256,
    )

    async def test_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app: FastAPI = create_app()
    app.dependency_overrides[get_db_session] = test_session
    app.dependency_overrides[get_settings] = lambda: settings

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, upload_dir

    async with engine.begin() as connection:
        await connection.execute(
            text("DELETE FROM documents WHERE filename LIKE 'integration-%'")
        )
    await engine.dispose()


async def test_ingestion_persists_vectors_and_deletion_cascades(
    postgres_client: tuple[AsyncClient, Path],
) -> None:
    client, upload_dir = postgres_client
    txt_content = (
        b"Refund Policy\n\nCustomers may request a refund within thirty days.\n\n"
        b"Proof of purchase is required for every refund request."
    )
    markdown_content = b"# Support\n\nEmail support is available during business hours."

    txt_response = await client.post(
        "/api/v1/documents",
        files={"file": ("integration-policy.txt", txt_content, "text/plain")},
    )
    markdown_response = await client.post(
        "/api/v1/documents",
        files={"file": ("integration-guide.md", markdown_content, "text/markdown")},
    )

    assert txt_response.status_code == 201, txt_response.text
    assert markdown_response.status_code == 201, markdown_response.text
    txt_document = txt_response.json()
    markdown_document = markdown_response.json()
    txt_id = uuid.UUID(txt_document["id"])
    markdown_id = uuid.UUID(markdown_document["id"])
    assert txt_document["status"] == "ready"
    assert txt_document["chunk_count"] >= 2
    assert (upload_dir / f"{txt_id.hex}.txt").is_file()
    assert (upload_dir / f"{markdown_id.hex}.md").is_file()

    duplicate = await client.post(
        "/api/v1/documents",
        files={"file": ("integration-copy.txt", txt_content, "text/plain")},
    )
    listing = await client.get("/api/v1/documents?limit=100&offset=0")
    detail = await client.get(f"/api/v1/documents/{txt_id}")

    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["details"]["document_id"] == str(txt_id)
    assert listing.status_code == 200
    assert {item["id"] for item in listing.json()["items"]} >= {str(txt_id), str(markdown_id)}
    assert detail.status_code == 200
    assert "embedding" not in detail.json()

    database_url = os.environ["TEST_DATABASE_URL"]
    verification_engine = create_async_engine(database_url)
    async with verification_engine.connect() as connection:
        document_row = (
            await connection.execute(
                text("SELECT status::text, storage_key FROM documents WHERE id = :id"),
                {"id": txt_id},
            )
        ).one()
        chunk_row = (
            await connection.execute(
                text(
                    "SELECT count(*), bool_and(embedding IS NOT NULL), "
                    "min(vector_dims(embedding)) FROM document_chunks WHERE document_id = :id"
                ),
                {"id": txt_id},
            )
        ).one()

    assert document_row[0] == "ready"
    assert document_row[1] == f"{txt_id.hex}.txt"
    assert chunk_row[0] >= 2
    assert chunk_row[1] is True
    assert chunk_row[2] == 256

    deleted = await client.delete(f"/api/v1/documents/{txt_id}")
    missing = await client.get(f"/api/v1/documents/{txt_id}")
    assert deleted.status_code == 200
    assert missing.status_code == 404
    assert not (upload_dir / f"{txt_id.hex}.txt").exists()

    async with verification_engine.connect() as connection:
        document_count = await connection.scalar(
            text("SELECT count(*) FROM documents WHERE id = :id"), {"id": txt_id}
        )
        chunk_count = await connection.scalar(
            text("SELECT count(*) FROM document_chunks WHERE document_id = :id"),
            {"id": txt_id},
        )
    await verification_engine.dispose()

    assert document_count == 0
    assert chunk_count == 0
    assert (await client.delete(f"/api/v1/documents/{markdown_id}")).status_code == 200
