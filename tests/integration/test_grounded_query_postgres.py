import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.dependencies import get_generation_provider
from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.main import create_app
from app.providers.generation import DeterministicGenerationProvider, GenerationResult
from app.providers.generation.prompt import GroundedPrompt

pytestmark = pytest.mark.integration


class CountingGenerationProvider(DeterministicGenerationProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def generate_grounded_answer(self, prompt: GroundedPrompt) -> GenerationResult:
        self.calls += 1
        return await super().generate_grounded_answer(prompt)


@pytest.fixture
async def grounded_postgres_client(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, CountingGenerationProvider, str]]:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    engine = create_async_engine(database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=database_url,
        upload_dir=tmp_path / "uploads",
        max_upload_bytes=4096,
        chunk_target_chars=300,
        chunk_overlap_chars=20,
        retrieval_top_k=5,
        retrieval_max_top_k=10,
        retrieval_min_score=0.15,
        retrieval_max_query_chars=500,
        retrieval_max_returned_chars=2000,
    )
    generation = CountingGenerationProvider()

    async def test_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app: FastAPI = create_app()
    app.dependency_overrides[get_db_session] = test_session
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_generation_provider] = lambda: generation

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, generation, database_url

    async with engine.begin() as connection:
        await connection.execute(
            text("DELETE FROM query_runs WHERE question LIKE 'integration query:%'")
        )
        await connection.execute(
            text("DELETE FROM documents WHERE filename = 'grounded-integration-policy.txt'")
        )
    await engine.dispose()


async def test_grounded_query_persists_validated_citations_and_pre_generation_abstention(
    grounded_postgres_client: tuple[AsyncClient, CountingGenerationProvider, str],
) -> None:
    client, generation, database_url = grounded_postgres_client
    upload = await client.post(
        "/api/v1/documents",
        files={
            "file": (
                "grounded-integration-policy.txt",
                b"Remote work policy allows employees to work from home two days each week.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 201, upload.text
    document_id = uuid.UUID(upload.json()["id"])

    supported = await client.post(
        "/api/v1/query",
        json={
            "question": "integration query: remote work home two days",
            "min_score": 0.1,
        },
    )
    assert supported.status_code == 200, supported.text
    body = supported.json()
    assert body["abstained"] is False
    assert body["citations"][0]["source_id"] == "S1"
    assert body["citations"][0]["document_id"] == str(document_id)
    assert generation.calls == 1

    fetched = await client.get(f"/api/v1/query-runs/{body['run_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["citations"][0]["chunk_id"] == body["citations"][0]["chunk_id"]
    assert fetched.json()["retrieval_snapshot"][0]["source_id"] == "S1"

    weak = await client.post(
        "/api/v1/query",
        json={
            "question": "integration query: policy volcano",
            "min_score": 0.1,
        },
    )
    assert weak.status_code == 200, weak.text
    assert weak.json()["abstained"] is True
    assert weak.json()["diagnostics"]["abstention_category"] == (
        "insufficient_evidence_strength"
    )
    assert weak.json()["diagnostics"]["top_evidence_score"] < 0.26
    assert weak.json()["diagnostics"]["generation_ms"] == 0.0
    assert generation.calls == 1

    absent = await client.post(
        "/api/v1/query",
        json={
            "question": "integration query: volcano magma tectonic plates",
            "min_score": 0.99,
        },
    )
    assert absent.status_code == 200, absent.text
    assert absent.json()["abstained"] is True
    assert absent.json()["citations"] == []
    assert absent.json()["diagnostics"]["generation_ms"] == 0.0
    assert generation.calls == 1

    engine = create_async_engine(database_url)
    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                text(
                    "SELECT status::text, abstained, citation_snapshot, retrieval_snapshot, "
                    "generation_ms, evidence_policy_version, evidence_sufficient, "
                    "abstention_category FROM query_runs "
                    "WHERE id IN (:supported, :weak, :absent) "
                    "ORDER BY abstained"
                ),
                {
                    "supported": uuid.UUID(body["run_id"]),
                    "weak": uuid.UUID(weak.json()["run_id"]),
                    "absent": uuid.UUID(absent.json()["run_id"]),
                },
            )
        ).all()
    await engine.dispose()

    assert len(rows) == 3
    assert rows[0][0] == "completed" and rows[0][1] is False
    assert rows[0][2][0]["source_id"] == "S1"
    assert rows[0][3][0]["document_id"] == str(document_id)
    assert rows[0][5] == "evidence_policy_v1" and rows[0][6] is True
    assert all(row[0] == "completed" and row[1] is True for row in rows[1:])
    assert all(row[2] == [] and row[4] == 0.0 for row in rows[1:])
    assert {row[7] for row in rows[1:]} == {
        "insufficient_evidence_strength",
        "no_retrieval_evidence",
    }

    assert (await client.delete(f"/api/v1/documents/{document_id}")).status_code == 200
