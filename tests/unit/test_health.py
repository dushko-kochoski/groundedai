from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.db.health import DatabaseReadiness, check_database_readiness
from app.main import create_app


@pytest.fixture
async def app() -> AsyncIterator[FastAPI]:
    application = create_app()
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
        yield http_client


async def test_health_is_live_and_returns_request_id(client: AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "test-request-id"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"] == "test-request-id"


async def test_ready_when_database_and_pgvector_are_available(
    app: FastAPI, client: AsyncClient
) -> None:
    async def ready_database() -> DatabaseReadiness:
        return DatabaseReadiness(database=True, pgvector=True)

    app.dependency_overrides[check_database_readiness] = ready_database

    response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": True, "pgvector": True}


async def test_ready_returns_503_when_dependency_is_unavailable(
    app: FastAPI, client: AsyncClient
) -> None:
    async def unavailable_database() -> DatabaseReadiness:
        return DatabaseReadiness(database=False, pgvector=False)

    app.dependency_overrides[check_database_readiness] = unavailable_database

    response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "database": False,
        "pgvector": False,
    }

