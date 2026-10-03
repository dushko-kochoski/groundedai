from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.main import create_app

ACCESS_KEY = "test-workspace-key-not-a-secret-12345"


@pytest.fixture
def protected_app(monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    monkeypatch.setenv("API_ACCESS_KEY", ACCESS_KEY)
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    try:
        yield create_app()
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/v1/documents"),
        ("POST", "/api/v1/documents"),
        ("DELETE", "/api/v1/documents/00000000-0000-0000-0000-000000000001"),
        ("POST", "/api/v1/query"),
        ("GET", "/api/v1/query-runs/00000000-0000-0000-0000-000000000001"),
        ("POST", "/api/v1/retrieval/search"),
        ("POST", "/api/v1/evaluations/run"),
        ("GET", "/api/v1/evaluations/runs/00000000-0000-0000-0000-000000000001"),
        ("GET", "/api/v1/workspace"),
    ],
)
async def test_api_is_protected_before_body_parsing(
    protected_app: FastAPI,
    method: str,
    path: str,
) -> None:
    async with AsyncClient(transport=ASGITransport(app=protected_app), base_url="http://test") as c:
        response = await c.request(method, path, content=b"invalid request body")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers["cache-control"] == "no-store"
    assert "x-request-id" in response.headers


@pytest.mark.parametrize("authorization", ["", "Bearer wrong", f"Basic {ACCESS_KEY}"])
async def test_invalid_keys_are_rejected(protected_app: FastAPI, authorization: str) -> None:
    async with AsyncClient(transport=ASGITransport(app=protected_app), base_url="http://test") as c:
        response = await c.get("/api/v1/workspace", headers={"Authorization": authorization})
    assert response.status_code == 401
    assert ACCESS_KEY not in response.text


async def test_valid_key_returns_only_public_configuration(protected_app: FastAPI) -> None:
    async with AsyncClient(transport=ASGITransport(app=protected_app), base_url="http://test") as c:
        response = await c.get(
            "/api/v1/workspace",
            headers={"Authorization": f"bearer {ACCESS_KEY}"},
        )
    assert response.status_code == 200
    assert response.json()["access_protected"] is True
    assert set(response.json()) == {
        "embedding_provider",
        "generation_provider",
        "max_upload_bytes",
        "max_query_chars",
        "access_protected",
    }
    assert ACCESS_KEY not in response.text
    assert response.headers["cache-control"] == "no-store"


async def test_workspace_and_health_are_public_but_production_docs_are_off(
    protected_app: FastAPI,
) -> None:
    async with AsyncClient(transport=ASGITransport(app=protected_app), base_url="http://test") as c:
        home = await c.get("/")
        assert home.status_code == 200
        assert 'id="upload-form"' in home.text
        assert 'id="query-form"' in home.text
        assert "script-src 'self'" in home.headers["content-security-policy"]
        assert home.headers["x-frame-options"] == "DENY"
        assert (await c.get("/health")).status_code == 200
        for asset in ("styles.css", "workspace.js"):
            assert (await c.get(f"/assets/{asset}")).status_code == 200
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert (await c.get(path)).status_code == 404


@pytest.mark.parametrize("key", [None, ""])
def test_production_requires_access_key(key: str | None) -> None:
    with pytest.raises(ValidationError, match="API_ACCESS_KEY is required"):
        Settings(_env_file=None, app_env="production", api_access_key=key)


@pytest.mark.parametrize("key", ["too-short", "x" * 32 + " ", "é" * 32])
def test_access_key_validation(key: str) -> None:
    with pytest.raises(ValidationError, match="at least 32 ASCII"):
        Settings(_env_file=None, api_access_key=key)


async def test_local_workspace_without_key_remains_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("API_ACCESS_KEY", "")
    get_settings.cache_clear()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app()),
            base_url="http://test",
        ) as c:
            response = await c.get("/api/v1/workspace")
            assert response.status_code == 200
            assert response.json()["access_protected"] is False
    finally:
        get_settings.cache_clear()
