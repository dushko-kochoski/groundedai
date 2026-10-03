from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.documents import router as documents_router
from app.api.routes.evaluations import router as evaluations_router
from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.workspace import router as workspace_router
from app.core.config import get_settings
from app.core.exceptions import ApplicationError, application_error_handler
from app.core.logging import RequestContextMiddleware, configure_logging
from app.core.security import AccessControlMiddleware
from app.db.session import dispose_engine


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title="GroundedAI",
        version="0.1.0",
        description="Reliability-first RAG and AI evaluation platform",
        lifespan=lifespan,
        docs_url=None if settings.app_env == "production" else "/docs",
        redoc_url=None if settings.app_env == "production" else "/redoc",
        openapi_url=None if settings.app_env == "production" else "/openapi.json",
    )
    application.add_middleware(
        AccessControlMiddleware,
        access_key=settings.api_access_key.get_secret_value() if settings.api_access_key else "",
    )
    application.add_middleware(RequestContextMiddleware)
    application.add_exception_handler(ApplicationError, application_error_handler)
    application.include_router(health_router)
    application.include_router(documents_router)
    application.include_router(retrieval_router)
    application.include_router(query_router)
    application.include_router(evaluations_router)
    application.include_router(workspace_router)
    web_root = Path(__file__).parent / "web"
    application.mount("/assets", StaticFiles(directory=web_root), name="assets")

    @application.get("/", include_in_schema=False)
    async def workspace() -> FileResponse:
        return FileResponse(web_root / "index.html", headers={"Cache-Control": "no-cache"})

    return application


app = create_app()
