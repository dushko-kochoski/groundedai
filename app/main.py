from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.core.config import get_settings
from app.core.exceptions import ApplicationError, application_error_handler
from app.core.logging import RequestContextMiddleware, configure_logging
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
    )
    application.add_middleware(RequestContextMiddleware)
    application.add_exception_handler(ApplicationError, application_error_handler)
    application.include_router(health_router)
    application.include_router(documents_router)
    return application


app = create_app()
