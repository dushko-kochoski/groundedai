from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.config import Settings, get_settings

router = APIRouter(prefix="/api/v1", tags=["workspace"])


class WorkspaceConfiguration(BaseModel):
    embedding_provider: str
    generation_provider: str
    max_upload_bytes: int
    max_query_chars: int
    access_protected: bool


@router.get("/workspace", response_model=WorkspaceConfiguration)
async def workspace_configuration(
    settings: Annotated[Settings, Depends(get_settings)],
) -> WorkspaceConfiguration:
    return WorkspaceConfiguration(
        embedding_provider=settings.embedding_provider,
        generation_provider=settings.generation_provider,
        max_upload_bytes=settings.max_upload_bytes,
        max_query_chars=settings.retrieval_max_query_chars,
        access_protected=bool(
            settings.api_access_key and settings.api_access_key.get_secret_value()
        ),
    )
