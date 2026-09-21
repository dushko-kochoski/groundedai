from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql+asyncpg://groundedai:groundedai@localhost:5432/groundedai"
    database_echo: bool = False
    upload_dir: Path = Path("data/uploads")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    chunk_target_chars: int = Field(default=1000, ge=100, le=100_000)
    chunk_overlap_chars: int = Field(default=150, ge=0, le=10_000)
    embedding_dimension: int = Field(default=256, ge=1)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_min_score: float = Field(default=0.70, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> "Settings":
        if self.chunk_overlap_chars >= self.chunk_target_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_TARGET_CHARS")
        if self.embedding_dimension != 256:
            raise ValueError("EMBEDDING_DIMENSION must be 256 until a schema migration changes it")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
