from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["development", "test", "production"] = "development"
    api_access_key: SecretStr | None = None
    log_level: str = "INFO"
    database_url: str = "postgresql+asyncpg://groundedai:groundedai@localhost:5432/groundedai"
    database_echo: bool = False
    upload_dir: Path = Path("data/uploads")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    chunk_target_chars: int = Field(default=1000, ge=100, le=100_000)
    chunk_overlap_chars: int = Field(default=150, ge=0, le=10_000)
    embedding_provider: Literal["deterministic", "openai"] = "deterministic"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: SecretStr | None = None
    embedding_dimension: int = Field(default=1536, ge=1)
    embedding_timeout_seconds: float = Field(default=20.0, gt=0, le=300)
    embedding_max_retries: int = Field(default=2, ge=0, le=5)
    embedding_batch_size: int = Field(default=64, ge=1, le=2048)
    generation_provider: Literal["deterministic", "openai"] = "deterministic"
    generation_model: str = "gpt-5-mini"
    openai_api_key: SecretStr | None = None
    generation_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    generation_max_retries: int = Field(default=2, ge=0, le=5)
    generation_max_output_tokens: int = Field(default=800, ge=64, le=16_384)
    retrieval_top_k: int = Field(default=5, ge=1)
    retrieval_max_top_k: int = Field(default=20, ge=1, le=100)
    retrieval_min_score: float = Field(default=0.15, ge=-1.0, le=1.0)
    retrieval_max_query_chars: int = Field(default=2000, ge=1, le=100_000)
    retrieval_max_returned_chars: int = Field(default=6000, ge=1, le=1_000_000)
    evidence_deterministic_min_score: float = Field(default=0.26, ge=-1.0, le=1.0)
    evidence_openai_min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    evaluation_root: Path = Path("evals")

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> "Settings":
        access_key = self.api_access_key.get_secret_value() if self.api_access_key else ""
        if access_key and (
            len(access_key) < 32
            or not access_key.isascii()
            or any(character.isspace() for character in access_key)
        ):
            raise ValueError(
                "API_ACCESS_KEY must be at least 32 ASCII characters without whitespace"
            )
        if self.app_env == "production" and not access_key:
            raise ValueError("API_ACCESS_KEY is required in production")
        if self.chunk_overlap_chars >= self.chunk_target_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_TARGET_CHARS")
        if self.embedding_dimension != 1536:
            raise ValueError("EMBEDDING_DIMENSION must be 1536 to match the database schema")
        if self.embedding_provider == "openai":
            if self.embedding_model != "text-embedding-3-small":
                raise ValueError("EMBEDDING_MODEL must be text-embedding-3-small when using OpenAI")
            if self.embedding_api_key is None or not self.embedding_api_key.get_secret_value():
                raise ValueError("EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER=openai")
        if self.generation_provider == "openai":
            if not self.generation_model.strip():
                raise ValueError("GENERATION_MODEL must not be empty")
            if self.openai_api_key is None or not self.openai_api_key.get_secret_value():
                raise ValueError("OPENAI_API_KEY is required when GENERATION_PROVIDER=openai")
        if self.retrieval_top_k > self.retrieval_max_top_k:
            raise ValueError("RETRIEVAL_TOP_K must not exceed RETRIEVAL_MAX_TOP_K")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
