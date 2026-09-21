import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_accept_valid_retrieval_configuration() -> None:
    settings = Settings(retrieval_top_k=7, retrieval_min_score=0.5)

    assert settings.retrieval_top_k == 7
    assert settings.retrieval_min_score == 0.5


def test_settings_reject_invalid_retrieval_score() -> None:
    with pytest.raises(ValidationError):
        Settings(retrieval_min_score=1.1)


def test_settings_parse_embedding_dimension_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")

    assert Settings(_env_file=None).embedding_dimension == 1536


def test_settings_reject_embedding_dimension_drift() -> None:
    with pytest.raises(ValidationError, match="must be 1536"):
        Settings(embedding_dimension=256)


def test_openai_provider_requires_credentials() -> None:
    with pytest.raises(ValidationError, match="EMBEDDING_API_KEY is required"):
        Settings(embedding_provider="openai", embedding_api_key=None)


def test_deterministic_provider_does_not_require_credentials() -> None:
    settings = Settings(embedding_provider="deterministic", embedding_api_key=None)

    assert settings.embedding_provider == "deterministic"


def test_openai_generation_requires_credentials() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY is required"):
        Settings(_env_file=None, generation_provider="openai", openai_api_key=None)


def test_deterministic_generation_does_not_require_credentials() -> None:
    settings = Settings(_env_file=None, generation_provider="deterministic", openai_api_key=None)

    assert settings.generation_provider == "deterministic"
