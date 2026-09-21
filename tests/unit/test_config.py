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
    monkeypatch.setenv("EMBEDDING_DIMENSION", "256")

    assert Settings(_env_file=None).embedding_dimension == 256


def test_settings_reject_embedding_dimension_drift() -> None:
    with pytest.raises(ValidationError, match="must be 256"):
        Settings(embedding_dimension=128)
