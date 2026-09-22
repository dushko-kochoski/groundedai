import pytest

from app.schemas.retrieval import RetrievalResult
from app.services.generation.evidence import (
    EVIDENCE_POLICY_VERSION,
    AbstentionCategory,
    EvidenceSufficiencyPolicy,
    build_evidence_policy,
)


def retrieval_result(score: float = 0.8) -> RetrievalResult:
    return RetrievalResult.model_validate(
        {
            "chunk_id": "b73be784-25f7-4913-9766-4d05cd4c65b7",
            "document_id": "08f8fefb-e9dd-42da-83bb-f7ea34f02673",
            "filename": "policy.txt",
            "chunk_index": 0,
            "text": "Policy evidence.",
            "score": score,
            "page_number": None,
            "section": None,
        }
    )


def policy(minimum_score: float = 0.26) -> EvidenceSufficiencyPolicy:
    return EvidenceSufficiencyPolicy(
        version=EVIDENCE_POLICY_VERSION,
        embedding_provider="deterministic-feature-hashing",
        embedding_model="feature-hashing-v1",
        embedding_version="1",
        minimum_score=minimum_score,
        calibrated=True,
    )


def test_no_evidence_is_distinguished_from_weak_evidence() -> None:
    decision = policy().evaluate([])

    assert decision.sufficient is False
    assert decision.top_score is None
    assert decision.category == AbstentionCategory.NO_RETRIEVAL_EVIDENCE


@pytest.mark.parametrize(
    ("score", "sufficient"),
    [(0.259999, False), (0.26, True), (0.260001, True)],
)
def test_threshold_boundary(score: float, sufficient: bool) -> None:
    result = retrieval_result(score)
    decision = policy().evaluate([result])

    assert decision.sufficient is sufficient
    assert decision.category == (
        None if sufficient else AbstentionCategory.INSUFFICIENT_EVIDENCE_STRENGTH
    )


def test_deterministic_profile_is_calibrated_and_versioned() -> None:
    result = build_evidence_policy(
        embedding_provider="deterministic-feature-hashing",
        embedding_model="feature-hashing-v1",
        embedding_version="1",
        deterministic_minimum_score=0.26,
        openai_minimum_score=None,
    )

    assert result.version == EVIDENCE_POLICY_VERSION
    assert result.calibrated is True
    assert result.minimum_score == 0.26


def test_openai_profile_fails_safe_until_explicitly_configured() -> None:
    uncalibrated = build_evidence_policy(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_version="1",
        deterministic_minimum_score=0.26,
        openai_minimum_score=None,
    )
    evidence = retrieval_result(0.9)

    decision = uncalibrated.evaluate([evidence])

    assert uncalibrated.calibrated is False
    assert decision.sufficient is False
    assert decision.category == AbstentionCategory.UNCALIBRATED_EMBEDDING_PROFILE


def test_openai_profile_accepts_explicit_uncalibrated_override() -> None:
    uncalibrated = build_evidence_policy(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_version="1",
        deterministic_minimum_score=0.26,
        openai_minimum_score=None,
    )

    decision = uncalibrated.evaluate(
        [retrieval_result(0.31)],
        minimum_score_override=0.30,
    )

    assert decision.sufficient is True
    assert decision.minimum_score == 0.30
