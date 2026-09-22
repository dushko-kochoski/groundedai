from dataclasses import dataclass
from enum import StrEnum

from app.schemas.retrieval import RetrievalResult

EVIDENCE_POLICY_VERSION = "evidence_policy_v1"


class AbstentionCategory(StrEnum):
    NO_RETRIEVAL_EVIDENCE = "no_retrieval_evidence"
    INSUFFICIENT_EVIDENCE_STRENGTH = "insufficient_evidence_strength"
    UNCALIBRATED_EMBEDDING_PROFILE = "uncalibrated_embedding_profile"
    PROVIDER_ABSTENTION = "provider_abstention"


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    sufficient: bool
    top_score: float | None
    minimum_score: float | None
    category: AbstentionCategory | None


@dataclass(frozen=True, slots=True)
class EvidenceSufficiencyPolicy:
    version: str
    embedding_provider: str
    embedding_model: str
    embedding_version: str
    minimum_score: float | None
    calibrated: bool

    def evaluate(
        self,
        results: list[RetrievalResult],
        *,
        minimum_score_override: float | None = None,
    ) -> EvidenceDecision:
        top_score = results[0].score if results else None
        if top_score is None:
            return EvidenceDecision(
                sufficient=False,
                top_score=None,
                minimum_score=self.minimum_score,
                category=AbstentionCategory.NO_RETRIEVAL_EVIDENCE,
            )
        minimum_score = (
            self.minimum_score if minimum_score_override is None else minimum_score_override
        )
        if not self.calibrated and minimum_score_override is None:
            return EvidenceDecision(
                sufficient=False,
                top_score=top_score,
                minimum_score=None,
                category=AbstentionCategory.UNCALIBRATED_EMBEDDING_PROFILE,
            )
        if minimum_score is None:
            raise ValueError("a calibrated evidence policy requires a minimum score")
        if top_score < minimum_score:
            return EvidenceDecision(
                sufficient=False,
                top_score=top_score,
                minimum_score=minimum_score,
                category=AbstentionCategory.INSUFFICIENT_EVIDENCE_STRENGTH,
            )
        return EvidenceDecision(
            sufficient=True,
            top_score=top_score,
            minimum_score=minimum_score,
            category=None,
        )


def build_evidence_policy(
    *,
    embedding_provider: str,
    embedding_model: str,
    embedding_version: str,
    deterministic_minimum_score: float,
    openai_minimum_score: float | None,
) -> EvidenceSufficiencyPolicy:
    identity = (embedding_provider, embedding_model, embedding_version)
    deterministic_identity = ("deterministic-feature-hashing", "feature-hashing-v1", "1")
    if identity == deterministic_identity:
        return EvidenceSufficiencyPolicy(
            version=EVIDENCE_POLICY_VERSION,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            minimum_score=deterministic_minimum_score,
            calibrated=True,
        )
    if embedding_provider == "openai":
        return EvidenceSufficiencyPolicy(
            version=EVIDENCE_POLICY_VERSION,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            minimum_score=openai_minimum_score,
            calibrated=openai_minimum_score is not None,
        )
    return EvidenceSufficiencyPolicy(
        version=EVIDENCE_POLICY_VERSION,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_version=embedding_version,
        minimum_score=None,
        calibrated=False,
    )
