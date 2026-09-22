import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.db.models import EvaluationMode, EvaluationRunStatus


class ExpectedFact(BaseModel):
    id: str = Field(min_length=1, max_length=127)
    acceptable_phrases: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def phrases_not_blank(self) -> "ExpectedFact":
        if any(not phrase.strip() for phrase in self.acceptable_phrases):
            raise ValueError("acceptable fact phrases must not be blank")
        return self


class ExpectedLocation(BaseModel):
    source: str
    page_number: int | None = Field(default=None, ge=1)
    section: str | None = None


class EvaluationDocumentSpec(BaseModel):
    id: str = Field(min_length=1, max_length=127)
    filename: str = Field(min_length=1, max_length=255)


class GoldenCase(BaseModel):
    id: str = Field(min_length=1, max_length=127)
    question: str = Field(min_length=1)
    answerable: bool
    expected_facts: list[ExpectedFact] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    expected_locations: list[ExpectedLocation] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_expectations(self) -> "GoldenCase":
        if self.answerable and (not self.expected_facts or not self.expected_sources):
            raise ValueError("answerable cases require expected facts and sources")
        if not self.answerable and (self.expected_facts or self.expected_sources):
            raise ValueError("unanswerable cases must not define expected facts or sources")
        return self


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=127)
    version: str = Field(min_length=1, max_length=64)
    documents: list[EvaluationDocumentSpec] = Field(min_length=1)
    cases: list[GoldenCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids_and_sources(self) -> "GoldenDataset":
        document_ids = [document.id for document in self.documents]
        case_ids = [case.id for case in self.cases]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("evaluation document IDs must be unique")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation case IDs must be unique")
        known = set(document_ids)
        for case in self.cases:
            if not set(case.expected_sources) <= known:
                raise ValueError(f"case {case.id} references an unknown expected source")
        return self


class EvaluationRunRequest(BaseModel):
    dataset: str = "golden_v1"
    mode: EvaluationMode = EvaluationMode.END_TO_END
    top_k: int | None = Field(default=None, ge=1)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    evidence_min_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class EvaluationRunResponse(BaseModel):
    run_id: uuid.UUID
    dataset_name: str
    dataset_version: str
    mode: EvaluationMode
    status: EvaluationRunStatus
    retrieval_top_k: int
    retrieval_min_score: float
    embedding_provider: str
    embedding_model: str
    embedding_version: str
    generation_provider: str | None
    generation_model: str | None
    prompt_version: str | None
    evidence_policy_version: str
    evidence_min_score: float | None
    case_count: int
    aggregate_metrics: dict[str, object]
    started_at: datetime
    completed_at: datetime | None
    duration_ms: float | None
    error_category: str | None
    error_message: str | None
    created_at: datetime


class EvaluationCaseResponse(BaseModel):
    case_id: str
    question: str
    answerable: bool
    expected_sources: list[str]
    retrieved_sources: list[str]
    answer: str | None
    abstained: bool | None
    fact_coverage: float | None
    source_hit: bool
    source_recall: float
    reciprocal_rank: float
    citation_validity_rate: float | None
    citation_source_precision: float | None
    citation_source_recall: float | None
    retrieval_ms: float
    generation_ms: float | None
    total_ms: float
    query_run_id: uuid.UUID | None
    error_category: str | None


class EvaluationCaseListResponse(BaseModel):
    run_id: uuid.UUID
    items: list[EvaluationCaseResponse]
