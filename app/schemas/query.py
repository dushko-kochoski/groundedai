import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.db.models import QueryRunStatus


class GroundedQueryRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class ValidatedCitation(BaseModel):
    source_id: str
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    filename: str
    chunk_index: int
    page_number: int | None
    section: str | None
    score: float


class GroundedQueryDiagnostics(BaseModel):
    retrieved_count: int
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    provider: str
    model: str
    prompt_version: str
    evidence_policy_version: str
    evidence_min_score: float | None
    top_evidence_score: float | None
    evidence_sufficient: bool | None
    abstention_category: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class GroundedQueryResponse(BaseModel):
    run_id: uuid.UUID
    question: str
    answer: str | None
    abstained: bool
    abstention_reason: str | None
    citations: list[ValidatedCitation]
    diagnostics: GroundedQueryDiagnostics


class QueryRunResponse(BaseModel):
    run_id: uuid.UUID
    status: QueryRunStatus
    question: str
    answer: str | None
    abstained: bool | None
    abstention_reason: str | None
    citations: list[dict[str, Any]]
    retrieval_snapshot: list[dict[str, Any]]
    diagnostics: GroundedQueryDiagnostics
    error_code: str | None
    error_message: str | None
    created_at: datetime
