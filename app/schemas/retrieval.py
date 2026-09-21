import uuid

from pydantic import BaseModel, Field


class RetrievalSearchRequest(BaseModel):
    query: str
    top_k: int | None = Field(default=None, ge=1)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class RetrievalResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    chunk_index: int = Field(ge=0)
    text: str
    score: float = Field(ge=-1.0, le=1.0)
    page_number: int | None = None
    section: str | None = None


class RetrievalDiagnostics(BaseModel):
    provider: str
    requested_top_k: int
    database_candidates_considered: int
    returned_count: int
    min_score: float
    max_returned_chars: int
    returned_chars: int
    embedding_ms: float
    search_ms: float
    total_ms: float


class RetrievalSearchResponse(BaseModel):
    query: str
    results: list[RetrievalResult]
    diagnostics: RetrievalDiagnostics

