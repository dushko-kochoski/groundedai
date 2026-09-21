# GroundedAI

GroundedAI is a reliability-first knowledge assistant and RAG evaluation platform. The project
is being built in measured phases so that each capability is testable before the next is added.

## Current status

Phases 1 through 3 are implemented:

- FastAPI application with `/health` and dependency-aware `/ready` endpoints
- environment-driven configuration with validation
- JSON structured logging and request correlation IDs
- async SQLAlchemy engine and session factory
- PostgreSQL 17 with pgvector through Docker Compose
- Alembic migration enabling pgvector and creating the initial `documents` table
- synchronous TXT and Markdown ingestion with bounded reads and strict UTF-8 decoding
- SHA-256 deduplication enforced by a database uniqueness constraint
- paragraph-aware chunking and deterministic 256-dimensional local embeddings
- pgvector-backed document chunks and persistent original-file storage
- versioned document create/list/detail/delete APIs
- measured exact-cosine retrieval with thresholds, source metadata, and timing diagnostics
- unit, API, and live PostgreSQL/pgvector integration tests
- Docker image running as a non-root user

PDF/DOCX extraction, real embedding providers, RAG answer generation, evaluation, MCP, and exports
are **not implemented yet**. They belong to later phases.

## Architecture

The API layer is intentionally thin. Configuration and logging live in `app/core`, database
infrastructure and models in `app/db`, and externally visible data contracts in `app/schemas`.
Schema evolution is performed only through Alembic. The ingestion service coordinates validation,
storage, chunking, embedding, and repository writes in one transaction, with file cleanup when an
ingestion transaction fails. Readiness requires both a database connection and the installed
`vector` extension; liveness has no external dependency.

## Local development

Requirements: Python 3.12+, Docker, and Docker Compose.

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -e ".[dev]"
copy .env.example .env
docker compose up -d db
alembic upgrade head
uvicorn app.main:app --reload
```

On macOS/Linux, activate with `source .venv/bin/activate` and copy the environment file with
`cp .env.example .env`.

Verify the service:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

Run the complete stack with `docker compose up --build`. The migration must be applied
before `/ready` reports ready:

```bash
docker compose run --rm api alembic upgrade head
docker compose up --build
```

## Document API

Phase 2 accepts `.txt`, `.md`, and `.markdown` files containing UTF-8 or UTF-8-with-BOM text.
Uploads are synchronous and return a `ready` document only after its chunks and vectors commit.

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@notes.md;type=text/markdown"

curl "http://localhost:8000/api/v1/documents?limit=20&offset=0"
curl http://localhost:8000/api/v1/documents/DOCUMENT_ID
curl -X DELETE http://localhost:8000/api/v1/documents/DOCUMENT_ID
```

The default upload limit is 10 MiB. `MAX_UPLOAD_BYTES` changes it. Files are read incrementally and
rejected with HTTP 413 as soon as the limit is exceeded. Empty files, invalid UTF-8, unsupported
extensions, and incompatible content types return structured client errors.

Original bytes are stored below `UPLOAD_DIR` using application-generated UUID filenames; the
caller-provided filename is metadata only and is never used as a path. Docker uses the persistent
`upload_data` volume at `/app/data/uploads`. Deleting a document removes its original file, while
the database removes chunks through `ON DELETE CASCADE`.

SHA-256 is calculated over the original bytes. Uploading identical bytes again returns HTTP 409
with the existing document ID and does not create another file or chunk set. The database checksum
constraint remains the final authority under concurrent uploads.

Chunking combines adjacent paragraphs up to `CHUNK_TARGET_CHARS`, safely splits oversized
paragraphs, and prepends up to `CHUNK_OVERLAP_CHARS` from the preceding chunk. Embeddings use a
network-free token-feature hashing provider and L2 normalization. The current schema dimension is
fixed at 256; changing it requires a migration.

## Retrieval API

GroundedAI can retrieve evidence at this phase but does not yet generate answers. Retrieval uses
the same deterministic embedding-provider interface as ingestion and an exact PostgreSQL/pgvector
cosine-distance scan over chunks belonging to `ready` documents.

```bash
curl -X POST http://localhost:8000/api/v1/retrieval/search \
  -H "Content-Type: application/json" \
  -d '{"query":"refund purchase thirty days","top_k":5,"min_score":0.15}'
```

Representative response:

```json
{
  "query": "refund purchase thirty days",
  "results": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "filename": "refund-policy.txt",
      "chunk_index": 0,
      "text": "Refund requests are accepted within thirty days of purchase.",
      "score": 0.75,
      "page_number": null,
      "section": null
    }
  ],
  "diagnostics": {
    "provider": "deterministic-feature-hashing",
    "requested_top_k": 5,
    "database_candidates_considered": 2,
    "returned_count": 1,
    "min_score": 0.15,
    "max_returned_chars": 6000,
    "returned_chars": 62,
    "embedding_ms": 0.1,
    "search_ms": 2.0,
    "total_ms": 2.2
  }
}
```

The application score is always `1 - pgvector_cosine_distance`; higher values mean greater
similarity. `top_k` is the maximum number of database candidates and defaults to
`RETRIEVAL_TOP_K`. Overrides cannot exceed `RETRIEVAL_MAX_TOP_K`. Results below
`min_score` are omitted, and no qualifying evidence returns HTTP 200 with an empty result list.

Results are returned in descending score order. `RETRIEVAL_MAX_RETURNED_CHARS` keeps the longest
relevance-ordered prefix of complete chunks that fits the budget; chunk text is never cut. Queries
are rejected rather than truncated when they exceed `RETRIEVAL_MAX_QUERY_CHARS`.

The current feature-hashing embeddings are lexical and development-only. They are useful for
repeatable overlap-based retrieval tests, but they are not a production semantic embedding model.
No ANN index is present: exact search remains the measured baseline for the current dataset size.

## Quality gates

```bash
pytest
ruff check .
mypy .
```

The default suite uses dependency overrides and requires no paid or external AI API. The live
integration test is enabled explicitly against an already migrated PostgreSQL/pgvector database:

```bash
set TEST_DATABASE_URL=postgresql+asyncpg://groundedai:groundedai@localhost:5432/groundedai
pytest -m integration
```

On macOS/Linux use `export` instead of `set`.

## Configuration

Copy `.env.example` to `.env`. Development defaults contain local-only credentials; production
deployments must supply their own `DATABASE_URL` and other secrets. Ingestion settings include
`UPLOAD_DIR`, `MAX_UPLOAD_BYTES`, `CHUNK_TARGET_CHARS`, `CHUNK_OVERLAP_CHARS`, and the schema-fixed
`EMBEDDING_DIMENSION=256`. Retrieval settings include `RETRIEVAL_TOP_K`,
`RETRIEVAL_MAX_TOP_K`, `RETRIEVAL_MIN_SCORE`, `RETRIEVAL_MAX_QUERY_CHARS`, and
`RETRIEVAL_MAX_RETURNED_CHARS`.

## Next phase

Phase 4 should add grounded answer generation that consumes `RetrievalService` directly, validates
structured model output and source IDs, rejects citations not present in retrieved evidence, and
abstains when retrieval is insufficient.
