# GroundedAI

GroundedAI is a reliability-first knowledge assistant and RAG evaluation platform. The project
is being built in measured phases so that each capability is testable before the next is added.

## Current status

Phases 1 and 2 are implemented:

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
- unit, API, and live PostgreSQL/pgvector integration tests
- Docker image running as a non-root user

PDF/DOCX extraction, real embedding providers, semantic query retrieval, RAG generation,
evaluation, MCP, and exports are **not implemented yet**. They belong to later phases.

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
`EMBEDDING_DIMENSION=256`.

## Next phase

Phase 3 should add question embedding and measured pgvector similarity search with configurable
Top-K and relevance filtering, returning chunk/source metadata and retrieval diagnostics. It should
not add answer generation until retrieval behavior is evaluated independently.
