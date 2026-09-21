# GroundedAI

GroundedAI is a reliability-first knowledge assistant and RAG evaluation platform. The project
is being built in measured phases so that each capability is testable before the next is added.

## Current status

Phase 1 (foundation) is implemented:

- FastAPI application with `/health` and dependency-aware `/ready` endpoints
- environment-driven configuration with validation
- JSON structured logging and request correlation IDs
- async SQLAlchemy engine and session factory
- PostgreSQL 17 with pgvector through Docker Compose
- Alembic migration enabling pgvector and creating the initial `documents` table
- unit tests for configuration and health/readiness behavior
- Docker image running as a non-root user

Document upload, extraction, chunking, embeddings, retrieval, grounded chat, evaluation, MCP, and
exports are **not implemented yet**. They belong to later phases.

## Architecture

The API layer is intentionally thin. Configuration and logging live in `app/core`, database
infrastructure and models in `app/db`, and externally visible data contracts in `app/schemas`.
Schema evolution is performed only through Alembic. Readiness requires both a database connection
and the installed `vector` extension; liveness has no external dependency.

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

Run the complete Phase 1 stack with `docker compose up --build`. The migration must be applied
before `/ready` reports ready:

```bash
docker compose run --rm api alembic upgrade head
docker compose up --build
```

## Quality gates

```bash
pytest
ruff check .
mypy .
```

Tests use dependency overrides and do not require a running PostgreSQL instance. Live database and
migration validation does require Docker.

## Configuration

Copy `.env.example` to `.env`. Development defaults contain local-only credentials; production
deployments must supply their own `DATABASE_URL` and other secrets. Provider variables are included
as placeholders for later phases and are not used in Phase 1.

## Next phase

Phase 2 will add a bounded document-ingestion vertical slice: safe TXT/Markdown uploads,
checksum-based deduplication, structure-aware chunking, a deterministic embedding provider, chunk
storage using pgvector, document status APIs, and integration tests. PDF and DOCX extraction will
follow after the text path is verified.
