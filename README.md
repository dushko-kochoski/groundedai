# GroundedAI

GroundedAI is a reliability-first knowledge assistant and RAG evaluation platform. The project
is being built in measured phases so that each capability is testable before the next is added.

This is a personal portfolio project created to demonstrate production-minded applied AI
engineering: evidence-bounded answers, verifiable citation provenance, measurable retrieval and
abstention behavior, deterministic tests, and explicit failure analysis. It is not presented as a
hosted commercial product, and the documented results are local measurements over synthetic
repository-owned fixtures.

## Stack

- Python 3.12, FastAPI, Pydantic v2, and Uvicorn
- SQLAlchemy 2.x async sessions, Alembic, PostgreSQL 17, and pgvector
- Deterministic offline embedding/generation providers for development and tests
- Optional OpenAI embeddings and Responses API generation behind provider interfaces
- pytest, Ruff, and mypy for automated quality checks
- Docker and Docker Compose for the local API/database stack

## Verified capabilities

Phases 1 through 7 are implemented:

- FastAPI application with `/health` and dependency-aware `/ready` endpoints
- environment-driven configuration with validation
- JSON structured logging and request correlation IDs
- async SQLAlchemy engine and session factory
- PostgreSQL 17 with pgvector through Docker Compose
- Alembic migration enabling pgvector and creating the initial `documents` table
- synchronous TXT, Markdown, text-PDF, and DOCX ingestion with bounded reads
- SHA-256 deduplication enforced by a database uniqueness constraint
- paragraph-aware chunking with PDF page and DOCX heading metadata
- configurable deterministic or OpenAI embedding providers using one 1,536-dimensional space
- pgvector-backed document chunks and persistent original-file storage
- versioned document create/list/detail/delete APIs
- measured exact-cosine retrieval with thresholds, source metadata, and timing diagnostics
- grounded deterministic or OpenAI Responses API generation behind a provider interface
- application-validated ephemeral citation aliases, deterministic abstention, and persisted query runs
- versioned golden datasets with retrieval-only and end-to-end evaluation modes
- persisted run/case metrics for retrieval, facts, citations, abstention, latency, and token usage
- evaluation API and CLI using the same production services as normal queries
- versioned, embedding-specific evidence sufficiency before generation
- persisted evidence decisions and a reproducible threshold-calibration CLI/artifact
- unit, API, and live PostgreSQL/pgvector integration tests
- Docker image running as a non-root user

OCR, claim-level entailment evaluation, Langfuse, MCP, and exports are **not implemented yet**.
PDF support is limited to documents with extractable text; scanned/image-only PDFs are rejected.

## Current limitations

- The default feature-hashing embeddings and extractive generator are deterministic development
  providers, not production semantic AI models.
- Retrieval is an exact cosine scan; there is no approximate-nearest-neighbor index, hybrid search,
  reranker, or query rewriting.
- Citation validation proves that citations came from retrieved evidence, but it does not prove
  claim-level semantic entailment.
- The calibrated deterministic evidence threshold did not generalize well to the frozen holdout
  dataset. The measured failures are documented in `docs/EVALUATION.md`.
- OpenAI-backed behavior requires user-supplied credentials and an operator-configured evidence
  threshold; it is not exercised by the normal automated test suite.
- Ingestion is synchronous. OCR, authentication/authorization, a browser UI, distributed jobs,
  hosted deployment configuration, tracing, exports, and MCP integrations are not implemented.

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

The endpoint accepts `.txt`, `.md`, `.markdown`, `.pdf`, and `.docx`. Text formats must contain
UTF-8 or UTF-8-with-BOM text. PDF extraction is page-by-page and preserves one-based page numbers.
DOCX extraction preserves heading names as section metadata where Word styles identify them.
Legacy `.doc`, macros, OCR, and text inside images are not supported. Uploads are synchronous and
return a `ready` document only after its chunks and vectors commit.

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@notes.md;type=text/markdown"

curl "http://localhost:8000/api/v1/documents?limit=20&offset=0"
curl http://localhost:8000/api/v1/documents/DOCUMENT_ID
curl -X DELETE http://localhost:8000/api/v1/documents/DOCUMENT_ID
```

The default upload limit is 10 MiB. `MAX_UPLOAD_BYTES` changes it. Files are read incrementally and
rejected with HTTP 413 as soon as the limit is exceeded. Empty files, invalid UTF-8,
malformed/encrypted PDFs, malformed DOCX files, unsupported extensions, and incompatible content
types return structured client errors.

Original bytes are stored below `UPLOAD_DIR` using application-generated UUID filenames; the
caller-provided filename is metadata only and is never used as a path. Docker uses the persistent
`upload_data` volume at `/app/data/uploads`. Deleting a document removes its original file, while
the database removes chunks through `ON DELETE CASCADE`.

SHA-256 is calculated over the original bytes. Uploading identical bytes again returns HTTP 409
with the existing document ID and does not create another file or chunk set. The database checksum
constraint remains the final authority under concurrent uploads.

Chunking combines adjacent paragraphs up to `CHUNK_TARGET_CHARS`, safely splits oversized
paragraphs, and prepends up to `CHUNK_OVERLAP_CHARS` from the preceding chunk. PDF chunks never
cross page boundaries; DOCX chunks never cross parsed heading-section boundaries.

The default `deterministic` embedding provider uses network-free, L2-normalized token feature
hashing for repeatable development and CI. It is lexical, not a production semantic model. The
`openai` provider uses `text-embedding-3-small`, validates responses, batches inputs, and applies
bounded retries to timeouts, transport errors, HTTP 429, and transient 5xx responses. Credentials
come only from settings and are never logged.

```text
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=<configured outside the repository>
EMBEDDING_DIMENSION=1536
```

Provider, model, dimension, and implementation version are stored on every chunk and enforced by
retrieval filters. The database uses one `VECTOR(1536)` column. Migration `20260921_0003` refuses
to run while documents exist because old 256-dimensional vectors cannot be reused; delete them
through the API before migration, then re-ingest. Any future model or dimension change likewise
requires re-embedding.

## Retrieval API

Retrieval remains independently available. It uses the same embedding-provider interface as
ingestion and an exact PostgreSQL/pgvector cosine-distance scan over chunks belonging to `ready`
documents.

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
      "model": "feature-hashing-v1",
      "dimension": 1536,
      "version": "1",
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

The default feature-hashing embeddings are lexical and development-only. They are useful for
repeatable overlap-based retrieval tests, but they are not a production semantic embedding model.
No ANN index is present: exact search remains the measured baseline for the current dataset size.

## Grounded query API

`POST /api/v1/query` calls `RetrievalService.retrieve()` first, uses only its bounded complete
chunks, assigns ephemeral aliases (`S1`, `S2`, ...), and then invokes the configured generation
provider. Retrieved document text is serialized as untrusted JSON data and is explicitly separated
from the versioned application instructions (`grounded_answer_v1`).

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the remote work policy?","top_k":5,"min_score":0.25}'

curl http://localhost:8000/api/v1/query-runs/RUN_ID
```

The model may return only aliases supplied in the prompt. GroundedAI rejects unknown aliases and
non-abstained answers without citations. Duplicate citations are normalized in first-use order;
filenames, document IDs, chunk IDs, pages, sections, and scores always come from trusted retrieval
results rather than model output. This validates citation existence and provenance, but does not
prove that every answer sentence is semantically entailed by its citation. Phase 6 deliberately
records this as a remaining limitation rather than inventing a deterministic groundedness score.

Zero qualifying retrieval results produce a deterministic abstention without calling generation.
Retrieved results also pass through `evidence_policy_v1`; the calibrated deterministic profile
requires a top score of at least 0.26. Weak evidence abstains before generation, records a distinct
category, and keeps generation latency at zero. Evidence policies are tied to embedding
provider/model/version. OpenAI embeddings have no calibrated default and fail safe unless an
explicit `EVIDENCE_OPENAI_MIN_SCORE` operator override is configured. With sufficient evidence,
the provider may also return a validated structured abstention. Completed
answers and abstentions are persisted in `query_runs`; provider, malformed-output, and citation
integrity failures are persisted with safe error categories and returned without upstream bodies.
Snapshots contain source metadata, not embeddings or full document text.

The default `deterministic` generation provider is an offline extractive provider for local
development and CI. Its `extractive-v2` behavior can use one passage per distinct document for
conjoined questions; it remains a test double, not semantic generation. Production generation uses
the OpenAI Responses API with strict JSON Schema
Structured Outputs, disabled provider-side response storage, bounded transient retries, and token
usage capture when reported:

```text
GENERATION_PROVIDER=openai
GENERATION_MODEL=gpt-5-mini
OPENAI_API_KEY=<configured outside the repository>
GENERATION_TIMEOUT_SECONDS=30
GENERATION_MAX_RETRIES=2
GENERATION_MAX_OUTPUT_TOKENS=800
```

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
`UPLOAD_DIR`, `MAX_UPLOAD_BYTES`, `CHUNK_TARGET_CHARS`, and `CHUNK_OVERLAP_CHARS`. Embedding settings
include `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, the schema-fixed
`EMBEDDING_DIMENSION=1536`, `EMBEDDING_TIMEOUT_SECONDS`, `EMBEDDING_MAX_RETRIES`, and
`EMBEDDING_BATCH_SIZE`. Retrieval settings include `RETRIEVAL_TOP_K`,
`RETRIEVAL_MAX_TOP_K`, `RETRIEVAL_MIN_SCORE`, `RETRIEVAL_MAX_QUERY_CHARS`, and
`RETRIEVAL_MAX_RETURNED_CHARS`. Generation settings include `GENERATION_PROVIDER`,
`GENERATION_MODEL`, `OPENAI_API_KEY`, `GENERATION_TIMEOUT_SECONDS`,
`GENERATION_MAX_RETRIES`, and `GENERATION_MAX_OUTPUT_TOKENS`. Evidence settings include the
calibrated `EVIDENCE_DETERMINISTIC_MIN_SCORE` and optional, explicitly uncalibrated
`EVIDENCE_OPENAI_MIN_SCORE` override.

## Automated evaluation

The application-controlled `golden_v1` dataset contains 30 synthetic cases over repository-owned
TXT, Markdown, text-PDF, and DOCX fixtures. Twenty cases are answerable and ten are explicitly
unanswerable. Expected facts use normalized acceptable phrases, and expected sources use stable
fixture IDs. At run time, those IDs are mapped to the trusted UUIDs of checksum-reused fixture
documents, so matching is independent of local paths and unrelated user documents are excluded
from evaluation retrieval.

Run the full grounded path or retrieval in isolation:

```bash
python -m scripts.evaluate --dataset golden_v1 --mode end_to_end
python -m scripts.evaluate --dataset golden_v1 --mode retrieval_only --json
python -m scripts.calibrate --run-id BASELINE_RUN_ID --output evals/results/calibration.json

curl -X POST http://localhost:8000/api/v1/evaluations/run \
  -H "Content-Type: application/json" \
  -d '{"dataset":"golden_v1","mode":"end_to_end"}'
curl http://localhost:8000/api/v1/evaluations/runs/RUN_ID
curl http://localhost:8000/api/v1/evaluations/runs/RUN_ID/cases
```

Dataset names resolve through a fixed registry; the API never accepts arbitrary filesystem paths.
Repeated runs reuse ready fixture documents by SHA-256 checksum and never delete unrelated user
documents. Evaluation-owned documents are metadata-tagged, excluded from ordinary retrieval, and
included only through the evaluator's explicit UUID scope. Run aggregates and individual case
results are persisted separately. Retrieval metrics
are Hit@K, Recall@K, MRR, zero-result rate, and result count. End-to-end metrics add deterministic
fact coverage, citation provenance/source matching, abstention classification, latency, and token
totals where providers report them. Cost remains null unless real token usage and a separately
versioned matching price configuration exist.

See [docs/EVALUATION.md](docs/EVALUATION.md) for metric definitions, the measured deterministic
baseline, failure analysis, dataset versioning rules, and current limitations.

## Next phase

Phase 8 should add vendor-neutral tracing/observability around the already measured ingestion,
retrieval, evidence-policy, generation, and evaluation paths. It should preserve
request/query/evaluation correlation without making core services depend directly on a tracing
vendor.
