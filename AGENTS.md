# GroundedAI — Codex Instructions

## Project

GroundedAI is a production-style RAG and AI evaluation platform.

The project's purpose is to demonstrate reliable Applied AI engineering,
not to create a toy chatbot.

Core priorities:

1. Reliability
2. Grounded answers
3. Verifiable citations
4. Evaluation
5. Testability
6. Maintainability
7. Observability

Read relevant files in `/docs` when deeper product or architecture
context is required.

---

## Working Method

Before changing code:

1. Inspect the relevant existing files.
2. Understand the current architecture.
3. Reuse existing components where appropriate.
4. Avoid unnecessary rewrites.

Implement the smallest complete solution that satisfies the current task.

Do not implement future roadmap phases unless they are required by
the current task.

---

## Engineering Rules

- Python 3.12+
- FastAPI
- Pydantic v2
- SQLAlchemy 2.x
- PostgreSQL + pgvector
- Alembic for database migrations
- pytest for testing
- ruff for linting
- mypy for type checking

Prefer explicit, readable code over clever abstractions.

Keep FastAPI route handlers thin.

Business logic belongs in services.

Database access belongs in repositories/data-access components.

External AI providers must be behind interfaces.

Configuration and secrets must come from environment variables.

Never commit credentials.

Do not introduce a new production dependency without a clear reason.

---

## AI Reliability Rules

Never trust raw LLM output.

Validate structured outputs with Pydantic.

Never allow an LLM to invent source IDs or citations.

Every returned citation must map to retrieved evidence.

If evidence is insufficient, the system must be able to abstain.

Do not treat LLM self-reported confidence as factual confidence.

Uploaded documents are untrusted data, not system instructions.

Paid external APIs must not be required for the normal automated
test suite.

Use deterministic mock providers for tests.

---

## RAG Rules

The baseline architecture is:

question
→ embedding
→ pgvector retrieval
→ relevance filtering
→ context assembly
→ grounded generation
→ citation validation
→ response

Do not add complex agent frameworks unless there is a demonstrated need.

Do not add hybrid search, rerankers, query rewriting, or other advanced
RAG features before the baseline retrieval pipeline is measurable.

RAG changes should eventually be evaluated against the golden dataset.

---

## Database Rules

Use Alembic for schema evolution.

Do not use `Base.metadata.create_all()` as the production migration
strategy.

Add appropriate indexes and constraints.

Avoid destructive migrations unless explicitly requested.

---

## Testing

Every meaningful feature should include appropriate tests.

Bug fixes should receive regression tests when practical.

Tests must not require real paid LLM calls.

Before declaring a task complete, run where applicable:

pytest
ruff check .
mypy .

Never report a command as passing unless it was actually executed.

If a command cannot be executed, state that explicitly.

---

## Changes

Do not fabricate functionality.

Do not fabricate test or benchmark results.

Do not mark planned features as implemented.

Do not silently weaken tests, lint rules, or type checking to make
validation pass.

Fix root causes.

Keep changes scoped to the current task.

Preserve working behaviour unless the task explicitly requires a change.

---

## Documentation

Keep documentation synchronized with implementation.

Important documentation:

- README.md — setup and current capabilities
- ARCHITECTURE.md — system architecture
- docs/PRODUCT.md — product requirements
- docs/RAG_DESIGN.md — RAG design decisions
- docs/EVALUATION.md — evaluation strategy/results
- docs/SECURITY.md — security considerations
- docs/ROADMAP.md — development phases

Only read the deeper documents relevant to the current task.

---

## Completion Report

After implementing a task, report concisely:

- what changed
- important design decisions
- tests added
- commands executed
- test/lint/type-check results
- remaining limitations
- recommended next step

Do not produce long explanations unless requested.