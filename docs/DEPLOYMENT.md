# Hosting GroundedAI

## Current status

The browser workspace and API run in one FastAPI process. This avoids a second build,
CORS configuration and exposing provider credentials in frontend code. The existing
Docker image packages the workspace assets. This update has not been deployed publicly.

A real deployment needs:

- PostgreSQL with pgvector, with `alembic upgrade head` run as a release step.
- Durable storage for original uploads. `LocalDocumentStorage` currently requires a
  mounted persistent directory; database persistence alone does not retain original files.
- HTTPS, `APP_ENV=production`, a strong `API_ACCESS_KEY`, and private database credentials.
- Provider credentials only when opting into OpenAI, plus an explicitly calibrated
  evidence threshold for OpenAI embeddings.

## Lowest-change deployment

Run the supplied Docker image on a host supporting persistent volumes and a reachable
PostgreSQL/pgvector database. Mount a volume at `/app/data/uploads`, set `UPLOAD_DIR` to
that directory, and ensure the non-root container user can write there. Set runtime
configuration through the host's secret/environment settings.

Apply migrations using the same image and database configuration before starting traffic.
Check `/health`, `/ready`, and a complete upload → query → cited answer or abstention →
delete flow. `/ready` currently checks database connectivity and the vector extension,
not migration head; it does not replace the migration or functional checks.

The supplied Compose configuration is for local development, with disposable database
credentials and a published database port. Do not use it unchanged as a public production
configuration. Keep database ports private, back up the database and upload volume, and
verify recovery of both.

## Vercel assessment

Vercel supports FastAPI, so Python itself is not the blocker. The current local-file
storage implementation needs replacement with durable object storage before running the
backend as a serverless deployment. Using a temporary upload directory would not satisfy
GroundedAI's promise to retain original files. A managed PostgreSQL/pgvector connection
and migration process would also be needed.

A future Vercel integration should:

1. Introduce a storage interface and a tested durable object-storage implementation.
2. Configure PostgreSQL connection pooling and migrations for the selected host.
3. Check current request-size/duration limits against uploads and synchronous evaluation.
4. Verify packaged assets, upload/query/delete behavior and storage cleanup in a preview.
5. Set production secrets privately and keep preview data separate from production.

No placeholder deployment configuration is shipped that would claim persistence works.
The connected Vercel account returned no teams during this update, so no project was created.

Official framework reference: <https://vercel.com/docs/frameworks/backend/fastapi>

## Access model and remaining security work

`API_ACCESS_KEY` is a single shared operator credential. All holders can upload, query,
list and delete any document and trigger evaluations. It is not multi-tenant authentication.
Use a generated random key (for example, `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
and store it outside the repository. Rotate it by changing the environment value and
restarting the service. Do not embed it in the HTML, a URL, or a public demo link.

API authentication runs before multipart bodies are parsed. The UI renders untrusted
filenames, answers and citation metadata as text, and serves a restrictive same-origin
Content Security Policy. API responses use `Cache-Control: no-store`.

Before opening access to untrusted users, implement user identity and document isolation,
request and spend quotas, reverse-proxy body limits, and stronger parser resource isolation.
The current application-level upload limit does not bound the entire multipart request at
the proxy, and there is no distributed rate limiter. Use deterministic providers and
non-sensitive synthetic documents for an operator-controlled portfolio demonstration.
