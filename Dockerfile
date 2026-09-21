FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system groundedai && adduser --system --ingroup groundedai groundedai \
    && mkdir -p /app/data/uploads \
    && chown -R groundedai:groundedai /app/data

COPY pyproject.toml README.md ./
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY evals ./evals
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install .

USER groundedai
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
