import hashlib
from io import BytesIO
from pathlib import Path
from typing import Protocol

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.db.models import DocumentStatus
from app.repositories.documents import DocumentRepository
from app.schemas.evaluation import GoldenDataset
from app.services.evaluation.datasets import DatasetRegistry
from app.services.ingestion.service import DocumentServiceProtocol

_CONTENT_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class EvaluationFixturePreparerProtocol(Protocol):
    async def prepare(self, dataset: GoldenDataset) -> dict[str, str]: ...


class EvaluationFixturePreparer:
    """Reuse checksum-identical ready fixtures without deleting unrelated documents."""

    def __init__(
        self,
        *,
        registry: DatasetRegistry,
        documents: DocumentRepository,
        ingestion: DocumentServiceProtocol,
    ) -> None:
        self.registry = registry
        self.documents = documents
        self.ingestion = ingestion

    async def prepare(self, dataset: GoldenDataset) -> dict[str, str]:
        document_to_fixture: dict[str, str] = {}
        for fixture in dataset.documents:
            path = self.registry.fixture_path(fixture.filename)
            content = path.read_bytes()
            checksum = hashlib.sha256(content).hexdigest()
            existing = await self.documents.get_by_checksum(checksum)
            if existing is not None and existing.status == DocumentStatus.READY:
                await self.documents.mark_evaluation_fixture(existing, fixture.id)
                document_to_fixture[str(existing.id)] = fixture.id
                continue
            response = await self.ingestion.ingest(
                UploadFile(
                    filename=fixture.filename,
                    file=BytesIO(content),
                    headers=Headers({"content-type": _content_type(path)}),
                )
            )
            created = await self.documents.get_by_checksum(checksum)
            if created is None:
                raise RuntimeError("ingested evaluation fixture could not be reloaded")
            await self.documents.mark_evaluation_fixture(created, fixture.id)
            document_to_fixture[str(response.id)] = fixture.id
        return document_to_fixture


def _content_type(path: Path) -> str:
    try:
        return _CONTENT_TYPES[path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported evaluation fixture extension: {path.suffix}") from exc
