import os
import uuid
from pathlib import Path


class LocalDocumentStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def write(self, document_id: uuid.UUID, extension: str, content: bytes) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        key = f"{document_id.hex}{extension}"
        destination = self._resolve(key)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return key

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def path_for(self, key: str) -> Path:
        return self._resolve(key)

    def _resolve(self, key: str) -> Path:
        if not key or Path(key).is_absolute():
            raise ValueError("storage key must be a relative filename")
        candidate = (self.root / key).resolve()
        if candidate.parent != self.root:
            raise ValueError("storage key escapes the configured upload directory")
        return candidate
