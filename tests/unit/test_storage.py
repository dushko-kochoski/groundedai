import uuid
from pathlib import Path

import pytest

from app.services.storage import LocalDocumentStorage


def test_storage_uses_generated_safe_filename_and_deletes(tmp_path: Path) -> None:
    storage = LocalDocumentStorage(tmp_path)
    document_id = uuid.uuid4()

    key = storage.write(document_id, ".txt", b"trusted bytes")

    assert key == f"{document_id.hex}.txt"
    assert storage.path_for(key).parent == tmp_path.resolve()
    assert storage.path_for(key).read_bytes() == b"trusted bytes"
    storage.delete(key)
    assert not storage.exists(key)


@pytest.mark.parametrize("key", ["../escape.txt", "folder/file.txt", ""])
def test_storage_rejects_unsafe_keys(tmp_path: Path, key: str) -> None:
    storage = LocalDocumentStorage(tmp_path)

    with pytest.raises(ValueError):
        storage.path_for(key)

