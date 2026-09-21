import hashlib
from io import BytesIO

import pytest
from fastapi import UploadFile

from app.core.exceptions import ApplicationError
from app.services.ingestion.validation import (
    decode_and_normalize_text,
    read_upload_bounded,
    validate_upload_metadata,
)


@pytest.mark.parametrize("filename", ["notes.txt", "README.md", "guide.markdown"])
def test_supported_extensions_are_accepted(filename: str) -> None:
    result = validate_upload_metadata(filename, "text/plain")

    assert result.filename == filename


def test_user_path_is_removed_from_display_filename() -> None:
    result = validate_upload_metadata("../../notes.txt", "text/plain")

    assert result.filename == "notes.txt"


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(ApplicationError, match="Only .txt") as error:
        validate_upload_metadata("report.pdf", "application/pdf")

    assert error.value.status_code == 415
    assert error.value.code == "unsupported_file_type"


def test_unsupported_content_type_is_rejected() -> None:
    with pytest.raises(ApplicationError) as error:
        validate_upload_metadata("report.txt", "application/pdf")

    assert error.value.code == "unsupported_content_type"


async def test_bounded_read_calculates_sha256() -> None:
    content = b"grounded evidence"
    upload = UploadFile(BytesIO(content), filename="notes.txt")

    result = await read_upload_bounded(upload, max_bytes=len(content))

    assert result.content == content
    assert result.checksum == hashlib.sha256(content).hexdigest()


async def test_bounded_read_rejects_oversized_file() -> None:
    upload = UploadFile(BytesIO(b"12345"), filename="notes.txt")

    with pytest.raises(ApplicationError) as error:
        await read_upload_bounded(upload, max_bytes=4)

    assert error.value.status_code == 413
    assert error.value.code == "upload_too_large"


async def test_bounded_read_rejects_empty_file() -> None:
    upload = UploadFile(BytesIO(b""), filename="notes.txt")

    with pytest.raises(ApplicationError) as error:
        await read_upload_bounded(upload, max_bytes=10)

    assert error.value.code == "empty_file"


def test_invalid_utf8_is_rejected() -> None:
    with pytest.raises(ApplicationError) as error:
        decode_and_normalize_text(b"\xff\xfe\xfa")

    assert error.value.code == "invalid_text_encoding"


def test_utf8_bom_and_line_endings_are_normalized() -> None:
    result = decode_and_normalize_text(b"\xef\xbb\xbfFirst  \r\n\r\n\r\nSecond\r\n")

    assert result == "First\n\nSecond"


def test_whitespace_only_document_is_rejected() -> None:
    with pytest.raises(ApplicationError) as error:
        decode_and_normalize_text(b" \n\n\t")

    assert error.value.code == "empty_document"

