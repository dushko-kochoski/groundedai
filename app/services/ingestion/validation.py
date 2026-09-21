import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath

from fastapi import UploadFile

from app.core.exceptions import ApplicationError

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})
SUPPORTED_CONTENT_TYPES = frozenset(
    {
        "application/octet-stream",
        "application/x-markdown",
        "text/markdown",
        "text/plain",
    }
)
READ_BLOCK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    filename: str
    extension: str
    content_type: str


@dataclass(frozen=True, slots=True)
class UploadBytes:
    content: bytes
    checksum: str


def validate_upload_metadata(filename: str | None, content_type: str | None) -> ValidatedUpload:
    normalized_name = PurePosixPath((filename or "").replace("\\", "/")).name
    if not normalized_name or len(normalized_name) > 255:
        raise ApplicationError(
            status_code=400,
            code="invalid_filename",
            message="A valid filename of at most 255 characters is required.",
        )

    extension = PurePosixPath(normalized_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ApplicationError(
            status_code=415,
            code="unsupported_file_type",
            message="Only .txt, .md, and .markdown files are supported.",
            details={"extension": extension or None},
        )

    normalized_content_type = (content_type or "application/octet-stream").lower()
    normalized_content_type = normalized_content_type.split(";", maxsplit=1)[0].strip()
    if normalized_content_type not in SUPPORTED_CONTENT_TYPES:
        raise ApplicationError(
            status_code=415,
            code="unsupported_content_type",
            message="The uploaded content type is not supported for a text document.",
            details={"content_type": normalized_content_type},
        )
    return ValidatedUpload(normalized_name, extension, normalized_content_type)


async def read_upload_bounded(upload: UploadFile, max_bytes: int) -> UploadBytes:
    content = bytearray()
    digest = hashlib.sha256()
    while True:
        block = await upload.read(min(READ_BLOCK_BYTES, max_bytes + 1 - len(content)))
        if not block:
            break
        content.extend(block)
        if len(content) > max_bytes:
            raise ApplicationError(
                status_code=413,
                code="upload_too_large",
                message="The uploaded file exceeds the configured size limit.",
                details={"max_upload_bytes": max_bytes},
            )
        digest.update(block)

    if not content:
        raise ApplicationError(
            status_code=400,
            code="empty_file",
            message="The uploaded file is empty.",
        )
    return UploadBytes(bytes(content), digest.hexdigest())


def decode_and_normalize_text(content: bytes) -> str:
    try:
        text = content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ApplicationError(
            status_code=400,
            code="invalid_text_encoding",
            message="The uploaded document must contain valid UTF-8 text.",
        ) from exc

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    while "\n\n\n" in normalized:
        normalized = normalized.replace("\n\n\n", "\n\n")
    normalized = normalized.strip()
    if not normalized:
        raise ApplicationError(
            status_code=400,
            code="empty_document",
            message="The uploaded file does not contain any usable text.",
        )
    return normalized

