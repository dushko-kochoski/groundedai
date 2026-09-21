from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Protocol
from zipfile import BadZipFile

from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import ApplicationError
from app.services.ingestion.validation import decode_and_normalize_text


@dataclass(frozen=True, slots=True)
class SourceUnit:
    text: str
    page_number: int | None = None
    section: str | None = None


class DocumentParser(Protocol):
    def parse(self, content: bytes) -> list[SourceUnit]: ...


class TextDocumentParser:
    def parse(self, content: bytes) -> list[SourceUnit]:
        return [SourceUnit(text=decode_and_normalize_text(content))]


class PdfDocumentParser:
    def parse(self, content: bytes) -> list[SourceUnit]:
        try:
            reader = PdfReader(BytesIO(content), strict=False)
            if reader.is_encrypted:
                raise ApplicationError(
                    status_code=400,
                    code="encrypted_pdf",
                    message="Encrypted PDF documents are not supported.",
                )
            units = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = self._normalize_extracted_text(page.extract_text() or "")
                if text:
                    units.append(SourceUnit(text=text, page_number=page_number))
        except ApplicationError:
            raise
        except (PdfReadError, ValueError, TypeError, OSError) as exc:
            raise ApplicationError(
                status_code=400,
                code="invalid_pdf",
                message="The uploaded PDF is malformed or unsupported.",
            ) from exc
        if not units:
            raise ApplicationError(
                status_code=400,
                code="pdf_no_extractable_text",
                message="The PDF does not contain extractable text. OCR is not supported.",
            )
        return units

    @staticmethod
    def _normalize_extracted_text(text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
        while "\n\n\n" in normalized:
            normalized = normalized.replace("\n\n\n", "\n\n")
        return normalized.strip()


class DocxDocumentParser:
    def parse(self, content: bytes) -> list[SourceUnit]:
        try:
            document = DocxDocument(BytesIO(content))
        except (PackageNotFoundError, BadZipFile, ValueError, KeyError, OSError) as exc:
            raise ApplicationError(
                status_code=400,
                code="invalid_docx",
                message="The uploaded DOCX is malformed or unsupported.",
            ) from exc

        units: list[SourceUnit] = []
        current_section: str | None = None
        section_parts: list[str] = []

        def flush() -> None:
            if section_parts:
                units.append(
                    SourceUnit(text="\n\n".join(section_parts), section=current_section)
                )
                section_parts.clear()

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = paragraph.style.name if paragraph.style is not None else ""
            if style_name.casefold().startswith("heading"):
                flush()
                current_section = text
                section_parts.append(text)
            else:
                section_parts.append(text)
        flush()

        if not units:
            raise ApplicationError(
                status_code=400,
                code="empty_document",
                message="The uploaded file does not contain any usable text.",
            )
        return units


class DocumentParserRegistry:
    def __init__(self) -> None:
        text_parser = TextDocumentParser()
        self._parsers: dict[str, DocumentParser] = {
            ".txt": text_parser,
            ".md": text_parser,
            ".markdown": text_parser,
            ".pdf": PdfDocumentParser(),
            ".docx": DocxDocumentParser(),
        }

    def parse(self, extension: str, content: bytes) -> list[SourceUnit]:
        return self._parsers[extension].parse(content)
