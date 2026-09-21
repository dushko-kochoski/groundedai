from io import BytesIO

import pytest
from docx import Document
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from app.core.exceptions import ApplicationError
from app.services.ingestion.parsers import DocxDocumentParser, PdfDocumentParser


def pdf_bytes(*pages: str) -> bytes:
    output = BytesIO()
    document = canvas.Canvas(output)
    for text in pages:
        document.drawString(72, 720, text)
        document.showPage()
    document.save()
    return output.getvalue()


def docx_bytes() -> bytes:
    output = BytesIO()
    document = Document()
    document.add_heading("Remote Work", level=1)
    document.add_paragraph("Employees may work from home two days each week.")
    document.add_heading("Security", level=1)
    document.add_paragraph("Use a company-managed device.")
    document.save(output)
    return output.getvalue()


def test_pdf_parser_preserves_one_based_page_numbers() -> None:
    units = PdfDocumentParser().parse(pdf_bytes("First page", "Second page"))

    assert [(unit.page_number, unit.text) for unit in units] == [
        (1, "First page"),
        (2, "Second page"),
    ]


def test_pdf_parser_rejects_empty_and_malformed_documents() -> None:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(output)

    with pytest.raises(ApplicationError, match="extractable text"):
        PdfDocumentParser().parse(output.getvalue())
    with pytest.raises(ApplicationError, match="malformed"):
        PdfDocumentParser().parse(b"%PDF-not-a-document")


def test_pdf_parser_rejects_encrypted_documents() -> None:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("not-used-by-parser")
    writer.write(output)

    with pytest.raises(ApplicationError, match="Encrypted") as error:
        PdfDocumentParser().parse(output.getvalue())

    assert error.value.code == "encrypted_pdf"


def test_docx_parser_preserves_heading_sections() -> None:
    units = DocxDocumentParser().parse(docx_bytes())

    assert [unit.section for unit in units] == ["Remote Work", "Security"]
    assert "work from home" in units[0].text


def test_docx_parser_rejects_malformed_document() -> None:
    with pytest.raises(ApplicationError, match="malformed"):
        DocxDocumentParser().parse(b"PK-not-a-docx")
