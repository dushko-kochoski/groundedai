"""Build the repository-owned PDF and DOCX golden-dataset fixtures."""

from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "evals" / "fixtures"


def build_pdf() -> None:
    path = FIXTURES / "employee-handbook.pdf"
    canvas = Canvas(str(path), pagesize=letter)
    pages = [
        (
            "Flexible Work Policy",
            [
                "Employees may work remotely two days each week.",
                "Remote-work days require manager approval and must be recorded "
                "in the team calendar.",
                "Core collaboration hours are 10:00 to 15:00 local time.",
            ],
        ),
        (
            "Annual Leave Policy",
            [
                "Full-time employees receive 20 days of paid annual leave each calendar year.",
                "Leave requests longer than five consecutive days require two weeks of notice.",
                "Unused annual leave may carry over up to five days into the next year.",
            ],
        ),
        (
            "Information Security Policy",
            [
                "Multi-factor authentication is required for all company systems.",
                "Suspected security incidents must be reported within one hour of discovery.",
                "Company laptops must use full-disk encryption.",
            ],
        ),
    ]
    width, height = letter
    for page_number, (title, paragraphs) in enumerate(pages, start=1):
        canvas.setFont("Helvetica-Bold", 18)
        canvas.drawString(Inches(0.8), height - Inches(0.9), title)
        canvas.setFont("Helvetica", 11)
        y = height - Inches(1.45)
        for paragraph in paragraphs:
            canvas.drawString(Inches(0.8), y, paragraph)
            y -= Inches(0.42)
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(width - Inches(0.7), Inches(0.55), f"Page {page_number}")
        canvas.showPage()
    canvas.save()


def build_docx() -> None:
    path = FIXTURES / "expense-policy.docx"
    document = Document()
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    title = document.add_paragraph(style="Title")
    title.add_run("Employee Expense Policy")
    document.add_paragraph(
        "This policy defines approval, evidence, and reimbursement rules for employee expenses."
    )
    content = [
        (
            "Travel",
            [
                "Air travel must be booked in economy class for journeys under six hours.",
                "International travel requires manager approval before booking.",
            ],
        ),
        (
            "Meals",
            [
                "The daily meal allowance is 60 US dollars while travelling.",
                "An itemized receipt is required for every meal reimbursement.",
            ],
        ),
        (
            "Equipment",
            [
                "Standard company laptops are replaced every four years.",
                "Home-office equipment purchases above 250 US dollars require manager approval.",
            ],
        ),
    ]
    for heading, paragraphs in content:
        document.add_heading(heading, level=1)
        for paragraph in paragraphs:
            document.add_paragraph(paragraph)
    styles = document.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(11)
    document.save(str(path))


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    build_pdf()
    build_docx()


if __name__ == "__main__":
    main()
