from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


class PdfGenerationError(RuntimeError):
    """Raised when a PDF cannot be generated."""


def generate_pdf_from_markdown(markdown_text: str, output_path: Path) -> None:
    if not markdown_text.strip():
        raise PdfGenerationError("Cannot generate PDF from empty Markdown")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        document = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        for raw_line in markdown_text.splitlines():
            line = raw_line.strip()
            if not line:
                story.append(Spacer(1, 8))
                continue
            if line.startswith("# "):
                story.append(Paragraph(line[2:], styles["Title"]))
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], styles["Heading2"]))
            elif line.startswith("### "):
                story.append(Paragraph(line[4:], styles["Heading3"]))
            else:
                story.append(Paragraph(line, styles["BodyText"]))
        document.build(story)
    except Exception as exc:  # pragma: no cover - defensive wrapper for native/font issues.
        raise PdfGenerationError(str(exc)) from exc
