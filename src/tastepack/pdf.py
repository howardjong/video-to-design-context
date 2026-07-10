from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as ReportLabImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

IMAGE_LINE_PATTERN = re.compile(r"^!\[[^]]*\]\((?P<path>[^)]+)\)$")


class PdfGenerationError(RuntimeError):
    """Raised when a PDF cannot be generated."""


def _frame_image(relative_path: str, asset_root: Path | None) -> ReportLabImage | None:
    if asset_root is None:
        return None
    resolved_root = asset_root.resolve()
    resolved_path = (resolved_root / relative_path).resolve()
    if not resolved_path.is_relative_to(resolved_root) or not resolved_path.is_file():
        return None
    try:
        image = ReportLabImage(str(resolved_path))
        scale = min(468 / image.imageWidth, 360 / image.imageHeight, 1.0)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        return image
    except Exception:
        return None


def generate_pdf_from_markdown(
    markdown_text: str,
    output_path: Path,
    asset_root: Path | None = None,
) -> None:
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
            image_match = IMAGE_LINE_PATTERN.match(line)
            if image_match:
                image = _frame_image(image_match.group("path"), asset_root)
                if image is not None:
                    story.append(image)
                    continue
                frame_reference = escape(f"Frame reference: {image_match.group('path')}")
                story.append(Paragraph(frame_reference, styles["BodyText"]))
                continue
            if line.startswith("# "):
                story.append(Paragraph(escape(line[2:]), styles["Title"]))
            elif line.startswith("## "):
                story.append(Paragraph(escape(line[3:]), styles["Heading2"]))
            elif line.startswith("### "):
                story.append(Paragraph(escape(line[4:]), styles["Heading3"]))
            else:
                story.append(Paragraph(escape(line), styles["BodyText"]))
        document.build(story)
    except Exception as exc:  # pragma: no cover - defensive wrapper for native/font issues.
        raise PdfGenerationError(str(exc)) from exc
