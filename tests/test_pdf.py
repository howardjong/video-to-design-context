import pytest
from pypdf import PdfReader

from tastepack.pdf import PdfGenerationError, generate_pdf_from_markdown


def test_pdf_generation_succeeds_from_markdown(tmp_path):
    pdf_path = tmp_path / "taste_packet.pdf"

    generate_pdf_from_markdown("# Taste Packet\n\nA useful packet.", pdf_path)

    assert pdf_path.exists()
    assert len(PdfReader(str(pdf_path)).pages) >= 1


def test_pdf_generation_fails_gracefully_for_empty_markdown(tmp_path):
    with pytest.raises(PdfGenerationError):
        generate_pdf_from_markdown("", tmp_path / "empty.pdf")
