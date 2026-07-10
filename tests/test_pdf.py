import pytest
from PIL import Image
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


def test_pdf_embeds_available_markdown_frame_images(tmp_path):
    frame_path = tmp_path / "frames" / "frame.jpg"
    frame_path.parent.mkdir()
    Image.new("RGB", (12, 8), "white").save(frame_path, format="JPEG")
    pdf_path = tmp_path / "taste_packet.pdf"

    generate_pdf_from_markdown(
        "# Taste Packet\n\n![Frame from asset](frames/frame.jpg)\n",
        pdf_path,
        asset_root=tmp_path,
    )

    reader = PdfReader(str(pdf_path))
    image_objects = []
    for page in reader.pages:
        xobjects = page.get("/Resources", {}).get("/XObject", {})
        for xobject in xobjects.values():
            if xobject.get_object().get("/Subtype") == "/Image":
                image_objects.append(xobject)
    assert image_objects
