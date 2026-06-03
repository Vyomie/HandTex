import json

from handtex.ocr import get_backend
from handtex.ocr.mock import MockBackend
from handtex.layout import layout_glyphs
from handtex.latex import document_to_latex, standalone_document


def test_mock_backend_demo_scene():
    glyphs = MockBackend().recognize("<demo>")
    assert {g.name for g in glyphs} >= {"x", "two", "alpha", "y", "one"}


def test_mock_backend_sidecar(tmp_path):
    scene = {"glyphs": [
        {"name": "a", "text": "a", "bbox": [0, 0, 10, 10]},
        {"name": "b", "text": "b", "bbox": [12, 0, 10, 10]},
    ]}
    img = tmp_path / "wb.png"
    img.write_bytes(b"not really an image")
    (tmp_path / "wb.png.glyphs.json").write_text(json.dumps(scene))

    glyphs = get_backend("mock").recognize(str(img))
    assert [g.name for g in glyphs] == ["a", "b"]


def test_full_pipeline_to_latex():
    glyphs = MockBackend().recognize("<demo>")
    doc = layout_glyphs(glyphs)
    assert document_to_latex(doc) == "x^{2} + \\alpha = y_{1}"


def test_standalone_document_wraps_font():
    glyphs = MockBackend().recognize("<demo>")
    doc = layout_glyphs(glyphs)
    tex = standalone_document(doc, font_path="HandTex-Regular.ttf")
    assert "\\documentclass" in tex
    assert "unicode-math" in tex
    assert "HandTex-Regular.ttf" in tex
