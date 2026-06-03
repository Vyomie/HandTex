from handtex.types import BBox, RecognizedGlyph, Role
from handtex.layout import layout_glyphs
from handtex.latex import document_to_latex


def g(name, text, x, y, w, h):
    return RecognizedGlyph(name=name, text=text, bbox=BBox(x, y, w, h))


def test_superscript_and_subscript_detection():
    glyphs = [
        g("x", "x", 20, 60, 40, 50),
        g("two", "2", 62, 38, 22, 28),   # small + raised
        g("plus", "+", 110, 64, 40, 40),
        g("y", "y", 310, 60, 40, 60),
        g("one", "1", 352, 92, 18, 28),  # small + lowered
    ]
    doc = layout_glyphs(glyphs)
    assert len(doc.lines) == 1
    line = doc.lines[0]
    names = [t.name for t in line.tokens]
    assert names == ["x", "plus", "y"]

    x_tok = line.tokens[0]
    assert [s.name for s in x_tok.superscript] == ["two"]
    assert x_tok.superscript[0].role == Role.SUPERSCRIPT

    y_tok = line.tokens[2]
    assert [s.name for s in y_tok.subscript] == ["one"]
    assert y_tok.subscript[0].role == Role.SUBSCRIPT

    assert document_to_latex(doc) == "x^{2} + y_{1}"


def test_small_centered_glyph_stays_base():
    # An equals sign is short but vertically centered: must NOT be a script.
    glyphs = [
        g("x", "x", 20, 60, 40, 50),
        g("equals", "=", 80, 70, 44, 28),
        g("y", "y", 150, 60, 40, 50),
    ]
    doc = layout_glyphs(glyphs)
    names = [t.name for t in doc.lines[0].tokens]
    assert names == ["x", "equals", "y"]


def test_high_mark_not_treated_as_superscript():
    # A small, raised apostrophe is an intrinsic-high mark, not an exponent.
    glyphs = [
        g("x", "x", 20, 60, 40, 50),
        g("apostrophe", "'", 62, 50, 10, 18),
    ]
    doc = layout_glyphs(glyphs)
    line = doc.lines[0]
    assert [t.name for t in line.tokens] == ["x", "apostrophe"]
    assert not line.tokens[0].superscript


def test_two_lines_grouped_separately():
    glyphs = [
        g("a", "a", 10, 10, 30, 30),
        g("b", "b", 50, 10, 30, 30),
        g("c", "c", 10, 120, 30, 30),
    ]
    doc = layout_glyphs(glyphs)
    assert len(doc.lines) == 2
    assert [t.name for t in doc.lines[0].tokens] == ["a", "b"]
    assert [t.name for t in doc.lines[1].tokens] == ["c"]
