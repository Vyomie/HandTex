import os

from fontTools.ttLib import TTFont

from handtex.font import symbols
from handtex.font.pipeline import build_font_from_cells, synth_cells


def test_build_font_has_unicode_cmap(tmp_path):
    names = [s.name for s in symbols.all_symbols() if s.name != "space"]
    cells = synth_cells(names)
    assert cells, "synth_cells produced no glyphs (no reference font found?)"

    out = os.path.join(tmp_path, "Hand.ttf")
    build_font_from_cells(cells, out)
    assert os.path.exists(out)

    font = TTFont(out)
    cmap = font.getBestCmap()
    # ASCII + a math symbol must all be reachable by their real code points.
    assert ord("x") in cmap
    assert 0x03B1 in cmap          # alpha
    assert 0x2211 in cmap          # sum
    # The font is structurally valid.
    assert "glyf" in font
    assert font["head"].unitsPerEm == 1000


def test_demo_cli_end_to_end(tmp_path):
    from handtex.cli import main

    out = os.path.join(tmp_path, "out")
    rc = main(["demo", "--out", out])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "HandTex-Regular.ttf"))
    assert os.path.exists(os.path.join(out, "whiteboard.tex"))
    assert os.path.exists(os.path.join(out, "whiteboard.png"))
