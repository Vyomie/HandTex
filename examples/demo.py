"""End-to-end walkthrough, as a script (mirrors `python -m handtex demo`).

Builds a font from synthesized stand-in handwriting, "reads" a built-in
whiteboard scene, lays it out, and emits both LaTeX and a rendered PNG.

    python examples/demo.py
"""

from __future__ import annotations

import os
import sys

# Allow running as a plain script (`python examples/demo.py`) without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtex.font import symbols
from handtex.font.pipeline import build_font_from_cells, synth_cells
from handtex.latex import document_to_latex, standalone_document
from handtex.layout import layout_glyphs
from handtex.ocr.mock import MockBackend
from handtex.render import render_document


def main(out_dir: str = "out") -> None:
    os.makedirs(out_dir, exist_ok=True)
    font_path = os.path.join(out_dir, "HandTex-Regular.ttf")

    # 1. Build a Unicode font from stand-in handwriting.
    names = [s.name for s in symbols.all_symbols() if s.name != "space"]
    build_font_from_cells(synth_cells(names), font_path)
    print(f"font   -> {font_path}")

    # 2. Recognize a whiteboard (offline demo scene): x^2 + alpha = y_1
    glyphs = MockBackend().recognize("<demo>")

    # 3. Lay out + emit LaTeX.
    doc = layout_glyphs(glyphs)
    print(f"latex  -> {document_to_latex(doc)}")
    tex_path = os.path.join(out_dir, "whiteboard.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(standalone_document(doc, font_path=os.path.basename(font_path)) + "\n")
    print(f"tex    -> {tex_path}")

    # 4. Render back in the handwritten font.
    png_path = os.path.join(out_dir, "whiteboard.png")
    render_document(doc, font_path, png_path)
    print(f"render -> {png_path}")


if __name__ == "__main__":
    main()
