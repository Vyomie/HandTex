"""Typeset a Document in a HandTex font (the "render back in my hand" output).

Draws each base glyph on the baseline; superscripts smaller and raised,
subscripts smaller and lowered. Intrinsically high/low marks (apostrophe,
comma, prime, ...) are nudged by their ``vhint`` so they land where they
should — the rendering counterpart of the layout engine's placement logic.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from ..types import Document, Line, Token
from ..font import symbols
from ..font.symbols import VHint


class _Pen:
    def __init__(self, font_path: str, base_size: int):
        self.base = ImageFont.truetype(font_path, base_size)
        self.small = ImageFont.truetype(font_path, int(base_size * 0.68))
        self.base_size = base_size

    def width(self, text: str, small: bool = False) -> float:
        font = self.small if small else self.base
        return font.getlength(text)


def _vhint_offset(tok: Token, base_size: int) -> float:
    sym = symbols.resolve(tok.name) or symbols.resolve(tok.text)
    if sym is None:
        return 0.0
    if sym.vhint == VHint.HIGH:
        return -0.42 * base_size  # raise toward the cap line
    if sym.vhint == VHint.LOW:
        return 0.0  # already sits low by design
    return 0.0


def _draw_token(draw: ImageDraw.ImageDraw, pen: _Pen, tok: Token, x: float, baseline: float) -> float:
    """Draw a base token (with its scripts) and return the new pen x."""
    ch = tok.text
    dy = _vhint_offset(tok, pen.base_size)
    draw.text((x, baseline + dy), ch, fill="black", font=pen.base, anchor="ls")
    x += pen.width(ch)

    sup_x = x
    for s in tok.superscript:
        draw.text((sup_x, baseline - 0.45 * pen.base_size), s.text,
                  fill="black", font=pen.small, anchor="ls")
        sup_x += pen.width(s.text, small=True)
    sub_x = x
    for s in tok.subscript:
        draw.text((sub_x, baseline + 0.22 * pen.base_size), s.text,
                  fill="black", font=pen.small, anchor="ls")
        sub_x += pen.width(s.text, small=True)

    return max(sup_x, sub_x)


def render_document(
    doc: Document,
    font_path: str,
    out_path: str,
    base_size: int = 96,
    padding: int = 40,
) -> str:
    """Render ``doc`` to a PNG at ``out_path`` using the font at ``font_path``."""
    pen = _Pen(font_path, base_size)
    line_height = int(base_size * 1.8)

    # First pass: measure required width.
    max_w = 0.0
    for line in doc.lines:
        x = 0.0
        for tok in line.tokens:
            x += pen.width(tok.text)
            x += max(
                sum(pen.width(s.text, small=True) for s in tok.superscript),
                sum(pen.width(s.text, small=True) for s in tok.subscript),
            )
            x += base_size * 0.12  # inter-token gap
        max_w = max(max_w, x)

    W = int(max_w + 2 * padding)
    H = int(len(doc.lines) * line_height + 2 * padding)
    img = Image.new("RGB", (max(W, 1), max(H, 1)), "white")
    draw = ImageDraw.Draw(img)

    for row, line in enumerate(doc.lines):
        baseline = padding + row * line_height + base_size
        x = float(padding)
        for tok in line.tokens:
            x = _draw_token(draw, pen, tok, x, baseline)
            x += base_size * 0.12

    img.save(out_path)
    return out_path
