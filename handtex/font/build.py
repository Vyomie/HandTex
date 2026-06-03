"""Contours -> a real font file.

Assembles per-glyph outlines into a TrueType font with a proper Unicode
``cmap``. Because every glyph is bound to its real code point (Greek, math
operators, punctuation included), the resulting ``.ttf``:

  * renders the symbol when you type that code point in *any* app, and
  * is picked up unchanged by XeLaTeX/LuaLaTeX (``fontspec`` / ``unicode-math``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from . import symbols

Point = Tuple[float, float]
Contour = List[Point]


@dataclass
class GlyphOutline:
    """One glyph ready to be placed in the font."""

    name: str               # canonical symbol name (see symbols table)
    contours: List[Contour] # in font units, baseline at y=0, y up
    advance_width: int


@dataclass
class FontMeta:
    family: str = "HandTex"
    style: str = "Regular"
    units_per_em: int = 1000
    ascent: int = 800
    descent: int = -200
    version: str = "0.1"


def _draw(pen: TTGlyphPen, contours: List[Contour]) -> None:
    for contour in contours:
        if len(contour) < 3:
            continue
        pen.moveTo((round(contour[0][0]), round(contour[0][1])))
        for pt in contour[1:]:
            pen.lineTo((round(pt[0]), round(pt[1])))
        pen.closePath()


def _notdef_outline(meta: FontMeta) -> List[Contour]:
    # A simple hollow box so missing glyphs are visible.
    w = meta.units_per_em // 2
    h = meta.ascent
    inset = meta.units_per_em // 20
    outer = [(inset, 0), (w - inset, 0), (w - inset, h), (inset, h)]
    inner = [
        (2 * inset, 2 * inset),
        (2 * inset, h - 2 * inset),
        (w - 2 * inset, h - 2 * inset),
        (w - 2 * inset, 2 * inset),
    ]
    return [outer, inner]


def build_font(outlines: Dict[str, GlyphOutline], out_path: str, meta: FontMeta | None = None) -> str:
    """Build a TTF from ``{symbol_name: GlyphOutline}`` and write ``out_path``.

    Returns the path written.
    """
    meta = meta or FontMeta()

    # Resolve glyph names + cmap from the symbol table.
    glyph_order = [".notdef"]
    glyphs_pens: Dict[str, object] = {}
    advances: Dict[str, int] = {}
    cmap: Dict[int, str] = {}

    # .notdef
    pen = TTGlyphPen(None)
    _draw(pen, _notdef_outline(meta))
    glyphs_pens[".notdef"] = pen.glyph()
    advances[".notdef"] = meta.units_per_em // 2

    # Always include a space.
    space = symbols.by_name("space")
    if space:
        gname = space.glyph_name
        glyph_order.append(gname)
        p = TTGlyphPen(None)
        glyphs_pens[gname] = p.glyph()
        advances[gname] = meta.units_per_em // 3
        cmap[space.codepoint] = gname

    for name, outline in outlines.items():
        sym = symbols.by_name(name) or symbols.resolve(name)
        if sym is None:
            # Unknown symbol: skip rather than emit an unmapped glyph.
            continue
        gname = sym.glyph_name
        if gname in glyphs_pens:
            continue
        glyph_order.append(gname)
        p = TTGlyphPen(None)
        _draw(p, outline.contours)
        glyphs_pens[gname] = p.glyph()
        advances[gname] = int(outline.advance_width)
        cmap[sym.codepoint] = gname

    fb = FontBuilder(meta.units_per_em, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs_pens)

    metrics = {}
    for gname in glyph_order:
        glyph = glyphs_pens[gname]
        xmin = getattr(glyph, "xMin", 0) or 0
        metrics[gname] = (advances.get(gname, meta.units_per_em // 2), xmin)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=meta.ascent, descent=meta.descent)

    name = f"{meta.family} {meta.style}".strip()
    fb.setupNameTable(
        {
            "familyName": meta.family,
            "styleName": meta.style,
            "fullName": name,
            "psName": name.replace(" ", ""),
            "version": meta.version,
        }
    )
    fb.setupOS2(sTypoAscender=meta.ascent, sTypoDescender=meta.descent)
    fb.setupPost()
    fb.save(out_path)
    return out_path
