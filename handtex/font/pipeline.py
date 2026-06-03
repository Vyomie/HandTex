"""Glue: capture-sheet cells -> outlines -> a font file.

Also provides :func:`synth_cells`, which renders stand-in glyphs from a
reference font so the font pipeline (and the end-to-end demo) can run with no
handwriting scans yet — swap in real :func:`~handtex.font.extract.extract_cells`
output and nothing downstream changes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

from . import symbols
from ._fonts import load_reference_font
from .build import FontMeta, GlyphOutline, build_font
from .extract import CellBitmap
from .template import Geometry
from .trace import trace_bitmap


def _advance_from_mask(mask: np.ndarray, units_per_pixel: float, side_bearing: float) -> int:
    """Advance width = ink width + symmetric side bearings, in font units."""
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return int(units_per_pixel * mask.shape[1] * 0.4)  # blankish glyph
    ink_w = (cols.max() - cols.min() + 1) * units_per_pixel
    return int(ink_w + 2 * side_bearing)


def cells_to_outlines(
    cells: Dict[str, CellBitmap],
    meta: Optional[FontMeta] = None,
    cell_size: int = Geometry().cell,
) -> Dict[str, GlyphOutline]:
    """Trace each cell bitmap into a :class:`GlyphOutline`."""
    meta = meta or FontMeta()
    upp = meta.units_per_em / float(cell_size)
    side_bearing = meta.units_per_em * 0.05
    outlines: Dict[str, GlyphOutline] = {}
    for name, cell in cells.items():
        contours = trace_bitmap(cell.mask, cell.baseline_row, upp)
        if not contours:
            continue
        # Shift so the left side bearing is consistent.
        cols = np.where(cell.mask.any(axis=0))[0]
        if len(cols):
            dx = side_bearing - cols.min() * upp
            contours = [[(x + dx, y) for (x, y) in c] for c in contours]
        outlines[name] = GlyphOutline(
            name=name,
            contours=contours,
            advance_width=_advance_from_mask(cell.mask, upp, side_bearing),
        )
    return outlines


def synth_cells(
    names: List[str],
    geometry: Optional[Geometry] = None,
) -> Dict[str, CellBitmap]:
    """Render stand-in glyphs from a reference font into capture cells."""
    geo = geometry or Geometry()
    font = load_reference_font(int(geo.cell * 0.6))
    inset = max(2, geo.cell // 40)
    size = geo.cell - 2 * inset
    baseline_row = geo.baseline_row - inset

    cells: Dict[str, CellBitmap] = {}
    for name in names:
        sym = symbols.by_name(name)
        if sym is None or sym.char == " ":
            continue
        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)
        if font is not None:
            _, top, _, bottom = draw.textbbox((0, 0), sym.char, font=font)
            gx = size * 0.12
            gy = baseline_row - (bottom - top) - top
            draw.text((gx, gy), sym.char, fill=255, font=font)
        mask = np.asarray(img) > 64
        if mask.any():
            cells[name] = CellBitmap(mask=mask, baseline_row=baseline_row)
    return cells


def build_font_from_cells(
    cells: Dict[str, CellBitmap],
    out_path: str,
    meta: Optional[FontMeta] = None,
    cell_size: int = Geometry().cell,
) -> str:
    meta = meta or FontMeta()
    outlines = cells_to_outlines(cells, meta=meta, cell_size=cell_size)
    return build_font(outlines, out_path, meta=meta)
