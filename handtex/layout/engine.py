"""The layout engine.

Given a flat list of :class:`RecognizedGlyph` (each with a pixel bbox), infer:

  1. **Lines** — group glyphs that share a horizontal band.
  2. **Baseline & base height** — the dominant text size on each line.
  3. **Roles** — for each small, vertically-offset glyph decide whether it is
     a *superscript* (raised exponent) or *subscript* (lowered index), and
     attach it to the base glyph it belongs to.

Intrinsically high/low marks (apostrophe, comma, prime, ...) are recognized
via the symbol table's ``vhint`` so they are NOT mistaken for exponents or
indices — that directly answers the "up or down for , ' \" *" requirement.

The thresholds live in :class:`LayoutParams` so they are easy to tune for a
particular hand.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List

from ..types import BBox, Document, Line, RecognizedGlyph, Role, Token
from ..font import symbols
from ..font.symbols import VHint


@dataclass
class LayoutParams:
    # A glyph is "small" (candidate super/subscript) when its height is below
    # this fraction of the line's base height.
    small_ratio: float = 0.68
    # Percentile of glyph heights used to estimate the base (body) height.
    base_height_pct: float = 70.0
    # Fraction of base height a small glyph's center must rise above / drop
    # below the base center to count as super/subscript.
    raise_ratio: float = 0.12
    drop_ratio: float = 0.10
    # Vertical-overlap fraction (of the smaller box height) for two glyphs to
    # be considered on the same line.
    line_overlap_ratio: float = 0.35


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _group_lines(glyphs: List[RecognizedGlyph], params: LayoutParams) -> List[List[RecognizedGlyph]]:
    """Cluster glyphs into horizontal lines.

    Two stages so raised exponents / lowered indices don't spuriously form
    their own lines: seed lines from the *tall* (base-sized) glyphs by
    vertical overlap, then attach each remaining small glyph to the nearest
    line band.
    """
    if not glyphs:
        return []

    base_h = _percentile([g.bbox.h for g in glyphs], params.base_height_pct)
    small_cut = params.small_ratio * base_h
    tall = [g for g in glyphs if g.bbox.h >= small_cut]
    small = [g for g in glyphs if g.bbox.h < small_cut]
    if not tall:  # everything is small; fall back to treating all as seeds
        tall, small = glyphs, []

    # Stage 1: seed lines from tall glyphs by vertical overlap.
    lines: List[List[RecognizedGlyph]] = []
    for g in sorted(tall, key=lambda g: g.bbox.cy):
        placed = False
        for line in lines:
            top = min(x.bbox.top for x in line)
            bot = max(x.bbox.bottom for x in line)
            band = BBox(0, top, 1, bot - top)
            if band.vertical_overlap(g.bbox) >= params.line_overlap_ratio * min(band.h, g.bbox.h):
                line.append(g)
                placed = True
                break
        if not placed:
            lines.append([g])

    # Stage 2: attach each small glyph to the nearest line (by center gap),
    # unless it is implausibly far (then it starts its own line).
    def line_center(line: List[RecognizedGlyph]) -> float:
        return statistics.median(g.bbox.cy for g in line)

    for g in small:
        if not lines:
            lines.append([g])
            continue
        nearest = min(lines, key=lambda ln: abs(line_center(ln) - g.bbox.cy))
        if abs(line_center(nearest) - g.bbox.cy) <= 1.5 * base_h:
            nearest.append(g)
        else:
            lines.append([g])

    for line in lines:
        line.sort(key=lambda g: g.bbox.left)
    lines.sort(key=lambda line: statistics.median(g.bbox.cy for g in line))
    return lines


def _intrinsic_vhint(g: RecognizedGlyph) -> VHint:
    sym = symbols.resolve(g.name) or symbols.resolve(g.text)
    return sym.vhint if sym else VHint.NORMAL


def _classify_line(glyphs: List[RecognizedGlyph], params: LayoutParams) -> Line:
    heights = [g.bbox.h for g in glyphs]
    base_h = _percentile(heights, params.base_height_pct)
    small_cut = params.small_ratio * base_h

    # Base glyphs define the baseline & body center.
    base_glyphs = [g for g in glyphs if g.bbox.h >= small_cut] or glyphs
    baseline = statistics.median(g.bbox.bottom for g in base_glyphs)
    base_cy = statistics.median(g.bbox.cy for g in base_glyphs)
    x_height = statistics.median(g.bbox.h for g in base_glyphs)

    raise_thr = params.raise_ratio * base_h
    drop_thr = params.drop_ratio * base_h

    line = Line(baseline=baseline, x_height=x_height)
    last_base: Token | None = None

    for g in sorted(glyphs, key=lambda g: g.bbox.left):
        vhint = _intrinsic_vhint(g)
        is_small = g.bbox.h < small_cut
        role = Role.BASE

        # Intrinsically high/low marks keep their meaning; they are not
        # exponents/indices even though they sit off the baseline.
        if vhint == VHint.NORMAL and is_small and last_base is not None:
            above = g.bbox.cy < base_cy - raise_thr
            below = g.bbox.cy > base_cy + drop_thr
            if above and g.bbox.bottom <= baseline - drop_thr:
                role = Role.SUPERSCRIPT
            elif below:
                role = Role.SUBSCRIPT

        token = Token(glyph=g, role=role)
        if role == Role.SUPERSCRIPT:
            last_base.superscript.append(token)
        elif role == Role.SUBSCRIPT:
            last_base.subscript.append(token)
        else:
            line.tokens.append(token)
            last_base = token

    return line


def layout_glyphs(glyphs: List[RecognizedGlyph], params: LayoutParams | None = None) -> Document:
    """Turn recognized glyphs into a structured :class:`Document`."""
    params = params or LayoutParams()
    doc = Document()
    for line_glyphs in _group_lines(glyphs, params):
        doc.lines.append(_classify_line(line_glyphs, params))
    return doc
