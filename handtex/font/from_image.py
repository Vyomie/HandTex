"""Build a font from a photo of handwriting — full OCR, end to end.

Unlike :mod:`handtex.font.extract` (which trusts a known template layout),
this path runs the actual OCR model: it segments the image into glyphs,
*classifies* each one to learn which character it is, then traces the very
same ink into an outline and assembles a Unicode font. So pointing it at an
arbitrary handwriting sheet yields a real ``.ttf`` with no manifest required.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import symbols
from .build import FontMeta, GlyphOutline, build_font
from .trace import trace_bitmap
from ..ocr.local.model import has_weights, load_model, predict_topk
from ..ocr.local.normalize import normalize_glyph
from ..ocr.local.refine import Candidate, refine
from ..ocr.local.segment import Segment, segment_image


@dataclass
class RecognizedCell:
    name: str
    char: str
    confidence: float
    segment: Segment


def recognize_cells(image_path: str, top_crop_frac: float = 0.0,
                    min_confidence: float = 0.0) -> List[RecognizedCell]:
    """Segment + classify every glyph in the image (full OCR)."""
    if not has_weights():
        raise RuntimeError(
            "no trained OCR weights. Train first: python -m handtex.ocr.local.train"
        )
    segments = segment_image(image_path)
    if top_crop_frac > 0 and segments:
        max_y = max(s.bbox.bottom for s in segments)
        cutoff = top_crop_frac * max_y
        segments = [s for s in segments if s.bbox.top >= cutoff]
    if not segments:
        return []

    model, labels = load_model()
    batch = np.stack([normalize_glyph(s.ink) for s in segments])
    idxk, confk = predict_topk(model, batch, k=4)

    # Refinement with situational awareness; a character sheet is expected to
    # contain each glyph once, so resolve duplicates by uniqueness.
    cands = [
        Candidate(index=n, bbox=segments[n].bbox,
                  options=[(labels[int(idxk[n][j])], float(confk[n][j]))
                           for j in range(idxk.shape[1])])
        for n in range(len(segments))
    ]
    resolved = refine(cands, unique_labels=True)

    cells: List[RecognizedCell] = []
    for r in resolved:
        if r.confidence < min_confidence:
            continue
        sym = symbols.by_name(r.label)
        cells.append(RecognizedCell(name=r.label, char=sym.char if sym else r.label,
                                    confidence=r.confidence, segment=segments[r.index]))
    return cells


def font_from_image(
    image_path: str,
    out_path: str,
    meta: Optional[FontMeta] = None,
    top_crop_frac: float = 0.0,
    min_confidence: float = 0.0,
) -> Tuple[str, List[RecognizedCell]]:
    """Recognize handwriting in ``image_path`` and build a font at ``out_path``.

    Returns ``(out_path, recognized_cells)``. When the same character is
    recognized more than once, the highest-confidence instance wins.
    """
    meta = meta or FontMeta()
    cells = recognize_cells(image_path, top_crop_frac=top_crop_frac,
                            min_confidence=min_confidence)
    if not cells:
        raise RuntimeError("no glyphs recognized in image")

    # Common scale so relative glyph sizes are preserved (median glyph ~ x-height).
    median_h = statistics.median(c.segment.bbox.h for c in cells) or 1.0
    upp = meta.units_per_em / (1.5 * median_h)
    side_bearing = meta.units_per_em * 0.06

    # Keep the best instance per character.
    best: Dict[str, RecognizedCell] = {}
    for c in cells:
        prev = best.get(c.name)
        if prev is None or c.confidence > prev.confidence:
            best[c.name] = c

    outlines: Dict[str, GlyphOutline] = {}
    for name, cell in best.items():
        mask = cell.segment.mask
        baseline_row = mask.shape[0] - 1  # glyph bottom sits on the baseline
        contours = trace_bitmap(mask, baseline_row, upp)
        if not contours:
            continue
        cols = np.where(mask.any(axis=0))[0]
        if len(cols):
            dx = side_bearing - cols.min() * upp
            contours = [[(x + dx, y) for (x, y) in c] for c in contours]
        ink_w = (cols.max() - cols.min() + 1) * upp if len(cols) else median_h * upp
        outlines[name] = GlyphOutline(
            name=name, contours=contours, advance_width=int(ink_w + 2 * side_bearing)
        )

    build_font(outlines, out_path, meta=meta)
    return out_path, cells
