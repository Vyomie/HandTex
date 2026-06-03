"""Slice a filled-in capture sheet back into per-glyph bitmaps.

Uses the manifest geometry to find each cell deterministically, binarizes the
ink (dark strokes on light paper), drops the faint printed guides, and returns
a clean mask plus the baseline row for each glyph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from PIL import Image

from .template import Manifest


@dataclass
class CellBitmap:
    mask: np.ndarray      # bool, True = ink
    baseline_row: float   # row (within the cropped cell) mapping to y=0


def _binarize(cell: np.ndarray, threshold: int) -> np.ndarray:
    """Dark pixels become ink. ``cell`` is grayscale uint8."""
    return cell < threshold


def extract_cells(filled_image: str, manifest: Manifest, threshold: int = 128) -> Dict[str, CellBitmap]:
    """Return ``{symbol_name: CellBitmap}`` from a filled sheet."""
    geo = manifest.geometry
    img = Image.open(filled_image).convert("L")
    arr = np.asarray(img)

    out: Dict[str, CellBitmap] = {}
    inset = max(2, geo.cell // 40)  # trim the printed cell border
    for i, name in enumerate(manifest.names):
        ox, oy = geo.cell_origin(i)
        x0, y0 = ox + inset, oy + inset
        x1, y1 = ox + geo.cell - inset, oy + geo.cell - inset
        # Guard against sheets photographed at a different size.
        if y1 > arr.shape[0] or x1 > arr.shape[1]:
            continue
        cell = arr[y0:y1, x0:x1]
        mask = _binarize(cell, threshold)
        baseline_row = geo.baseline_row - inset
        out[name] = CellBitmap(mask=mask, baseline_row=baseline_row)
    return out
