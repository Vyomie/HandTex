"""Segment a whiteboard image into individual glyph regions.

Classic connected-component analysis (no ML): binarize ink, label components,
drop noise, then merge components that clearly belong to one glyph — stacked
marks like ``=`` ``i`` ``j`` ``÷`` whose parts overlap horizontally. Vertically
offset neighbours (an exponent next to its base) are deliberately *not* merged,
so the layout engine can still see them as separate boxes and reason about
super/subscripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

from ...types import BBox


@dataclass
class Segment:
    bbox: BBox
    mask: np.ndarray   # boolean ink mask, cropped to bbox
    ink: np.ndarray    # ink intensity (0..1, high = ink), cropped to bbox


def _ink_and_binary(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ink, binary)``: ink intensity (0..1, high = ink) and a mask.

    Otsu separates dark strokes from the (possibly light-gray) background, so
    faint printed grid lines on a capture sheet are treated as background.
    """
    try:
        thr8 = float(threshold_otsu(gray))
    except Exception:
        thr8 = 128.0
    g = gray.astype(np.float32) / 255.0
    thr = thr8 / 255.0
    ink = np.clip((thr - g) / max(thr, 1e-3), 0.0, 1.0)
    binary = gray < thr8
    return ink, binary


def _hsub(a: BBox, b: BBox) -> float:
    """Horizontal overlap as a fraction of the narrower box."""
    inter = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    return inter / max(1.0, min(a.w, b.w))


def segment_image(image_path: str, min_area_frac: float = 0.0001,
                  min_area_abs: float = 20.0, preprocess: bool = True) -> List[Segment]:
    """Return glyph segments left-to-right.

    By default the image is cleaned first (resize, illumination flattening,
    contrast stretch, deskew) so faint or tilted glyphs are still detected;
    pass ``preprocess=False`` to segment the raw image.
    """
    if preprocess:
        from ...preprocess import preprocess as _preprocess
        gray = _preprocess(image_path)
    else:
        gray = np.asarray(Image.open(image_path).convert("L"))
    ink, binary = _ink_and_binary(gray)

    lbl = label(binary)
    H, W = binary.shape
    min_area = max(min_area_abs, min_area_frac * H * W)

    boxes = []
    for r in regionprops(lbl):
        minr, minc, maxr, maxc = r.bbox
        if r.area < min_area:
            continue
        boxes.append(BBox(minc, minr, maxc - minc, maxr - minr))

    # Drop non-glyph blobs (page edges, shadows, margin rules) that are far
    # taller than the text — they would otherwise span many rows and collapse
    # them into one line. Tall real glyphs (integrals, brackets) stay well
    # under the threshold.
    if len(boxes) >= 5:
        import statistics
        med_h = statistics.median(b.h for b in boxes)
        boxes = [b for b in boxes if b.h <= 6.0 * med_h]

    boxes = _merge_stacked(boxes)
    boxes.sort(key=lambda b: b.left)

    segs: List[Segment] = []
    for b in boxes:
        y0, y1 = int(b.top), int(b.bottom)
        x0, x1 = int(b.left), int(b.right)
        sub_ink = ink[y0:y1, x0:x1]
        segs.append(Segment(bbox=b, mask=sub_ink > 0.35, ink=sub_ink))
    return segs


def _merge_stacked(boxes: List[BBox]) -> List[BBox]:
    """Union-merge components that stack vertically into one glyph.

    The vertical-gap tolerance is scaled by the *line's* character height (the
    median box height), not each component's own height — otherwise thin parts
    like the two bars of ``=`` (each only a few px tall) never merge.
    """
    n = len(boxes)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            # strong horizontal overlap + small vertical gap => same glyph.
            # Scale the gap tolerance by the pair's largest dimension, so thin
            # parts (e.g. the two wide bars of '=') still merge even though
            # each bar is only a few pixels tall.
            if _hsub(a, b) > 0.55:
                gap = max(a.top, b.top) - min(a.bottom, b.bottom)
                ref = max(a.w, b.w, a.h, b.h)
                if gap < 0.45 * ref:
                    union(i, j)

    groups: dict[int, List[BBox]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])

    merged: List[BBox] = []
    for grp in groups.values():
        left = min(b.left for b in grp)
        top = min(b.top for b in grp)
        right = max(b.right for b in grp)
        bottom = max(b.bottom for b in grp)
        merged.append(BBox(left, top, right - left, bottom - top))
    return merged
