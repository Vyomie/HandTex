"""Bitmap -> vector contours, in font units.

A glyph bitmap is a 2-D boolean array (``True`` = ink). We extract its
outlines with marching squares, simplify them, map pixel space into font
units with the baseline at ``y = 0`` (y grows up), and fix contour winding so
holes (e.g. the inside of an "o") subtract correctly under the non-zero fill
rule.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from skimage import measure

Point = Tuple[float, float]
Contour = List[Point]


def _signed_area(poly: Contour) -> float:
    """Shoelace signed area; >0 means counter-clockwise (y-up coords)."""
    area = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return area / 2.0


def _point_in_poly(pt: Point, poly: Contour) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def trace_bitmap(
    mask: np.ndarray,
    baseline_row: float,
    units_per_pixel: float,
    simplify_tolerance: float = 1.0,
) -> List[Contour]:
    """Trace ``mask`` into font-unit contours.

    Parameters
    ----------
    mask:
        2-D boolean array, ``True`` where there is ink.
    baseline_row:
        Pixel row that should map to font ``y = 0`` (the baseline).
    units_per_pixel:
        Scale from pixels to font units (e.g. ``em / cell_height``).
    simplify_tolerance:
        Polygon simplification tolerance, in pixels.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return []

    # Pad so contours that touch the border still close cleanly.
    padded = np.pad(mask.astype(float), 1, mode="constant", constant_values=0.0)
    raw = measure.find_contours(padded, 0.5)

    polys: List[Contour] = []
    for c in raw:
        c = measure.approximate_polygon(c, tolerance=simplify_tolerance)
        if len(c) < 3:
            continue
        poly: Contour = []
        for row, col in c:
            # undo the 1px pad; map to font units, flip y to baseline-relative
            x = (col - 1) * units_per_pixel
            y = (baseline_row - (row - 1)) * units_per_pixel
            poly.append((x, y))
        # drop duplicate closing point if present
        if len(poly) > 1 and poly[0] == poly[-1]:
            poly.pop()
        if len(poly) >= 3:
            polys.append(poly)

    # Fix winding by nesting depth: outer (even depth) clockwise, holes
    # (odd depth) counter-clockwise — opposite orientations so non-zero fill
    # cuts the holes out.
    fixed: List[Contour] = []
    for i, poly in enumerate(polys):
        depth = 0
        probe = poly[0]
        for j, other in enumerate(polys):
            if i != j and _point_in_poly(probe, other):
                depth += 1
        area = _signed_area(poly)
        want_ccw = (depth % 2) == 1  # holes counter-clockwise
        is_ccw = area > 0
        if is_ccw != want_ccw:
            poly = list(reversed(poly))
        fixed.append(poly)
    return fixed
