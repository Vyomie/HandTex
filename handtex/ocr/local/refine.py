"""Situational-awareness refinement of raw detections.

Raw segmentation+classification produces one guess per box. This step adds
context so the result is cleaner:

  1. **Size awareness** — boxes whose area is a wild outlier versus the median
     glyph (tiny specks, merged blobs) are dropped *if* the model is also
     unsure about them.
  2. **Doodle removal** — anything the model classifies as ``doodle``
     (gibberish/scribble) is discarded.
  3. **Overlap dedup** — when two boxes cover the same region, the
     lower-confidence one is dropped.
  4. **Duplicate resolution** (sheets, ``unique_labels=True``) — if two boxes
     claim the same label, the most confident keeps it and the other is demoted
     to its next-best guess; if that next guess is ``doodle`` / too weak / also
     already taken, the box is assumed bad and removed.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ...types import BBox

DOODLE = "doodle"


@dataclass
class Candidate:
    """A detection with its top-k guesses, ``[(label, prob), ...]`` desc."""

    index: int
    bbox: BBox
    options: List[Tuple[str, float]]


@dataclass
class Resolved:
    index: int
    bbox: BBox
    label: str
    confidence: float


def _iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    iy = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def refine(
    candidates: Sequence[Candidate],
    *,
    unique_labels: bool,
    doodle_label: str = DOODLE,
    doodle_reject_prob: float = 0.60,
    min_prob: float = 0.12,
    small_ratio: float = 0.10,
    large_ratio: float = 8.0,
    overlap_iou: float = 0.55,
) -> List[Resolved]:
    """Apply the refinement rules; return the surviving, resolved detections.

    A box is only thrown out as gibberish when ``doodle`` is the top guess
    *and* its probability clears ``doodle_reject_prob``. Otherwise the
    ``doodle`` option is ignored and the box keeps its best real guess — so a
    merely-uncertain real glyph is never silently deleted.
    """
    if not candidates:
        return []

    median_area = statistics.median(c.bbox.area for c in candidates) or 1.0

    # (3) Overlap dedup: keep the most confident of any heavily-overlapping set.
    by_conf = sorted(candidates, key=lambda c: -c.options[0][1])
    kept: List[Candidate] = []
    for c in by_conf:
        if any(_iou(c.bbox, k.bbox) > overlap_iou for k in kept):
            continue
        kept.append(c)

    # (1)+(2) Doodle removal (only when confident) + size awareness.
    survivors: List[Candidate] = []
    for c in kept:
        top_label, top_prob = c.options[0]
        if top_label == doodle_label and top_prob >= doodle_reject_prob:
            continue  # confident gibberish -> remove
        # Drop the doodle option so an uncertain real glyph keeps its best
        # real guess instead of being deleted.
        real_opts = [(l, p) for (l, p) in c.options if l != doodle_label]
        if not real_opts:
            continue
        ratio = c.bbox.area / median_area
        if (ratio < small_ratio or ratio > large_ratio) and real_opts[0][1] < 0.55:
            continue
        survivors.append(Candidate(c.index, c.bbox, real_opts))

    # (4) Duplicate resolution by demotion to the next-best guess.
    results: List[Resolved] = []
    taken = set()
    for c in sorted(survivors, key=lambda c: -c.options[0][1]):
        chosen = None
        for label, prob in c.options:
            if prob < min_prob:
                break  # nothing plausible left -> bad
            if not unique_labels or label not in taken:
                chosen = (label, prob)
                break
        if chosen is None:
            continue  # assumed bad, removed
        if unique_labels:
            taken.add(chosen[0])
        results.append(Resolved(c.index, c.bbox, chosen[0], chosen[1]))

    return results
