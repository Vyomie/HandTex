"""Offline OCR backend.

Lets the whole pipeline run with no network and no model. It recognizes a
scene from a JSON sidecar next to the image (``<image>.glyphs.json``) when
present, otherwise returns a small built-in demo scene. The sidecar format
is also exactly what :func:`handtex.ocr.vision.VisionBackend` is asked to
produce, so the two backends are drop-in compatible.

Sidecar schema::

    {"glyphs": [
        {"name": "x", "text": "x", "bbox": [x, y, w, h], "confidence": 1.0},
        ...
    ]}
"""

from __future__ import annotations

import json
import os
from typing import List

from ..types import BBox, RecognizedGlyph
from .base import OCRBackend


def glyphs_from_json(data: dict) -> List[RecognizedGlyph]:
    out: List[RecognizedGlyph] = []
    for g in data.get("glyphs", []):
        x, y, w, h = g["bbox"]
        out.append(
            RecognizedGlyph(
                name=g.get("name", g.get("text", "?")),
                text=g.get("text", g.get("name", "?")),
                bbox=BBox(float(x), float(y), float(w), float(h)),
                confidence=float(g.get("confidence", 1.0)),
            )
        )
    return out


# A built-in scene roughly equivalent to:  x^{2} + α = y_{1}
# Boxes are hand-placed so the layout engine has real geometry to reason about:
# the "2" is small and raised (superscript); the "1" is small and lowered
# (subscript). Coordinates are pixels on a ~600x160 whiteboard crop.
_DEMO_SCENE = {
    "glyphs": [
        {"name": "x",      "text": "x", "bbox": [20, 60, 40, 50]},
        {"name": "two",    "text": "2", "bbox": [62, 38, 22, 28]},   # raised + small
        {"name": "plus",   "text": "+", "bbox": [110, 64, 40, 40]},
        {"name": "alpha",  "text": "α", "bbox": [170, 60, 44, 50]},
        {"name": "equals", "text": "=", "bbox": [240, 70, 44, 28]},
        {"name": "y",      "text": "y", "bbox": [310, 60, 40, 60]},
        {"name": "one",    "text": "1", "bbox": [352, 92, 18, 28]},  # lowered + small
    ]
}


class MockBackend(OCRBackend):
    name = "mock"

    def __init__(self, scene: dict | None = None):
        self._scene = scene

    def recognize(self, image_path: str) -> List[RecognizedGlyph]:
        if self._scene is not None:
            return glyphs_from_json(self._scene)

        sidecar = f"{image_path}.glyphs.json"
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as fh:
                return glyphs_from_json(json.load(fh))

        # Also accept a plain "<stem>.glyphs.json".
        stem, _ = os.path.splitext(image_path)
        alt = f"{stem}.glyphs.json"
        if os.path.exists(alt):
            with open(alt, "r", encoding="utf-8") as fh:
                return glyphs_from_json(json.load(fh))

        return glyphs_from_json(_DEMO_SCENE)
