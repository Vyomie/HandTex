"""The small-model OCR backend: segment + classify.

Localizes glyphs with connected components, then classifies each one with the
tiny :class:`~handtex.ocr.local.model.GlyphCNN`. Returns the same
``RecognizedGlyph`` list every other backend produces, so layout/LaTeX/render
are unchanged.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ...font import symbols
from ...types import RecognizedGlyph
from ..base import OCRBackend
from .model import GlyphCNN, has_weights, load_model, predict
from .normalize import normalize_glyph
from .segment import segment_image


class LocalBackend(OCRBackend):
    name = "local"

    def __init__(self, min_confidence: float = 0.0, device: str = "cpu"):
        if not has_weights():
            raise RuntimeError(
                "no trained weights found. Train the model first:\n"
                "    python -m handtex.ocr.local.train --epochs 12\n"
                "or use the 'mock' backend for offline runs."
            )
        self._model, self._labels = load_model(device=device)
        self._device = device
        self._min_confidence = min_confidence

    def recognize(self, image_path: str) -> List[RecognizedGlyph]:
        segments = segment_image(image_path)
        if not segments:
            return []

        batch = np.stack([normalize_glyph(seg.ink) for seg in segments])
        idx, conf = predict(self._model, batch, device=self._device)

        out: List[RecognizedGlyph] = []
        for seg, i, c in zip(segments, idx, conf):
            if c < self._min_confidence:
                continue
            name = self._labels[int(i)]
            sym = symbols.by_name(name)
            out.append(
                RecognizedGlyph(
                    name=name,
                    text=sym.char if sym else name,
                    bbox=seg.bbox,
                    confidence=float(c),
                )
            )
        return out
