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
from .model import GlyphCNN, has_weights, load_model, predict_topk
from .normalize import normalize_glyph
from .refine import Candidate, refine
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
        idxk, confk = predict_topk(self._model, batch, k=4, device=self._device)

        # Refine: drop doodles, size-outliers and overlapping duplicates.
        # Repeated characters are legitimate when reading text, so do NOT
        # enforce label uniqueness here.
        cands = [
            Candidate(index=n, bbox=segments[n].bbox,
                      options=[(self._labels[int(idxk[n][j])], float(confk[n][j]))
                               for j in range(idxk.shape[1])])
            for n in range(len(segments))
        ]
        resolved = refine(cands, unique_labels=False)

        out: List[RecognizedGlyph] = []
        for r in resolved:
            if r.confidence < self._min_confidence:
                continue
            sym = symbols.by_name(r.label)
            out.append(
                RecognizedGlyph(name=r.label, text=sym.char if sym else r.label,
                                bbox=segments[r.index].bbox, confidence=r.confidence)
            )
        return out
