"""Shared glyph normalization.

Both training-data synthesis and inference funnel every glyph through the
*same* normalization so the model sees a consistent input: an ink-intensity
image (high = ink) is cropped to its content, scaled to fit a fixed content
box preserving aspect ratio, and centered on a square canvas.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

OUT_SIZE = 32      # model input is OUT_SIZE x OUT_SIZE
CONTENT = 24       # glyph is scaled to fit a CONTENT x CONTENT box, centered
_INK_THRESHOLD = 0.15


def normalize_glyph(ink: np.ndarray, out_size: int = OUT_SIZE, content: int = CONTENT) -> np.ndarray:
    """Normalize an ink-intensity image to ``out_size x out_size`` float32.

    ``ink`` is a 2-D array where higher values mean more ink. Returns an array
    in ``[0, 1]`` with the glyph centered; an empty input yields all zeros.
    """
    ink = np.asarray(ink, dtype=np.float32)
    if ink.max() > 1.0:
        ink = ink / 255.0

    ys, xs = np.where(ink > _INK_THRESHOLD)
    canvas = np.zeros((out_size, out_size), dtype=np.float32)
    if len(xs) == 0:
        return canvas

    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = content / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    resized = np.asarray(img, dtype=np.float32) / 255.0

    y0 = (out_size - new_h) // 2
    x0 = (out_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas
