"""Visualize OCR results: draw each glyph's box colored by confidence.

Confidence is shown purely through color — a continuous red (low) → yellow →
green (high) scale — so you can read reliability at a glance without numbers
cluttering the image.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import BBox, RecognizedGlyph

Item = Tuple[BBox, str, float]  # (bbox, label, confidence)


def confidence_color(conf: float) -> Tuple[int, int, int]:
    """Map confidence in [0, 1] to a red→yellow→green RGB color."""
    t = max(0.0, min(1.0, float(conf)))
    red = (220, 50, 50)
    yellow = (235, 200, 40)
    green = (40, 175, 70)
    if t < 0.5:
        a, b, f = red, yellow, t / 0.5
    else:
        a, b, f = yellow, green, (t - 0.5) / 0.5
    return tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))


def _to_items(objs: Iterable) -> List[Item]:
    """Accept RecognizedGlyph, RecognizedCell, or (bbox, label, conf) tuples."""
    items: List[Item] = []
    for o in objs:
        if isinstance(o, RecognizedGlyph):
            items.append((o.bbox, o.text, o.confidence))
        elif isinstance(o, tuple) and len(o) == 3:
            items.append(o)
        elif hasattr(o, "segment"):  # RecognizedCell
            items.append((o.segment.bbox, getattr(o, "char", "?"), o.confidence))
        elif hasattr(o, "bbox"):
            items.append((o.bbox, getattr(o, "text", "?"), getattr(o, "confidence", 1.0)))
    return items


def _legend(draw: ImageDraw.ImageDraw, x: int, y: int, w: int = 240, h: int = 14) -> None:
    for i in range(w):
        draw.line([(x + i, y), (x + i, y + h)], fill=confidence_color(i / (w - 1)))
    draw.rectangle([x, y, x + w, y + h], outline=(0, 0, 0))


def annotate_recognition(
    image: Union[str, np.ndarray, Image.Image],
    objs: Sequence,
    out_path: str,
    draw_label: bool = True,
    box_width: int = 3,
    legend: bool = True,
) -> str:
    """Draw confidence-colored boxes (and optional char labels) onto ``image``."""
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = image.convert("RGB")

    draw = ImageDraw.Draw(img)
    items = _to_items(objs)
    size = max(14, int(0.02 * img.height))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        font = ImageFont.load_default()

    for bbox, label, conf in items:
        color = confidence_color(conf)
        draw.rectangle([bbox.left, bbox.top, bbox.right, bbox.bottom],
                       outline=color, width=box_width)
        if draw_label:
            draw.text((bbox.left, bbox.top - size - 2), label, fill=color, font=font)

    if legend and items:
        _legend(draw, 12, 12)

    img.save(out_path)
    return out_path
