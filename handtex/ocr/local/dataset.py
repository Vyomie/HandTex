"""Synthetic training data for the glyph classifier.

We have no labeled handwriting yet, so we bootstrap from fonts: render every
symbol in every font that actually covers it, apply light augmentations
(rotation, scale, translation, stroke weight, noise, thresholding), and
normalize. This is the standard cold-start; fine-tuning on the user's own
capture-sheet glyphs (see ``handtex.font.extract``) closes the print→hand gap.
"""

from __future__ import annotations

import glob
import os
import random
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ...font import symbols
from .normalize import OUT_SIZE, normalize_glyph

_FONT_GLOBS = [
    "/usr/share/fonts/**/*.ttf",
    "/usr/share/fonts/**/*.otf",
]


@lru_cache(maxsize=1)
def discover_fonts() -> List[str]:
    paths: List[str] = []
    for pat in _FONT_GLOBS:
        paths.extend(glob.glob(pat, recursive=True))
    return sorted(set(paths))


@lru_cache(maxsize=256)
def _font_cmap(path: str) -> frozenset:
    try:
        return frozenset(TTFont(path, fontNumber=0).getBestCmap().keys())
    except Exception:
        return frozenset()


def font_covers(path: str, char: str) -> bool:
    return ord(char) in _font_cmap(path)


def class_names() -> List[str]:
    """Stable, ordered list of class labels (symbol names) for the model."""
    return [s.name for s in symbols.all_symbols() if s.char != " "]


def _render_glyph(char: str, font: ImageFont.FreeTypeFont, canvas: int = 64) -> np.ndarray:
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    try:
        l, t, r, b = draw.textbbox((0, 0), char, font=font)
    except Exception:
        return np.zeros((canvas, canvas), dtype=np.float32)
    gw, gh = r - l, b - t
    if gw <= 0 or gh <= 0:
        return np.zeros((canvas, canvas), dtype=np.float32)
    draw.text(((canvas - gw) / 2 - l, (canvas - gh) / 2 - t), char, fill=255, font=font)
    return np.asarray(img, dtype=np.float32) / 255.0


def _augment(ink: np.ndarray, rng: random.Random) -> np.ndarray:
    """Light, glyph-preserving augmentation on a 64x64 ink image."""
    img = Image.fromarray((np.clip(ink, 0, 1) * 255).astype(np.uint8))
    # rotation + scale + translation via affine
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.82, 1.12)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    if abs(scale - 1.0) > 1e-3:
        w, h = img.size
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)

    # stroke weight: occasional dilate/erode (MaxFilter thickens, MinFilter thins)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.MaxFilter(3) if rng.random() < 0.5 else ImageFilter.MinFilter(3))

    out = np.asarray(img, dtype=np.float32) / 255.0
    norm = normalize_glyph(out)
    # translation jitter
    dx, dy = rng.randint(-2, 2), rng.randint(-2, 2)
    norm = np.roll(norm, (dy, dx), axis=(0, 1))
    # noise + random binarization
    if rng.random() < 0.4:
        norm = norm + rng.uniform(0.0, 0.08) * np.random.randn(*norm.shape).astype(np.float32)
    if rng.random() < 0.3:
        norm = (norm > rng.uniform(0.3, 0.6)).astype(np.float32)
    return np.clip(norm, 0.0, 1.0)


def build_dataset(
    per_class_per_font: int = 4,
    max_fonts: int = 40,
    seed: int = 0,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Render the full synthetic dataset.

    Returns ``(X, y, labels)`` where ``X`` is ``(N, OUT_SIZE, OUT_SIZE)``
    float32, ``y`` is ``(N,)`` int64 class indices, and ``labels`` maps index
    to symbol name.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    labels = class_names()
    label_to_idx = {n: i for i, n in enumerate(labels)}
    fonts = discover_fonts()[:max_fonts]
    if not fonts:
        raise RuntimeError("no fonts found to synthesize training data")

    X: List[np.ndarray] = []
    y: List[int] = []
    for name in labels:
        sym = symbols.by_name(name)
        char = sym.char
        covering = [f for f in fonts if font_covers(f, char)]
        if not covering:
            continue
        for path in covering:
            try:
                font = ImageFont.truetype(path, 44)
            except Exception:
                continue
            base = _render_glyph(char, font)
            if base.max() <= 0:
                continue
            # one clean sample + augmented variants
            X.append(normalize_glyph(base)); y.append(label_to_idx[name])
            for _ in range(per_class_per_font):
                X.append(_augment(base, rng)); y.append(label_to_idx[name])
        if progress:
            print(f"  {name}: {len([1 for yi in y if yi == label_to_idx[name]])} samples")

    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64), labels
