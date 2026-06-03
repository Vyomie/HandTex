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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ...font import symbols
from .normalize import OUT_SIZE, normalize_glyph

_FONT_GLOBS = [
    "/usr/share/fonts/**/*.ttf",
    "/usr/share/fonts/**/*.otf",
]


_REPO_ROOT = Path(__file__).resolve().parents[3]
_HANDWRITING_GLOBS = [
    str(_REPO_ROOT / "assets" / "handwriting_fonts" / "*.ttf"),
    str(_REPO_ROOT / "examples" / "handwriting_to_font" / "MyHandwriting.ttf"),
]


@lru_cache(maxsize=1)
def discover_fonts() -> List[str]:
    paths: List[str] = []
    for pat in _FONT_GLOBS:
        paths.extend(glob.glob(pat, recursive=True))
    return sorted(set(paths))


@lru_cache(maxsize=1)
def discover_handwriting_fonts() -> List[str]:
    """Handwriting fonts (downloaded set + the user's own) used to close the
    print→handwriting domain gap. Reproduce the downloaded set with
    ``scripts/fetch_handwriting_fonts.sh``."""
    paths: List[str] = []
    for pat in _HANDWRITING_GLOBS:
        paths.extend(glob.glob(pat))
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


DOODLE_LABEL = "doodle"


def _make_doodle(rng: random.Random) -> np.ndarray:
    """Synthesize a random scribble/blob: a 'gibberish' reject sample.

    Doodles are intentionally *complex* (several crossing/squiggly strokes or
    blobs) so they stay visually distinct from clean single-stroke glyphs like
    ``/ 1 - l``, which a real classifier must keep.
    """
    S = 56
    img = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(img)
    kind = rng.random()
    if kind < 0.55:                      # multi-stroke squiggle (crossing)
        for _ in range(rng.randint(3, 6)):
            pts = [(rng.randint(2, S - 2), rng.randint(2, S - 2))
                   for _ in range(rng.randint(3, 7))]
            d.line(pts, fill=255, width=rng.randint(2, 6), joint="curve")
    elif kind < 0.8:                     # dense short-stroke scribble
        for _ in range(rng.randint(8, 18)):
            x, y = rng.randint(6, S - 6), rng.randint(6, S - 6)
            d.line([(x, y), (x + rng.randint(-16, 16), y + rng.randint(-16, 16))],
                   fill=255, width=rng.randint(2, 5))
    else:                                # blobs / fills
        for _ in range(rng.randint(1, 3)):
            x0, y0 = rng.randint(2, S - 26), rng.randint(2, S - 26)
            d.ellipse([x0, y0, x0 + rng.randint(10, 30), y0 + rng.randint(10, 30)], fill=255)
        for _ in range(rng.randint(1, 3)):
            pts = [(rng.randint(2, S - 2), rng.randint(2, S - 2)) for _ in range(rng.randint(3, 6))]
            d.line(pts, fill=255, width=rng.randint(2, 6), joint="curve")

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if rng.random() < 0.25:              # speckle noise
        arr = np.clip(arr + (np.random.rand(S, S) < 0.02).astype(np.float32), 0, 1)
    return _augment_array(arr, rng)


def _augment_array(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Normalize + light jitter (shared tail of glyph augmentation)."""
    norm = normalize_glyph(arr)
    dx, dy = rng.randint(-2, 2), rng.randint(-2, 2)
    return np.clip(np.roll(norm, (dy, dx), axis=(0, 1)), 0.0, 1.0)


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
    handwriting_fonts: Optional[List[str]] = None,
    handwriting_oversample: int = 5,
    math_oversample: int = 4,
    doodle_samples: int = 1500,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Render the full synthetic dataset.

    Print fonts give broad coverage (incl. Greek/math); handwriting fonts are
    *oversampled* so the classifier generalizes to real handwriting for the
    Latin letters, digits and punctuation they cover. A ``doodle`` reject class
    of random scribbles lets the model flag gibberish for removal.

    Returns ``(X, y, labels)`` where ``X`` is ``(N, OUT_SIZE, OUT_SIZE)``
    float32, ``y`` is ``(N,)`` int64 class indices, and ``labels`` maps index
    to symbol name (the last label is ``doodle`` when ``doodle_samples`` > 0).
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    labels = class_names()
    if doodle_samples > 0:
        labels = labels + [DOODLE_LABEL]
    label_to_idx = {n: i for i, n in enumerate(labels)}

    if handwriting_fonts is None:
        handwriting_fonts = discover_handwriting_fonts()
    print_fonts = discover_fonts()[:max_fonts]
    if not print_fonts and not handwriting_fonts:
        raise RuntimeError("no fonts found to synthesize training data")

    # (path, is_handwriting); handwriting fonts get more augmented variants.
    fonts: List[Tuple[str, bool]] = [(f, False) for f in print_fonts]
    fonts += [(f, True) for f in handwriting_fonts]

    X: List[np.ndarray] = []
    y: List[int] = []
    for name in labels:
        if name == DOODLE_LABEL:
            continue  # synthesized separately below, not from a font
        sym = symbols.by_name(name)
        char = sym.char
        # Greek/math symbols (non-ASCII) are the weak spot — no handwriting
        # fonts cover them — so oversample their (printed) renderings.
        is_focus = not char.isascii()
        for path, is_hw in fonts:
            if not font_covers(path, char):
                continue
            try:
                font = ImageFont.truetype(path, 44)
            except Exception:
                continue
            base = _render_glyph(char, font)
            if base.max() <= 0:
                continue
            variants = per_class_per_font * (handwriting_oversample if is_hw else 1)
            if is_focus:
                variants *= math_oversample
            # one clean sample + augmented variants
            X.append(normalize_glyph(base)); y.append(label_to_idx[name])
            for _ in range(variants):
                X.append(_augment(base, rng)); y.append(label_to_idx[name])
        if progress:
            print(f"  {name}: {sum(1 for yi in y if yi == label_to_idx[name])} samples")

    if doodle_samples > 0:
        didx = label_to_idx[DOODLE_LABEL]
        for _ in range(doodle_samples):
            X.append(_make_doodle(rng)); y.append(didx)

    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64), labels
