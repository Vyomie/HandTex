"""Image preprocessing for robust glyph detection.

A photo of a whiteboard or a sheet of paper is rarely clean: it can be tilted,
unevenly lit, low-contrast, or the wrong size. This module fixes those before
segmentation so *every* letter and symbol is detectable:

    resize  →  flatten illumination  →  stretch contrast  →  deskew

Each step is conservative and a no-op on already-clean input, so it is safe to
run by default (see ``handtex.ocr.local.segment.segment_image``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from PIL import Image
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian, threshold_otsu
from skimage.transform import rotate

ImageLike = Union[str, np.ndarray, Image.Image]


@dataclass
class PreprocessParams:
    min_height: int = 900       # upscale shorter images so glyphs are big enough
    max_height: int = 1600      # downscale taller images for speed
    illumination: bool = True   # flatten shadows / lighting gradients
    contrast: bool = True       # stretch to the full tonal range
    deskew: bool = True         # rotate the page upright
    max_skew_deg: float = 12.0  # search range for skew estimation
    min_skew_deg: float = 0.3   # ignore skew below this (avoid needless blur)


def to_grayscale(image: ImageLike) -> np.ndarray:
    """Load anything into a uint8 grayscale array."""
    if isinstance(image, str):
        return np.asarray(Image.open(image).convert("L"))
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"))
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = np.asarray(Image.fromarray(arr).convert("L"))
    return arr.astype(np.uint8)


def resize_for_ocr(gray: np.ndarray, min_height: int, max_height: int) -> np.ndarray:
    """Scale so the image height lands in ``[min_height, max_height]``."""
    h = gray.shape[0]
    if h == 0:
        return gray
    scale = 1.0
    if h < min_height:
        scale = min_height / h
    elif h > max_height:
        scale = max_height / h
    if abs(scale - 1.0) < 1e-3:
        return gray
    img = Image.fromarray(gray)
    new = (max(1, round(gray.shape[1] * scale)), max(1, round(h * scale)))
    return np.asarray(img.resize(new, Image.LANCZOS))


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide out smooth lighting gradients (shadows, vignetting).

    Estimates the local paper brightness with a large Gaussian blur and
    normalizes against it, so dark ink stands out evenly across the frame.
    """
    g = gray.astype(np.float32)
    sigma = max(gray.shape) / 30.0
    background = gaussian(g, sigma=sigma, preserve_range=True)
    background = np.maximum(background, 1.0)
    norm = g / background
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255).astype(np.uint8)


def stretch_contrast(gray: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Percentile contrast stretch to the full 0–255 range."""
    lo, hi = np.percentile(gray, (low_pct, high_pct))
    if hi - lo < 1e-3:
        return gray
    out = rescale_intensity(gray, in_range=(lo, hi), out_range=(0, 255))
    return out.astype(np.uint8)


def estimate_skew(gray: np.ndarray, max_skew_deg: float = 12.0, step: float = 0.5) -> float:
    """Estimate page skew in degrees via projection-profile sharpness.

    The correct rotation makes text rows align horizontally, which maximizes
    the variance of the row-sum profile of the ink mask.
    """
    # Work on a small binary copy for speed.
    small = Image.fromarray(gray)
    scale = 600.0 / max(gray.shape)
    if scale < 1.0:
        small = small.resize((max(1, int(gray.shape[1] * scale)),
                              max(1, int(gray.shape[0] * scale))), Image.BILINEAR)
    arr = np.asarray(small)
    try:
        thr = threshold_otsu(arr)
    except Exception:
        thr = 128
    ink = (arr < thr).astype(np.float32)
    if ink.sum() < 10:
        return 0.0

    best_angle, best_score = 0.0, -1.0
    angle = -max_skew_deg
    while angle <= max_skew_deg + 1e-6:
        rotated = rotate(ink, angle, resize=False, order=0, preserve_range=True)
        profile = rotated.sum(axis=1)
        score = float(np.var(np.diff(profile)))
        if score > best_score:
            best_score, best_angle = score, angle
        angle += step
    return best_angle


def deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by ``angle`` degrees, keeping the paper background white."""
    # cval is in the same 0–255 range as the data (preserve_range): fill the
    # exposed corners with white paper, not black.
    rotated = rotate(gray, angle, resize=True, order=1, mode="constant",
                     cval=255.0, preserve_range=True)
    return np.clip(rotated, 0, 255).astype(np.uint8)


def preprocess(image: ImageLike, params: PreprocessParams | None = None) -> np.ndarray:
    """Return a cleaned uint8 grayscale image ready for segmentation."""
    p = params or PreprocessParams()
    gray = to_grayscale(image)
    gray = resize_for_ocr(gray, p.min_height, p.max_height)
    if p.illumination:
        gray = flatten_illumination(gray)
    if p.contrast:
        gray = stretch_contrast(gray)
    if p.deskew:
        angle = estimate_skew(gray, p.max_skew_deg)
        if abs(angle) >= p.min_skew_deg:
            gray = deskew(gray, angle)
    return gray
