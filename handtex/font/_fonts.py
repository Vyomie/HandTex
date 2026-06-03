"""Locate a Unicode reference font (for templates and offline synthesis).

We need *some* TrueType font that covers Greek/math so the capture sheet can
show reference glyphs and the demo/tests can synthesize stand-in handwriting.
Falls back to PIL's bitmap default if nothing is found.
"""

from __future__ import annotations

import glob
import os
from functools import lru_cache
from typing import Optional

from PIL import ImageFont

_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


@lru_cache(maxsize=1)
def find_reference_font_path() -> Optional[str]:
    for path in _CANDIDATES:
        if os.path.exists(path):
            return path
    for pattern in (
        "/usr/share/fonts/**/DejaVuSans.ttf",
        "/usr/share/fonts/**/*Sans*.ttf",
        "/usr/share/fonts/**/*.ttf",
    ):
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def load_reference_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    path = find_reference_font_path()
    if path is None:
        try:
            return ImageFont.load_default()
        except Exception:
            return None
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return None
