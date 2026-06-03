"""Font building: handwriting bitmaps -> a Unicode .ttf/.otf.

  * symbols  — the name/char/codepoint/LaTeX table.
  * template — generate a printable sheet to write your glyphs on.
  * extract  — slice a filled-in sheet back into per-glyph bitmaps.
  * trace    — bitmap -> vector contours.
  * build    — contours -> a real font file (Unicode cmap + LaTeX-ready).
"""

from . import symbols  # noqa: F401

__all__ = ["symbols"]
