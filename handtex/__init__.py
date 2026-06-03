"""HandTex — turn handwriting into a font and read whiteboards into LaTeX.

Three subsystems, wired into one pipeline:

  * font   — build a Unicode .ttf/.otf from samples of your handwriting,
             mapping math symbols to real codepoints so the *same* file
             works as a normal font in any app AND in LaTeX (fontspec).
  * ocr    — pluggable recognition backends (mock for offline runs, a
             vision-model backend for the real thing).
  * layout — turn recognized glyphs + bounding boxes into a structured
             document: line grouping, baseline detection, super/subscripts
             and punctuation placement.

See ``handtex.cli`` for the command-line entry point and ``examples/demo.py``
for an end-to-end synthetic run.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
