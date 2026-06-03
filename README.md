# HandTex

Turn **your handwriting** into a real font, then point a camera at a
**whiteboard** and get back **LaTeX** — typeset in that same hand.

One font file, three jobs:

- works as a **normal font** in any application (you type `α`, you get *your* α),
- works in **LaTeX** unchanged (XeLaTeX/LuaLaTeX via `fontspec` / `unicode-math`),
- is the font HandTex uses to **re-render** whatever it reads off a whiteboard.

> Status: **end-to-end skeleton.** Every stage runs today with an offline mock
> recognizer and stand-in handwriting, so you can see the whole pipeline work
> before plugging in your real samples and a vision model. See
> [Roadmap](#roadmap) for what's stubbed vs. real.

## The pipeline

```
whiteboard image
      │  ocr        (handtex.ocr)      pluggable: mock | vision model
      ▼
list[RecognizedGlyph]                  each: name + char + pixel bbox + confidence
      │  layout     (handtex.layout)   line grouping, baseline, super/subscripts,
      ▼                                 high/low mark placement (, ' " *)
Document  ──► latex  (handtex.latex)   →  x^{2} + \alpha = y_{1}
          └─► render (handtex.render)  →  PNG typeset in your handwritten font
                                            ▲
your handwriting  ──►  font  (handtex.font)  ──►  HandTex.ttf  (Unicode + LaTeX-ready)
```

## Quickstart

```bash
pip install -e .            # fonttools, pillow, numpy, scikit-image

# See the whole thing run with synthetic handwriting + a built-in scene:
python -m handtex demo --out out/
#   out/HandTex-Regular.ttf   a real font (Greek + math mapped to Unicode)
#   out/whiteboard.tex        x^{2} + \alpha = y_{1}
#   out/whiteboard.png        that equation rendered in the font
```

### Make a font from *your* handwriting

```bash
# 1. Print a capture sheet (one cell per glyph, with baseline guides):
python -m handtex template --out sheet.png --manifest sheet.json

# 2. Write each glyph on the baseline, photograph/scan it back, then:
python -m handtex build-font --filled sheet_filled.png --manifest sheet.json \
       --out HandTex-Regular.ttf
```

### Read a whiteboard

```bash
# Offline (mock backend / sidecar JSON of recognized glyphs):
python -m handtex read photo.jpg --backend mock \
       --tex out.tex --png out.png --font HandTex-Regular.ttf

# Real recognition with a vision model:
export ANTHROPIC_API_KEY=...        # pip install 'handtex[vision]'
python -m handtex read photo.jpg --backend vision --tex out.tex
```

## How "one font works everywhere" works

Every glyph is bound to its true **Unicode code point** in the font's `cmap`
(`handtex/font/symbols.py`). Greek `α` → U+03B1, `∑` → U+2211, and so on. That
single mapping is why the math glyphs show up when you type the character in any
app *and* why LaTeX picks up the same outlines:

```latex
\documentclass{article}
\usepackage{fontspec}\usepackage{unicode-math}
\setmathfont{HandTex-Regular.ttf}
\begin{document}$x^{2} + \alpha = y_{1}$\end{document}   % compile with xelatex
```

## Layout: sizing & placement

The layout engine (`handtex/layout/engine.py`) works purely from glyph geometry:

- **Lines** are seeded from base-sized glyphs, then small glyphs attach to the
  nearest line — so a raised exponent never becomes its own line.
- **Baseline & body height** are estimated per line (robust percentile/median).
- **Super/subscripts**: a glyph that is *small* **and** clearly raised/lowered
  relative to the base center becomes `^{...}` / `_{...}` on the preceding base.
- **`,  '  "  *` placement**: these carry an intrinsic vertical hint (`VHint`)
  in the symbol table, so a high apostrophe is never mistaken for an exponent
  and a low comma is never mistaken for a subscript.

Thresholds live in `LayoutParams` — tune them for your hand.

## Project layout

```
handtex/
  types.py            core dataclasses (BBox, RecognizedGlyph, Token, Document)
  ocr/                recognition backends (base, mock, vision)
  layout/engine.py    geometry -> structured Document
  latex/emit.py       Document -> LaTeX
  render/render.py    Document + font -> PNG
  font/
    symbols.py        name ⟷ char ⟷ codepoint ⟷ LaTeX  (the keystone)
    template.py       printable capture sheet
    extract.py        filled sheet -> per-glyph bitmaps
    trace.py          bitmap -> vector contours
    build.py          contours -> .ttf (Unicode cmap, LaTeX-ready)
    pipeline.py       glue + offline glyph synthesis
  cli.py              `handtex template | build-font | read | demo`
tests/                unit + end-to-end coverage
```

## Roadmap

Real today: font building (template → trace → Unicode TTF), layout
(lines/baseline/scripts/mark-placement), LaTeX emission, PNG rendering, the
offline mock OCR, and the vision-backend wiring.

Next:

- **Vision backend hardening** — calibrate bbox accuracy, add retries/caching.
- **Capture UX** — auto-detect/deskew a photographed sheet (currently geometry
  is read from the manifest; perspective correction is TODO).
- **Outlines** — curve fitting (quadratics) instead of polygon simplification;
  optional `.otf`/CFF output; stroke smoothing.
- **Layout** — fractions, roots, matrices, and multi-glyph base spans.
- **Training your own recognizer** — the backend interface is ready for a
  fine-tuned/custom model if you outgrow the API.

## License

MIT
