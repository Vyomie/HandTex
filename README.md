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
      │  ocr        (handtex.ocr)      pluggable: mock | local CNN | vision model
      ▼
list[RecognizedGlyph]                  each: name + char + pixel bbox + confidence
      │  layout     (handtex.layout)   line grouping, baseline, super/subscripts,
      ▼                                 high/low mark placement (, ' " *)
Document  ──► latex  (handtex.latex)   →  x^{2} + \alpha = y_{1}
          └─► render (handtex.render)  →  PNG typeset in your handwritten font
                                            ▲
your handwriting  ──►  font  (handtex.font)  ──►  HandTex.ttf  (Unicode + LaTeX-ready)
```

### The small local OCR model

The default real recognizer (`--backend local`) is a **tiny CNN** that runs
entirely offline. It works because the layout engine already handles spatial
reasoning, so the model only has to classify one glyph at a time:

```
image ─► segment (connected components, merge stacked marks like =, i, ÷)
      ─► normalize each glyph to 32×32
      ─► GlyphCNN  (~174k params)  ─► symbol name + confidence + bbox
```

It is bootstrapped on **synthetic data** — every symbol rendered across all
installed fonts with augmentations (`handtex train-ocr`) — so no hand-labeling
is needed to start. There is a real print→handwriting domain gap; fine-tuning
on your own capture-sheet glyphs is the way to close it (roadmap).

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

### Build a font straight from a handwriting photo (full OCR)

No template/manifest needed — the OCR model recognizes each glyph and traces it:

```bash
pip install -e '.[local]'                      # adds torch
python -m handtex train-ocr --epochs 12        # one-time (weights are committed)
python -m handtex font-from-image sheet.jpg --out MyHand.ttf --show
```

### Read a whiteboard

```bash
# Small local CNN (offline, no API):
python -m handtex read photo.jpg --backend local \
       --tex out.tex --png out.png --font MyHand.ttf

# Or fully offline with hand-authored glyph boxes (mock + sidecar JSON):
python -m handtex read photo.jpg --backend mock --tex out.tex

# Or a vision model:
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
    local/            small CNN: segment + normalize + classify + train
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
    from_image.py     full-OCR: recognize a handwriting photo -> font
  cli.py              `handtex template | build-font | font-from-image | train-ocr | read | demo`
tests/                unit + end-to-end coverage
```

## Roadmap

Real today: font building (template → trace → Unicode TTF), layout
(lines/baseline/scripts/mark-placement), LaTeX emission, PNG rendering, the
small local OCR model (segment + CNN, trained on synthetic data) plus
`font-from-image`, the offline mock OCR, and the vision-backend wiring.

Next:

- **Close the print→handwriting gap** — fine-tune the CNN on the user's own
  capture-sheet glyphs (the labels are known from the template), which is the
  single biggest accuracy win for real handwriting.
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
