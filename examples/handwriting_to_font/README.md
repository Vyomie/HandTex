# Example: handwriting → font → LaTeX

A real run of HandTex on actual handwriting.

## Files

| file | what it is |
|------|------------|
| `my_handwriting_grid.png` | The handwriting capture sheet (A–Z, a–z, 0–9, punctuation) fed into HandTex. |
| `MyHandwriting.ttf` | A reference handwriting font (Calligraphr build of the same hand) — used here to show the font working as a **normal font** and in **LaTeX**. |
| `MyHand-FromOCR.ttf` | The font **HandTex built itself** by running its OCR end-to-end on `my_handwriting_grid.png` (`handtex font-from-image`): segment → classify → trace → build. |
| `MyHandwriting_normal.png` | `MyHandwriting.ttf` rendered as an ordinary font (PIL): pangram, digits, punctuation, and an `x² − y₁` super/subscript expression. |
| `MyHandwriting_latex_hybrid.png` | `MyHandwriting.ttf` compiled in **real LaTeX** (XeLaTeX via tectonic + `mathastext`): `E = mc²`, `y = x² + 3x + 1`, `H₂O`, `(a+b)/2`. Letters/digits are handwritten; math operators fall back to the math font — a **hybrid**. |
| `MyHandwriting_latex_hybrid.tex` | The LaTeX source for the above (compile with `xelatex`/`lualatex`/`tectonic`). |

## Reproduce

```bash
# Build a font from the handwriting sheet using HandTex's own OCR:
python -m handtex font-from-image examples/handwriting_to_font/my_handwriting_grid.png \
       --out MyHand-FromOCR.ttf --top-crop 0.04 --show

# Compile the hybrid LaTeX (needs a unicode-aware engine):
tectonic examples/handwriting_to_font/MyHandwriting_latex_hybrid.tex
```

## End-to-end OCR run on the hybrid LaTeX image

Running the full pipeline (`segment → classify → layout → build font`) on
`MyHandwriting_latex_hybrid.png` (re-rasterized at high resolution):

`e2e_hybrid_annotated.png` shows the result — green box = confident, orange =
low confidence, red label = predicted character.

| stage | result |
|-------|--------|
| segmentation | **18 glyphs** found, mostly one box per glyph |
| classification | printed operators/digits perfect (`= + 2 3`); handwritten letters mostly right (`E m y x H`); errors: `c`→⊂, `1`→l, `O`→o, and the fraction's `a+b` merged into one box |
| layout | over-splits into super/subscripts here, because the handwritten lowercase letters are genuinely much shorter than the printed digits/capitals, so they look like scripts |
| build font | succeeds — a `.ttf` is produced from the recognized glyphs |

Takeaways: **segmentation and printed-glyph recognition are solid**; the two
weak spots are (1) the classifier on handwritten letters (domain gap — fix by
training on handwriting fonts) and (2) fraction structure (roadmap). The
layout over-subscripting is specific to size-disparate *hybrid* images and is
far milder on uniform handwriting.

## Notes

`MyHand-FromOCR.ttf` is the honest output of the current OCR: segmentation is
solid (65/70 glyphs found and separated), but the classifier — trained only on
*printed* fonts so far — mislabels some handwritten letters (e.g. `B`→β,
`C`→⊂, `D`→p). Training the classifier on real handwriting fonts (see
`scripts/fetch_handwriting_fonts.sh`) is what closes that gap.
