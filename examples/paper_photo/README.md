# Example: a real photo of handwriting on paper

A phone photo (`paper.jpg`, 4080×2296) of handwritten uppercase, lowercase,
digits, a row of superscript math (`x¹ + y² + z³ + α⁴ + β⁵ + …`), and symbols
(`÷ ∈ ⊂ { } [ ] ( ) + = ≈`). It exercises the whole pipeline on messy,
real-world input.

| file | what it is |
|------|------------|
| `paper.jpg` | the original phone photo (off-white paper, shadow, slight tilt) |
| `paper_preprocessed.png` | after `handtex.preprocess`: resized, illumination-flattened, contrast-stretched, deskewed |
| `paper_annotated.png` | OCR result after **refinement** — each glyph boxed and **colored by confidence** (red = low → green = high; legend top-left). The inkblot after "MN" is gone (classified `doodle` and removed); duplicates were resolved so every box is a unique glyph. |

Reproduce:

```bash
python -m handtex preprocess paper.jpg --out clean.png
python -m handtex font-from-image paper.jpg --out PaperHand.ttf --annotate annotated.png
```

## What works / what doesn't (honest)

- **Preprocessing**: handles the shadow, off-white paper and tilt well.
- **Segmentation**: excellent — ~108 glyphs found, including the small
  superscripts and the tall integral.
- **Classification**: solid on uppercase/lowercase letters and many digits
  (handwriting-trained model); weaker on Greek/math symbols (still mostly
  print-trained) and on genuinely ambiguous upper/lower pairs (`C/c`, `O/o`,
  `V/v`, `X/x`).
- **Refinement (situational awareness)**: a `doodle` reject class removes the
  inkblot and stray marks — but only when the model is *confidently* doodle, so
  real glyphs are kept (4 removed here, down from 16 before the threshold +
  cleaner doodle training). Size-outlier + low-confidence boxes are pruned, and
  on this character sheet duplicate labels are resolved by confidence (the
  loser falls back to its 2nd-best guess, or is dropped) so every surviving box
  is a unique glyph (84 kept).
- **Multi-row page layout**: improved (a spurious full-height shadow no longer
  collapses everything; rows separate), but a dense 9-row page is still harder
  than a single equation — per-row super/subscript grouping on a full page is
  the next layout milestone. Super/subscript detection is reliable on single
  equations (see the `read` command and tests).
