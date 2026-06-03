# Example: a real photo of handwriting on paper

A phone photo (`paper.jpg`, 4080×2296) of handwritten uppercase, lowercase,
digits, a row of superscript math (`x¹ + y² + z³ + α⁴ + β⁵ + …`), and symbols
(`÷ ∈ ⊂ { } [ ] ( ) + = ≈`). It exercises the whole pipeline on messy,
real-world input.

| file | what it is |
|------|------------|
| `paper.jpg` | the original phone photo (off-white paper, shadow, slight tilt) |
| `paper_preprocessed.png` | after `handtex.preprocess`: resized, illumination-flattened, contrast-stretched, deskewed |
| `paper_annotated.png` | OCR result — each glyph boxed and **colored by confidence** (red = low → green = high; legend top-left). No numbers, just color. |

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
- **Multi-row page layout**: improved (a spurious full-height shadow no longer
  collapses everything; rows separate), but a dense 9-row page is still harder
  than a single equation — per-row super/subscript grouping on a full page is
  the next layout milestone. Super/subscript detection is reliable on single
  equations (see the `read` command and tests).
