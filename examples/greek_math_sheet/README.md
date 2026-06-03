# Example: Greek/math sheet with deliberate scribbles

A phone photo (`sheet.jpg`) of handwritten uppercase, lowercase, a row of
superscript Greek/math (`x¹ y² z³ α⁴ β⁵ δ⁶ θ⁷ φ⁸ + - / ~ =`), operators and
brackets (`∫ ∂ ° % { } ( ) ∈ ⊂ [ ]`), **and several intentional scribbles**
(after `z³`, after `mn`, and in the bottom row) to test doodle rejection.

`sheet_annotated.png` is the OCR result after refinement, with boxes colored by
confidence (red→green). Produced with the model trained for 50 epochs and a
×6 oversample on Greek/math classes (synthetic val_acc ≈ 0.93).

```bash
python -m handtex train-ocr --epochs 50 --math-oversample 6
python -m handtex font-from-image sheet.jpg --out PaperHand.ttf --annotate sheet_annotated.png
```

## What this shows

- **Doodle rejection works**: the 9 scribbles/blobs are dropped (unboxed),
  while real glyphs around them are kept.
- **Greek/math recognized**: with the focused training the Greek letters and
  math symbols are detected and boxed (α, β, δ, θ, φ, ∫, ∂, ∈, ⊂, …), where
  the earlier print-only model mostly failed on them.
- Honest limits: handwritten Greek is still trained from *printed* fonts (just
  oversampled), so some confusions remain (e.g. `α`↔`∝`); and upper/lower
  look-alikes (`C/c`, `X/x`, `O/o`) stay ambiguous. Real handwritten samples of
  these glyphs are the next accuracy lever.
