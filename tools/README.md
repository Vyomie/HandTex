# HandTex tools

## `handwriting_studio.html` — draw your font in the browser

A self-contained page to **trace every glyph by hand** and export an initial
font. No build step, no OCR — just open it.

```
# open it directly, or serve the repo and browse to tools/handwriting_studio.html
python -m http.server   # then visit http://localhost:8000/tools/handwriting_studio.html
```

### What it does
- Steps through all 149 HandTex symbols (A–Z, a–z, 0–9, punctuation, Greek,
  math) one at a time — **◀ Prev / Next ▶** or the arrow keys.
- Each glyph has a tracing canvas with **baseline, x-height, cap and
  descender guides** and a faint **reference character to trace over**
  (toggle with “trace guide”).
- Adjustable **pen width**; **Undo** (`U`) / **Clear** (`C`); a live grid
  showing which glyphs are done. Work **autosaves** to your browser.

### Exports
- **⬇ Download .ttf** — builds a real Unicode font right in the browser
  (via opentype.js) from your strokes. Math symbols map to their true code
  points, so the file works as a normal font *and* in LaTeX. (Needs internet
  the first time to load opentype.js.)
- **⬇ Sheet .png + manifest.json** — a structured capture sheet matching the
  HandTex template geometry, so you can also build the font with the Python
  pipeline (works fully offline):

  ```bash
  handtex build-font --filled handwriting_sheet.png --manifest manifest.json \
         --out MyHand-Regular.ttf
  ```

Each pen stroke is turned into a filled outline (non-zero winding of overlapping
quads + round caps), so closed shapes like `o`, `a`, `e` get their holes
naturally.
