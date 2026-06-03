"""HandTex command-line interface.

    handtex template  --out sheet.png --manifest sheet.json
    handtex build-font --filled sheet.png --manifest sheet.json --out hand.ttf
    handtex build-font --synth --out hand.ttf          # offline, no scans
    handtex read IMAGE --backend mock --tex out.tex --png out.png --font hand.ttf
    handtex demo --out out/                              # full synthetic run
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .font import symbols
from .font.build import FontMeta
from .font.extract import extract_cells
from .font.pipeline import build_font_from_cells, synth_cells
from .font.template import Manifest, generate_template
from .layout import layout_glyphs
from .latex import document_to_latex, standalone_document
from .ocr import get_backend
from .render import render_document


def _cmd_template(args: argparse.Namespace) -> int:
    names = None
    if args.set == "ascii":
        names = [s.name for s in symbols.all_symbols()
                 if s.char.isascii() and s.name != "space"]
    generate_template(args.out, args.manifest, names=names)
    print(f"wrote template image -> {args.out}")
    print(f"wrote manifest       -> {args.manifest}")
    return 0


def _cmd_build_font(args: argparse.Namespace) -> int:
    meta = FontMeta(family=args.family)
    if args.synth:
        names = [s.name for s in symbols.all_symbols() if s.name != "space"]
        cells = synth_cells(names)
    else:
        if not (args.filled and args.manifest):
            print("error: provide --filled and --manifest (or use --synth)", file=sys.stderr)
            return 2
        with open(args.manifest, encoding="utf-8") as fh:
            manifest = Manifest.from_json(fh.read())
        cells = extract_cells(args.filled, manifest)
    build_font_from_cells(cells, args.out, meta=meta)
    print(f"built font ({len(cells)} glyphs) -> {args.out}")
    return 0


def _cmd_preprocess(args: argparse.Namespace) -> int:
    from PIL import Image
    from .preprocess import preprocess

    cleaned = preprocess(args.image)
    Image.fromarray(cleaned).save(args.out)
    print(f"cleaned image ({cleaned.shape[1]}x{cleaned.shape[0]}) -> {args.out}")
    return 0


def _cmd_font_from_image(args: argparse.Namespace) -> int:
    from .font.from_image import font_from_image

    meta = FontMeta(family=args.family)
    _, cells = font_from_image(args.image, args.out, meta=meta,
                               top_crop_frac=args.top_crop, min_confidence=args.min_conf)
    uniq = len({c.name for c in cells})
    print(f"recognized {len(cells)} glyphs ({uniq} unique) -> built {args.out}")
    if args.annotate:
        from .visualize import annotate_recognition
        from .preprocess import preprocess
        annotate_recognition(preprocess(args.image), cells, args.annotate)
        print(f"wrote confidence-colored annotation -> {args.annotate}")
    if args.show:
        for c in sorted(cells, key=lambda c: (c.segment.bbox.cy, c.segment.bbox.left)):
            print(f"  {c.char!r:6} conf={c.confidence:.2f}")
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    backend = get_backend(args.backend)
    glyphs = backend.recognize(args.image)
    doc = layout_glyphs(glyphs)
    tex = document_to_latex(doc)
    print(tex)

    if args.tex:
        content = standalone_document(doc, font_path=args.font) if args.standalone else tex
        with open(args.tex, "w", encoding="utf-8") as fh:
            fh.write(content + "\n")
        print(f"wrote LaTeX -> {args.tex}", file=sys.stderr)
    if args.png:
        if not args.font:
            print("error: --png requires --font (a HandTex .ttf)", file=sys.stderr)
            return 2
        render_document(doc, args.font, args.png)
        print(f"wrote render -> {args.png}", file=sys.stderr)
    return 0


def _cmd_train_ocr(args: argparse.Namespace) -> int:
    from .ocr.local.train import train

    acc = train(epochs=args.epochs, per_class_per_font=args.per_class_per_font,
                max_fonts=args.max_fonts, math_oversample=args.math_oversample)
    print(f"final val_acc={acc:.3f}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .ocr.mock import MockBackend

    os.makedirs(args.out, exist_ok=True)
    font_path = os.path.join(args.out, "HandTex-Regular.ttf")
    tex_path = os.path.join(args.out, "whiteboard.tex")
    png_path = os.path.join(args.out, "whiteboard.png")

    # 1. Build a font from synthesized stand-in handwriting.
    names = [s.name for s in symbols.all_symbols() if s.name != "space"]
    cells = synth_cells(names)
    build_font_from_cells(cells, font_path)
    print(f"[1/4] built font ({len(cells)} glyphs) -> {font_path}")

    # 2. "Read" a whiteboard (built-in demo scene): x^2 + alpha = y_1
    glyphs = MockBackend().recognize("<demo>")
    print(f"[2/4] recognized {len(glyphs)} glyphs")

    # 3. Lay out -> LaTeX.
    doc = layout_glyphs(glyphs)
    tex = standalone_document(doc, font_path=os.path.basename(font_path))
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    print(f"[3/4] layout -> LaTeX: {document_to_latex(doc)!r} -> {tex_path}")

    # 4. Render back in the handwritten font.
    render_document(doc, font_path, png_path)
    print(f"[4/4] rendered -> {png_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="handtex", description="Handwriting -> font + whiteboard OCR -> LaTeX")
    p.add_argument("--version", action="version", version=f"handtex {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("template", help="generate a handwriting capture sheet")
    t.add_argument("--out", default="handtex_sheet.png")
    t.add_argument("--manifest", default="handtex_sheet.json")
    t.add_argument("--set", choices=["all", "ascii"], default="all")
    t.set_defaults(func=_cmd_template)

    b = sub.add_parser("build-font", help="build a .ttf from a filled sheet (or --synth)")
    b.add_argument("--filled", help="photo/scan of the filled capture sheet")
    b.add_argument("--manifest", help="manifest written by `template`")
    b.add_argument("--synth", action="store_true", help="synthesize stand-in glyphs (offline)")
    b.add_argument("--family", default="HandTex")
    b.add_argument("--out", default="HandTex-Regular.ttf")
    b.set_defaults(func=_cmd_build_font)

    pp = sub.add_parser("preprocess", help="clean an image (resize, deskew, flatten light, contrast)")
    pp.add_argument("image")
    pp.add_argument("--out", default="cleaned.png")
    pp.set_defaults(func=_cmd_preprocess)

    fi = sub.add_parser("font-from-image", help="OCR a handwriting photo and build a font (end-to-end)")
    fi.add_argument("image")
    fi.add_argument("--out", default="HandFromImage-Regular.ttf")
    fi.add_argument("--family", default="HandTex")
    fi.add_argument("--top-crop", type=float, default=0.0, help="ignore this top fraction (e.g. a title bar)")
    fi.add_argument("--min-conf", type=float, default=0.0, help="drop glyphs below this confidence")
    fi.add_argument("--show", action="store_true", help="print recognized glyphs")
    fi.add_argument("--annotate", help="write a confidence-colored annotated image to this path")
    fi.set_defaults(func=_cmd_font_from_image)

    r = sub.add_parser("read", help="read a whiteboard image -> LaTeX / render")
    r.add_argument("image")
    r.add_argument("--backend", default="mock", choices=["mock", "local", "vision"])
    r.add_argument("--tex", help="write LaTeX to this path")
    r.add_argument("--standalone", action="store_true", help="emit a full compilable .tex")
    r.add_argument("--png", help="render result to this PNG (needs --font)")
    r.add_argument("--font", help="a HandTex .ttf to render with")
    r.set_defaults(func=_cmd_read)

    tr = sub.add_parser("train-ocr", help="train the small local OCR model on synthetic data")
    tr.add_argument("--epochs", type=int, default=12)
    tr.add_argument("--per-class-per-font", type=int, default=4)
    tr.add_argument("--max-fonts", type=int, default=40)
    tr.add_argument("--math-oversample", type=int, default=4,
                    help="extra sample multiplier for Greek/math symbol classes")
    tr.set_defaults(func=_cmd_train_ocr)

    d = sub.add_parser("demo", help="run the full synthetic pipeline end-to-end")
    d.add_argument("--out", default="out")
    d.set_defaults(func=_cmd_demo)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
