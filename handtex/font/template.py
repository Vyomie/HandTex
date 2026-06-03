"""Generate a printable sheet to capture your handwriting.

Each glyph gets a cell with a baseline guide and a faint reference character
plus a label. You print it, write your version of each glyph sitting on the
baseline, photograph/scan it, and feed it to :mod:`handtex.font.extract`.

The grid geometry is deterministic and saved to a manifest JSON so extraction
needs no image-analysis guesswork to find the cells.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import List, Optional

from PIL import Image, ImageDraw

from . import symbols
from ._fonts import load_reference_font


@dataclass
class Geometry:
    cell: int = 256
    cols: int = 8
    margin: int = 40
    baseline_frac: float = 0.78  # baseline row within a cell, from the top
    rows: int = 0                # filled in when the template is generated

    @property
    def baseline_row(self) -> float:
        return self.cell * self.baseline_frac

    def cell_origin(self, index: int) -> tuple[int, int]:
        r, c = divmod(index, self.cols)
        return self.margin + c * self.cell, self.margin + r * self.cell


@dataclass
class Manifest:
    geometry: Geometry
    names: List[str]

    def to_json(self) -> str:
        return json.dumps({"geometry": asdict(self.geometry), "names": self.names}, indent=2)

    @staticmethod
    def from_json(text: str) -> "Manifest":
        d = json.loads(text)
        return Manifest(geometry=Geometry(**d["geometry"]), names=list(d["names"]))


def generate_template(
    out_image: str,
    out_manifest: str,
    names: Optional[List[str]] = None,
    geometry: Optional[Geometry] = None,
) -> Manifest:
    """Render the capture sheet and write its manifest. Returns the manifest."""
    if names is None:
        names = [s.name for s in symbols.all_symbols() if s.name != "space"]
    geo = geometry or Geometry()
    geo.rows = (len(names) + geo.cols - 1) // geo.cols

    W = geo.margin * 2 + geo.cols * geo.cell
    H = geo.margin * 2 + geo.rows * geo.cell
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    ref_font = load_reference_font(int(geo.cell * 0.6))
    label_font = load_reference_font(int(geo.cell * 0.12))
    guide = (210, 210, 210)
    faint = (232, 232, 232)

    for i, name in enumerate(names):
        ox, oy = geo.cell_origin(i)
        # cell border
        draw.rectangle([ox, oy, ox + geo.cell, oy + geo.cell], outline=guide, width=1)
        # baseline guide
        by = oy + geo.baseline_row
        draw.line([ox + 6, by, ox + geo.cell - 6, by], fill=guide, width=1)
        # faint reference glyph, sitting on the baseline
        sym = symbols.by_name(name)
        if sym and ref_font is not None:
            _, top, _, bottom = draw.textbbox((0, 0), sym.char, font=ref_font)
            gx = ox + geo.cell * 0.18
            gy = by - (bottom - top) - top
            draw.text((gx, gy), sym.char, fill=faint, font=ref_font)
        # label
        if label_font is not None:
            draw.text((ox + 6, oy + 4), name, fill=guide, font=label_font)

    img.save(out_image)
    manifest = Manifest(geometry=geo, names=names)
    with open(out_manifest, "w", encoding="utf-8") as fh:
        fh.write(manifest.to_json())
    return manifest
