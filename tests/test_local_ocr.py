import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from handtex.ocr.local.normalize import OUT_SIZE, normalize_glyph
from handtex.ocr.local.segment import segment_image
from handtex.font._fonts import find_reference_font_path


def _render_glyphs(path, glyphs, size=(600, 200)):
    fp = find_reference_font_path()
    font = ImageFont.truetype(fp, 90)
    img = Image.new("L", size, 255)
    d = ImageDraw.Draw(img)
    for ch, (x, y) in glyphs:
        d.text((x, y), ch, font=font, fill=0)
    img.save(path)


def test_normalize_centers_and_shapes():
    ink = np.zeros((40, 40), dtype=np.float32)
    ink[5:15, 5:10] = 1.0  # off-center blob
    out = normalize_glyph(ink)
    assert out.shape == (OUT_SIZE, OUT_SIZE)
    assert out.max() <= 1.0 and out.min() >= 0.0
    # center of mass should be near the middle after normalization
    ys, xs = np.where(out > 0.2)
    assert abs(ys.mean() - OUT_SIZE / 2) < 4
    assert abs(xs.mean() - OUT_SIZE / 2) < 4


def test_segmentation_separates_glyphs(tmp_path):
    img = tmp_path / "g.png"
    _render_glyphs(str(img), [("A", (30, 60)), ("B", (200, 60)), ("C", (380, 60))])
    segs = segment_image(str(img))
    assert len(segs) == 3
    # left-to-right ordering
    xs = [s.bbox.left for s in segs]
    assert xs == sorted(xs)


def test_segmentation_merges_equals_sign(tmp_path):
    img = tmp_path / "eq.png"
    _render_glyphs(str(img), [("=", (250, 60))])
    segs = segment_image(str(img))
    # the two bars of '=' must merge into a single glyph
    assert len(segs) == 1


def test_font_from_image_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from handtex.ocr.local.model import has_weights
    if not has_weights():
        pytest.skip("no trained weights committed")
    from handtex.font.from_image import font_from_image
    from fontTools.ttLib import TTFont

    img = tmp_path / "abc.png"
    # printed glyphs the model was trained on -> should classify reliably
    _render_glyphs(str(img), [("A", (30, 60)), ("E", (200, 60)), ("H", (380, 60))])
    out = tmp_path / "hand.ttf"
    _, cells = font_from_image(str(img), str(out))
    assert len(cells) == 3
    font = TTFont(str(out))
    assert len(font.getBestCmap()) >= 1
