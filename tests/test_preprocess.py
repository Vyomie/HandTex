import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from handtex.font._fonts import find_reference_font_path
from handtex.preprocess import estimate_skew, preprocess
from handtex.ocr.local.segment import segment_image


def _font(size):
    return ImageFont.truetype(find_reference_font_path(), size)


def test_resize_brings_small_image_up():
    tiny = np.full((50, 80), 255, dtype=np.uint8)
    out = preprocess(tiny)
    assert out.shape[0] >= 900


def test_estimate_skew_recovers_angle(tmp_path):
    # horizontal text rotated by a known angle -> skew estimate ~ -angle
    img = Image.new("L", (800, 240), 255)
    d = ImageDraw.Draw(img)
    d.text((40, 90), "A B C D E F G", font=_font(90), fill=0)
    rotated = img.rotate(6, expand=True, fillcolor=255)
    angle = estimate_skew(np.asarray(rotated))
    assert abs(angle - (-6)) < 1.5


def test_uneven_illumination_lets_all_glyphs_be_found(tmp_path):
    # White page with a strong shadow gradient toward the right.
    W, H = 800, 200
    grad = np.tile(np.linspace(255, 110, W).astype(np.uint8), (H, 1))
    img = Image.fromarray(grad)
    d = ImageDraw.Draw(img)
    for i, ch in enumerate("ABCD"):
        d.text((60 + i * 180, 60), ch, font=_font(90), fill=0)
    p = tmp_path / "shadow.png"
    img.save(p)

    raw = segment_image(str(p), preprocess=False)
    clean = segment_image(str(p), preprocess=True)
    # Preprocessing should recover all four letters; the raw shadowed image
    # loses at least one (flooded by the dark background under global Otsu).
    assert len(clean) == 4
    assert len(clean) > len(raw)


def test_preprocess_is_safe_on_clean_image(tmp_path):
    img = Image.new("L", (700, 200), 255)
    d = ImageDraw.Draw(img)
    d.text((40, 60), "X Y Z", font=_font(90), fill=0)
    p = tmp_path / "clean.png"
    img.save(p)
    assert len(segment_image(str(p), preprocess=True)) == 3
