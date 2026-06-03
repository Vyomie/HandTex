from handtex.types import BBox, RecognizedGlyph
from handtex.visualize import annotate_recognition, confidence_color


def test_confidence_color_endpoints_and_monotonic():
    low = confidence_color(0.0)
    high = confidence_color(1.0)
    # low is reddish (R dominant), high is greenish (G dominant)
    assert low[0] > low[1]
    assert high[1] > high[0]
    # "greenness" (green minus red) rises monotonically with confidence
    greenness = [confidence_color(c)[1] - confidence_color(c)[0]
                 for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert greenness == sorted(greenness)


def test_annotate_writes_image(tmp_path):
    import numpy as np
    img = np.full((120, 200), 255, dtype=np.uint8)
    glyphs = [
        RecognizedGlyph("x", "x", BBox(10, 10, 30, 40), confidence=0.95),
        RecognizedGlyph("y", "y", BBox(60, 10, 30, 40), confidence=0.30),
    ]
    out = tmp_path / "anno.png"
    annotate_recognition(img, glyphs, str(out))
    assert out.exists()
