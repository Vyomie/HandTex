from handtex.types import BBox
from handtex.ocr.local.refine import Candidate, refine


def C(i, x, y, w, h, options):
    return Candidate(index=i, bbox=BBox(x, y, w, h), options=options)


def test_doodle_is_removed():
    cands = [
        C(0, 0, 0, 40, 40, [("x", 0.9)]),
        C(1, 50, 0, 40, 40, [("doodle", 0.8), ("y", 0.1)]),
    ]
    out = refine(cands, unique_labels=False)
    labels = {r.label for r in out}
    assert labels == {"x"}


def test_duplicate_demoted_to_second_best():
    # both call "x"; higher-conf keeps it, the other falls back to its 2nd guess
    cands = [
        C(0, 0, 0, 40, 40, [("x", 0.9), ("plus", 0.05)]),
        C(1, 50, 0, 40, 40, [("x", 0.6), ("plus", 0.3)]),
    ]
    out = sorted(refine(cands, unique_labels=True), key=lambda r: r.index)
    assert out[0].label == "x"
    assert out[1].label == "plus"


def test_duplicate_removed_when_no_good_alternative():
    # the loser's only fallback is doodle -> assumed bad, removed
    cands = [
        C(0, 0, 0, 40, 40, [("x", 0.9)]),
        C(1, 50, 0, 40, 40, [("x", 0.55), ("doodle", 0.4)]),
    ]
    out = refine(cands, unique_labels=True)
    assert [r.label for r in out] == ["x"]


def test_repeats_allowed_when_not_unique():
    cands = [
        C(0, 0, 0, 40, 40, [("x", 0.9)]),
        C(1, 80, 0, 40, 40, [("x", 0.85)]),
    ]
    out = refine(cands, unique_labels=False)
    assert sorted(r.label for r in out) == ["x", "x"]


def test_tiny_lowconf_box_dropped_but_confident_small_kept():
    # median area is ~1600; a tiny uncertain speck is dropped, a tiny
    # *confident* mark (a period) is kept.
    cands = [
        C(0, 0, 0, 40, 40, [("a", 0.9)]),
        C(1, 50, 0, 40, 40, [("b", 0.9)]),
        C(2, 100, 0, 6, 6, [("comma", 0.2), ("period", 0.1)]),   # tiny + unsure -> drop
        C(3, 150, 30, 6, 6, [("period", 0.95)]),                 # tiny + sure -> keep
    ]
    out = {r.label for r in refine(cands, unique_labels=False)}
    assert "period" in out
    assert "comma" not in out


def test_overlapping_boxes_keep_higher_confidence():
    cands = [
        C(0, 0, 0, 40, 40, [("x", 0.6)]),
        C(1, 5, 5, 40, 40, [("y", 0.95)]),  # overlaps #0 heavily
    ]
    out = refine(cands, unique_labels=False)
    assert [r.label for r in out] == ["y"]
