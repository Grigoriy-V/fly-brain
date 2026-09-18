"""Offline checks of the pure helpers in flydream.data; no data, no network."""
import numpy as np

from flydream.data.columns_roi import parse
from flydream.data.export import hex_symmetries
from flydream.data.optic_lobe import weighted_median


def test_hex_symmetries_are_twelve_distinct_bijections():
    syms = hex_symmetries()
    assert len(syms) == 12
    pts = [(q, r) for q in range(-2, 3) for r in range(-2, 3)]
    images = set()
    for _, f in syms:
        img = tuple(f(q, r) for q, r in pts)
        assert len(set(img)) == len(pts)          # injective on the patch
        images.add(img)
    assert len(images) == 12                      # all distinct
    rot1 = dict(syms)["rot1"]
    p = (1, 0)
    for _ in range(6):
        p = rot1(*p)
    assert p == (1, 0)                            # six rotations return home


def test_weighted_median_picks_the_heavy_side():
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 10.0])) == 3.0
    assert weighted_median(np.array([5.0, 1.0]), np.array([1.0, 1.0])) in (1.0, 5.0)


def test_parse_column_rois_takes_the_heaviest_column_and_keeps_per_neuropil_maxima():
    info = {
        "ME_R_col_15_08": {"pre": 8, "post": 61},
        "ME_R_col_14_08": {"pre": 4, "post": 45},
        "LOP_R_col_14_07": {"pre": 23, "post": 16},
        "ME_R_layer_01": {"pre": 1, "post": 1},
        "ME(R)": {"pre": 20, "post": 200},
    }
    row = parse(65399, info)
    assert (row["np"], row["side"], row["hex1"], row["hex2"], row["syn"]) == ("ME", "R", 15, 8, 69)
    assert (row["me_hex1"], row["me_hex2"], row["me_syn"]) == (15, 8, 69)
    assert (row["lop_hex1"], row["lop_hex2"], row["lop_syn"]) == (14, 7, 39)
    assert row["lo_syn"] == 0 and row["lo_hex1"] is None
    assert parse(1, {"ME(R)": {"pre": 1, "post": 1}}) is None
