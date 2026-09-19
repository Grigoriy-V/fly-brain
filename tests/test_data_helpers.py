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


def test_video_hex_eye_chain_and_selection_scores():
    """The eye chain on a synthetic clip: shapes, the scores that select a
    window, and that a hex rotation moves the lattice without changing the
    clip's statistics. No file, no download."""
    from flydream.data import video_hex as V

    t, h, w = 12, 200, 500
    y, x = np.mgrid[0:h, 0:w]
    frames = np.stack([0.5 + 0.4 * np.sin(2 * np.pi * (x - 12 * k) / 60.0) for k in range(t)]).astype(np.float32)
    assert frames.shape == (t, h, w)

    rgb = np.stack([np.full((4, 4), 255.0), np.zeros((4, 4)), np.zeros((4, 4))], -1)
    assert np.allclose(V.to_luminance(rgb), 0.299, atol=2e-3)          # PIL's "L" weights, not a channel mean

    hexals = V.to_hexals(frames, 24.0, 0.02, splits=2)
    assert hexals.shape[0] == 2 and hexals.shape[2] == 721             # two views, the fly's 721 hexals
    assert hexals.shape[1] == int(np.ceil(50 / 24 * t))                # resampled to 1/dt
    assert np.isfinite(hexals).all() and hexals.min() >= 0

    one = hexals[0]
    assert V.motion(one) > 0.002 and V.contrast(one) > 0.05            # a moving grating moves and has contrast
    assert V.cut_score(one) < 10                                       # ... and contains no cut
    with_cut = one.copy()
    with_cut[len(one) // 2:] = 1.0 - with_cut[len(one) // 2:]          # a hard cut in the middle
    assert V.cut_score(with_cut) > V.cut_score(one) * 3

    rot = V.augment(one, n_rot=2)
    assert rot.shape == one.shape and abs(float(rot.std()) - float(one.std())) < 1e-5
    assert not np.allclose(rot, one)                                   # the lattice really moved
    assert np.allclose(V.augment(one, n_rot=0), one)

    ws = V.windows(one, 5, 5)
    assert len(ws) == len(one) // 5 and ws[0].shape == (5, 721)
