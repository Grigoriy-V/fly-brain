"""Offline: the decoder's arithmetic, on synthetic data.

No download, no credential, no Modal. The hexagonal lattice comes from flyvis's
`hex_utils`, which is pure integer geometry, and everything else is made here.
"""
import numpy as np
import pytest

from flydream.decode import hexraster as H
from flydream.decode import metrics as M
from flydream.decode import pairs as P
from flydream.decode import ridge as R


# ------------------------------------------------------------------ metrics


def test_pixel_correlation_is_one_for_a_rescaled_copy():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(7, 40))
    assert M.pixel_correlation(3.0 * t + 5.0, t) == pytest.approx(1.0)
    assert M.pixel_correlation(-t, t) == pytest.approx(-1.0)


def test_pixel_correlation_of_independent_noise_is_near_zero():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(200, 60)), rng.normal(size=(200, 60))
    assert abs(M.pixel_correlation(a, b)) < 0.05


def test_r2_punishes_the_offset_that_correlation_forgives():
    """The reason both are reported: a decoder can have the shape and not the scale."""
    rng = np.random.default_rng(2)
    t = rng.normal(size=(20, 30))
    assert M.pixel_correlation(3.0 * t + 5.0, t) == pytest.approx(1.0)
    assert M.r2(3.0 * t + 5.0, t) < 0.0


def test_identification_is_at_chance_for_a_blind_prediction():
    rng = np.random.default_rng(3)
    t = rng.normal(size=(300, 50))
    blind = np.tile(t.mean(axis=0), (300, 1))       # the mean frame, the same every time
    acc, chance, used = M.identification(t, t, n_candidates=10)
    assert acc == pytest.approx(1.0)
    assert chance == pytest.approx(0.1)
    assert used == 300
    acc_blind, _, _ = M.identification(blind + 1e-9 * rng.normal(size=t.shape), t, n_candidates=10)
    assert acc_blind < 0.3


def test_identification_excludes_frames_with_no_spatial_structure():
    """A synthetic protocol pads its condition with a constant grey screen, and
    no decoder can say which grey frame it is looking at. Counting those frames
    as misses caps a perfect reconstruction far below 1 for a reason that has
    nothing to do with the decoder."""
    rng = np.random.default_rng(8)
    structured = rng.normal(size=(40, 30))
    flat = np.full((60, 30), 0.5)                    # 60% of the frames carry no pattern
    t = np.concatenate([structured, flat])
    acc, _, used = M.identification(t, t, n_candidates=10)
    assert used == 40
    assert acc == pytest.approx(1.0)                 # a perfect prediction now scores perfectly


def test_hex_ssim_saturates_when_the_window_is_smaller_than_a_hexal():
    """Why `ssim_win` is a setting with a floor: a window inside one flat Voronoi
    cell sees zero local variance and reports perfect similarity for anything."""
    rng = np.random.default_rng(4)
    a = rng.normal(size=(1, 721))
    b = rng.normal(size=(1, 721))
    honest = M.hex_ssim(a, b, pix_per_hex=3, win_size=9)
    too_small = M.hex_ssim(a, b, pix_per_hex=9, win_size=3)
    assert too_small > honest


# ------------------------------------------------------------------- ridge


def _linear_problem(n=400, p=40, q=15, noise=0.05, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p)) * scale
    w = rng.normal(size=(p, q))
    y = x @ w / scale + noise * rng.normal(size=(n, q))
    return x, y


def test_ridge_recovers_a_linear_map():
    x, y = _linear_problem()
    model = R.fit(x[:300], y[:300])
    assert M.r2(model.predict(x[300:]), y[300:]) > 0.98


def test_the_penalty_grid_makes_the_fit_scale_invariant():
    """The claim that lets the map read raw voltages instead of dividing by
    flyvis's per-cell-type normalisation constants.

    Ridge is scale-equivariant: multiply the features by c and the optimal
    penalty scales by c squared while the fit is unchanged. That only survives
    in practice if the penalty grid is relative to the data, which is why the
    grid here is alpha * mean(s^2) over the design matrix's own singular values.
    A cell type 1000 times louder than another must not therefore score
    differently.
    """
    scores = []
    for scale in (1e-3, 1.0, 1e3):
        x, y = _linear_problem(scale=scale, seed=7)
        model = R.fit(x[:300], y[:300])
        scores.append(M.r2(model.predict(x[300:]), y[300:]))
    assert max(scores) - min(scores) < 0.01, scores


def test_a_shared_absolute_penalty_would_have_ranked_by_loudness():
    """The control for the test above: with one fixed penalty for every cell
    type, the same data at two scales scores differently, which is the ranking
    artefact the relative grid removes."""
    fixed = (1.0,)
    scores = []
    for scale in (1e-3, 1e3):
        x, y = _linear_problem(scale=scale, seed=7)
        xc, yc = x[:300] - x[:300].mean(0), y[:300] - y[:300].mean(0)
        w = np.linalg.solve(xc.T @ xc + fixed[0] * np.eye(x.shape[1]), xc.T @ yc)
        pred = (x[300:] - x[:300].mean(0)) @ w + y[:300].mean(0)
        scores.append(M.r2(pred, y[300:]))
    assert abs(scores[0] - scores[1]) > 0.1, scores


def test_shuffle_time_keeps_every_frame_and_only_moves_it():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(3, 12, 4))
    b = R.shuffle_time(a, seed=1)
    assert a.shape == b.shape
    for i in range(a.shape[0]):
        assert np.array_equal(np.sort(a[i], axis=0), np.sort(b[i], axis=0))
    assert not np.array_equal(a, b)


def test_shuffle_samples_leaves_no_sample_with_its_own_activity():
    """Removes what the time shuffle leaves behind, the identity of the
    condition, while keeping the frame index aligned; a score here near the real
    one is the signature of a stimulus set that is a function of frame index."""
    rng = np.random.default_rng(9)
    a = rng.normal(size=(6, 5, 3))
    b = R.shuffle_samples(a, seed=2)
    assert a.shape == b.shape
    for i in range(a.shape[0]):
        assert not np.array_equal(a[i], b[i])
    # the set of samples is unchanged, only which stimulus each is paired with
    assert np.array_equal(np.sort(a.sum(axis=(1, 2))), np.sort(b.sum(axis=(1, 2))))


def test_subsets_are_within_range_and_sized_as_asked():
    got = R.subsets(50, [5, 20, 80], seed=0)
    assert set(got) == {5, 20, 50}
    for c, sel in got.items():
        assert len(set(sel.tolist())) == c and sel.max() < 50


# ------------------------------------------------------------------- pairs


def _pairs(s=4, t=10, h=6, cells=5):
    rng = np.random.default_rng(6)
    act = rng.normal(size=(s, t, 2 * cells)).astype(np.float32)
    return P.Pairs(stimulus=rng.normal(size=(s, t, h)).astype(np.float32),
                   activity=act, groups=np.array([f"g{i // 2}" for i in range(s)]),
                   cell_types=["A", "B"],
                   index={"A": np.arange(cells), "B": np.arange(cells, 2 * cells)})


def test_by_type_slices_the_cells_of_that_type():
    pr = _pairs()
    assert pr.by_type("A").shape == (4, 10, 5)
    assert np.array_equal(pr.by_type("B"), pr.activity[:, :, 5:])


def test_frames_aligns_the_lag_and_drops_the_frames_it_cannot_fill():
    pr = _pairs()
    x0, y0, g0 = pr.frames("A", np.array([0, 1]), lags=(0,))
    assert x0.shape == (20, 5) and y0.shape == (20, 6) and len(g0) == 20
    x2, y2, _ = pr.frames("A", np.array([0, 1]), lags=(0, 2))
    assert x2.shape == (16, 10) and y2.shape == (16, 6)     # two frames lost per sample
    # the second block is the activity two frames later than the first
    assert np.allclose(x2[0, 5:], pr.by_type("A")[0, 2])
    assert np.allclose(x2[0, :5], pr.by_type("A")[0, 0])


def test_frames_refuses_a_lag_wider_than_the_sample():
    with pytest.raises(ValueError):
        _pairs(t=3).frames("A", np.array([0]), lags=(0, 5))


def test_split_by_group_never_puts_a_group_on_both_sides():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    train, test = P.split_by_group(groups, test_fraction=0.25, seed=0)
    assert set(groups[train]).isdisjoint(set(groups[test]))
    assert len(train) + len(test) == len(groups)


def test_scene_labels_fall_back_to_one_group_per_sample_without_a_name_column():
    class Bare:
        arg_df = None

    got = P.scene_labels(Bare(), np.array([3, 5]))
    assert list(got) == ["sample_3", "sample_5"]


def test_drop_nan_rows_removes_the_padding_of_unequal_length_samples():
    from flydream.decode.map import drop_nan_rows

    x = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]])
    y = np.array([[1.0], [1.0], [np.nan]])
    xk, yk, gk, dropped = drop_nan_rows(x, y, np.array(["a", "b", "c"]))
    assert dropped == 2 and xk.shape == (1, 2) and list(gk) == ["a"]


def test_pairs_survive_a_round_trip_through_disk(tmp_path):
    pr = _pairs()
    back = P.Pairs.load(pr.save(tmp_path / "pairs.npz"))
    assert np.array_equal(back.stimulus, pr.stimulus)
    assert np.array_equal(back.activity, pr.activity)
    assert list(back.groups) == list(pr.groups)
    assert back.cell_types == pr.cell_types
    assert {k: v.tolist() for k, v in back.index.items()} == {k: v.tolist() for k, v in pr.index.items()}


# --------------------------------------------------------------- hexraster


def test_the_raster_covers_the_lattice_and_nothing_beyond_it():
    index, mask, shape = H.raster_map(721, pix_per_hex=3)
    assert index.shape == mask.shape == shape
    assert 0.5 < mask.mean() < 1.0          # a hexagon inside its bounding box
    assert index[mask].min() >= 0 and index[mask].max() == 720


def test_a_constant_lattice_rasterises_to_a_constant_image():
    img = H.to_raster(np.full(721, 0.4), 721, 3, fill=np.nan)
    assert np.nanmin(img) == pytest.approx(0.4) and np.nanmax(img) == pytest.approx(0.4)


def test_neighbours_are_mutual_and_opposite():
    """E/W, NE/SW and NW/SE must be inverses, or a smoothness penalty built on
    this index would pull each hexal toward the wrong side of itself."""
    nb = H.neighbour_index(721)
    assert nb.shape == (721, 6)
    for d in range(6):
        back = (d + 3) % 6
        for i in range(721):
            j = nb[i, d]
            if j >= 0:
                assert nb[j, back] == i


def test_the_axial_map_is_the_shape_the_reference_decoder_stores():
    """flyvis's own hexal storage is 31x31 at extent 15 (task/decoder.py:304)."""
    out = H.to_axial(np.arange(721, dtype=np.float32))
    assert out.shape == (31, 31)
    assert np.isfinite(out).sum() == 721
