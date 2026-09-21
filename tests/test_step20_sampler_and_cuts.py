"""Offline tests for item 20 (sampler fixes) and 21 (state cuts, hex geometry).

No download, no Modal, no credential: the flow is a two-line analytic field and
the PCA dictionary is built by hand, so every number here is checkable by eye.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from flydream.decode import hexraster as H
from flydream.generate import cut19 as C
from flydream.generate import fix19 as F
from flydream.generate import hexcov19 as X
from flydream.generate import pca19 as P
from flydream.generate import prior17 as R


class Constant(torch.nn.Module):
    """v(x, t) = c, so the exact solution of the flow is x + c·(t1 − t0)."""

    def __init__(self, c: float = 2.0):
        super().__init__()
        self.c = c
        self.seen: list[float] = []

    def forward(self, x, t, y=None):
        self.seen.append(float(t[0]))
        return torch.full_like(x, self.c)


# ------------------------------------------------------------------ the knobs

def test_shifted_keeps_the_interval_and_is_identity_at_one():
    assert R.shifted(0.0, 3.0) == 0.0 and R.shifted(1.0, 3.0) == pytest.approx(1.0)
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert R.shifted(u, 1.0) == pytest.approx(u)
    nodes = [R.shifted(i / 8, 3.0) for i in range(9)]
    assert all(b > a for a, b in zip(nodes, nodes[1:]))               # monotone
    assert R.shifted(0.5, 3.0) > 0.5                                 # s > 1 pulls nodes towards t = 1
    assert R.shifted(0.5, 0.5) < 0.5
    with pytest.raises(ValueError):
        R.shifted(0.5, 0.0)


def test_integrate_unchanged_at_shift_one_and_exact_for_a_constant_field():
    m = Constant(2.0)
    x = torch.zeros(3, 1, 1, 4)
    out = R.integrate(m, x, steps=20)
    assert out.allclose(x + 2.0, atol=1e-6)
    assert m.seen == pytest.approx([i / 20 for i in range(20)], abs=1e-6)   # the old grid, node for node
    m2 = Constant(2.0)
    out2 = R.integrate(m2, x, steps=20, shift=3.0)
    assert out2.allclose(x + 2.0, atol=1e-5)                          # the interval is the same
    assert m2.seen[1] > m.seen[1]                                     # but the nodes moved


def test_scaled_divides_the_field_and_is_the_model_itself_at_one():
    m = Constant(2.0)
    assert R.scaled(m, 1.0) is m
    x = torch.zeros(2, 1, 1, 3)
    assert R.integrate(R.scaled(m, 2.0), x, steps=10).allclose(x + 1.0, atol=1e-6)


def test_parse_spec_and_name():
    assert F.parse_spec("base") == {"steps": 0, "vscale": 1.0, "shift": 1.0, "sdproj": 0}
    assert F.parse_spec("sdproj")["sdproj"] == 1                      # a bare flag, as default_specs writes it
    assert all(F.parse_spec(s) for s in F.default_specs())            # every default parses
    assert F.parse_spec("vscale=1.1,steps=100")["vscale"] == pytest.approx(1.1)
    assert F.parse_spec("vscale=1.1,steps=100")["steps"] == 100
    assert F.name_of(F.parse_spec("base"), 20) == "базовый"
    assert F.name_of(F.parse_spec("vscale=1.1,shift=2"), 20) == "vscale=1.1+shift=2"
    assert F.name_of(F.parse_spec("steps=100"), 20) == "steps=100"
    with pytest.raises(ValueError):
        F.parse_spec("nonsense=1")
    with pytest.raises(ValueError):
        F.parse_spec("vscale")


def test_integrate_fixed_matches_the_plain_sampler_without_a_knob():
    m, x = Constant(1.5), torch.zeros(2, 1, 1, 8)
    base = R.integrate(m, x, steps=10)
    same = F.integrate_fixed(m, x, steps=10, spec=F.parse_spec("base"), sd_data=1.0, tokens=8)
    assert torch.equal(base, same)


def test_sdproj_lands_on_the_ideal_curve():
    """The projection's whole point: whatever the field does, the batch's sd at
    t = 1 is the data's. A constant field would otherwise leave it at 1.0."""
    m = Constant(3.0)
    x = torch.randn(64, 1, 1, 16, generator=torch.Generator().manual_seed(0))
    out = F.integrate_fixed(m, x, steps=20, spec=F.parse_spec("sdproj"), sd_data=0.5, tokens=16)
    assert float(out.std()) == pytest.approx(0.5, rel=1e-3)
    assert F.ideal_sd(0.0, 0.5) == pytest.approx(1.0)
    assert F.ideal_sd(1.0, 0.5) == pytest.approx(0.5)


def test_velocity_profile_reports_every_step():
    m = Constant(2.0)
    v = F.velocity_profile(m, torch.zeros(4, 1, 1, 6), steps=5, tokens=6)
    assert len(v["marks"]) == 5 and v["marks"][0]["t"] == 0.0
    assert v["marks"][0]["v_norm"] == pytest.approx(2.0 * np.sqrt(6), rel=1e-5)


# ------------------------------------------------------------------- the cuts

def test_parse_cut():
    assert C.parse_cut("types=T4+complete") == {"kind": "types", "value": "T4",
                                                "complete": True, "mask": False}
    assert C.parse_cut(" dct=8 ")["complete"] is False
    with pytest.raises(ValueError):
        C.parse_cut("pca=512+complete")
    with pytest.raises(ValueError):
        C.parse_cut("colour=red")
    with pytest.raises(ValueError):
        C.parse_cut("T4")


def test_parse_cut_mask_is_the_generators_own_input():
    """`+mask` says "this type is not given" through 13B's own eight mask bits,
    which it trained on; without it a zeroed half is passed off as present."""
    from flydream.generate.gen13b import named_mask

    c = C.parse_cut("types=T4+mask")
    assert c == {"kind": "types", "value": "T4", "complete": False, "mask": True}
    assert list(named_mask("t4")) == [1, 1, 1, 1, 0, 0, 0, 0]
    assert list(named_mask("t5")) == [0, 0, 0, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError):                                   # a completed type is present
        C.parse_cut("types=T4+complete+mask")
    with pytest.raises(ValueError):                                   # the mask is per type only
        C.parse_cut("dct=8+mask")


def test_mask_of_counts_what_it_keeps():
    shape = (16, 8, 721)
    D = 16 * 8 * 721
    t4 = C.mask_of(C.parse_cut("types=T4"), shape)
    assert t4.numel() == D and int(t4.sum()) == D // 2                # four of eight types
    d8 = C.mask_of(C.parse_cut("dct=8"), shape)
    assert int(d8.sum()) == D // 2
    d1 = C.mask_of(C.parse_cut("dct=1"), shape)
    assert int(d1.sum()) == 8 * 721
    r0 = C.mask_of(C.parse_cut("rings=0"), shape)
    assert int(r0.sum()) == 16 * 8 * int((H.ring_of(721) == 0).sum())
    full = C.mask_of(C.parse_cut("rings=99"), shape)
    assert int(full.sum()) == D
    with pytest.raises(ValueError):
        C.mask_of(C.parse_cut("dct=99"), shape)
    with pytest.raises(ValueError):
        C.mask_of(C.parse_cut("types=Lp"), shape)


def _dictionary(D: int, k: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    B = torch.linalg.qr(torch.randn(D, k, generator=g))[0]
    return {"basis": B, "lam": torch.linspace(4.0, 1.0, k), "mean": torch.randn(D, generator=g),
            "dims": D, "k": k}


def test_completion_is_exact_for_a_state_inside_the_subspace():
    """A state that lies in the PCA subspace is fully determined by k
    coordinates, so half the numbers are enough to recover the other half."""
    p = _dictionary(200, 8)
    g = torch.Generator().manual_seed(1)
    c = torch.randn(3, 8, generator=g)
    x = p["mean"] + c @ p["basis"].T
    mask = torch.zeros(200, dtype=torch.bool); mask[:100] = True
    hat, resid = C.complete_in_subspace(x, mask, p)
    assert resid < 1e-4
    assert torch.allclose(hat, x, atol=1e-3)
    assert not torch.allclose(hat[:, 100:], torch.zeros_like(hat[:, 100:]))


def test_completion_beats_zeroing_on_the_dropped_half():
    p = _dictionary(200, 8)
    g = torch.Generator().manual_seed(2)
    x = p["mean"] + torch.randn(4, 8, generator=g) @ p["basis"].T
    shape = (1, 2, 100)                                               # 1 x 2 x 100 = 200 numbers
    cut = {"kind": "types", "value": "T4", "complete": True}
    mask = torch.zeros(200, dtype=torch.bool); mask[:100] = True
    hat, _ = C.complete_in_subspace(x, mask, p)
    zero = torch.zeros_like(x); zero[:, mask] = x[:, mask]
    err_hat = float((hat[:, 100:] - x[:, 100:]).norm())
    err_zero = float((zero[:, 100:] - x[:, 100:]).norm())
    assert err_hat < err_zero
    assert cut["complete"] and shape[2] == 100                        # the spec this mirrors


def test_apply_cut_reports_how_many_numbers_are_given():
    p = _dictionary(2 * 3 * 4, 5)
    flat = torch.randn(2, 24, generator=torch.Generator().manual_seed(3))
    got = C.apply_cut(flat, C.parse_cut("pca=5"), p, (2, 3, 4))
    assert got["given"] == 5 and got["x"].shape == flat.shape and got["latent"]["dims"] == 5
    got = C.apply_cut(flat, C.parse_cut("dct=1"), p, (2, 3, 4))
    assert got["given"] == 12 and torch.equal(got["x"][:, 12:], torch.zeros(2, 12))
    assert torch.equal(got["x"][:, :12], flat[:, :12])


# ------------------------------------------------- hex geometry and covariance

def test_hex_distance_and_rings_agree_with_the_neighbour_table():
    d = H.hex_distance(721)
    nb = H.neighbour_index(721)
    assert d.shape == (721, 721) and (d == d.T).all() and (np.diag(d) == 0).all()
    for i in (0, 100, 360, 720):
        for j in nb[i][nb[i] >= 0]:
            assert d[i, j] == 1
    assert (d[0] == 1).sum() == (nb[0] >= 0).sum()                    # no extra neighbours
    ring = H.ring_of(721)
    assert ring.min() == 0 and int((ring == 0).sum()) == 1
    assert int((ring == 1).sum()) == 6 and int((ring == 2).sum()) == 12
    assert ring.max() == 15


def test_corr_of_and_by_distance_read_a_local_covariance():
    """A covariance built to decay with distance must come back decaying, and
    the shuffled control must come back flat — the control the run prints."""
    dist = H.hex_distance(721)
    S = torch.as_tensor(np.exp(-dist / 2.0), dtype=torch.float32)
    Cm = X.corr_of(S)
    assert torch.allclose(Cm.diagonal(), torch.ones(721), atol=1e-5)
    m, a = X.by_distance(Cm, dist, 8)
    assert m[0] == pytest.approx(1.0, abs=1e-5)
    assert m[1] > m[2] > m[4] > m[8]
    assert X.locality_length(a) == 3.0                                # exp(−d/2) halves by d = 3
    rng = np.random.default_rng(0)
    perm = rng.permutation(721)
    ms, _ = X.by_distance(Cm[np.ix_(perm, perm)], dist, 8)
    assert abs(ms[4] - ms[8]) < 0.05                                  # flat once space is destroyed


def test_locality_length_is_nan_when_nothing_decays():
    assert np.isnan(X.locality_length(np.ones(10)))
    assert np.isnan(X.locality_length(np.array([1.0, 0.0])))


def test_geometry_of_a_whitened_gaussian_matches_its_own_expectations():
    """Guard for the target fix19 aims at: the geometry helper's expectations
    are the ones the sweep is scored against."""
    z = torch.randn(4096, 64, generator=torch.Generator().manual_seed(4))
    g = P.geometry(z)
    assert g["sd"] == pytest.approx(1.0, abs=0.02)
    assert g["radius_mean"] == pytest.approx(np.sqrt(64), abs=0.2)
    assert g["radius_sd"] == pytest.approx(np.sqrt(0.5), abs=0.1)
    assert g["kurtosis_mean"] == pytest.approx(3.0, abs=0.2)
