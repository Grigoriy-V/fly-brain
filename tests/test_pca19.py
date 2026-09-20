"""Offline checks of the linear first stage (19.0); no data, no GPU, no Modal.

The properties tested are the ones the step leans on: the basis is orthonormal,
whitening makes the *training* coordinates unit-variance exactly, a planted
low-rank signal is recovered at its own rank, and the saved/loaded pair is the
same map (including truncation to a smaller k, which is how 1,024 and 512 get
checked without refitting).
"""
import numpy as np
import torch

from flydream.generate import pca19 as P


def planted(n=60, d=40, r=5, noise=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, r, generator=g)
    B = torch.randn(r, d, generator=g)
    X = A @ B + 3.0                                                   # a mean offset the fit must remove
    if noise:
        X = X + noise * torch.randn(n, d, generator=g)
    return X


def test_basis_is_orthonormal_and_whitening_is_exact_on_the_fit_set():
    X = planted()
    p = P.fit(X, 5, block=7, log=lambda *_: None)
    gram = p["basis"].T @ p["basis"]
    assert torch.allclose(gram, torch.eye(5), atol=1e-4)
    z = P.encode(X, p, block=7)
    assert torch.allclose(z.std(0, correction=1), torch.ones(5), atol=1e-4)
    assert abs(float(z.mean())) < 1e-4


def test_a_planted_rank_is_recovered_at_its_own_rank():
    X = planted(r=5)
    p = P.fit(X, 5, block=7, log=lambda *_: None)
    back = P.decode(P.encode(X, p, block=7), p, block=7)
    assert float((back - X).pow(2).mean()) < 1e-6                     # rank 5 data, 5 components: exact
    e = P.explained(X, p, (1, 3, 5), block=7)
    assert e[5]["explained"] > 0.999
    assert e[1]["explained"] < e[3]["explained"] < e[5]["explained"]  # monotone in k
    assert e[5]["mse"] < e[3]["mse"] < e[1]["mse"]


def test_fewer_components_than_the_rank_lose_exactly_the_tail():
    X = planted(r=5)
    p = P.fit(X, 5, block=7, log=lambda *_: None)
    lam = p["lam"]
    e = P.explained(X, p, (2,), block=7)
    kept = float(lam[:2].sum() / lam.sum())                           # the eigenvalues say what k=2 keeps
    assert abs(e[2]["explained"] - kept) < 1e-4


def test_held_out_coordinates_are_measured_not_assumed():
    """The fit set is unit-variance by construction; a held-out set is not, and
    that gap is the number 19.0 exists to report."""
    X, Y = planted(seed=0, noise=0.5), planted(seed=1, noise=0.5)
    p = P.fit(X, 5, block=7, log=lambda *_: None)
    g_fit = P.geometry(P.encode(X, p, block=7))
    assert abs(g_fit["sd_axis_mean"] - 1.0) < 1e-3
    assert abs(g_fit["radius_mean"] - g_fit["typical_radius"]) < 0.25
    g_out = P.geometry(P.encode(Y, p, block=7))
    assert g_out["dims"] == 5 and g_out["n"] == len(Y)
    assert len(g_out["profile"]) >= 1                                 # the tail profile is reported, whatever it says


def test_save_and_load_round_trip_including_truncation():
    X = planted(r=5)
    p = P.fit(X, 5, block=7, log=lambda *_: None)
    z = {k: v for k, v in P.to_numpy(p, basis_dtype=np.float32).items()}
    q = P.from_numpy(z, torch.device("cpu"))
    assert torch.allclose(P.encode(X, p, block=7), P.encode(X, q, block=7), atol=1e-4)
    cut = P.from_numpy(z, torch.device("cpu"), k=2)
    assert cut["k"] == 2
    assert torch.allclose(P.encode(X, cut, block=7), P.encode(X, p, block=7)[:, :2], atol=1e-4)


def test_half_precision_input_is_accepted():
    """The maps on the volume are float16; the fit must never cast the whole
    set to float32 at once."""
    X = planted(r=5)
    p = P.fit(X.half(), 5, block=7, log=lambda *_: None)
    assert p["basis"].dtype == torch.float32
    back = P.decode(P.encode(X.half(), p, block=7), p, block=7)
    assert float((back - X).pow(2).mean()) < 1e-2
