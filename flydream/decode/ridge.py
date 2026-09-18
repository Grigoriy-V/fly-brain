"""Rung one of the decoder ladder: ridge regression from activity to stimulus.

Linear first, because at single-neuron resolution linear decoders come within
about 0.01 of nonlinear ones in retina and mouse V1, so a nonlinear decoder that
beats ridge by less than that has told us nothing about the code.

Two choices here carry the weight of the whole per-cell-type comparison.

**The penalty is tuned per cell type, on the training split only.** Ridge is
scale-equivariant: multiply a cell type's activity by c and the optimal penalty
scales by c^2, while the fit is unchanged. FlyVis's per-cell-type response
amplitudes span about 76x within one model and thousands of times across the
ensemble, so a single penalty shared across cell types would not compare cell
types, it would rank them by loudness. Tuning per type removes that, and with
it the need to divide activity by a precomputed normalisation constant, which
would have equalised exactly the amplitude the map is trying to measure.

**The penalty grid is relative, not absolute**, for the same reason: the grid is
`alpha * mean(s^2)` over the singular values of that cell type's own design
matrix, so the same grid means the same thing for a loud type and a quiet one.

The fit itself is one SVD per cell type, reused across the whole penalty grid,
which is why a 65-type map with cross-validation is cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The grid must be wide enough that no cell type's search ends at its edge: a
# type whose best penalty is the largest on offer has not been fitted, it has
# been truncated. Measured on the first Sintel map with a -6..3 grid, exactly
# two types (Mi2, Tm20) chose the top value, and they were exactly the two whose
# score fell below their own time-shuffle control with an R2 of -0.44; two
# others (R3, R8) chose the bottom. Widened to -9..6, and `fit` now reports when
# a search still ends at an edge.
ALPHAS = tuple(float(10.0 ** e) for e in np.arange(-9.0, 6.01, 0.5))


@dataclass
class Ridge:
    """A fitted linear map from activity to the hexal stimulus."""

    w: np.ndarray                  # (p, n_targets)
    x_mean: np.ndarray             # (p,)
    y_mean: np.ndarray             # (n_targets,)
    alpha: float                   # the chosen relative penalty
    lam: float                     # the absolute penalty it became
    n_train: int
    n_features: int
    val_score: float = float("nan")
    at_grid_edge: str = ""         # "low", "high" or "": the search was truncated
    alpha_scores: dict = field(default_factory=dict)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.x_mean) @ self.w + self.y_mean


def _solve_grid(x: np.ndarray, y: np.ndarray, alphas) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float]:
    """Ridge solutions for every alpha from one SVD. Returns (ws, x_mean, y_mean, scale)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - x_mean, y - y_mean
    # the SVD in float32 (about twice as fast; 2026-09-18 night), the rest in float64
    u, s, vt = np.linalg.svd(xc.astype(np.float32), full_matrices=False)
    u, s, vt = u.astype(np.float64), s.astype(np.float64), vt.astype(np.float64)
    scale = float(np.mean(s ** 2)) or 1.0
    uty = u.T @ yc
    ws = []
    for a in alphas:
        lam = a * scale
        d = s / (s ** 2 + lam)
        ws.append(vt.T @ (d[:, None] * uty))
    return ws, x_mean, y_mean, scale


def fit_gcv(x: np.ndarray, y: np.ndarray, *, alphas=ALPHAS, dtype=np.float32) -> Ridge:
    """Fit with the penalty chosen by generalised cross-validation, one SVD.

    GCV(lam) = n * ||y - H y||^2 / (n - tr H)^2 with H the ridge hat matrix,
    all of it from the singular values, so the whole grid and the final weights
    cost one SVD instead of the two of `fit` (choose on a held-out slice, refit
    on everything). In float32 the SVD is about twice as fast again; the
    weights are returned in float64 for the predictions. Golub, Heath & Wahba
    1979. Written 2026-09-18 night when the map's cost was 58% SVD.
    """
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    n = x64.shape[0]
    x_mean, y_mean = x64.mean(axis=0), y64.mean(axis=0)
    xc = (x64 - x_mean).astype(dtype, copy=False)
    yc = (y64 - y_mean)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    u, s, vt = u.astype(np.float64), s.astype(np.float64), vt.astype(np.float64)
    scale = float(np.mean(s ** 2)) or 1.0
    uty = u.T @ yc                                   # (r, n_targets)
    sst = float((yc ** 2).sum())
    yy = float((yc ** 2).sum())
    uty2 = (uty ** 2).sum(axis=1)                    # per component
    scores, best = {}, (np.inf, 0)
    for k, a in enumerate(alphas):
        lam = a * scale
        f = s ** 2 / (s ** 2 + lam)                  # shrinkage per component
        # ||y - Hy||^2 = ||y||^2 - 2 sum f_i |u_i^T y|^2 + sum f_i^2 |u_i^T y|^2
        rss = yy - 2.0 * float((f * uty2).sum()) + float((f ** 2 * uty2).sum())
        df = float(f.sum())
        gcv = n * max(rss, 0.0) / max(n - df, 1e-9) ** 2
        scores[float(a)] = -gcv
        if gcv < best[0]:
            best = (gcv, k)
    alpha = float(alphas[best[1]])
    lam = alpha * scale
    w = vt.T @ ((s / (s ** 2 + lam))[:, None] * uty)
    f = s ** 2 / (s ** 2 + lam)
    rss = yy - 2.0 * float((f * uty2).sum()) + float((f ** 2 * uty2).sum())
    edge = "low" if best[1] == 0 else ("high" if best[1] == len(alphas) - 1 else "")
    return Ridge(w=w, x_mean=x_mean, y_mean=y_mean, alpha=alpha, lam=lam, n_train=n,
                 n_features=x64.shape[1], val_score=float(1.0 - rss / sst) if sst else float("nan"),
                 at_grid_edge=edge, alpha_scores=scores)


def fit(x: np.ndarray, y: np.ndarray, *, alphas=ALPHAS, val_fraction: float = 0.2,
        seed: int = 0, groups: np.ndarray | None = None, method: str = "holdout") -> Ridge:
    """Fit with the penalty chosen on a held-out slice of the training data
    (default, two SVDs) or by GCV (`method="gcv"`, one SVD).

    GCV is NOT the default on purpose: frames of one clip are strongly
    autocorrelated, so leave-one-frame-out is over-optimistic and GCV
    under-penalises by two to three orders of magnitude (measured 2026-09-18
    night at the 80 ms window: L3 alpha 1e-3 against 0.3 by scene hold-out,
    test PixCorr 0.62 against 0.80; Mi4 0.56 against 0.87; T4a 0.40 against
    0.58). The scene-grouped hold-out is the right validation for this data.

    `groups` (one label per row, e.g. the Sintel scene) keeps every row of a
    group on the same side of the validation cut, so the penalty is not chosen
    on frames whose neighbours it was fitted on.
    """
    if method == "gcv":
        return fit_gcv(x, y, alphas=alphas)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.permutation(n)
        cut = max(1, int(round(val_fraction * n)))
        val_i, fit_i = idx[:cut], idx[cut:]
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        cut = max(1, int(round(val_fraction * len(uniq))))
        held = set(uniq[:cut].tolist())
        mask = np.array([g in held for g in groups])
        val_i, fit_i = np.flatnonzero(mask), np.flatnonzero(~mask)

    ws, xm, ym, scale = _solve_grid(x[fit_i], y[fit_i], alphas)
    xv, yv = x[val_i], y[val_i]
    scores = {}
    best = (-np.inf, 0)
    for k, (a, w) in enumerate(zip(alphas, ws)):
        pred = (xv - xm) @ w + ym
        sse = float(((pred - yv) ** 2).sum())
        sst = float(((yv - yv.mean()) ** 2).sum())
        sc = -np.inf if sst == 0 else 1.0 - sse / sst
        scores[float(a)] = sc
        if sc > best[0]:
            best = (sc, k)

    alpha = float(alphas[best[1]])
    edge = "low" if best[1] == 0 else ("high" if best[1] == len(alphas) - 1 else "")
    # refit on everything with the chosen penalty, so no data is wasted
    ws_full, xm_f, ym_f, scale_f = _solve_grid(x, y, [alpha])
    return Ridge(w=ws_full[0], x_mean=xm_f, y_mean=ym_f, alpha=alpha,
                 lam=alpha * scale_f, n_train=n, n_features=x.shape[1],
                 val_score=float(best[0]), at_grid_edge=edge, alpha_scores=scores)


def shuffle_time(activity: np.ndarray, seed: int = 0) -> np.ndarray:
    """Control: permute the frame order within each sample.

    Every marginal statistic of the activity survives, only the pairing with
    the stimulus dies. A decoder that still scores on this is reading something
    other than the stimulus, which is the failure mode this control exists for.

    This is the control that holds on a synchronised protocol, where every
    sample runs the same schedule at the same speed and the stimulus is
    therefore a function of the frame index alone. Permuting within the sample
    breaks that function, so a decoder that recovers the frame index from the
    activity still has a random target. What survives is the sample's identity:
    the decoder can recognise which condition it is looking at and draw that
    condition's average frame, which on the moving-edge set is worth a PixCorr
    of 0.47. Read that as the floor, not as zero.
    """
    a = np.array(activity, copy=True)
    rng = np.random.default_rng(seed)
    for i in range(a.shape[0]):
        a[i] = a[i][rng.permutation(a.shape[1])]
    return a


def shuffle_samples(activity: np.ndarray, seed: int = 0) -> np.ndarray:
    """Control: pair each sample's stimulus with a different sample's activity.

    A cyclic shift by a non-zero offset, so no sample keeps its own activity.
    It removes what `shuffle_time` leaves behind, the identity of the sample,
    but it keeps the frame index aligned, and that is its weakness: on a
    protocol where every sample runs the same schedule, frame t of one condition
    determines frame t of every other, so the shift is a consistent relabelling
    that a linear decoder undoes completely. Measured on the moving-edge set it
    scores exactly what the real fit scores, 1.000 for the photoreceptors.

    So it is not a floor. It is a test of whether the stimulus set is a function
    of the frame index: a score here near the real one says the protocol is
    synchronised and cannot support a decodability comparison, whatever the
    real number looks like. On stimuli that are not synchronised to each other,
    naturalistic scenes among them, it becomes the stronger of the two controls.
    Both are reported for exactly this reason.
    """
    a = np.asarray(activity)
    n = a.shape[0]
    if n < 2:
        return np.array(a, copy=True)
    k = int(np.random.default_rng(seed).integers(1, n))
    return np.roll(a, k, axis=0).copy()


def subsets(n_cells: int, counts, seed: int = 0) -> dict[int, np.ndarray]:
    """Control: random cell subsets, for the quality-against-count curve."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in counts:
        c = int(min(c, n_cells))
        out[c] = rng.choice(n_cells, size=c, replace=False)
    return out
