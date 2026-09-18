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
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    scale = float(np.mean(s ** 2)) or 1.0
    uty = u.T @ yc
    ws = []
    for a in alphas:
        lam = a * scale
        d = s / (s ** 2 + lam)
        ws.append(vt.T @ (d[:, None] * uty))
    return ws, x_mean, y_mean, scale


def fit(x: np.ndarray, y: np.ndarray, *, alphas=ALPHAS, val_fraction: float = 0.2,
        seed: int = 0, groups: np.ndarray | None = None) -> Ridge:
    """Fit with the penalty chosen on a held-out slice of the training data.

    `groups` (one label per row, e.g. the Sintel scene) keeps every row of a
    group on the same side of the validation cut, so the penalty is not chosen
    on frames whose neighbours it was fitted on.
    """
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
