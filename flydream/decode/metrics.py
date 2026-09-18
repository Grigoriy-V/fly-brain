"""How a reconstruction is scored.

The metrics are the ones the references report for neural reconstruction, and
deliberately not the ones vision papers report for natural images: CLIP or any
other semantic score means nothing for a fly, which has no object categories to
agree about.

- `pixel_correlation` (PixCorr): Pearson r between predicted and target hexal
  values, computed per frame on the 721-vector and then averaged. Computed on
  the vector and not on the raster because Voronoi cells differ by about +-10%
  in pixel count, which would weight some hexals more than others.
- `r2`: fraction of target variance explained, the quantity ridge minimises;
  reported because a correlation is blind to scale and offset.
- `hex_ssim`: structural similarity on the rasterised pair, which needs a local
  neighbourhood and therefore a raster. `win_size` must span at least three
  hexals or the window sits inside one flat Voronoi cell, local variance is zero
  and SSIM saturates at 1 regardless of the reconstruction.
- `identification`: the test the decoding literature uses when a correlation is
  hard to interpret. Among `n_candidates` frames drawn from the same test set,
  is the true target the one the prediction correlates with most? Chance is
  1/n_candidates, which makes it the one metric with a baseline that needs no
  control run.
"""
from __future__ import annotations

import numpy as np

from flydream.decode.hexraster import to_raster


def _flat(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, a.shape[-1])


def pixel_correlation(pred: np.ndarray, targ: np.ndarray) -> float:
    """Mean over frames of the within-frame Pearson r. NaN frames are dropped."""
    p, t = _flat(pred), _flat(targ)
    p = p - p.mean(axis=-1, keepdims=True)
    t = t - t.mean(axis=-1, keepdims=True)
    num = (p * t).sum(axis=-1)
    den = np.sqrt((p * p).sum(axis=-1) * (t * t).sum(axis=-1))
    r = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return float(np.nanmean(r))


def r2(pred: np.ndarray, targ: np.ndarray) -> float:
    """1 - SSE/SST against the target's overall mean, pooled over all frames."""
    p, t = _flat(pred), _flat(targ)
    sse = float(((p - t) ** 2).sum())
    sst = float(((t - t.mean()) ** 2).sum())
    return float("nan") if sst == 0 else 1.0 - sse / sst


def spatiotemporal_correlation(pred: np.ndarray, targ: np.ndarray) -> float:
    """One Pearson r over the whole (frames x hexals) block, space and time together."""
    p, t = _flat(pred).ravel(), _flat(targ).ravel()
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() < 2:
        return float("nan")
    return float(np.corrcoef(p[m], t[m])[0, 1])


def hex_ssim(pred: np.ndarray, targ: np.ndarray, pix_per_hex: int = 3,
             win_size: int = 9, data_range: float | None = None) -> float:
    """Mean SSIM over frames, on the rasterised lattice.

    `data_range` defaults to the target's own range, since a rendered hexal
    stimulus is a contrast around mid-grey and not a 0..1 image.
    """
    from skimage.metrics import structural_similarity

    p, t = _flat(pred), _flat(targ)
    n_hex = t.shape[-1]
    if data_range is None:
        data_range = float(np.nanmax(t) - np.nanmin(t)) or 1.0
    out = []
    for i in range(t.shape[0]):
        pi = to_raster(p[i], n_hex, pix_per_hex, fill=0.0)
        ti = to_raster(t[i], n_hex, pix_per_hex, fill=0.0)
        out.append(structural_similarity(ti, pi, data_range=data_range, win_size=win_size))
    return float(np.mean(out))


def identification(pred: np.ndarray, targ: np.ndarray, n_candidates: int = 20,
                   seed: int = 0, min_std: float = 1e-3) -> tuple[float, float, int]:
    """Return (accuracy, chance, n_frames_used).

    For each frame the prediction is correlated with the true target and with
    `n_candidates - 1` other target frames drawn from the same test set; a hit
    is the true one ranking first.

    Frames whose target has no spatial structure are excluded, and this is not a
    detail: a synthetic protocol pads its condition with a constant grey screen
    (41% of the frames of the moving-edge set), and no decoder, however perfect,
    can say which grey frame it is looking at. Leaving them in caps the score
    far below 1 for reasons that have nothing to do with the reconstruction. The
    count is returned so a report can say how many frames the number rests on.
    """
    p, t = _flat(pred), _flat(targ)
    keep = t.std(axis=-1) >= min_std
    p, t = p[keep], t[keep]
    n = t.shape[0]
    k = min(n_candidates, n)
    if k < 2:
        return float("nan"), float("nan"), n
    pz = p - p.mean(axis=-1, keepdims=True)
    tz = t - t.mean(axis=-1, keepdims=True)
    pz /= np.maximum(np.linalg.norm(pz, axis=-1, keepdims=True), 1e-12)
    tz /= np.maximum(np.linalg.norm(tz, axis=-1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    hits = 0
    for i in range(n):
        others = rng.choice(np.delete(np.arange(n), i), size=k - 1, replace=False)
        cand = np.concatenate([[i], others])
        scores = tz[cand] @ pz[i]
        hits += int(np.argmax(scores) == 0)
    return hits / n, 1.0 / k, n


def all_metrics(pred: np.ndarray, targ: np.ndarray, *, pix_per_hex: int = 3,
                win_size: int = 9, n_candidates: int = 20, seed: int = 0) -> dict:
    acc, chance, n_id = identification(pred, targ, n_candidates, seed)
    return {
        "pixcorr": pixel_correlation(pred, targ),
        "r2": r2(pred, targ),
        "stcorr": spatiotemporal_correlation(pred, targ),
        "ssim": hex_ssim(pred, targ, pix_per_hex, win_size),
        "identification": acc,
        "identification_chance": chance,
        "identification_frames": n_id,
    }
