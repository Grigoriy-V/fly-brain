"""Pictures of the map: what comes back out of each stage, as an image.

    python -m flydream.decode.figures --run 2026-09-18_decode_sintel_flyvis

A table of PixCorr values says how much is recoverable; it does not show what
is recoverable, and the two are not the same question. This renders the ladder:
the stimulus the photoreceptors actually saw, then the reconstruction from each
stage beside it, on the same frames, at the same scale.

Resolution is not a free choice and needs no invention. The eye is 721
ommatidia on a 31 by 31 hexagonal lattice at 5.8 degrees each, so at four
pixels per ommatidium the raster is 125 by 125: a 128-pixel image is the
lattice drawn at its own resolution, not an upsampling. Anything larger is a
generative guess and must be labelled as one.

Every panel here is the 721 values, nothing more. The grey ring outside the
hexagon is off-lattice, where the eye has no receptors.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tomllib

import numpy as np

from flydream.model import ROOT

from flydream.decode import metrics as M
from flydream.decode import pairs as P
from flydream.decode import ridge as R
from flydream.decode.hexraster import raster_map

# A ladder through the hierarchy: one or two types per stage, in processing order.
LADDER = ["R1", "L1", "L3", "Mi4", "Tm5a", "T4a", "T5a"]
STAGE = {"R1": "photoreceptor", "L1": "lamina", "L3": "lamina", "Mi4": "medulla",
         "Tm5a": "medulla", "T4a": "motion", "T5a": "motion"}


def settings() -> dict:
    cfg = tomllib.loads((ROOT / "config.toml").read_text())
    return cfg.get("decode", {})


def fit_ladder(pairs: P.Pairs, types, train_i, test_i, s: dict, lags=(0,)):
    """Refit the decoders and keep their test predictions, which the map only scored."""
    out = {}
    for t in types:
        if t not in pairs.index:
            print(f"  {t}: not a cell type of this connectome, skipped")
            continue
        xtr, ytr, gtr = pairs.frames(t, train_i, lags)
        xte, yte, _ = pairs.frames(t, test_i, lags)
        keep = np.isfinite(xtr).all(1) & np.isfinite(ytr).all(1)
        xtr, ytr, gtr = xtr[keep], ytr[keep], gtr[keep]
        keep = np.isfinite(xte).all(1) & np.isfinite(yte).all(1)
        xte, yte = xte[keep], yte[keep]
        model = R.fit(xtr, ytr, val_fraction=s.get("val_fraction", 0.2),
                      seed=s.get("seed", 0), groups=gtr)
        pred = model.predict(xte)
        out[t] = {"pred": pred, "targ": yte,
                  "pixcorr": M.pixel_correlation(pred, yte),
                  "alpha": model.alpha, "edge": model.at_grid_edge}
        print(f"  {t:<6} PixCorr {out[t]['pixcorr']:+.3f}"
              f"{'  (penalty at grid edge: ' + model.at_grid_edge + ')' if model.at_grid_edge else ''}",
              flush=True)
    return out


def ladder_figure(fits: dict, frames, path: pathlib.Path, pix_per_hex: int = 4,
                  title: str = "") -> pathlib.Path:
    """Rows are frames, columns are the target and each stage's reconstruction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from flydream.decode.hexraster import to_raster

    _, mask, shape = raster_map(721, pix_per_hex)
    types = list(fits)
    targ = fits[types[0]]["targ"]
    ncol, nrow = len(types) + 1, len(frames)
    fig, ax = plt.subplots(nrow, ncol, figsize=(1.35 * ncol, 1.45 * nrow),
                           squeeze=False, facecolor="white")

    def draw(a, vec, vmin, vmax):
        img = to_raster(vec, 721, pix_per_hex, fill=np.nan)
        a.imshow(np.where(mask, img, np.nan), cmap="gray", vmin=vmin, vmax=vmax,
                 interpolation="nearest")
        a.set_facecolor("0.85")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)

    for r, fr in enumerate(frames):
        t_vec = targ[fr]
        lo, hi = float(np.nanmin(t_vec)), float(np.nanmax(t_vec))
        draw(ax[r][0], t_vec, lo, hi)
        if r == 0:
            ax[r][0].set_title("stimulus\n(what the eye saw)", fontsize=7.5)
        ax[r][0].set_ylabel(f"frame {fr}", fontsize=7)
        for c, t in enumerate(types, 1):
            p = fits[t]["pred"][fr]
            # each panel on its own scale: a decoder can recover the pattern and
            # miss the absolute level, and the question here is the pattern
            draw(ax[r][c], p, float(np.nanmin(p)), float(np.nanmax(p)))
            if r == 0:
                ax[r][c].set_title(f"{t}\n{STAGE.get(t, '')}  r={fits[t]['pixcorr']:+.2f}",
                                   fontsize=7.5)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)
    return path


def ladder_animation(fits: dict, path: pathlib.Path, n_frames: int = 40,
                     pix_per_hex: int = 4, fps: int = 12) -> pathlib.Path | None:
    """The same ladder as a moving clip: one row, time running."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("  imageio not available, skipping the clip")
        return None
    from flydream.decode.hexraster import to_raster

    _, mask, _ = raster_map(721, pix_per_hex)
    types = list(fits)
    targ = fits[types[0]]["targ"]
    n = min(n_frames, len(targ))

    def panel(vec):
        img = to_raster(vec, 721, pix_per_hex, fill=np.nan)
        lo, hi = float(np.nanmin(vec)), float(np.nanmax(vec))
        img = (img - lo) / max(hi - lo, 1e-9)
        img = np.where(mask, img, 0.55)
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    out = []
    for f in range(n):
        strip = [panel(targ[f])] + [panel(fits[t]["pred"][f]) for t in types]
        sep = np.full((strip[0].shape[0], 3), 255, np.uint8)
        row = strip[0]
        for s in strip[1:]:
            row = np.concatenate([row, sep, s], axis=1)
        out.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, out, fps=fps, loop=0)
    return path


def main(argv=None) -> int:
    s = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="a run under data/decode/ holding pairs.npz")
    p.add_argument("--types", nargs="*", default=LADDER)
    p.add_argument("--frames", nargs="*", type=int, default=[8, 18, 28])
    p.add_argument("--pix-per-hex", type=int, default=4, help="4 gives a 125x125 raster")
    p.add_argument("--no-clip", action="store_true")
    a = p.parse_args(argv)

    rundir = ROOT / "data" / "decode" / a.run
    cache = rundir / "pairs.npz"
    if not cache.exists():
        print(f"no pairs at {cache}; run flydream.decode.map with --cache first")
        return 1
    print(f"loading {cache} ({cache.stat().st_size / 1e9:.2f} GB)")
    pairs = P.Pairs.load(cache)
    print(f"pairs: {pairs.n_samples} samples x {pairs.n_frames} frames, "
          f"{len(np.unique(pairs.groups))} scenes")
    train_i, test_i = P.split_by_group(pairs.groups, s.get("test_fraction", 0.2),
                                       s.get("seed", 0))
    print(f"split: {len(train_i)} train / {len(test_i)} test samples")
    print("fitting the ladder:")
    fits = fit_ladder(pairs, a.types, train_i, test_i, s, tuple(s.get("lags", [0])))
    if not fits:
        print("nothing fitted")
        return 1

    figdir = ROOT / "reports" / "figures"
    png = ladder_figure(fits, a.frames, figdir / f"{a.run}_ladder.png",
                        pix_per_hex=a.pix_per_hex,
                        title=f"Reconstruction of the rendered stimulus, stage by stage  "
                              f"({a.run}, 721 ommatidia at {a.pix_per_hex} px each)")
    print(f"\nwrote {png}")
    if not a.no_clip:
        gif = ladder_animation(fits, figdir / f"{a.run}_ladder.gif", pix_per_hex=a.pix_per_hex)
        if gif:
            print(f"wrote {gif}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
