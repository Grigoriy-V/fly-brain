"""The decodability map: for every cell type, how much of the stimulus comes back.

    python -m flydream.decode.map --stimuli edges --run 2026-09-18_decode_edges

One row per cell type per decoder per control. The point of the map is the
ordering and the shape of the curve across the visual hierarchy, not any single
number, so nothing here reports a score without the controls beside it:

- `shuffled_time`: the frame order of the activity permuted within each sample.
  Every marginal statistic survives; only the pairing with the stimulus dies.
- `subset_<n>`: the same fit on n randomly chosen cells of the type, which turns
  the score into a curve against how much of the lattice is read.
- `identification_chance`: 1/n_candidates, a baseline that needs no run.

The model and the connectome are arguments, never hard-coded (DECISIONS,
2026-09-18): the same command runs on a pretrained FlyVis member and on the
MaleCNS model zero, and the report names which one produced each figure.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import tomllib

import numpy as np
import pandas as pd

from flydream.model import ROOT, configure_flyvis_root

configure_flyvis_root()

import flyvis  # noqa: E402
from flydream.decode import metrics as M  # noqa: E402
from flydream.decode import pairs as P  # noqa: E402
from flydream.decode import ridge as R  # noqa: E402

DEFAULTS = {
    "model": "flow/0000/000",
    "dt": 0.02,
    "lags": [0],
    "test_fraction": 0.2,
    "val_fraction": 0.2,
    "seed": 0,
    "pix_per_hex": 3,
    "ssim_win": 9,
    "identification_candidates": 20,
    "subset_counts": [10, 30, 90, 270],
    "batch_size": 4,
}


def settings() -> dict:
    cfg = tomllib.loads((ROOT / "config.toml").read_text())
    out = dict(DEFAULTS)
    out.update(cfg.get("decode", {}))
    return out


# ----------------------------------------------------------------- stimuli


def stimulus_set(name: str, dt: float):
    """Return (dataset, stim_key, t_pre, t_fade_in).

    `edges` is the debug and motion stimulus: a single speed, so every sample
    has the same number of frames and no NaN padding, and one edge angle per
    sample, so a sample is an independent condition and a row-level split is
    honest. `sintel` is the naturalistic half, split by scene.
    """
    if name == "edges":
        from flyvis.datasets.moving_bar import MovingEdge

        ds = MovingEdge(offsets=(-10, 11), intensities=[0, 1], speeds=[19.0], height=80,
                        post_pad_mode="continue", dt=dt, device=flyvis.device,
                        t_pre=1.0, t_post=1.0)
        return ds, None, 1.0, 0.0
    if name == "bars":
        from flyvis.datasets.moving_bar import MovingBar

        ds = MovingBar(widths=[2, 4, 8], offsets=(-10, 11), intensities=[0, 1],
                       speeds=[19.0], height=9, post_pad_mode="continue", dt=dt,
                       device=flyvis.device, t_pre=1.0, t_post=1.0)
        return ds, None, 1.0, 0.0
    if name == "sintel":
        from flyvis.datasets.sintel import AugmentedSintel

        # augment must stay True: augment=False short-circuits get_item and skips
        # the 24 fps -> 1/dt resampling. The dataset is deterministic anyway,
        # its augmentation defaults are all identity and baked per sample.
        ds = AugmentedSintel(tasks=["lum", "flow"], n_frames=19, dt=dt,
                             temporal_split=True, interpolate=True,
                             flip_axes=[0], n_rotations=[0], augment=True,
                             vertical_splits=3, center_crop_fraction=0.7,
                             boxfilter=dict(extent=15, kernel_size=13))
        return ds, "lum", 0.0, 2.0
    raise ValueError(f"unknown stimulus set: {name}")


# ----------------------------------------------------------------- scoring


def drop_nan_rows(x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Datasets of unequal-length samples come back NaN padded; those rows carry
    no stimulus and must not enter a fit. Returns the kept rows and how many went."""
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    return x[keep], y[keep], groups[keep], int((~keep).sum())


def broken_pairings(pairs: P.Pairs, seed: int) -> list[tuple[str, P.Pairs]]:
    """The two controls, built once for the whole map.

    Each one copies the entire activity array, which on a Sintel run is 1.4 GB,
    so building them per cell type would have copied 180 GB over 65 types for
    no change in the result: both shuffles are deterministic in the seed.
    """
    return [("shuffled_time", R.shuffle_time(pairs.activity, seed)),
            ("shuffled_samples", R.shuffle_samples(pairs.activity, seed))]


def score_one(pairs: P.Pairs, cell_type: str, train_i, test_i, s: dict,
              controls: list | None = None) -> list[dict]:
    """Fit and score one cell type, with its controls. One dict per row of the map."""
    lags = tuple(s["lags"])
    xtr, ytr, gtr = pairs.frames(cell_type, train_i, lags)
    xte, yte, _ = pairs.frames(cell_type, test_i, lags)
    xtr, ytr, gtr, drop_tr = drop_nan_rows(xtr, ytr, gtr)
    xte, yte, _, drop_te = drop_nan_rows(xte, yte, np.zeros(len(xte)))
    if len(xtr) < 10 or len(xte) < 10:
        return [{"cell_type": cell_type, "decoder": "ridge", "control": "none",
                 "note": "too few finite rows", "n_train": len(xtr), "n_test": len(xte)}]

    n_cells = pairs.by_type(cell_type).shape[-1]
    common = {"cell_type": cell_type, "decoder": "ridge", "n_cells": n_cells,
              "n_features": xtr.shape[1], "n_train": len(xtr), "n_test": len(xte),
              "dropped_train": drop_tr, "dropped_test": drop_te, "lags": str(lags)}
    mkw = dict(pix_per_hex=s["pix_per_hex"], win_size=s["ssim_win"],
               n_candidates=s["identification_candidates"], seed=s["seed"])
    rows = []

    model = R.fit(xtr, ytr, val_fraction=s["val_fraction"], seed=s["seed"], groups=gtr)
    rows.append({**common, "control": "none", "alpha": model.alpha, "lam": model.lam,
                 "val_r2": model.val_score, "grid_edge": model.at_grid_edge,
                 **M.all_metrics(model.predict(xte), yte, **mkw)})

    # controls that break the pairing: the frame order inside each sample, and
    # then which sample's activity goes with which sample's stimulus. Neither is
    # the floor on its own; read them together. The time shuffle leaves the
    # condition's identity readable, and the sample shuffle leaves the frame
    # index aligned, which a synchronised protocol lets a decoder exploit
    # completely (see ridge.shuffle_samples).
    for name, broken in (controls if controls is not None else broken_pairings(pairs, s["seed"])):
        sh = P.Pairs(stimulus=pairs.stimulus, activity=broken, groups=pairs.groups,
                     cell_types=pairs.cell_types, index=pairs.index)
        sxtr, sytr, sgtr = sh.frames(cell_type, train_i, lags)
        sxte, syte, _ = sh.frames(cell_type, test_i, lags)
        sxtr, sytr, sgtr, _ = drop_nan_rows(sxtr, sytr, sgtr)
        sxte, syte, _, _ = drop_nan_rows(sxte, syte, np.zeros(len(sxte)))
        if len(sxtr) < 10 or len(sxte) < 10:
            continue
        sm = R.fit(sxtr, sytr, val_fraction=s["val_fraction"], seed=s["seed"], groups=sgtr)
        rows.append({**common, "control": name, "alpha": sm.alpha, "lam": sm.lam,
                     "val_r2": sm.val_score, "grid_edge": sm.at_grid_edge,
                     **M.all_metrics(sm.predict(sxte), syte, **mkw)})

    # control: random cell subsets, i.e. quality against how much of the lattice is read
    for c, sel in R.subsets(n_cells, [c for c in s["subset_counts"] if c < n_cells], s["seed"]).items():
        cols = np.concatenate([sel + k * n_cells for k in range(len(lags))])
        sub = R.fit(xtr[:, cols], ytr, val_fraction=s["val_fraction"], seed=s["seed"], groups=gtr)
        rows.append({**common, "control": f"subset_{c}", "n_cells": c,
                     "n_features": len(cols), "alpha": sub.alpha, "lam": sub.lam,
                     "val_r2": sub.val_score, "grid_edge": sub.at_grid_edge,
                     **M.all_metrics(sub.predict(xte[:, cols]), yte, **mkw)})
    return rows


# ----------------------------------------------------------------- driver


def main(argv=None) -> int:
    s = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--stimuli", default="edges", choices=["edges", "bars", "sintel"])
    p.add_argument("--model", default=s["model"], help="a flyvis NetworkView name, or 'malecns'")
    p.add_argument("--types", nargs="*", default=None, help="cell types, default all")
    p.add_argument("--n-samples", type=int, default=None, help="cap the number of samples")
    p.add_argument("--run", default=None, help="run id; default <date>_decode_<stimuli>")
    p.add_argument("--cache", action="store_true", help="save/reuse the simulated pairs")
    p.add_argument("--lags", nargs="*", type=int, default=None,
                   help="decoder window as frame offsets, e.g. 0 2 4; default config.toml [decode] lags")
    p.add_argument("--pairs-from", default=None,
                   help="run id whose pairs.npz to reuse (the simulation is identical across lag windows)")
    a = p.parse_args(argv)
    if a.lags is not None:
        s["lags"] = list(a.lags)

    run = a.run or f"{time.strftime('%Y-%m-%d')}_decode_{a.stimuli}"
    outdir = ROOT / "data" / "decode" / run
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"model: {a.model}")
    if a.model == "malecns":
        from flydream.model.zero import build_network, content_addressed

        from flydream.data.export import filters_path

        cfg = tomllib.loads((ROOT / "config.toml").read_text())
        net = build_network(content_addressed(filters_path(cfg)), extent=cfg["data"]["extent"])
    else:
        net = flyvis.NetworkView(a.model).init_network(checkpoint="best")
    net.eval()

    ds, stim_key, t_pre, t_fade_in = stimulus_set(a.stimuli, s["dt"])
    idx = np.arange(len(ds))
    if a.n_samples:
        idx = idx[:a.n_samples]
    print(f"stimuli: {a.stimuli}, {len(idx)} of {len(ds)} samples, dt={s['dt']}")

    cache = outdir / "pairs.npz"
    source = (ROOT / "data" / "decode" / a.pairs_from / "pairs.npz") if a.pairs_from else cache
    if (a.cache or a.pairs_from) and source.exists():
        pairs = P.Pairs.load(source)
        print(f"pairs from {source}: {pairs.n_samples} samples x {pairs.n_frames} frames")
    else:
        pairs = P.build(net, ds, dt=s["dt"], indices=idx, batch_size=s["batch_size"],
                        t_pre=t_pre, t_fade_in=t_fade_in,
                        stim_key=stim_key or "lum", keep_types=a.types)
        if a.cache:
            pairs.save(cache)
    print(f"pairs: stimulus {pairs.stimulus.shape}, activity {pairs.activity.shape}, "
          f"{len(np.unique(pairs.groups))} groups")

    train_i, test_i = P.split_by_group(pairs.groups, s["test_fraction"], s["seed"])
    print(f"split: {len(train_i)} train / {len(test_i)} test samples")

    types = a.types or pairs.cell_types
    controls = broken_pairings(pairs, s["seed"])
    rows = []
    for i, t in enumerate(types, 1):
        started = time.time()
        got = score_one(pairs, t, train_i, test_i, s, controls)
        rows.extend(got)
        base = next((r for r in got if r.get("control") == "none"), {})
        c_t = next((r for r in got if r.get("control") == "shuffled_time"), {})
        c_s = next((r for r in got if r.get("control") == "shuffled_samples"), {})
        print(f"[{i}/{len(types)}] {t:<8} pixcorr {base.get('pixcorr', float('nan')):+.3f} "
              f"(time {c_t.get('pixcorr', float('nan')):+.3f} / "
              f"sample {c_s.get('pixcorr', float('nan')):+.3f})  "
              f"r2 {base.get('r2', float('nan')):+.3f}  id {base.get('identification', float('nan')):.2f}"
              f"  {time.time() - started:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    csv = outdir / "map.csv"
    df.to_csv(csv, index=False)
    meta = {"run": run, "model": a.model, "stimuli": a.stimuli, "settings": s,
            "n_samples": int(pairs.n_samples), "n_frames": int(pairs.n_frames),
            "n_types": len(types), "seconds": round(time.time() - t0, 1),
            "flyvis": flyvis.__version__ if hasattr(flyvis, "__version__") else "1.2.0"}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{csv}  ({len(df)} rows, {meta['seconds']}s)")

    base = df[df.control == "none"].sort_values("pixcorr", ascending=False)
    c_t = df[df.control == "shuffled_time"].set_index("cell_type")
    c_s = df[df.control == "shuffled_samples"].set_index("cell_type")
    print("\nby PixCorr, with both controls [time / sample]:")
    shown = pd.concat([base.head(8), base.tail(5)]).drop_duplicates("cell_type")
    for _, r in shown.iterrows():
        print(f"  {r.cell_type:<8} {r.pixcorr:+.3f}  "
              f"[{c_t.pixcorr.get(r.cell_type, float('nan')):+.3f} / "
              f"{c_s.pixcorr.get(r.cell_type, float('nan')):+.3f}]  "
              f"r2 {r.r2:+.3f}  id {r.identification:.2f} of {r.identification_chance:.2f}")
    print(f"\nspread over {len(base)} types: min {base.pixcorr.min():+.3f}, "
          f"median {base.pixcorr.median():+.3f}, max {base.pixcorr.max():+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
