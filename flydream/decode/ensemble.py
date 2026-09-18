"""The map read over members and splits: a number with its interval per cell type.

    python -m flydream.decode.ensemble --glob "2026-09-18_decode_sintel_m*_s*_lag_*" --tag 2026-09-18_sintel_ensemble_map

Each run under data/decode/ is one (member, split, window) of `flydream.decode.map`
(run names `<date>_decode_sintel_m<member>_s<seed>_lag_<lags>`, the shape
`deploy/modal/decode_app.py` writes). Per cell type and window this joins the
real score with its two controls from the same run, forms the frame component
(real − time-shuffle) and the scene component (time-shuffle − sample-shuffle),
and reports the median with the 10th–90th percentile over the (member, split)
pairs: the ensemble spread the contract asks for beside every number
(AGENTS.md, "A result is a number beside its control"). Writes
`by_type_window.csv` (every run), `summary.csv` (median and interval per type
and window), `by_stage.csv`, and the figure `reports/figures/<tag>.png`: the
frame component per type at the reported windows, ordered by stage, interval
as a bar, the 0 ms and 80 ms windows side by side (ISS-0004: the window is
part of the measurement, never reported alone).
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd

from flydream.model import ROOT
from flydream.decode.sweep import STAGES, SLOTS, INK, INK2, MUTED, GRID, SURFACE, stage_of, window_ms

RUN_RE = re.compile(r"_m(?P<member>\d+)_s(?P<seed>\d+)_lag_(?P<lags>[\d_]+)$")


def parse_run(name: str) -> dict | None:
    m = RUN_RE.search(name)
    if not m:
        return None
    return {"member": int(m["member"]), "seed": int(m["seed"]), "lags": m["lags"].replace("_", " ")}


def load(names: list[str]) -> pd.DataFrame:
    rows = []
    for r in names:
        p = ROOT / "data" / "decode" / r / "map.csv"
        meta = parse_run(r)
        if meta is None or not p.exists():
            print(f"  {r}: skipped ({'no map.csv' if meta else 'name does not carry member/seed/lags'})")
            continue
        df = pd.read_csv(p)
        w = df.pivot_table(index="cell_type", columns="control", values="pixcorr")
        for t, row in w.iterrows():
            real, tsh, ssh = row.get("none", np.nan), row.get("shuffled_time", np.nan), row.get("shuffled_samples", np.nan)
            rows.append({"run": r, **meta, "window_ms": window_ms(meta["lags"]), "cell_type": t,
                         "stage": stage_of(t), "real": real, "time_shuffle": tsh, "sample_shuffle": ssh,
                         "frame": real - tsh, "scene": tsh - ssh})
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame) -> pd.DataFrame:
    def q(x, k):
        return float(np.nanpercentile(x, k)) if len(x) else np.nan
    g = t.groupby(["cell_type", "stage", "window_ms"])
    out = g.agg(n=("real", "size"), n_members=("member", "nunique"), n_splits=("seed", "nunique"),
                real=("real", "median"), real_lo=("real", lambda x: q(x, 10)), real_hi=("real", lambda x: q(x, 90)),
                frame=("frame", "median"), frame_lo=("frame", lambda x: q(x, 10)), frame_hi=("frame", lambda x: q(x, 90)),
                scene=("scene", "median"), time_shuffle=("time_shuffle", "median")).reset_index()
    return out.sort_values(["window_ms", "stage", "frame"], ascending=[True, True, False])


def by_stage(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.groupby(["window_ms", "stage"]).agg(n_types=("cell_type", "size"), real=("real", "median"),
                                                        frame=("frame", "median"), frame_lo=("frame_lo", "median"),
                                                        frame_hi=("frame_hi", "median")).reset_index()


def figure(summary: pd.DataFrame, path, title: str, windows: list[float] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ws = windows or sorted(summary.window_ms.unique())[:2]
    stages = [s for s, _ in STAGES] + ["6 lobula/other"]
    order = (summary[summary.window_ms == ws[0]].set_index("cell_type")
             .assign(k=lambda d: d.stage.map({s: i for i, s in enumerate(stages)}))
             .sort_values(["k", "frame"], ascending=[True, False]).index.tolist())
    fig, axes = plt.subplots(1, len(ws), figsize=(3.6 * len(ws) + 1.2, 0.17 * len(order) + 1.6),
                             sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    y = np.arange(len(order))
    for ax, w in zip(axes, ws):
        ax.set_facecolor(SURFACE)
        d = summary[summary.window_ms == w].set_index("cell_type").reindex(order)
        colors = [SLOTS[min(stages.index(s), len(SLOTS) - 1)] if s in stages else MUTED for s in d.stage]
        ax.hlines(y, d.frame_lo, d.frame_hi, color=colors, linewidth=2, alpha=0.55, zorder=2)
        ax.scatter(d.frame, y, s=22, color=colors, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.axvline(0, color=GRID, linewidth=1, zorder=1)
        ax.grid(True, axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.set_title(f"window {w:.0f} ms", color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("frame component of PixCorr (real − time-shuffle)", color=INK2, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=6.5, color=INK2)
    axes[0].invert_yaxis()
    n = summary[["n_members", "n_splits"]].max()
    fig.suptitle(f"{title}  (median, 10–90% over {int(n.n_members)} members × {int(n.n_splits)} splits)",
                 color=INK, fontsize=10, x=0.02, ha="left")
    handles = [plt.Line2D([], [], color=SLOTS[i], marker="o", linestyle="", label=s[2:]) for i, s in enumerate(stages[:5])]
    fig.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK2, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="*_decode_sintel_m*_s*_lag_*", help="runs under data/decode/ to join")
    p.add_argument("--runs", nargs="*", default=None, help="explicit run names instead of --glob")
    p.add_argument("--tag", default="sintel_ensemble_map")
    p.add_argument("--windows", nargs="*", type=float, default=None, help="windows (ms) to draw, default the first two")
    a = p.parse_args(argv)
    names = a.runs or sorted(d.name for d in (ROOT / "data" / "decode").glob(a.glob) if d.is_dir())
    t = load(names)
    if t.empty:
        print("nothing to read"); return 1
    out = ROOT / "data" / "decode" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "by_type_window.csv", index=False)
    s = summarise(t)
    s.to_csv(out / "summary.csv", index=False)
    st = by_stage(s)
    st.to_csv(out / "by_stage.csv", index=False)
    print(f"{len(names)} runs, {t.member.nunique()} members, {t.seed.nunique()} splits, "
          f"windows {sorted(t.window_ms.unique())}")
    print("\nmedian frame component per stage [10–90% of the per-type medians]:")
    for w, d in st.groupby("window_ms"):
        print(f"  {w:.0f} ms: " + ", ".join(f"{r.stage[2:]} {r.frame:+.3f}" for r in d.itertuples()))
    print("\ntop and bottom types by frame component at each window:")
    for w, d in s.groupby("window_ms"):
        d = d.sort_values("frame", ascending=False)
        print(f"  {w:.0f} ms: " + ", ".join(f"{r.cell_type} {r.frame:+.2f} [{r.frame_lo:+.2f}, {r.frame_hi:+.2f}]"
                                            for r in d.head(5).itertuples())
              + "  ...  " + ", ".join(f"{r.cell_type} {r.frame:+.2f}" for r in d.tail(3).itertuples()))
    figure(s, ROOT / "reports" / "figures" / f"{a.tag}.png",
           "Frame decodability per cell type", a.windows)
    print(f"\nwrote {out} and reports/figures/{a.tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
