"""The lag sweep, read together: does the decoder's window change the map's order?

    python -m flydream.decode.sweep --runs 2026-09-18_decode_sintel_lag_0 2026-09-18_decode_sintel_lag_0_1 ...

Each run is one window of `flydream.decode.map` on the same cached pairs. This
joins them into one table per cell type and window, with the two controls, and
draws the one figure the question needs: the control-corrected frame component,
median per stage, against the window length. If the lines cross, the stage
curve at any single window is a property of the window (ISSUES ISS-0004).

Colour follows the dataviz method of the workspace: five stages take
categorical slots 1-5 in fixed order (validated adjacent-pair light mode:
worst CVD delta E 9.1, normal-vision 19.6), 2 px lines, 8 px end markers with a
surface ring, hairline solid grid, labels in ink not in series colour, a legend
plus direct end labels. A PNG for the report is static, so there is no hover
layer; the CSV beside it is the table view.
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd

from flydream.model import ROOT

STAGES = [("1 photoreceptor", r"^R\d"), ("2 lamina", r"^(L\d|Lawf|Am|C2|C3|T1$)"),
          ("3 medulla Mi", r"^(Mi|CT1)"), ("4 medulla Tm/TmY", r"^(Tm|TmY)"),
          ("5 T4/T5 motion", r"^(T4|T5)")]
# categorical slots 1-5 of the reference palette (light), fixed order, never cycled
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def stage_of(name: str) -> str:
    for label, pat in STAGES:
        if re.match(pat, name):
            return label
    return "6 lobula/other"


def window_ms(lags: str, dt: float = 0.02) -> float:
    """`(0, 2, 4)` at dt 0.02 s is an 80 ms reach after the frame."""
    xs = [int(x) for x in re.findall(r"-?\d+", str(lags))]
    return (max(xs) - min(xs)) * dt * 1000.0 if xs else 0.0


def load(runs: list[str]) -> pd.DataFrame:
    rows = []
    for r in runs:
        p = ROOT / "data" / "decode" / r / "map.csv"
        if not p.exists():
            print(f"  {r}: no map.csv yet, skipped")
            continue
        df = pd.read_csv(p)
        w = df.pivot_table(index="cell_type", columns="control", values="pixcorr")
        lags = df["lags"].iloc[0]
        for t, row in w.iterrows():
            real, tsh, ssh = row.get("none", np.nan), row.get("shuffled_time", np.nan), row.get("shuffled_samples", np.nan)
            rows.append({"run": r, "lags": lags, "window_ms": window_ms(lags), "cell_type": t,
                         "stage": stage_of(t), "real": real, "time_shuffle": tsh, "sample_shuffle": ssh,
                         "frame": real - tsh, "scene": tsh - ssh})
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby(["window_ms", "stage"]).agg(n=("real", "size"), real=("real", "median"),
                                              frame=("frame", "median"), scene=("scene", "median"),
                                              time_shuffle=("time_shuffle", "median")).reset_index()
    return g.sort_values(["stage", "window_ms"])


def rank_shift(t: pd.DataFrame) -> pd.DataFrame:
    """Per cell type: frame component and its rank at the shortest and longest window."""
    ws = sorted(t.window_ms.unique())
    a, b = t[t.window_ms == ws[0]].set_index("cell_type"), t[t.window_ms == ws[-1]].set_index("cell_type")
    out = pd.DataFrame({"stage": a.stage, f"frame_{ws[0]:.0f}ms": a.frame, f"frame_{ws[-1]:.0f}ms": b.frame.reindex(a.index)})
    out["gain"] = out.iloc[:, 2] - out.iloc[:, 1]
    out[f"rank_{ws[0]:.0f}ms"] = out.iloc[:, 1].rank(ascending=False).astype(int)
    out[f"rank_{ws[-1]:.0f}ms"] = out.iloc[:, 2].rank(ascending=False).astype(int)
    out["rank_change"] = out.iloc[:, 4] - out.iloc[:, 5]
    return out.sort_values("gain", ascending=False)


def figure(summary: pd.DataFrame, path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [s for s, _ in STAGES if s in set(summary.stage)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID); ax.spines[sp].set_linewidth(1)
    ax.grid(True, axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ends = []
    for i, s in enumerate(stages):
        d = summary[summary.stage == s].sort_values("window_ms")
        ax.plot(d.window_ms, d.frame, color=SLOTS[i], linewidth=2, solid_joinstyle="round", solid_capstyle="round",
                label=s[2:], zorder=3)
        ax.scatter(d.window_ms, d.frame, s=64, color=SLOTS[i], edgecolor=SURFACE, linewidth=2, zorder=4)
        ends.append((float(d.window_ms.iloc[-1]), float(d.frame.iloc[-1]), s[2:]))
    # direct end labels in ink, nudged apart only if they collide
    ends.sort(key=lambda e: e[1])
    ys, min_gap = [], 0.028
    for x, y, label in ends:
        yy = y if not ys or y - ys[-1] >= min_gap else ys[-1] + min_gap
        ys.append(yy)
        ax.annotate(label, (x, y), xytext=(8, (yy - y) * 300), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=1) if abs(yy - y) > 1e-9 else None)
    ax.set_xlabel("decoder window after the stimulus frame, ms", color=INK2, fontsize=9)
    ax.set_ylabel("frame component of PixCorr, median per stage", color=INK2, fontsize=8.5)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_xticks(sorted(summary.window_ms.unique()))
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    # the legend below the plot, never on the lines
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower center",
               ncol=len(labels), bbox_to_anchor=(0.5, 0.0))
    ax.margins(x=0.18)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--tag", default="2026-09-18_sintel_lag_sweep")
    a = p.parse_args(argv)
    t = load(a.runs)
    if t.empty:
        print("nothing to read"); return 1
    out = ROOT / "data" / "decode" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "by_type_window.csv", index=False)
    s = summarise(t)
    s.to_csv(out / "by_stage_window.csv", index=False)
    print("median frame component per stage and window:")
    print(s.pivot(index="stage", columns="window_ms", values="frame").round(3).to_string())
    print("\nmedian PixCorr (real):")
    print(s.pivot(index="stage", columns="window_ms", values="real").round(3).to_string())
    if t.window_ms.nunique() > 1:
        rs = rank_shift(t)
        rs.to_csv(out / "rank_shift.csv")
        print("\nlargest gains from the window (frame component):")
        print(rs.head(8).round(3).to_string())
        print("\nlargest losses:")
        print(rs.tail(5).round(3).to_string())
        from scipy.stats import spearmanr
        print(f"\nSpearman(frame at shortest, gain) = {spearmanr(rs.iloc[:, 1], rs.gain).statistic:+.3f}")
    figure(s, ROOT / "reports" / "figures" / f"{a.tag}.png",
           "Decodability of the frame by stage depends on the decoder's window")
    print(f"\nwrote {out} and reports/figures/{a.tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
