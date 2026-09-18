"""Figure: the training-optimisation benchmark on a T4 beside its own noise.

    python tools/fig_optimization_bench.py [--run 2026-09-18_bench_150043_689575_229624b2a4]

Left: seconds per iteration per variant, both repeats as points (batch 16,
30 iterations, checkpoints included). Right: how far each variant's final
state is from its paired baseline (decoder head, max abs), beside the
distance between the two baseline repeats themselves, which is CUDA's own
run-to-run noise; the network's free parameters agree to 3e-8 everywhere and
are not drawn. Reads data/benchmarks/<run>/{records.json, *final.pt}.
Dataviz method of the workspace: slots 1-2, ink labels, legend below.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
VARIANTS = ["baseline", "stats", "relu", "stats_relu"]


def state_gap(a: dict, b: dict, part: str) -> float:
    m = 0.0
    def walk(u, v):
        nonlocal m
        if isinstance(u, dict):
            for k in u:
                walk(u[k], v[k])
        else:
            m = max(m, float((u.double() - v.double()).abs().max()))
    walk(a[part], b[part])
    return m


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="2026-09-18_bench_150043_689575_229624b2a4")
    a = p.parse_args()
    d = ROOT / "data" / "benchmarks" / a.run
    recs = [r for r in json.loads((d / "records.json").read_text()) if not r["profile"]]
    by = {v: [r["s_per_iter"] for r in recs if r["variant"] == v] for v in VARIANTS}
    tags = {(r["variant"], r["repeat"]): r["tag"] for r in recs}
    states = {}
    for (v, rep), tag in tags.items():
        f = d / tag / "evidence" / "final.pt"
        if f.exists():
            states[(v, rep)] = torch.load(f, weights_only=True)
    gaps = {}
    if ("baseline", 0) in states and ("baseline", 1) in states:
        gaps["baseline r0 vs r1\n(CUDA noise)"] = state_gap(states[("baseline", 0)], states[("baseline", 1)], "decoder")
    for v in VARIANTS[1:]:
        if (v, 0) in states and ("baseline", 0) in states:
            gaps[f"{v}\nvs baseline"] = state_gap(states[(v, 0)], states[("baseline", 0)], "decoder")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.1, 1]})
    for axx in (ax, ax2):
        axx.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            axx.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            axx.spines[sp].set_color(GRID)
        axx.grid(True, axis="y", color=GRID, linewidth=1)
        axx.set_axisbelow(True)
        axx.tick_params(colors=MUTED, labelsize=8.5)
    x = np.arange(len(VARIANTS))
    med = [float(np.median(by[v])) for v in VARIANTS]
    ax.bar(x, med, width=0.6, color=SLOTS[0], alpha=0.85, zorder=2, label="median of two repeats")
    for i, v in enumerate(VARIANTS):
        ax.scatter([i - 0.1, i + 0.1][: len(by[v])], by[v], s=30, color=INK, zorder=3, label="repeat" if i == 0 else None)
        ax.annotate(f"{med[i]:.3f}\n{med[0] / med[i]:.2f}×", (i, med[i]), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS)
    ax.set_ylim(0, max(med) * 1.3)
    ax.set_ylabel("seconds per iteration, Tesla T4, batch 16", color=INK2, fontsize=8.5)
    ax.set_title("Two source-level changes to flyvis's step: 1.17× together", color=INK, fontsize=9.5, loc="left")
    labels = list(gaps)
    vals = [gaps[k] for k in labels]
    colors = [SLOTS[2]] + [SLOTS[1]] * (len(labels) - 1)
    ax2.bar(np.arange(len(labels)), vals, width=0.6, color=colors, alpha=0.85, zorder=2)
    for i, v in enumerate(vals):
        ax2.annotate(f"{v:.1e}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylim(0, max(vals) * 1.35 if vals else 1)
    ax2.set_ylabel("max |Δ| of the decoder head after 30 iterations", color=INK2, fontsize=8.5)
    ax2.set_title("The change is no larger than the reference's own noise", color=INK, fontsize=9.5, loc="left")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ROOT / "reports" / "figures" / "2026-09-18_optimization_bench_t4_noise.png"
    fig.savefig(out, dpi=190, facecolor=SURFACE)
    print(f"wrote {out}; medians {dict(zip(VARIANTS, [round(m, 3) for m in med]))}; gaps {dict((k.split(chr(10))[0], round(v, 6)) for k, v in gaps.items())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
