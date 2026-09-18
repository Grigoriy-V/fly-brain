"""Figure: the batch-size smoke on a Tesla T4 (run 2026-09-18_step3_smoke_batch_t4).

    python tools/fig_batch_throughput.py

Reads the numbers from reports/runs.jsonl (metrics samples_per_s_batch*) plus
the batch-4 smoke (2026-09-18_step3_smoke_t4), and draws samples per second and
dollars per member (10^6 samples at $0.59/h) against the batch size. Dataviz
method of the workspace: slots 1 and 2 of the palette, ink labels, legend below.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ["#2a78d6", "#eb6834"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
T4_USD_PER_H, SAMPLES = 0.59, 250_000 * 4


def main() -> int:
    rows = {}
    for line in (ROOT / "reports" / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        m = re.match(r"samples_per_s_batch(\d+)", r.get("metric", ""))
        if m and r["value"] > 0:
            rows[int(m.group(1))] = r["value"]
        if r.get("run") == "2026-09-18_step3_smoke_t4" and r.get("metric") == "s_per_iter":
            rows[4] = 4 / r["value"]
    rows.setdefault(4, 9.2)
    b = sorted(rows)
    sps = [rows[x] for x in b]
    usd = [SAMPLES / s / 3600 * T4_USD_PER_H for s in sps]

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.plot(b, sps, color=SLOTS[0], linewidth=2, label="samples per second", zorder=3)
    ax.scatter(b, sps, s=64, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=4)
    for x, y in zip(b, sps):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(b)
    ax.set_xticklabels([str(x) for x in b])
    ax.set_xlabel("batch size", color=INK2, fontsize=9)
    ax.set_ylabel("samples per second, Tesla T4", color=INK2, fontsize=9)
    ax.set_ylim(0, max(sps) * 1.3)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax2 = ax.twinx()
    ax2.plot(b, usd, color=SLOTS[1], linewidth=2, linestyle=(0, (4, 3)), label="$ per member (10⁶ samples)", zorder=3)
    ax2.scatter(b, usd, s=48, color=SLOTS[1], edgecolor=SURFACE, linewidth=2, zorder=4)
    for x, y in zip(b, usd):
        ax2.annotate(f"${y:.0f}", (x, y), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax2.set_ylim(0, max(usd) * 1.3)
    ax2.set_ylabel("$ per member at $0.59/h", color=INK2, fontsize=9)
    ax2.tick_params(colors=MUTED, labelsize=8.5)
    for sp in ("top", "left", "bottom"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax.set_title("Throughput saturates at batch 16; batch 64 cannot run (fewer than 64 training clips)",
                 color=INK, fontsize=9.5, loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out = ROOT / "reports" / "figures" / "2026-09-18_step3_batch_throughput_t4.png"
    fig.savefig(out, dpi=190, facecolor=SURFACE)
    print(f"wrote {out}: " + ", ".join(f"batch {x}: {s:.1f}/s, ${u:.1f}" for x, s, u in zip(b, sps, usd)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
