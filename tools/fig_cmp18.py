"""Two priors, one row: the best arm by the gate beside the weighted K = 32.

    python tools/fig_cmp18.py

Three samples from each, taken at the same quantiles of each prior's own gate
distribution (0.35 / 0.5 / 0.65) — the middle, deliberately not the best
three, and the same rule on both sides so the comparison is not a selection.
Every cell carries its own round trip, because a generated video without that
number is a picture, not a result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fig_gen13b_pick import row  # noqa: E402

ARMS = [("samples18_corpus_dct16_w192_lr1e3_c", "K=16"),
        ("samples18_corpus_dct32_w192_lr1e3_w1_c", "K=32 + вес")]


def pick(S: dict, qs) -> list[str]:
    sc = S["scores"]
    ranked = sorted([n for n in sc if sc[n]["kind"] == "prior"], key=lambda n: sc[n]["round_trip"])
    return [ranked[int(round(q * (len(ranked) - 1)))] for q in qs]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--prefix", default="2026-09-20_malecns_cmp18")
    p.add_argument("--per", type=int, default=3)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    qs = [0.35, 0.5, 0.65][:a.per]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    cells, meds, frames, clip = [], [], None, None
    for tag, label in ARMS:
        S = json.loads((run / f"{tag}.json").read_text(encoding="utf-8"))
        z = np.load(run / f"{tag}.npz")
        frames = S["frames"]
        g = S["gates"]
        meds.append((label, g["prior"]["round_trip"]["median"]))
        clip = g["clip"]["round_trip"]["median"]
        sc = S["scores"]
        for i, n in enumerate(pick(S, qs)):
            rt = sc[n]["round_trip"]
            cells.append((z[f"video__{n}"], f"{label} — {i + 1}", f"прогонка {ru(f'{rt:.3f}')}"))
    slow = f"{frames} кадров по 20 мс (0.8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    pair = ", ".join(f"{lab} {ru(f'{v:.3f}')}" for lab, v in meds)
    title = (f"Шум → прайор → состояние T4/T5 → 13B, исходного клипа нет нигде. "
             f"Медианы прогонки по 16 сэмплам: {pair}; у настоящего отложенного клипа {ru(f'{clip:.3f}')}. "
             f"По три сэмпла из середины каждого распределения, не лучшие. {slow}.")
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
