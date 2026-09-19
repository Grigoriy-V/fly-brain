"""Four generated videos, nothing else (the human, 2026-09-20: "просто из 4").

    python tools/fig_new_video18.py --run data/prior18 --name samples18_local

Each cell is a video 13B drew from a state the prior sampled out of noise —
no clip anywhere in the chain. The samples are taken across the middle of the
distribution, not the best four; the round trip stays under each cell, because
a generated video without that number is a picture, not a result. The state
maps, the controls and the scale are in
`reports/figures/2026-09-20_malecns_prior18.gif`.
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--name", default="samples18_local")
    p.add_argument("--prefix", default="2026-09-20_malecns_new_video18")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    S = json.loads((run / f"{a.name}.json").read_text(encoding="utf-8"))
    z = np.load(run / f"{a.name}.npz")
    sc, frames = S["scores"], S["frames"]
    ranked = sorted([n for n in sc if sc[n]["kind"] == "prior"], key=lambda n: sc[n]["round_trip"])
    pick = [ranked[int(round(q * (len(ranked) - 1)))] for q in (0.3, 0.45, 0.6, 0.75)][:a.n]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    cells = [(z[f"video__{n}"], f"новое видео {i + 1}", f"прогонка {sc[n]['round_trip']:.3f}")
             for i, n in enumerate(pick)]
    slow = f"{frames} кадров по 20 мс (0.8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    med = S.get("gates", {}).get("prior", {}).get("round_trip", {}).get("median")
    clip = S.get("gates", {}).get("clip", {}).get("round_trip", {}).get("median")
    title = (f"Видео, порождённые без исходного клипа: шум → прайор → состояние T4/T5 → 13B. "
             f"Медиана прогонки по 16 сэмплам {med:.3f}, у настоящего отложенного клипа {clip:.3f}. {slow}.")
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
