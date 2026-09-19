"""18.4e in one row: the same noise, four ways out of it.

    python tools/fig_classcmp18.py

Cell 1 is the unconditional prior; cells 2-4 are the class-conditional prior
given the same noise vector and three different labels. 13B renders all four
with the same z, so the only thing that differs between the cells is the
prior's response to the label. Under each cell: its round trip, and its
correlation to cell 1 — the number that says whether the label was read at all.
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
    p.add_argument("--name", default="classcmp18_local")
    p.add_argument("--others", default="classcmp18_s1,classcmp18_s2,classcmp18_s3",
                   help="the same comparison on other noises; only their numbers are used")
    p.add_argument("--prefix", default="2026-09-20_malecns_classcmp18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    S = json.loads((run / f"{a.name}.json").read_text(encoding="utf-8"))
    z = np.load(run / f"{a.name}.npz")
    frames = S["frames"]
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    keys = [c["key"] for c in S["cells"]]
    base = keys[0]
    cells = []
    for c in S["cells"]:
        k, rt = c["key"], c["round_trip"]
        r = S["corr_video"].get(f"{base}|{k}")
        foot = f"прогонка {ru(f'{rt:.3f}')}"
        if r is not None:
            foot += f", r к ячейке 1 {ru(f'{r:+.2f}')}"
        cells.append((z[f"video__{k}"], c["title"], foot))

    runs = [S] + [json.loads((run / f"{t}.json").read_text(encoding="utf-8"))
                  for t in a.others.split(",") if t.strip() and (run / f"{t}.json").exists()]
    by_key = {k: [r["cells"][i]["round_trip"] for r in runs] for i, k in enumerate(keys)}
    among = [v for r in runs for k, v in r["corr_video"].items() if not k.startswith(base)]
    means = ", ".join(f"{c['title'].replace('класс ', '')} {ru(f'{np.mean(by_key[k]):.3f}')}"
                      for c, k in zip(S["cells"], keys))
    slow = f"{frames} кадров по 20 мс (0.8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (f"Один и тот же шум, четыре выхода: метка читается, но лишь смещает сэмпл. "
             f"Между метками видео совпадают на r {ru(f'{np.mean(among):+.2f}')}, "
             f"тогда как два разных шума дают r ≈ 0,02. "
             f"Среднее по {len(runs)} шумам: {means}. {slow}.")
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
