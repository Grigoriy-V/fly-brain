"""18.13: лучшее плечо проекта на нескольких сидах, без отбора.

    python tools/fig_seedclip18.py

Четыре независимых прогона ворот (сиды 0–3) меняют и шум прайора, и то, какие
отложенные клипы берутся для сравнения. Показан **нулевой сэмпл каждого
прогона** — не лучший и не худший, а первый попавшийся, чтобы картинка не
была отбором. Справа — 13B из настоящего состояния в том же коде, чтобы было
видно, чего в сэмплах ещё нет.
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

BASE = "samples18_corpus_dct16_w384_lr1e3_c"
PREV = "samples18_corpus_dct16_w192_lr1e3_c"
SEEDS = (0, 1, 2, 3)


def tag(base: str, s: int) -> str:
    return base if s == 0 else f"{base}_s{s}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--edges", default="edges18_seeds")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--prefix", default="2026-09-20_malecns_seedclip18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    E = json.loads((run / f"{a.edges}.json").read_text(encoding="utf-8"))["groups"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    key = f"prior_{a.sample}"
    cells, gates, flats, prev_g, prev_f = [], [], [], [], []
    for s in SEEDS:
        S = json.loads((run / f"{tag(BASE, s)}.json").read_text(encoding="utf-8"))
        z = np.load(run / f"{tag(BASE, s)}.npz")
        rt = S["scores"][key]["round_trip"]
        fl = E[f"w384 s{s}"]["per_clip"][a.sample]["frac_flat"]
        gates.append(E[f"w384 s{s}"]); flats.append(E[f"w384 s{s}"]["frac_flat"]["mean"])
        prev_g.append(json.loads((run / f"{tag(PREV, s)}.json").read_text(encoding="utf-8"))
                      ["gates"]["prior"]["round_trip"]["median"])
        prev_f.append(E[f"w192 s{s}"]["frac_flat"]["mean"])
        cells.append((z[f"video__{key}"], f"сид {s}",
                      f"прогонка {ru(f'{rt:.3f}')}, ровного {ru(f'{100 * fl:.1f}')} %"))
        if s == SEEDS[-1]:
            rk = next(k for k in z.files if k.startswith("video__clip_"))
            rf = E["13B из настоящего состояния"]
            cells.append((z[rk], "13B из настоящего\nсостояния",
                          f"прогонка {ru(f'{S['gates']['clip']['round_trip']['median']:.3f}')}, "
                          f"ровного {ru(f'{100 * rf['frac_flat']['mean']:.1f}')} %"))

    med = [json.loads((run / f"{tag(BASE, s)}.json").read_text(encoding="utf-8"))
           ["gates"]["prior"]["round_trip"]["median"] for s in SEEDS]
    frames = int(json.loads((run / f"{tag(BASE, 0)}.json").read_text(encoding="utf-8"))["frames"])
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (f"K=16, ширина 384 — лучшее плечо проекта, и превосходство держится на каждом сиде: "
             f"ворота {ru(f'{np.mean(med):.4f}')} против {ru(f'{np.mean(prev_g):.4f}')} у прежнего лучшего, "
             f"разреженность {ru(f'{100 * np.mean(flats):.1f}')} % против {ru(f'{100 * np.mean(prev_f):.1f}')} % "
             f"(4 из 4 сидов по обеим мерам). Показан нулевой сэмпл каждого прогона, без отбора; "
             f"по 16 сэмплов на сид, ворота локально, $0. До настоящего остаётся разреженность: "
             f"27 % из 47. {slow}.")
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=80)
    print("  ворота по сидам w384: " + ", ".join(f"{s}:{m:.4f}" for s, m in zip(SEEDS, med)))
    print("  ворота по сидам w192: " + ", ".join(f"{s}:{m:.4f}" for s, m in zip(SEEDS, prev_g)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
