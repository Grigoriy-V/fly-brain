"""Генерации новой модели рядом с прежним лучшим плечом и с эталоном.

    python tools/fig_new384.py

Первая ячейка — 13B из настоящего состояния корпуса: то, как выглядит сцена
в этом представлении. Дальше по два сэмпла из K=32 при ширине 384 и из
прежнего лучшего плеча (K=16, ширина 192), отобранные по одному правилу —
середина собственного распределения ворот. Под каждой ячейкой два числа:
прогонка через мозг и доля ровных переходов (18.9), то есть совместимость и
структура порознь.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flydream.generate import learned as L  # noqa: E402
from flydream.generate.edges18 import edges  # noqa: E402
from tools.fig_cmp18 import pick  # noqa: E402
from tools.fig_gen13b_pick import row  # noqa: E402

ARMS = [("samples18_corpus_dct32_w384_lr1e3_w1_c", "K=32, ширина 384"),
        ("samples18_corpus_dct16_w192_lr1e3_c", "K=16, ширина 192")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--prefix", default="2026-09-20_malecns_new384")
    p.add_argument("--per", type=int, default=2)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    nb = np.asarray(L.neighbour_index(721))
    qs = [0.35, 0.6, 0.45][:a.per]

    cells, meds, frames = [], [], None
    for i, (tag, label) in enumerate(ARMS):
        S = json.loads((run / f"{tag}.json").read_text(encoding="utf-8"))
        z = np.load(run / f"{tag}.npz")
        sc, frames = S["scores"], S["frames"]
        meds.append((label, S["gates"]["prior"]["round_trip"]["median"],
                     100 * float(np.mean([edges(z[f"video__{k}"], nb)["frac_flat"]
                                          for k in sc if sc[k]["kind"] == "prior"]))))
        if not cells:                                                # эталон один раз, из того же прогона
            c = [k for k in sc if sc[k]["kind"] == "clip"][0]
            v = z[f"video__{c}"]
            rt_c = sc[c]["round_trip"]
            fl_c = 100 * edges(v, nb)["frac_flat"]
            cells.append((v, "13B из настоящего состояния",
                          f"прогонка {ru(f'{rt_c:.3f}')}, ровного {ru(f'{fl_c:.0f}')} %"))
        for j, n in enumerate(pick(S, qs)):
            v = z[f"video__{n}"]
            fl = 100 * edges(v, nb)["frac_flat"]
            rt = sc[n]["round_trip"]
            cells.append((v, f"{label} — {j + 1}",
                          f"прогонка {ru(f'{rt:.3f}')}, ровного {ru(f'{fl:.0f}')} %"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    slow = f"{frames} кадров по 20 мс (0.8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    pair = "; ".join(f"{lab}: ворота {ru(f'{g:.4f}')}, ровного {ru(f'{f:.1f}')} %" for lab, g, f in meds)
    title = (f"Шум → прайор → состояние T4/T5 → 13B, исходного клипа нет нигде. {pair}. "
             f"У настоящего: ровного 46,6 %. По два сэмпла из середины каждого распределения, не лучшие. {slow}.")
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
