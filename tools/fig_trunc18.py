"""Потолок усечения DCT: ниже K = 16 сцена рассыпается.

    python tools/fig_trunc18.py

Настоящее состояние ужато до K временных коэффициентов, развёрнуто обратно в
40 кадров и отрисовано тем же 13B с тем же z. Слева — сырое видео корпуса, то,
с чем сравнивают. Это **верхняя граница** представления: ниже неё обученная
модель на таком K не поднимется.

Заодно пятая по счёту дискредитация статистик структуры: при K = 2 разреженность
49,2 % — ближе к сырым 49,8 %, чем у K = 16, — а корреляция с сырым клипом
0,127, то есть сцены нет вовсе. Здесь это видно прямо, потому что у теста есть
эталон.
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

SHOW = ["K = 40 (без усечения)", "K = 16", "K = 8", "K = 4"]
HEAD = {"K = 40 (без усечения)": "без усечения\nD = 230 720", "K = 16": "K = 16 (сейчас)\nD = 92 288",
        "K = 8": "K = 8\nD = 46 144", "K = 4": "K = 4\nD = 23 072"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--tag", default="trunc18")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-20_malecns_trunc18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    z = np.load(run / f"{a.tag}.npz")
    i = a.show
    frames = len(z[f"raw__{i}"])

    cells = [(z[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
              f"ровного {ru(f'{100 * S['raw']['frac_flat']:.1f}')} %")]
    for g in SHOW:
        w = S["groups"][g]
        cells.append((z[f"video__{g}|{i}"], HEAD[g],
                      f"r к сырому {ru(f'{w['r_to_raw']:.3f}')}, ровного {ru(f'{100 * w['frac_flat']:.1f}')} %"))

    g16, g8, g2 = (S["groups"][k] for k in ("K = 16", "K = 8", "K = 2"))
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Потолок усечения DCT по времени: сцена держится до K = 16 и рассыпается сразу под ним. "
        f"Корреляция с сырым клипом {ru(f'{g16['r_to_raw']:.3f}')} при K = 16 → "
        f"{ru(f'{g8['r_to_raw']:.3f}')} при K = 8, ворота {ru(f'{g16['gate']:.4f}')} → "
        f"{ru(f'{g8['gate']:.4f}')} (в 8 раз хуже). Значит половинную размерность этим энкодером не купить: "
        f"обученная модель на K = 8 не поднимется выше этой клетки. "
        f"И пятая дискредитация статистик структуры: при K = 2 разреженность "
        f"{ru(f'{100 * g2['frac_flat']:.1f}')} % — ближе к сырым "
        f"{ru(f'{100 * S['raw']['frac_flat']:.1f}')} %, чем K = 16, — при корреляции "
        f"{ru(f'{g2['r_to_raw']:.3f}')}, то есть сцены нет вовсе. "
        f"6 отложенных клипов, один z у 13B; показан образец {i + 1} из шести. Локально, $0. {slow}.")
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 132), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:24} {:>7} {:>8} {:>11} {:>8} {:>9}".format("представление", "D", "ошибка", "r к сырому", "ворота", "ровного"))
    for name, v in S["groups"].items():
        print("{:24} {:7d} {:7.2f}% {:11.3f} {:8.4f} {:8.1f}%".format(
            name, v["dims"], 100 * v["trunc_error"], v["r_to_raw"], v["gate"], 100 * v["frac_flat"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
