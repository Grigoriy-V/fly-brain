"""19.1: линейная первая стадия против обучаемой, на одних и тех же клипах.

    python tools/fig_pca19.py

Слева направо латент беднеет: 2 048 → 1 024 → 512 компонент. Всё, что видно,
получено одним и тем же 13B с одним и тем же z; отличается только состояние,
которое ему дали.

Главное сравнение — не между клетками, а с прошлой неделей работы: обучаемый
энкодер-декодер на 2 163 измерения давал 0,687 при любой силе KL, включая
нулевую (18.24c). Линейная PCA **на 512** обходит его, а на 2 048 забирает
86,7 % потолка цепочки.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fig_gen13b_pick import row  # noqa: E402

VAE_BEST = 0.687                                                      # 18.24c, те же шесть клипов, любая beta


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="pcaval19")
    p.add_argument("--fit", default="pca2048")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_pca19")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.tag}.npz")
    F = json.loads(str(np.load(run / f"{a.fit}.npz")["summary"]))     # сводка самого разложения, 19.0
    i, ks = a.show, S["ks"]
    frames = len(Z[f"raw__{i}"])
    ceil = S["groups"]["настоящее состояние"]["r_to_raw"]
    ex = {k: 100 * F["explained"]["test"][str(k)]["explained"] for k in ks}
    g = F["geometry"]["test"]

    cells = [
        (Z[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
         f"ровного {ru(f'{100 * S['raw']['frac_flat']:.1f}')} %"),
        (Z[f"video__настоящее состояние|{i}"], "13B из настоящего\nсостояния (потолок цепочки)",
         f"r {ru(f'{ceil:.3f}')}"),
    ]
    for k in ks:
        v = S["groups"][f"через PCA k = {k}"]
        cells.append((Z[f"video__через PCA k = {k}|{i}"],
                      f"через PCA, k = {k}\nдисперсии {ru(f'{ex[k]:.1f}')} %",
                      f"r {ru(f'{v['r_to_raw']:.3f}')}, "
                      f"{ru(f'{100 * v['r_to_raw'] / ceil:.0f}')} % потолка"))
    best = S["groups"][f"через PCA k = {ks[0]}"]["r_to_raw"]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Первая стадия сделана линейной — и обходит обучаемую. PCA на {ks[0]} компонент возвращает отложенный "
        f"клип с r = {ru(f'{best:.3f}')}, это {ru(f'{100 * best / ceil:.0f}')} % потолка цепочки "
        f"({ru(f'{ceil:.3f}')}); обучаемый энкодер-декодер на 2 163 измерения давал {ru(f'{VAE_BEST:.3f}')} при "
        f"любой силе KL, включая нулевую (18.24c), то есть {ru(f'{100 * VAE_BEST / ceil:.0f}')} % потолка. "
        f"Даже k = {ks[-1]} ({ru(f'{S['groups'][f'через PCA k = {ks[-1]}']['r_to_raw']:.3f}')}) его обходит. "
        f"Шесть отложенных клипов те же, что у VAE, и они из {10} классов UCF101, которых обучение не видело "
        f"вовсе: сплит сделан по классам (train и val по 91 классу, test — 10, пересечений файлов нет), так что "
        f"это перенос на незнакомые категории, а не на незнакомые клипы. Доля дисперсии на 1 246 отложенных "
        f"состояниях почти не отличается от val: {ru(f'{ex[ks[0]]:.1f}')} % против "
        f"{ru(f'{100 * F['explained']['val'][str(ks[0])]['explained']:.1f}')} %. Что латент пока НЕ гауссов и что "
        f"чинить потоку: ст. откл. по осям {ru(f'{g['sd_axis_mean']:.3f}')}, радиус "
        f"{ru(f'{g['radius_mean']:.1f}')} ± {ru(f'{g['radius_sd']:.1f}')} при √k = "
        f"{ru(f'{g['typical_radius']:.1f}')} ± 0,71, эксцесс {ru(f'{g['kurtosis_mean']:.2f}')} против 3,0. "
        f"Размытие остаётся: ровного {ru(f'{100 * S['groups'][f'через PCA k = {ks[0]}']['frac_flat']:.1f}')} % "
        f"против {ru(f'{100 * S['raw']['frac_flat']:.1f}')} % у сырого. Один z у 13B во всех клетках; показан "
        f"образец {i + 1} из шести. Разложение — 43 с на T4, 0,02 доллара; приёмка локально, даром. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 146), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:26} {:>11} {:>9} {:>8} {:>9}".format("группа", "r к сырому", "% потолка", "ворота", "ровного"))
    for k, v in S["groups"].items():
        print("{:26} {:11.3f} {:8.0f}% {:8.4f} {:8.1f}%".format(
            k, v["r_to_raw"], 100 * v["r_to_raw"] / ceil, v["gate"], 100 * v["frac_flat"]))
    print("{:26} {:11.3f} {:8.0f}% {:>8} {:>9}".format("VAE 2 163 (18.24c)", VAE_BEST,
                                                       100 * VAE_BEST / ceil, "0,047", "—"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
