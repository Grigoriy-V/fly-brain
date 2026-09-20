"""18.24: сид стал разыгрываемым, но декодер сцену из него не делает.

    python tools/fig_vae18.py

Две половины приёмки, заданные человеком. **B** выполнена: латент отложенных
состояний и свежий розыгрыш лежат в одном и том же распределении, то есть сид
наконец разыгрываемый. **A** нет: через энкодер-декодер клип узнаётся хуже
(r 0,604 против потолка 0,952), а из разыгранного латента сцены не выходит.

Все клетки — один и тот же 13B с одним и тем же z; отличается только
состояние, которое ему дали.
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

CELLS = [("настоящее состояние", "13B из настоящего\nсостояния"),
         ("через VAE (A)", "через энкодер-декодер\n(сид задан клипом)"),
         ("розыгрыш латента (B)", "из РАЗЫГРАННОГО\nлатента")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--tag", default="vaeval18")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-20_malecns_vae18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    z = np.load(run / f"{a.tag}.npz")
    i, lt = a.show, S["latent"]
    frames = len(z[f"raw__{i}"])

    cells = [(z[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
              f"ровного {ru(f'{100 * S['raw']['frac_flat']:.1f}')} %")]
    for key, head in CELLS:
        v = S["groups"][key]
        cells.append((z[f"video__{key}|{i}"], head,
                      f"r к сырому {ru(f'{v['r_to_raw']:.3f}')}, ровного {ru(f'{100 * v['frac_flat']:.1f}')} %"))

    gA, gB, g0 = (S["groups"][k] for k, _ in CELLS[::-1][::-1][1:] + [CELLS[0]])
    A, B, Z0 = S["groups"]["через VAE (A)"], S["groups"]["розыгрыш латента (B)"], S["groups"]["настоящее состояние"]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Половина приёмки взята, половина нет. **Сид стал разыгрываемым**: латент отложенных состояний "
        f"лежит на радиусе {ru(f'{lt['enc_radius']:.1f}')} при ст. отклонении "
        f"{ru(f'{lt['enc_sd']:.3f}')}, свежий розыгрыш — {ru(f'{lt['draw_radius']:.1f}')} и "
        f"{ru(f'{lt['draw_sd']:.3f}')}, √D = {ru(f'{lt['typical_radius']:.1f}')}; у потока прообразы стояли на "
        f"0,85 от оболочки и не двигались. Но декодер платит за это картинкой: через энкодер-декодер "
        f"r = {ru(f'{A['r_to_raw']:.3f}')} против {ru(f'{Z0['r_to_raw']:.3f}')} у потолка, а из разыгранного "
        f"латента — {ru(f'{B['r_to_raw']:.3f}')}, ровного {ru(f'{100 * B['frac_flat']:.1f}')} % против "
        f"{ru(f'{100 * S['raw']['frac_flat']:.1f}')} % и контраст {ru(f'{B['sd']:.3f}')} против "
        f"{ru(f'{S['raw']['sd']:.3f}')}. Причина в самом латенте: сигнал в среднем даёт "
        f"{ru(f'{lt['mu_sd']:.3f}')}, шум — {ru(f'{lt['sigma']:.3f}')}, то есть меньше половины латента несёт "
        f"информацию, и это прямое следствие силы KL. 6 отложенных клипов, один z у 13B; показан образец "
        f"{i + 1} из шести. Локально, $0. {slow}.").replace("**", "")
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 132), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:24} {:>11} {:>9} {:>9} {:>10}".format("группа", "r к сырому", "ровного", "контраст", "ближайшее"))
    for k, v in S["groups"].items():
        print("{:24} {:11.3f} {:8.1f}% {:9.3f} {:10.3f}".format(
            k, v["r_to_raw"], 100 * v["frac_flat"], v["sd"], v["nearest_r"]))
    print("{:24} {:>11} {:8.1f}% {:9.3f}".format("сырое видео", "—", 100 * S["raw"]["frac_flat"], S["raw"]["sd"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
