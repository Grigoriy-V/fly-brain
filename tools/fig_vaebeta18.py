"""18.24: два конца ручки KL, и оба не дают сцену.

    python tools/fig_vaebeta18.py

Сильный KL (β = 1e-4) кладёт латент ровно на оболочку и голодом морит декодер:
сигнал — 19 % дисперсии латента. Слабый (β = 1e-5) отдаёт декодеру 87 %
сигнала, но латент уезжает с оболочки, и розыгрыш даёт состояния, до которых
мозгу не дотянуться (ворота 0,26 против 0,011 у настоящего).

Главное, что видно при сравнении: рост сигнала в латенте в 4,6 раза поднял
реконструкцию всего с 0,604 до 0,651 при потолке 0,952 и 0,890 у линейной PCA
той же размерности. Значит узкое место — **не шум от KL**, а сам
энкодер-декодер.
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
    p.add_argument("--strong", default="vaeval18")
    p.add_argument("--weak", default="vaeval18_b1e5")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-20_malecns_vaebeta18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.strong}.json").read_text(encoding="utf-8"))
    W = json.loads((run / f"{a.weak}.json").read_text(encoding="utf-8"))
    ZS = np.load(run / f"{a.strong}.npz")
    ZW = np.load(run / f"{a.weak}.npz")
    i = a.show
    frames = len(ZS[f"raw__{i}"])
    ls, lw = S["latent"], W["latent"]

    def sig(lt):
        return 100 * lt["mu_sd"] ** 2 / (lt["mu_sd"] ** 2 + lt["sigma"] ** 2)

    cells = [
        (ZS[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
         f"ровного {ru(f'{100 * S['raw']['frac_flat']:.1f}')} %"),
        (ZS[f"video__настоящее состояние|{i}"], "13B из настоящего\nсостояния (потолок)",
         f"r {ru(f'{S['groups']['настоящее состояние']['r_to_raw']:.3f}')}"),
        (ZS[f"video__через VAE (A)|{i}"], "через VAE, β = 1e-4\nсигнала в латенте 19 %",
         f"r {ru(f'{S['groups']['через VAE (A)']['r_to_raw']:.3f}')}, "
         f"ст.откл. z {ru(f'{ls['enc_sd']:.3f}')}"),
        (ZW[f"video__через VAE (A)|{i}"], "через VAE, β = 1e-5\nсигнала 87 %",
         f"r {ru(f'{W['groups']['через VAE (A)']['r_to_raw']:.3f}')}, "
         f"ст.откл. z {ru(f'{lw['enc_sd']:.3f}')}"),
        (ZW[f"video__розыгрыш латента (B)|{i}"], "розыгрыш латента,\nβ = 1e-5",
         f"ворота {ru(f'{W['groups']['розыгрыш латента (B)']['gate']:.4f}')}, "
         f"ровного {ru(f'{100 * W['groups']['розыгрыш латента (B)']['frac_flat']:.1f}')} %"),
    ]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Ручка KL обрезана с обеих сторон. Сильный KL: латент ровно на оболочке "
        f"(ст. откл. {ru(f'{ls['enc_sd']:.3f}')}, радиус {ru(f'{ls['enc_radius']:.1f}')} при √D = 46,5), но сигнал "
        f"— лишь {ru(f'{sig(ls):.0f}')} % дисперсии латента, декодер голодает. Слабый KL: сигнала "
        f"{ru(f'{sig(lw):.0f}')} %, зато латент уехал — радиус {ru(f'{lw['enc_radius']:.1f}')} ± "
        f"{ru(f'{lw['enc_radius_sd']:.1f}')}, и розыгрыш даёт состояния, до которых мозг не дотягивается: "
        f"ворота {ru(f'{W['groups']['розыгрыш латента (B)']['gate']:.4f}')} против "
        f"{ru(f'{S['groups']['настоящее состояние']['gate']:.4f}')} у настоящего. "
        f"И главное: сигнал вырос в 4,6 раза, а реконструкция — лишь с "
        f"{ru(f'{S['groups']['через VAE (A)']['r_to_raw']:.3f}')} до "
        f"{ru(f'{W['groups']['через VAE (A)']['r_to_raw']:.3f}')} при потолке "
        f"{ru(f'{S['groups']['настоящее состояние']['r_to_raw']:.3f}')} и 0,890 у линейной PCA той же "
        f"размерности. Значит узкое место не шум от KL, а сам энкодер-декодер. "
        f"Оговорка: сильный прогон 20 000 шагов, слабый 8 000 — сравнение неполное. "
        f"6 отложенных клипов, один z у 13B; показан образец {i + 1} из шести. Локально, $0. {slow}.")
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 140), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:14} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "beta", "сигнал", "ст.откл z", "радиус", "разброс", "A: r", "B: ворота"))
    for nm, d_ in (("1e-4 (20k)", S), ("1e-5 (8k)", W)):
        lt = d_["latent"]
        print("{:14} {:7.0f}% {:9.3f} {:9.1f} {:9.2f} {:9.3f} {:9.4f}".format(
            nm, sig(lt), lt["enc_sd"], lt["enc_radius"], lt["enc_radius_sd"],
            d_["groups"]["через VAE (A)"]["r_to_raw"], d_["groups"]["розыгрыш латента (B)"]["gate"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
