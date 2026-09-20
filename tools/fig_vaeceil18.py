"""18.24c: потолок самой архитектуры — KL выключен совсем.

    python tools/fig_vaeceil18.py

18.24b показал, что ручка KL обрезана с обеих сторон, и поставил вопрос: а
сколько эта пара энкодер-декодер вытягивает, когда ей вообще ничего не мешает?
Здесь β = 0 — чистый автоэнкодер, те же 8 000 шагов, что у β = 1e-5, то есть
эта пара сравнивается честно, шаг в шаг.

Линейка — та же, что и во всём пункте 18: рендер против **сырого видео
корпуса** на отложенных клипах, а не процент дисперсии.
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

PCA_2048 = 0.890                                                      # 18.22, линейная граница той же размерности


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--strong", default="vaeval18")
    p.add_argument("--weak", default="vaeval18_b1e5")
    p.add_argument("--none", default="vaeval18_b0")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-20_malecns_vaeceil18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S, W, N = (json.loads((run / f"{t}.json").read_text(encoding="utf-8"))
               for t in (a.strong, a.weak, a.none))
    ZS, ZN = np.load(run / f"{a.strong}.npz"), np.load(run / f"{a.none}.npz")
    ZW = np.load(run / f"{a.weak}.npz")
    i = a.show
    frames = len(ZS[f"raw__{i}"])
    A_ = "через VAE (A)"; M_ = "через VAE, из среднего (A)"
    rS, rW, rN = (d["groups"][A_]["r_to_raw"] for d in (S, W, N))     # из разыгранного z
    mS, mW, mN = (d["groups"][M_]["r_to_raw"] for d in (S, W, N))     # из самого mu
    ceil = S["groups"]["настоящее состояние"]["r_to_raw"]

    def sig(d):
        lt = d["latent"]
        return 100 * lt["mu_sd"] ** 2 / (lt["mu_sd"] ** 2 + lt["sigma"] ** 2)

    cells = [
        (ZS[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
         f"ровного {ru(f'{100 * S['raw']['frac_flat']:.1f}')} %"),
        (ZS[f"video__настоящее состояние|{i}"], "13B из настоящего\nсостояния (потолок цепочки)",
         f"r {ru(f'{ceil:.3f}')}"),
        (ZS[f"video__{A_}|{i}"], "β = 1e-4, 20 000 шагов\nсигнала 19 %",
         f"r {ru(f'{rS:.3f}')} (из μ {ru(f'{mS:.3f}')})"),
        (ZW[f"video__{A_}|{i}"], "β = 1e-5, 8 000 шагов\nсигнала 87 %",
         f"r {ru(f'{rW:.3f}')} (из μ {ru(f'{mW:.3f}')})"),
        (ZN[f"video__{A_}|{i}"], "β = 0, 8 000 шагов\nKL выключен совсем",
         f"r {ru(f'{rN:.3f}')} (из μ {ru(f'{mN:.3f}')})"),
    ]
    best = max(mN, rN)                                                # потолок пары без всякого KL
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"KL выключен совсем — и потолок не сдвинулся. Лучшее, что пара энкодер-декодер отдаёт при β = 1e-5, "
        f"это {ru(f'{mW:.3f}')}; при β = 0, когда ей вообще ничто не мешает, — {ru(f'{best:.3f}')}. Одинаковые "
        f"8 000 шагов, один и тот же замер, разница в третьем знаке. Значит KL не был тем, что держало "
        f"реконструкцию: он решает только, разыгрывается ли латент. А держит её сама пара — {ru(f'{best:.3f}')} "
        f"против {ru(f'{PCA_2048:.3f}')} у ЛИНЕЙНОЙ PCA той же размерности (18.22) и {ru(f'{ceil:.3f}')} у "
        f"потолка цепочки. Попутно снят мой же довод: при сильном KL декодер не страдает от шума апостериора, "
        f"а нуждается в нём — тот же клип из μ даёт {ru(f'{mS:.3f}')} против {ru(f'{rS:.3f}')} из розыгрыша, "
        f"потому что μ (ст. откл. {ru(f'{S['latent']['mu_sd']:.3f}')}) для него код вне распределения. Цена "
        f"выключенного KL отдельно: латент β = 0 стоит на радиусе {ru(f'{N['latent']['enc_radius']:.1f}')} "
        f"± {ru(f'{N['latent']['enc_radius_sd']:.1f}')} при √D = {ru(f'{N['latent']['typical_radius']:.1f}')}, "
        f"ст. откл. {ru(f'{N['latent']['enc_sd']:.3f}')} — разыграть его нельзя по построению, ради этого KL и "
        f"вводился. Оговорка: потолок измерен на 8 000 шагах; единственное длинное плечо (20 000) — с другим "
        f"распределением кода, поэтому бюджет обучения этим прогоном не отделён. "
        f"6 отложенных клипов, один z у 13B; показан образец {i + 1} из шести. Локально, $0. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 146), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:16} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "beta", "сигнал", "ст.откл z", "радиус", "A: из mu", "A: розыгр", "B: ворота"))
    for nm, d_ in (("1e-4 (20k)", S), ("1e-5 (8k)", W), ("0 (8k)", N)):
        lt = d_["latent"]
        print("{:16} {:7.0f}% {:9.3f} {:9.1f} {:9.3f} {:9.3f} {:9.4f}".format(
            nm, sig(d_), lt["enc_sd"], lt["enc_radius"], d_["groups"][M_]["r_to_raw"],
            d_["groups"][A_]["r_to_raw"], d_["groups"]["розыгрыш латента (B)"]["gate"]))
    print(f"{'потолок цепочки':16} {'—':>8} {'—':>9} {'—':>9} {ceil:9.3f} {ceil:9.3f} "
          f"{S['groups']['настоящее состояние']['gate']:9.4f}")
    print(f"{'линейная PCA 2048':16} {'—':>8} {'—':>9} {'—':>9} {PCA_2048:9.3f} {'—':>9} {'—':>9}   (18.22)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
