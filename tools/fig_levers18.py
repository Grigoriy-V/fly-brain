"""Два рычага бьют в разные мишени: шаги чинят геометрию, ширина — картинку.

    python tools/fig_levers18.py

Проверка гипотезы человека (2026-09-20): раз пара «шум ↔ состояние» на
обучении берётся случайной заново каждый шаг, дефект прообразов — от неё, а
не от ёмкости. Прямой тест: то же плечо w128, та же конфигурация, только
шагов втрое больше.

Все четыре клетки — один и тот же 13B с одним и тем же z. Отличается только
состояние: настоящее у первой, у остальных — **обычный розыгрыш** N(0, I)
через приор, отличающийся обучением.

`sd прообраза` — ст. отклонение по осям того шума, в который обращается
настоящее состояние (18.17). У точно обученного потока он был бы 1,000.
Прибор проверен: состояние, сделанное самим приором из заведомо стандартного
шума, обращается назад в 1,001 с ошибкой 0,0–0,4 % (`selfinv18`).
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

ARMS = [("reach18_corpus_dct16_c", "ширина 128\n20 000 шагов"),
        ("reach18_60k", "ширина 128\n60 000 шагов  (×3)"),
        ("reach18_s100", "ширина 384\n20 000 шагов")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--self", dest="selfinv", default="selfinv18")
    p.add_argument("--show", type=int, default=0)
    p.add_argument("--prefix", default="2026-09-20_malecns_levers18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = {t: json.loads((run / f"{t}.json").read_text(encoding="utf-8")) for t, _ in ARMS}
    Z = {t: np.load(run / f"{t}.npz") for t, _ in ARMS}
    V = json.loads((run / f"{a.selfinv}.json").read_text(encoding="utf-8"))["rows"]

    base = S[ARMS[0][0]]
    frames = len(Z[ARMS[0][0]][f"video__настоящее состояние|{a.show}"])
    cells = [(Z[ARMS[0][0]][f"video__настоящее состояние|{a.show}"], "настоящее состояние",
              f"ровного {ru(f'{100 * base['groups']['настоящее состояние']['structure']['frac_flat']['mean']:.1f}')} %")]
    for t, head in ARMS:
        d = S[t]
        cells.append((Z[t][f"video__обычный розыгрыш|{a.show}"], head,
                      f"sd прообраза {ru(f'{d['noise']['sd_per_dim']:.3f}')}, "
                      f"ровного {ru(f'{100 * d['groups']['обычный розыгрыш']['structure']['frac_flat']['mean']:.1f}')} %"))

    sd = {t: S[t]["noise"]["sd_per_dim"] for t, _ in ARMS}
    fl = {t: S[t]["groups"]["обычный розыгрыш"]["structure"]["frac_flat"]["mean"] for t, _ in ARMS}
    t20, t60, w384 = (x[0] for x in ARMS)
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Втрое больше обучения на том же плече двигает геометрию шума и не двигает картинку: "
        f"sd прообраза {ru(f'{sd[t20]:.3f}')} → {ru(f'{sd[t60]:.3f}')} (закрыто "
        f"{ru(f'{100 * (sd[t60] - sd[t20]) / (1 - sd[t20]):.0f}')} % разрыва до 1,000), "
        f"а разреженность розыгрыша как была {ru(f'{100 * fl[t20]:.1f}')} %, так и осталась "
        f"{ru(f'{100 * fl[t60]:.1f}')} %. Втрое больше ширины — наоборот: геометрия стоит "
        f"({ru(f'{sd[w384]:.3f}')}), а разреженность растёт до {ru(f'{100 * fl[w384]:.1f}')} %. "
        f"Значит это две разные болезни, и sd прообраза не предсказывает качество сэмпла. "
        f"Прибор проверен: шум N(0, I) → состояние → обратно даёт "
        f"{ru(f'{np.mean([r['back_sd'] for r in V]):.3f}')} при ошибке "
        f"{ru(f'{100 * np.mean([r['back_rel_error'] for r in V]):.1f}')} %. "
        f"6 отложенных клипов, 100 шагов Эйлера; все числа — средние по шести. "
        f"Показан образец {a.show + 1} из шести: на первом настоящая сцена почти без контраста и по ней "
        f"ничего не видно, у розыгрышей выбор клетки ни на одно число не влияет. Локально, $0. {slow}.")
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 118), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print(f"{'плечо':26} {'sd прообраза':>13} {'радиус':>8} {'ровного розыгрыша':>19}")
    for t, head in ARMS:
        d = S[t]
        print(f"{head.replace(chr(10), ' / '):26} {d['noise']['sd_per_dim']:13.3f} "
              f"{d['noise']['radius_mean']:8.1f} {100 * fl[t]:18.1f}%")
    print("\nконтроль обращения (selfinv18):")
    for r in V:
        print(f"  {Path(r['prior_ckpt']).stem:34} вход {r['eps_sd']:.3f} → назад {r['back_sd']:.3f} "
              f"(радиус {r['back_radius']:.1f}, ошибка {100 * r['back_rel_error']:.1f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
