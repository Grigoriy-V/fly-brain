r"""22: три способа сжать блок против простой PCA, при сравнимом бюджете.

    python tools/fig_stage22.py

Клетки берутся из разных прогонов, но клипы во всех одни и те же (один сид
выбора), и 13B получает один z на клип в каждом: группы ровно по шесть, а
пачка равна группе (ISS-0009), поэтому клетки сравнимы.
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

# (файл прогона, имя группы, заголовок клетки)
CELLS = [("latent22real", "T4a+T4b без сжатия", "блок без сжатия"),
         ("latent22real", "k = 2048", "PCA 2048"),
         ("chan22", "каналов 4", "каналы 32 в 4\n(22.3)"),
         ("resid22", "PCA 1536 + остаток 4", "PCA 1536 + остаток\n(22.4)"),
         ("blockae22", "автоэнкодер 241x8", "автоэнкодер, решётка\n721 в 241 (22.5)")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_stage22")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    i = a.show
    S0 = json.loads((run / "latent22real.json").read_text(encoding="utf-8"))
    Z0 = np.load(run / "latent22real.npz")
    frames = len(Z0[f"raw__{i}"])
    cells = [(Z0[f"raw__{i}"], f"сырое видео корпуса\nклип {S0['clip_idx'][i]}",
              f"ровного {ru(f'{100 * S0['raw']['frac_flat']:.1f}')} %")]
    got = {}
    for tag, key, head in CELLS:
        S = json.loads((run / f"{tag}.json").read_text(encoding="utf-8"))
        Z = np.load(run / f"{tag}.npz")
        nums = ({f"каналов {m}": m * 721 for m in S.get("channels", [])}
                | {f"k = {k}": k for k in S["ks"]} | S.get("residual_numbers", {}))
        v = S["groups"][key]
        n = int(nums.get(key, S["block_numbers"]))
        got[head.split("\n")[0]] = (n, v["r_to_raw"], v["frac_flat"])
        cells.append((Z[f"video__{key}|{i}"], head,
                      f"{n} чисел\nr {ru(format(v['r_to_raw'], '.3f'))}, "
                      f"ровного {ru(format(100 * v['frac_flat'], '.1f'))} %"))
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Три попытки обойти простую PCA как первую ступень — ни одна не обошла. Бюджет сопоставим (1 928 - "
        f"4 420 чисел против 2 048 у PCA), клипы и шум 13B одни и те же. PCA-2048 даёт r "
        f"{ru(f'{got['PCA 2048'][1]:.3f}')}; сжатие каналов в колонке — {ru(f'{got['каналы 32 в 4'][1]:.3f}')} "
        f"при большем бюджете (22.3); обучаемый локальный остаток поверх замороженной PCA-1536 — "
        f"{ru(f'{got['PCA 1536 + остаток'][1]:.3f}')} при вдвое большем (22.4); автоэнкодер, прореживающий "
        f"решётку 721 в 241 и поднимающий каналы, — {ru(f'{got['автоэнкодер, решётка'][1]:.3f}')} (22.5). "
        f"Резкость не вернул никто: ровного поля 24-28 % у всех сжатий против "
        f"{ru(f'{100 * got['блок без сжатия'][2]:.1f}')} % у несжатого блока и "
        f"{ru(f'{100 * S0['raw']['frac_flat']:.1f}')} % у сырого видео. Что это значит: выигрыш лежит не в "
        f"первой ступени. Оговорка, которая обязана стоять рядом: автоэнкодеры здесь маленькие (116 072 "
        f"параметра) и коротко обученные (8 000 шагов, 266 с на T4), их кривая по валидации ещё росла — "
        f"утверждение «нелинейная ступень не догонит PCA» этими прогонами НЕ доказано, доказано лишь, что за "
        f"такую цену не догоняет. Шесть отложенных клипов из 10 классов UCF101, не виденных обучением; маска "
        f"13B объявляет T4c, T4d и все T5 отсутствующими; показан образец {i + 1} из шести. Обучение на карте "
        f"0,025 + 0,044 доллара, рендеры локально. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 165), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:26} {:>9} {:>11} {:>9}".format("способ", "чисел", "r к сырому", "ровного"))
    for k, (n, r, f) in got.items():
        print("{:26} {:9} {:11.3f} {:8.1f}%".format(k, n, r, 100 * f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
