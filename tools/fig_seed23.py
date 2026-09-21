r"""23: сид-тест гекс-локального потока над блоком, против плоского над PCA.

    python tools/fig_seed23.py

Шкала калибрована полами 22.8: 0 — белый шум, 100 — сырое видео корпуса.
Без неё «ровного 26,3 %» не читается никак.
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

SHOW = [("настоящее состояние блока", "13B из настоящего\nблока (потолок)"),
        ("сид от клипа через поток", "СИД ЭТОГО КЛИПА\nчерез поток"),
        ("свежий розыгрыш", "СВЕЖИЙ РОЗЫГРЫШ\nбез исходного клипа")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior23"))
    p.add_argument("--tag", default="seed23")
    p.add_argument("--floors", default=str(ROOT / "data" / "prior19" / "floors22.json"))
    p.add_argument("--old", default=str(ROOT / "data" / "prior19" / "seed22_ab1536.json"))
    p.add_argument("--accept", default=str(ROOT / "data" / "prior23" / "accept23.json"))
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_seed23")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.tag}.npz")
    F = json.loads(Path(a.floors).read_text(encoding="utf-8"))
    O = json.loads(Path(a.old).read_text(encoding="utf-8"))
    A = json.loads(Path(a.accept).read_text(encoding="utf-8"))
    lo, hi = F["arms"]["белый шум"], F["arms"]["настоящий клип"]
    cal = lambda v: 100 * (v - lo["frac_flat"]) / (hi["frac_flat"] - lo["frac_flat"])   # noqa: E731

    i, G = a.show, S["groups"]
    frames = len(Z[f"raw__{i}"])
    cells = [(Z[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
              f"ровного {ru(f'{100 * S['raw']['frac_flat']:.1f}')} %, шкала 100")]
    for key, head in SHOW:
        v = G[key]
        cells.append((Z[f"video__{key}|{i}"], head,
                      f"r {ru(format(v['r_to_raw'], '.3f'))}, "
                      f"резкость {ru(f'{cal(v['frac_flat']):.0f}')} из 100"))
    drw, old = G["свежий розыгрыш"], O["groups"]["свежий розыгрыш"]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Локальная архитектура сдвинула то, чего не сдвинуло уменьшение объекта. Поток работает над самим "
        f"блоком 721 × 32 с целой решёткой: 241 патч по три колонки, внимание только на шесть соседей, "
        f"относительное смещение вместо выученной позиции, 4 служебных токена как глобальный канал. "
        f"ЧТО ВЫИГРАНО. Сид клипа возвращает его клип за {ru(f'{G['сид от клипа через поток']['r_to_raw']:.3f}')} — "
        f"это РОВНО потолок ({ru(f'{G['настоящее состояние блока']['r_to_raw']:.3f}')}), потому что ступени PCA "
        f"больше нет и терять нечего; у прежнего потока над PCA-1536 тот же сид давал 0,809. Свежий розыгрыш: "
        f"резкость {ru(f'{cal(drw['frac_flat']):.0f}')} из 100 против {ru(f'{cal(old['frac_flat']):.0f}')} у "
        f"прежнего — вдвое; покоординатный эксцесс латента {ru(f'{A['draw']['kurtosis']:.2f}')} против "
        f"{ru(f'{A['training_subsamples']['kurtosis']['mean']:.2f}')} ± "
        f"{ru(f'{A['training_subsamples']['kurtosis']['sd']:.2f}')} у обучающих и "
        f"{ru(f'{A['gaussian']['kurtosis']:.2f}')} у гауссианы, то есть пройдено "
        f"{ru(f'{100 * A['kurtosis_closed']:.0f}')} % против 21 %; и главное — поток наконец ПЕРЕНОСИТ точку: "
        f"поворот {ru(f'{A['transport']['degrees']:.0f}')}° при 90° у случайного, тогда как прежний "
        f"поворачивал на 19° и потому выдавал наложения. ЧТО НЕ ВЫИГРАНО: сцены по-прежнему нет. Резкость "
        f"{ru(f'{cal(drw['frac_flat']):.0f}')} из 100 — это меньше четверти пути, контраст "
        f"{ru(f'{drw['sd']:.3f}')} против {ru(f'{S['raw']['sd']:.3f}')} у сырого, а эксцесс лежит ниже всего "
        f"диапазона сорока подвыборок обучающих. Копией розыгрыш не стал: ближайшее из 15 514 "
        f"{ru(f'{drw['nearest_r']:.3f}')} при {ru(f'{F['arms']['настоящий клип']['nearest_r']:.3f}')} у "
        f"настоящего отложенного клипа против того же банка. Колонка r у розыгрыша не показательна — её пол, "
        f"измеренный между двумя разными настоящими клипами, {ru('0,002')} ± {ru('0,100')} (22.8). Обучение "
        f"3 047 с на T4 при загрузке карты 99,9 %, 0,51 доллара; приёмка локально, даром. Шесть отложенных "
        f"клипов из 10 классов UCF101, маска 13B объявляет шесть типов из восьми отсутствующими; один z на "
        f"клип в каждой клетке; показан образец {i + 1} из шести. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 165), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:30} {:>11} {:>10} {:>9} {:>10}".format("группа", "r к сырому", "резкость", "контраст",
                                                    "ближайшее"))
    for k, v in G.items():
        print("{:30} {:11.3f} {:8.0f}/100 {:9.3f} {:10.3f}".format(
            k, v["r_to_raw"], cal(v["frac_flat"]), v["sd"], v["nearest_r"]))
    print("{:30} {:>11} {:8.0f}/100 {:9.3f}".format("сырое видео корпуса", "—",
                                                    cal(S["raw"]["frac_flat"]), S["raw"]["sd"]))
    print("{:30} {:>11} {:8.0f}/100".format("прежний поток, розыгрыш", f"{old['r_to_raw']:.3f}",
                                            cal(old["frac_flat"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
