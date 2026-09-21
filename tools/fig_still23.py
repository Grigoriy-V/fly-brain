r"""23.1: что рисует 13B из состояния без движения — потолок идеи «статическая картинка».

    python tools/fig_still23.py
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

SHOW = [(1, "ТОЛЬКО DC\nдвижения нет"), (2, "2 коэффициента"), (4, "4 коэффициента"),
        (16, "полный блок\n(потолок)")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior23"))
    p.add_argument("--tag", default="still23")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_still23")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.tag}.npz")
    i, G, W = a.show, S["groups"], S["raw"]
    frames = len(Z[f"raw__{i}"])
    cells = [(Z[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
              f"контраст {ru(f'{W['sd']:.3f}')}, кадр-к-кадру {ru(f'{W['lag1']:.3f}')}")]
    for k, head in SHOW:
        key = f"коэффициентов времени {k} из 16"
        v = G[key]
        cells.append((Z[f"video__{key}|{i}"], head,
                      f"{S['numbers'][key]} чисел\nконтраст {ru(f'{v['sd']:.3f}')}, "
                      f"кадр-к-кадру {ru(f'{v['lag1']:.3f}')}"))
    dc, full = G["коэффициентов времени 1 из 16"], G["коэффициентов времени 16 из 16"]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Статической картинки в нашем состоянии нет, и это измерено, а не выведено. Вопрос был: если "
        f"движение пока не интересует, нельзя ли рисовать только нулевой коэффициент DCT — 1 442 числа "
        f"вместо 23 072 — и получать сцену как неподвижный кадр? Потолок идеи проверен на НАСТОЯЩИХ блоках "
        f"отложенных клипов: оставлены первые K коэффициентов времени, остальные обнулены (латент выбелен, "
        f"ноль это среднее корпуса), и 13B отрисовал результат. Из одного DC получается почти пустое поле: "
        f"контраст {ru(f'{dc['sd']:.3f}')} против {ru(f'{W['sd']:.3f}')} у сырого видео и "
        f"{ru(f'{full['sd']:.3f}')} у полного блока, то есть впятеро слабее. И оно даже НЕ статично — "
        f"связность кадр-к-кадру {ru(f'{dc['lag1']:.3f}')} против {ru(f'{W['lag1']:.3f}')} у сырого видео, "
        f"то есть меняется БОЛЬШЕ, чем настоящее видео. Причина ровно та, которую и следовало ожидать: T4 и "
        f"T5 — детекторы движения, и постоянный во времени сигнал T4 означает не неподвижную сцену, а "
        f"равномерный поток; 13B честно рисует слабую ползущую дымку. Два коэффициента уже дают контраст "
        f"{ru(f'{G['коэффициентов времени 2 из 16']['sd']:.3f}')}, четыре — "
        f"{ru(f'{G['коэффициентов времени 4 из 16']['sd']:.3f}')}, и только все шестнадцать — "
        f"{ru(f'{full['sd']:.3f}')}. ВЫВОД: дорога к неподвижной сцене лежит не через обнуление времени, а "
        f"через другое представление — ранние слои зрительной системы (L1-L5, Mi, Tm), которые несут "
        f"яркость и которые модель считает, но на которых 13B никогда не обучался. Шесть отложенных клипов "
        f"из 10 классов UCF101, маска 13B объявляет шесть типов из восьми отсутствующими, один z на клип в "
        f"каждой клетке; показан образец {i + 1} из шести. Локально, даром. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 165), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
