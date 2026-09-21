r"""23.2: несёт ли цепочка одну статическую картинку. Валидный тест.

    python tools/fig_static23.py
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

CELLS = [("stim__неподвижный кадр", "СТИМУЛ:\nодин кадр, удержан"),
         ("video__неподвижный кадр", "13B из состояния\nэтого кадра"),
         ("stim__настоящий клип", "стимул: настоящий\nклип (контроль)"),
         ("video__настоящий клип", "13B из состояния\nклипа"),
         ("video__тот же кадр, плывущий", "13B: тот же кадр,\nплывущий по глазу")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior23"))
    p.add_argument("--tag", default="static23")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_static23")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.tag}.npz")
    i, G, W = a.show, S["groups"], S["stimulus"]
    frames = len(Z[f"stim__настоящий клип|{i}"])
    cells = []
    for key, head in CELLS:
        name, g = key.split("__", 1)
        v = (W if name == "stim" else G)[g]
        cap = (f"контраст {ru(f'{v['sd']:.3f}')}, кадр-к-кадру {ru(f'{v['lag1']:.3f}')}"
               if name == "stim" else
               f"r к стимулу {ru(f'{v['r_to_stimulus']:.3f}')}\n"
               f"контраст {ru(f'{v['sd']:.3f}')}, кадр-к-кадру {ru(f'{v['lag1']:.3f}')}")
        cells.append((Z[f"{key}|{i}"], head, cap))
    st, cl, dr = G["неподвижный кадр"], G["настоящий клип"], G["тот же кадр, плывущий"]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Одну статическую картинку цепочка несёт — и лучше, чем клип. Прошлый тест (23.1) был невалиден, и "
        f"человек это назвал: там у настоящего состояния обнулялись 15 коэффициентов DCT из 16, то есть 13B "
        f"получал состояние, какого не вызывал ни один клип. Здесь состояние не калечится — мозгу подаётся "
        f"стимул и снимается то, что он выдаёт. НЕПОДВИЖНЫЙ КАДР, удержанный всё окно, проходит цепочку "
        f"стимул → мозг → состояние → 13B за r {ru(f'{st['r_to_stimulus']:.3f}')} против "
        f"{ru(f'{cl['r_to_stimulus']:.3f}')} у настоящего клипа, контраст "
        f"{ru(f'{st['sd']:.3f}')} при {ru(f'{W['неподвижный кадр']['sd']:.3f}')} у самого стимула, и выход "
        f"действительно НЕПОДВИЖЕН: связность кадр-к-кадру {ru(f'{st['lag1']:.3f}')} при "
        f"{ru('1,000')} у стимула. Энергия состояния у неподвижного кадра такая же, как у клипа "
        f"(|T4a| {ru(f'{st['state_energy']['T4a']:.3f}')} против "
        f"{ru(f'{cl['state_energy']['T4a']:.3f}')}), так что довод «T4 детектор движения, значит статики в "
        f"нём нет» — неверен, и он был мой. Третья проверка, тот же кадр, равномерно плывущий по глазу на "
        f"шаг решётки в кадр: {ru(f'{dr['r_to_stimulus']:.3f}')} — хуже, сдвиг на такой скорости цепочке "
        f"даётся тяжелее неподвижности. ЧТО ЭТО ЗНАЧИТ: постановка «сцена как одна картинка, движение вне "
        f"задачи» осуществима на нынешнем отрисовщике, без переобучения 13B и без смены представления. "
        f"Потоку при этом не надо выдумывать движение — целая ось задачи уходит, а корпус умножается: "
        f"каждый кадр каждого клипа это отдельная статическая картинка. Шесть отложенных клипов из 10 "
        f"классов UCF101, маска 13B объявляет шесть типов из восьми отсутствующими; показан образец "
        f"{i + 1} из шести. Локально, 33 с, даром. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 165), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
