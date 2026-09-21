r"""22.0: что ещё можно убрать внутри T4 — время нельзя, поле зрения можно.

    python tools/fig_inside22.py

Шесть клеток: вход, полное состояние, половина T4, три направления из
четырёх, поле зрения обрезано до шестого кольца, и срез по времени. Последние
две задают почти одинаковое число координат, а выглядят противоположно —
в этом весь ответ.
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

SHOW = [("полное состояние", "13B из полного\nсостояния (потолок)"),
        ("T4/16/15", "половина T4\n4 направления"),
        ("T4a+T4b+T4c/16/15", "три направления\nиз четырёх"),
        ("T4/16/6", "поле зрения до\n6-го кольца из 15"),
        ("T4/8/15", "половина времени\n8 коэффициентов из 16")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="inside22")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_inside22")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.tag}.npz")
    i, G, D = a.show, S["groups"], S["dims"]
    frames = len(Z[f"raw__{i}"])

    cells = [(Z[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
              f"ровного {ru(f'{100 * S['raw']['frac_flat']:.1f}')} %")]
    for key, head in SHOW:
        v = G[key]
        zone = "" if v["arm"] is None or v["arm"]["rings"] >= 15 else f" на кольцах ≤ {v['arm']['rings']}"
        cells.append((Z[f"video__{key}|{i}"], head,
                      f"{v['given']} чисел ({ru(f'{100 * v['given'] / D:.1f}')} %)\n"
                      f"r {ru(f'{v['central_r']['own']:.3f}')}{zone}"))
    t4, d3 = G["T4/16/15"], G["T4a+T4b+T4c/16/15"]
    r6, t8 = G["T4/16/6"], G["T4/8/15"]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Внутри T4 дёшево режется пространство, дорого — время. Две последние клетки задают почти одинаковое "
        f"число координат ({r6['given']} против {t8['given']}), но обрезанное поле зрения сохраняет картинку на "
        f"том, что задано (r {ru(f'{r6['central_r']['own']:.3f}')} на кольцах ≤ 6 против "
        f"{ru(f'{t4['central_r']['6']:.3f}')} у полной T4 на той же зоне — потеря "
        f"{ru(f'{t4['central_r']['6'] - r6['central_r']['own']:.3f}')}), а срез времени рушит всё поле сразу "
        f"(r {ru(f'{t8['central_r']['own']:.3f}')} против {ru(f'{t4['central_r']['own']:.3f}')}). Порядок, в "
        f"котором стоит убирать: сначала T5 (даром, возвращается сам), потом поле зрения (это меньшая картинка, "
        f"а не худшая: 721 -> 331 колонка стоит {ru(f'{t4['central_r']['10'] - G['T4/16/10']['central_r']['own']:.3f}')} "
        f"на сохранённой зоне), потом одно направление из четырёх "
        f"({ru(f'{t4['central_r']['own']:.3f}')} -> {ru(f'{d3['central_r']['own']:.3f}')}), и НИКОГДА время. "
        f"Цепочка возвращает выброшенное тем хуже, чем ближе срез ко времени: состояние′ против настоящего "
        f"{ru(f'{t4['state_prime_vs_real']['err']:.4f}')} у половины T4, "
        f"{ru(f'{d3['state_prime_vs_real']['err']:.4f}')} у трёх направлений, "
        f"{ru(f'{t8['state_prime_vs_real']['err']:.4f}')} у среза времени при собственном поле цепочки "
        f"{ru(f'{G['полное состояние']['state_prime_vs_real']['err']:.4f}')}. ВАЖНО: у арм с кольцами r считан на "
        f"той зоне, которую арма задаёт — серую периферию, которую мы сами решили не задавать, штрафовать нечестно; "
        f"по всему полю та же клетка даёт {ru(f'{r6['central_r']['15']:.3f}')}. Шесть отложенных клипов из 10 "
        f"классов UCF101, не виденных обучением; маска 13B честно объявляет отсутствующие типы; один z на клип; "
        f"показан образец {i + 1} из шести. Локально, {ru(f'{S['seconds']:.0f}')} с, даром. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 165), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:20} {:>8} {:>6} {:>10} {:>11} {:>14}".format(
        "арма", "задано", "от D", "r всё поле", "r своя зона", "состояние′"))
    for k, v in G.items():
        print("{:20} {:8} {:5.1f}% {:10.3f} {:11.3f} {:9.4f}/{:.3f}".format(
            k, v["given"], 100 * v["given"] / D, v["central_r"]["15"], v["central_r"]["own"],
            v["state_prime_vs_real"]["err"], v["state_prime_vs_real"]["r"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
