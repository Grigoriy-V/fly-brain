r"""21а: до какого среза состояния видео держится — и что значит «не задавать».

    python tools/fig_cut21.py

Шесть клеток, и во всех состояние НАСТОЯЩЕЕ (клип прогнан через мозг), меняется
только то, какую часть состояния мы оставляем заданной. Вопрос человека был про
экономию: сколько чисел надо разыграть, чтобы видео ещё держалось.
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

SHOW = ["настоящее состояние", "types=T4+complete", "types=T4", "pca=128", "rings=10+complete"]
HEAD = {"настоящее состояние": "13B из полного\nсостояния (потолок)",
        "types=T4+complete": "ПОЛОВИНА ТИПОВ,\nвторая достроена",
        "types=T4": "та же половина,\nвторая ОБНУЛЕНА",
        "pca=128": "128 чисел вместо\n92 288 (PCA)",
        "rings=10+complete": "без внешних колец,\nдостроены"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="cut19")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_cut21")
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
    for key in SHOW:
        v = G[key]
        cells.append((Z[f"video__{key}|{i}"], HEAD[key],
                      f"задано {v['given']} ({ru(f'{100 * v['given'] / D:.1f}')} %)\n"
                      f"r {ru(f'{v['r_to_raw']:.3f}')}, ворота {ru(f'{v['gate']:.4f}')}"))
    full, t4c, t4z = G["настоящее состояние"], G["types=T4+complete"], G["types=T4"]
    p128, p2048, r10 = G["pca=128"], G["pca=2048"], G["rings=10+complete"]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Часть состояния можно не задавать — если её ДОСТРАИВАТЬ, а не обнулять. Половина типов (T4, четыре "
        f"направления из восьми) плюс достройка второй половины по PCA-подпространству даёт r "
        f"{ru(f'{t4c['r_to_raw']:.3f}')} и ворота {ru(f'{t4c['gate']:.4f}')} — это уровень полного пути через "
        f"первую ступень ({ru(f'{p2048['r_to_raw']:.3f}')} и {ru(f'{p2048['gate']:.4f}')} при 2 048 числах). "
        f"Та же половина, но ОБНУЛЁННАЯ, даёт r {ru(f'{t4z['r_to_raw']:.3f}')} — картинка даже ближе к клипу, — "
        f"но ворота {ru(f'{t4z['gate']:.4f}')}, в {ru(f'{t4z['gate'] / full['gate']:.0f}')} раз хуже полного "
        f"состояния: мозг не может породить ON-путь без OFF-пути, и такое состояние для него невозможно. "
        f"Для генератора считать надо не проценты состояния, а ЧИСЛА, которые придётся разыграть: 128 главных "
        f"компонент (0,1 % от 92 288) держат r {ru(f'{p128['r_to_raw']:.3f}')} при воротах "
        f"{ru(f'{p128['gate']:.4f}')} — ворота не хуже, чем у 2 048, падает только верность конкретному клипу. "
        f"Внешние кольца поля зрения тоже восстановимы: без них и с достройкой r {ru(f'{r10['r_to_raw']:.3f}')}, "
        f"ворота {ru(f'{r10['gate']:.4f}')}. Оговорки: 13B учился на полном состоянии, поэтому любой срез — вход, "
        f"которого он не видел; обнулённое в z-scored координатах значит «среднее корпуса», а не «темнота»; "
        f"невязка достройки на заданной половине 31 % для T4, то есть подпространство ранга 2 048 держит не всё. "
        f"Шесть отложенных клипов из 10 классов UCF101, не виденных обучением; один z у 13B во всех клетках; "
        f"показан образец {i + 1} из шести. Локально, даром. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 165), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:24} {:>9} {:>7} {:>11} {:>8} {:>9} {:>10}".format(
        "срез", "задано", "от D", "r к сырому", "ворота", "ровного", "ближайшее"))
    for k, v in G.items():
        print("{:24} {:9} {:6.1f}% {:11.3f} {:8.4f} {:8.1f}% {:10.3f}".format(
            k, v["given"], 100 * v["given"] / D, v["r_to_raw"], v["gate"],
            100 * v["frac_flat"], v["nearest_r"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
