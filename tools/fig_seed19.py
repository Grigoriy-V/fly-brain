r"""19.3: сид над двухступенчатой цепочкой — заданный и разыгранный.

    python tools/fig_seed19.py

Пять клеток, и разница между ними только в том, откуда взялось состояние:
сырое видео корпуса; 13B из настоящего состояния (потолок цепочки); только
первая ступень (PCA туда и обратно); **сид этого клипа**, пропущенный через
поток назад и вперёд; и **свежий розыгрыш** ε ~ N(0, I) через ту же цепочку.
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

OLD_RADIUS, OLD_SHELL, OLD_SD = 252.4, 303.8, 0.833                   # пункт 17: прообразы над состояниями


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="seed19")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--steps", default="100")
    p.add_argument("--prefix", default="2026-09-21_malecns_seed19")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.tag}.npz")
    i = a.show
    frames = len(Z[f"raw__{i}"])
    pre = S["preimage"][a.steps]
    dat = S["latent_data"]
    G = S["groups"]
    ceil = G["настоящее состояние"]["r_to_raw"]

    order = [("настоящее состояние", "13B из настоящего\nсостояния (потолок цепочки)",
              lambda v: f"r {ru(f'{v['r_to_raw']:.3f}')}"),
             ("только PCA (19.1)", "только первая ступень\nPCA туда и обратно",
              lambda v: f"r {ru(f'{v['r_to_raw']:.3f}')}"),
             ("сид от клипа через поток", "СИД ЭТОГО КЛИПА\nчерез поток и обратно",
              lambda v: f"r {ru(f'{v['r_to_raw']:.3f}')}, ворота {ru(f'{v['gate']:.4f}')}"),
             ("свежий розыгрыш", "СВЕЖИЙ РОЗЫГРЫШ\nε ~ N(0, I)",
              lambda v: f"ближайшее из 15 514: {ru(f'{v['nearest_r']:.3f}')}, "
                        f"ворота {ru(f'{v['gate']:.4f}')}")]
    cells = [(Z[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
              f"ровного {ru(f'{100 * S['raw']['frac_flat']:.1f}')} %, "
              f"контраст {ru(f'{S['raw']['sd']:.3f}')}")]
    for key, head, sub in order:
        cells.append((Z[f"video__{key}|{i}"], head, sub(G[key])))
    d = G["свежий розыгрыш"]
    sc = S.get("draw_scale", 1.0)
    verdict = ("Он делает примерно половину переноса, то есть недоучен." if sc == 1.0 else
               f"Форму распределения он воспроизводит, а масштаб завышает, поэтому розыгрыш поправлен "
               f"в {ru(f'{sc:.3f}')} раза — это тот же приём, что scaling_factor у латентной диффузии. "
               f"Ворота от поправки улучшились, сценой розыгрыш не стал.")
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Сид стал разыгрываемым — впервые за всю линию. Прообразы {S['n_test']} ОТЛОЖЕННЫХ латентов лежат на "
        f"радиусе {ru(f'{pre['radius_mean']:.1f}')} ± {ru(f'{pre['radius_sd']:.2f}')} при оболочке "
        f"√k = {ru(f'{pre['typical_radius']:.1f}')} ± 0,71, ст. откл. по осям {ru(f'{pre['sd']:.3f}')}, эксцесс "
        f"{ru(f'{pre['kurtosis_mean']:.2f}')} против 3,0 у гауссова; обращение замыкается с ошибкой "
        f"{ru(f'{100 * pre['round_trip_error']:.1f}')} %. В пункте 17 над состояниями было "
        f"{ru(f'{OLD_RADIUS:.1f}')} при оболочке {ru(f'{OLD_SHELL:.1f}')} — на 73 σ внутрь. Поток сделал то, "
        f"ради чего он и нужен: сами данные имели эксцесс {ru(f'{dat['kurtosis_mean']:.2f}')} и разброс радиуса "
        f"{ru(f'{dat['radius_sd']:.2f}')}, прообразы — {ru(f'{pre['kurtosis_mean']:.2f}')} и "
        f"{ru(f'{pre['radius_sd']:.2f}')}. Третья клетка: сид, заданный клипом, проходит поток туда и обратно за "
        f"{ru(f'{G['сид от клипа через поток']['r_to_raw']:.3f}')} против "
        f"{ru(f'{G['только PCA (19.1)']['r_to_raw']:.3f}')} у одной PCA — вторая ступень не стоит почти ничего. "
        f"Четвёртая: свежий розыгрыш. Он НЕ копия — ближайшее из 15 514 обучающих видео "
        f"{ru(f'{d['nearest_r']:.3f}')} — и контраст у него {ru(f'{d['sd']:.3f}')}, выше, чем у сырого видео "
        f"({ru(f'{S['raw']['sd']:.3f}')}). Но сценой он не стал: ровного поля {ru(f'{100 * d['frac_flat']:.1f}')} % "
        f"против {ru(f'{100 * S['raw']['frac_flat']:.1f}')} %, а мозг за ним не поспевает — ворота "
        f"{ru(f'{d['gate']:.4f}')} против {ru(f'{G['настоящее состояние']['gate']:.4f}')} у настоящего "
        f"состояния, в {ru(f'{d['gate'] / G['настоящее состояние']['gate']:.1f}')} раза хуже. "
        f"И у этого есть адрес: на {S['draw_many']['n_draws']} розыгрышах поток "
        f"сдвигает точку лишь на {ru(f'{100 * S['draw_many']['moved']:.1f}')} % её нормы — радиус "
        f"{ru(f'{S['draw_many']['radius_start']:.1f}')} → {ru(f'{S['draw_many']['radius_mean']:.1f}')}, а нужно "
        f"было → {ru(f'{S['draw_many']['radius_needed']:.1f}')}, разброс {ru(f'{S['draw_many']['radius_sd']:.2f}')} "
        f"против {ru(f'{dat['radius_sd']:.2f}')} у данных. {verdict} "
        f"Шесть отложенных клипов из 10 классов UCF101, не виденных обучением; "
        f"один z у 13B во всех клетках; показан образец {i + 1} из шести. Локально, даром. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 150), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:28} {:>11} {:>9} {:>8} {:>9} {:>10}".format(
        "группа", "r к сырому", "% потолка", "ворота", "ровного", "ближайшее"))
    for k, v in G.items():
        print("{:28} {:11.3f} {:8.0f}% {:8.4f} {:8.1f}% {:10.3f}".format(
            k, v["r_to_raw"], 100 * v["r_to_raw"] / ceil, v["gate"], 100 * v["frac_flat"], v["nearest_r"]))
    print("\nпрообразы отложенных латентов против оболочки:")
    for st, v in S["preimage"].items():
        print(f"  {st:>3} шагов: радиус {v['radius_mean']:.1f} +- {v['radius_sd']:.2f} при "
              f"{v['typical_radius']:.1f} +- 0.71, ст. откл. {v['sd']:.3f}, эксцесс {v['kurtosis_mean']:.2f}")
    print(f"  сами данные:  радиус {dat['radius_mean']:.1f} +- {dat['radius_sd']:.2f}, "
          f"ст. откл. {dat['sd']:.3f}, эксцесс {dat['kurtosis_mean']:.2f}")
    print(f"  пункт 17:     радиус {OLD_RADIUS} при {OLD_SHELL}, ст. откл. {OLD_SD} — 73 сигмы внутрь")
    return 0


if __name__ == "__main__":
    sys.exit(main())
