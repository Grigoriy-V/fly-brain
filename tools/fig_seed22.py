r"""22.6: сид-тест над блоком T4a+T4b через PCA-1536.

    python tools/fig_seed22.py

Тот же тест, что 19.3, но объект вчетверо меньше: поток разыгрывает 1 536
координат блока вместо 2 048 координат полного состояния, а 13B получает
честную маску «даны только T4a и T4b».

Ворота здесь НЕ показаны намеренно: они меряют расстояние до заказанного
состояния, в котором шесть типов из восьми объявлены отсутствующими, то есть
до состояния, которого не бывает (21в). Судья — r к сырому видео, доля
ровного поля, контраст и расстояние до ближайшего обучающего видео.
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

SHOW = [("настоящее состояние", "13B из настоящего\nблока (потолок)"),
        ("только PCA (19.1)", "только первая ступень\nPCA 1536 туда-обратно"),
        ("сид от клипа через поток", "СИД ЭТОГО КЛИПА\nчерез поток"),
        ("свежий розыгрыш", "СВЕЖИЙ РОЗЫГРЫШ\nбез поправки"),
        ("розыгрыш + sdproj", "розыгрыш + поправка\nсэмплера (20)")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="seed22_ab1536")
    p.add_argument("--steps", default="100")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_seed22")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.tag}.npz")
    i, G = a.show, S["groups"]
    frames = len(Z[f"raw__{i}"])
    pre, dat = S["preimage"][a.steps], S["latent_data"]

    cells = [(Z[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
              f"ровного {ru(f'{100 * S['raw']['frac_flat']:.1f}')} %")]
    for key, head in SHOW:
        v = G[key]
        cells.append((Z[f"video__{key}|{i}"], head,
                      f"r {ru(format(v['r_to_raw'], '.3f'))}, "
                      f"ровного {ru(format(100 * v['frac_flat'], '.1f'))} %"))
    drw, prj = G["свежий розыгрыш"], G["розыгрыш + sdproj"]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Объект уменьшен вчетверо — блокиратор не сдвинулся. Поток обучен над 1 536 координатами блока "
        f"T4a+T4b вместо 2 048 координат полного состояния (301 с на T4, 0,05 доллара), 13B получает честную "
        f"маску «даны только a и b». Что работает: сид РИСУЕМ — прообразы {S['n_test']} отложенных латентов "
        f"лежат на радиусе {ru(f'{pre['radius_mean']:.1f}')} ± {ru(f'{pre['radius_sd']:.2f}')} при оболочке "
        f"√k = {ru(f'{pre['typical_radius']:.1f}')}, эксцесс {ru(f'{pre['kurtosis_mean']:.2f}')}, замыкание "
        f"{ru(f'{100 * pre['round_trip_error']:.1f}')} %; и сид КОНКРЕТНОГО клипа возвращает его клип за "
        f"{ru(f'{G['сид от клипа через поток']['r_to_raw']:.3f}')} — ровно столько же, сколько даёт одна первая "
        f"ступень ({ru(f'{G['только PCA (19.1)']['r_to_raw']:.3f}')}), то есть поток ничего не теряет. Что НЕ "
        f"работает, третий раз подряд: свежий розыгрыш не сцена. r к сырому {ru(f'{drw['r_to_raw']:.3f}')}, "
        f"ровного поля {ru(f'{100 * drw['frac_flat']:.1f}')} % при {ru(f'{100 * S['raw']['frac_flat']:.1f}')} % у "
        f"сырого видео, контраст {ru(f'{drw['sd']:.3f}')} против {ru(f'{S['raw']['sd']:.3f}')}; поправка "
        f"сэмплера чинит контраст ({ru(f'{prj['sd']:.3f}')}), но не сцену. Копией розыгрыш не стал: ближайшее из "
        f"15 514 обучающих {ru(f'{drw['nearest_r']:.3f}')}. Сам латент по-прежнему не гауссов: ст. откл. "
        f"{ru(f'{dat['sd']:.3f}')}, радиус {ru(f'{dat['radius_mean']:.1f}')} ± "
        f"{ru(f'{dat['radius_sd']:.2f}')}. ВЫВОД: уменьшение содержания вчетверо задачу потока не изменило — "
        f"размерность латента задаётся сложностью данных, а не размером объекта. Ворота намеренно не показаны: "
        f"они мерят расстояние до заказа, где шесть типов из восьми объявлены отсутствующими (21в). "
        f"Шесть отложенных клипов из 10 классов UCF101, не виденных обучением; один z на клип; показан "
        f"образец {i + 1} из шести. Приёмка локально, даром. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 165), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:28} {:>11} {:>9} {:>9} {:>10}".format("группа", "r к сырому", "ровного", "контраст", "ближайшее"))
    for k, v in G.items():
        print("{:28} {:11.3f} {:8.1f}% {:9.3f} {:10.3f}".format(
            k, v["r_to_raw"], 100 * v["frac_flat"], v["sd"], v["nearest_r"]))
    w = S["raw"]
    print("{:28} {:>11} {:8.1f}% {:9.3f}".format("сырое видео корпуса", "—", 100 * w["frac_flat"], w["sd"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
