r"""19.5: сид, которого нет ни у одного клипа — и два контроля без потока.

    python tools/fig_interp19.py

Розыгрыш просит у модели точку рядом с выученным множеством, и поле там
экстраполирует: профиль по времени (19.4) показал, что траектория из
розыгрыша отрывается от идеала уже на t = 0,1 и приходит на 59 % выше.
Середина сферического пути между прообразами **двух разных отложенных
клипов** — это сид, которого нет ни у одного клипа, но лежащий на той же
оболочке, что настоящие прообразы, и внутри области, где поле выучено.

Контроли отвечают на отдельный вопрос: нужен ли поток вообще. Латент отбелён,
поэтому `z ~ N(0, I)` прямо в PCA-обратно — уже модель первого порядка; второй
контроль добавляет к ней верный тяжёлый хвост радиуса.
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

SHOW = [("настоящее состояние", "13B из настоящего\nсостояния (потолок)"),
        ("сид от клипа через поток", "сид ЭТОГО клипа\nчерез поток"),
        ("середина двух прообразов", "СЕРЕДИНА между сидами\nДВУХ клипов"),
        ("свежий розыгрыш", "свежий розыгрыш\nчерез поток"),
        ("контроль: N(0, I) без потока", "контроль: N(0, I)\nбез потока")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="seed19_ctrl")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_interp19")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.tag}.npz")
    i, G = a.show, S["groups"]
    frames = len(Z[f"raw__{i}"])
    mid, drw = G["середина двух прообразов"], G["свежий розыгрыш"]
    iso = G["контроль: N(0, I) без потока"]
    rad = G["контроль: радиус из данных"]
    ig = S["interp_geometry"]
    pair = ig["pairs"][i]

    ca, cb = S["clip_idx"][i], S["clip_idx"][pair]
    sk = "сид от клипа через поток"
    cells = [
        (Z[f"raw__{i}"], f"сырое видео, РОДИТЕЛЬ A\nклип {ca}", "то, что снято"),
        (Z[f"raw__{pair}"], f"сырое видео, РОДИТЕЛЬ B\nклип {cb}", "то, что снято"),
        (Z[f"video__{sk}|{i}"], "из сида A\nчерез поток", f"r к своему клипу {ru(f'{G[sk]['r_to_raw']:.3f}')}"),
        (Z[f"video__{sk}|{pair}"], "из сида B\nчерез поток", "тот же путь для B"),
        (Z[f"video__середина двух прообразов|{i}"], "СЕРЕДИНА сидов A и B\nклипа с таким сидом нет",
         f"ближайшее из 15 514: {ru(f'{G['середина двух прообразов']['nearest_r']:.3f}')}"),
        (Z[f"video__свежий розыгрыш|{i}"], "свежий розыгрыш\nдля сравнения",
         f"ворота {ru(f'{drw['gate']:.4f}')}"),
    ]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Сид, которого нет ни у одного клипа. Пятая клетка — середина сферического пути между сидами ДВУХ разных "
        f"отложенных клипов: точка на той же оболочке, где живут настоящие прообразы (радиус "
        f"{ru(f'{ig['сид']['radius_mean']:.1f}')} при √k = 45,3), но не принадлежащая ни одному клипу. Она попадает "
        f"в данные точно — латент на радиусе {ru(f'{ig['латент']['radius_mean']:.1f}')} при "
        f"{ru(f'{S['latent_data']['radius_mean']:.1f}')} у настоящих, — и даёт лучшую структуру среди всего "
        f"сгенерированного: ровного поля {ru(f'{100 * mid['frac_flat']:.1f}')} % против "
        f"{ru(f'{100 * drw['frac_flat']:.1f}')} % у розыгрыша при {ru(f'{100 * S['raw']['frac_flat']:.1f}')} % у "
        f"сырого видео, ворота {ru(f'{mid['gate']:.4f}')} против {ru(f'{drw['gate']:.4f}')}. И это НЕ копия: "
        f"ближайшее из 15 514 обучающих видео {ru(f'{mid['nearest_r']:.3f}')}. Розыгрыш (правая клетка) просит "
        f"точку РЯДОМ с выученным множеством, и поле там экстраполирует: 19.4 измерил, что его траектория "
        f"отрывается от идеала на t = 0,1 и приходит на 59 % выше. Нужен ли поток вообще: без него N(0, I) даёт "
        f"ворота {ru(f'{iso['gate']:.4f}')} и ровного {ru(f'{100 * iso['frac_flat']:.1f}')} %, а с верным тяжёлым "
        f"хвостом радиуса — {ru(f'{rad['gate']:.4f}')} и {ru(f'{100 * rad['frac_flat']:.1f}')} %; поток лучше "
        f"обоих, но не решает. ОГОВОРКА: у этой пары родитель B почти пуст, поэтому середина наследует раскладку A; "
        f"честная проверка новизны — пара из двух структурных клипов и r к каждому родителю отдельно. "
        f"Шесть отложенных клипов из 10 классов UCF101, не виденных обучением; один z у 13B во всех клетках; "
        f"показан образец {i + 1} из шести. Локально, даром. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 150), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:30} {:>11} {:>8} {:>9} {:>9} {:>10}".format(
        "группа", "r к сырому", "ворота", "ровного", "контраст", "ближайшее"))
    for k, v in G.items():
        print("{:30} {:11.3f} {:8.4f} {:8.1f}% {:9.3f} {:10.3f}".format(
            k, v["r_to_raw"], v["gate"], 100 * v["frac_flat"], v["sd"], v["nearest_r"]))
    w = S["raw"]
    print("{:30} {:>11} {:>8} {:8.1f}% {:9.3f}".format("сырое видео корпуса", "—", "—",
                                                       100 * w["frac_flat"], w["sd"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
