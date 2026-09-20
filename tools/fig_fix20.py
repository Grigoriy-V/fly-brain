r"""20: поправка сэмплера — что она починила и что нет.

    python tools/fig_fix20.py

Шесть клеток: сырое видео корпуса, потолок цепочки (13B из настоящего
состояния), только первая ступень, и три розыгрыша — без поправки, с проекцией
на измеренную кривую (`sdproj`) и с Epsilon Scaling из литературы
(`vscale`). Судья — ворота, раунд-трип через замороженный мозг; r к сырому
видео у розыгрыша равен нулю по построению, потому что розыгрыш не обязан
совпадать с каким-то одним отложенным клипом.
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="seed19_fix")
    p.add_argument("--sweep", default="fix19")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_fix20")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    W = json.loads((run / f"{a.sweep}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.tag}.npz")
    i = a.show
    frames = len(Z[f"raw__{i}"])
    G, F = S["groups"], S["fix"]["arms"]
    tr = W["train"]
    base, proj, vsc = G["свежий розыгрыш"], G["розыгрыш + sdproj"], G["розыгрыш + vscale=1.35"]
    gp, gv = F["sdproj"], F["vscale=1.35"]
    raw = S["raw"]

    cells = [
        (Z[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
         f"контраст {ru(f'{raw['sd']:.3f}')}"),
        (Z[f"video__настоящее состояние|{i}"], "13B из настоящего\nсостояния (потолок)",
         f"ворота {ru(f'{G['настоящее состояние']['gate']:.4f}')}"),
        (Z[f"video__только PCA (19.1)|{i}"], "только первая ступень\nPCA туда и обратно",
         f"ворота {ru(f'{G['только PCA (19.1)']['gate']:.4f}')}"),
        (Z[f"video__свежий розыгрыш|{i}"], "РОЗЫГРЫШ\nбез поправки",
         f"ворота {ru(f'{base['gate']:.4f}')}, контраст {ru(f'{base['sd']:.3f}')}"),
        (Z[f"video__розыгрыш + sdproj|{i}"], "РОЗЫГРЫШ + проекция\nна измеренную кривую",
         f"ворота {ru(f'{proj['gate']:.4f}')}, контраст {ru(f'{proj['sd']:.3f}')}"),
        (Z[f"video__розыгрыш + vscale=1.35|{i}"], "РОЗЫГРЫШ + Epsilon\nScaling (литература)",
         f"ворота {ru(f'{vsc['gate']:.4f}')}, контраст {ru(f'{vsc['sd']:.3f}')}"),
    ]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Сэмплер был частью проблемы, но не блокиратором. 19.4 измерил, что траектория свежего розыгрыша "
        f"отрывается от идеала на t = 0,1 и приходит с завышенной нормой; поправка, выведенная прямо из этого "
        f"измерения (после каждого шага масштаб партии приводится к идеальной кривой), геометрию починила "
        f"ПОЛНОСТЬЮ: ст. откл. {ru(f'{gp['sd']:.3f}')} при {ru(f'{tr['sd']:.3f}')} у обучающего латента, радиус "
        f"{ru(f'{gp['radius_mean']:.1f}')} при {ru(f'{tr['radius_mean']:.1f}')}, и контраст видео "
        f"{ru(f'{proj['sd']:.3f}')} совпал с сырым корпусом {ru(f'{raw['sd']:.3f}')} — против "
        f"{ru(f'{base['sd']:.3f}')} у непоправленного. Ворота улучшились в "
        f"{ru(f'{base['gate'] / proj['gate']:.1f}')} раза ({ru(f'{base['gate']:.4f}')} -> "
        f"{ru(f'{proj['gate']:.4f}')}), НО настоящее состояние даёт {ru(f'{G['настоящее состояние']['gate']:.4f}')}, "
        f"а одна первая ступень {ru(f'{G['только PCA (19.1)']['gate']:.4f}')}: сценой розыгрыш не стал. "
        f"Epsilon Scaling из литературы (Ning et al., ICLR 2024) в нашей параметризации НЕ работает: радиус "
        f"{ru(f'{gv['radius_mean']:.1f}')} вместо {ru(f'{tr['radius_mean']:.1f}')}, разброс радиусов сжат до "
        f"{ru(f'{gv['radius_sd']:.2f}')} при {ru(f'{tr['radius_sd']:.2f}')} у данных — он душит разнообразие, "
        f"а не масштаб, и ворота почти не двигает ({ru(f'{vsc['gate']:.4f}')}). Что осталось несовпавшим после "
        f"поправки: эксцесс {ru(f'{gp['kurtosis_mean']:.2f}')} против {ru(f'{tr['kurtosis_mean']:.2f}')} у "
        f"обучающего латента — поток гауссианизировал смесь масштабов, и это следующий диагноз. "
        f"Шесть отложенных клипов из 10 классов UCF101, не виденных обучением; один z у 13B во всех клетках; "
        f"показан образец {i + 1} из шести; 100 шагов Эйлера. Локально, даром. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 165), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:30} {:>11} {:>8} {:>9} {:>9} {:>10}".format(
        "группа", "r к сырому", "ворота", "ровного", "контраст", "ближайшее"))
    for k, v in G.items():
        print("{:30} {:11.3f} {:8.4f} {:8.1f}% {:9.3f} {:10.3f}".format(
            k, v["r_to_raw"], v["gate"], 100 * v["frac_flat"], v["sd"], v["nearest_r"]))
    print("\n{:24} {:>9} {:>9} {:>9} {:>9}".format("геометрия", "ст.откл.", "радиус", "разброс", "эксцесс"))
    for k, v in list(W["arms"].items()):
        print("{:24} {:9.3f} {:9.1f} {:9.2f} {:9.2f}".format(
            k, v["sd"], v["radius_mean"], v["radius_sd"], v["kurtosis_mean"]))
    print("{:24} {:9.3f} {:9.1f} {:9.2f} {:9.2f}".format(
        "обучающий латент", tr["sd"], tr["radius_mean"], tr["radius_sd"], tr["kurtosis_mean"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
