"""Из шума в видео: шум настоящего клипа против обычного розыгрыша.

    python tools/fig_fromseed18.py

Человек, 2026-09-20: «я хочу получить гиф, где из шума сгенерится видео,
которое максимально близко к реальному».

Цепочка одна и та же во всех клетках, кроме первой: **шум → (приор, 100 шагов
Эйлера) → состояние → (13B, 20 шагов) → видео**. Меняется только шум.

- «шум этого клипа» — прообраз ε*, полученный обращением потока приора из
  настоящего состояния (18.17). Это и есть «сид» данного видео.
- «обычный розыгрыш» — torch.randn, то, что приор выдаёт сам по себе.

Первая клетка — настоящий клип корпуса, кадр в кадр (состояние считается с
того же серого покоя, без сдвига). Вторая — 13B прямо из настоящего
состояния, то есть потолок генератора без участия приора.

Показан клип с индексом 0 из четырёх, не лучший и не худший; r усреднён по
всем четырём. Массивы взяты из готового прогона walk18, ничего не считается
заново.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flydream.generate.invert import pixcorr_per_frame  # noqa: E402
from tools.fig_gen13b_pick import row  # noqa: E402

CELLS = [("real", "13B из настоящего\nсостояния"),
         ("1.0", "из шума этого клипа"),
         ("0.0", "из обычного розыгрыша")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--tag", default="walk18_local")
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--show", type=int, default=0)
    p.add_argument("--prefix", default="2026-09-20_malecns_fromseed18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    z = np.load(run / f"{a.tag}.npz")
    idx = S["clip_idx"]
    frames = len(z[f"video__real|0"])
    bank = np.load(Path(a.corpus) / "videos.npz")["videos"]
    ref = np.asarray(bank[idx][:, :frames], np.float32)               # кадр в кадр, t_pre — прогрев с серого

    r = {k: np.array([pixcorr_per_frame(np.asarray(z[f"video__{k}|{i}"], np.float32), ref[i]).mean()
                      for i in range(len(idx))]) for k, _ in CELLS}
    rows = {w["alpha"]: w for w in S["rows"]}
    flat = {"real": rows[None]["frac_flat"], "1.0": rows[1.0]["frac_flat"], "0.0": rows[0.0]["frac_flat"]}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    i = a.show
    cells = [(ref[i], f"что видел глаз:\nклип {idx[i]}", "настоящее видео корпуса")]
    for k, head in CELLS:
        cells.append((np.asarray(z[f"video__{k}|{i}"], np.float32), head,
                      f"r к настоящему {ru(f'{r[k][i]:+.2f}')}, ровного {ru(f'{100 * flat[k]:.1f}')} %"))
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (f"Шум → состояние → видео. Если шум взят как прообраз настоящего клипа, из него выходит "
             f"это же видео: r = {ru(f'{r['1.0'].mean():+.2f}')} по 4 клипам против "
             f"{ru(f'{r['0.0'].mean():+.2f}')} у обычного розыгрыша и {ru(f'{r['real'].mean():+.2f}')} "
             f"у 13B прямо из настоящего состояния. Приор умеет сцену — но такой шум лежит на "
             f"73 σ внутрь от оболочки, куда садится розыгрыш, и сам он его не выдаст. "
             f"Отложенные клипы, K=16/w384, локально, $0. {slow}.")
    import textwrap
    title = textwrap.fill(title, 118)                                 # иначе одна строка растягивает холст
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=110)
    print(f"{'условие':>26} {'r к настоящему':>16} {'ровного':>9}")
    for k, head in CELLS:
        print(f"{head.replace(chr(10), ' '):>26} {r[k].mean():>+10.3f} ± {r[k].std():.3f} {100 * flat[k]:8.1f}%")
    print(f"  по клипам {idx}: " + "  ".join(f"{k}=" + ",".join(f"{v:+.2f}" for v in r[k]) for k, _ in CELLS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
