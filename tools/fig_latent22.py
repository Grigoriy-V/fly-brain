r"""22.1: как выглядит клип при разном размере латента над блоком T4a+T4b.

    python tools/fig_latent22.py
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
    p.add_argument("--tag", default="latent22")
    p.add_argument("--cells", default="")
    p.add_argument("--title", default="")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_latent22")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.tag}.npz")
    i, G = a.show, S["groups"]
    frames = len(Z[f"raw__{i}"])
    base = f"{'+'.join(S['types'])} без сжатия"
    exp = {int(k): v for k, v in S["explained"].items()}

    cells = [(Z[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
              f"ровного {ru(f'{100 * S['raw']['frac_flat']:.1f}')} %"),
             (Z[f"video__{base}|{i}"], f"{'+'.join(S['types'])} целиком\nбез латента",
              f"{S['block_numbers']} чисел\nr {ru(f'{G[base]['r_to_raw']:.3f}')}")]
    nums = {f"каналов {m}": m * 721 for m in S.get("channels", [])} | {f"k = {k}": k for k in S["ks"]}
    nums |= {k: int(v) for k, v in S.get("residual_numbers", {}).items()}
    picked = [c.strip() for c in a.cells.split(";") if c.strip()] or [f"k = {k}" for k in S["ks"]]
    for key in picked:
        g = G[key]
        head = key if key.startswith("каналов") else f"латент {key}"
        cells.append((Z[f"video__{key}|{i}"], head,
                      f"{nums.get(key, S['block_numbers'])} чисел" + chr(10) +
                      f"r {ru(format(g['r_to_raw'], '.3f'))}, "
                      f"ровного {ru(format(100 * g['frac_flat'], '.1f'))} %"))
    ks = S["ks"]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Лестница размера латента над блоком T4a+T4b ({S['block_numbers']} чисел, все 16 коэффициентов, все 721 "
        f"колонка). Слева направо латент растёт, и r к сырому видео идёт "
        + ", ".join(f"{ru(f'{G[f'k = {k}']['r_to_raw']:.3f}')} при k = {k}" for k in ks)
        + f", против {ru(f'{G[base]['r_to_raw']:.3f}')} у того же блока без сжатия вовсе. Ни один размер латента "
        f"не возвращает несжатое: даже k = {ks[-1]} ({ru(f'{100 * exp[ks[-1]]:.0f}')} % отложенной дисперсии "
        f"блока) даёт {ru(f'{G[f'k = {ks[-1]}']['r_to_raw']:.3f}')} против {ru(f'{G[base]['r_to_raw']:.3f}')}. "
        f"Доля ровного поля "
        f"падает сразу и одинаково у всех k ({ru(f'{100 * G[f'k = {ks[0]}']['frac_flat']:.1f}')}–"
        f"{ru(f'{100 * G[f'k = {ks[-1]}']['frac_flat']:.1f}')} % против "
        f"{ru(f'{100 * G[base]['frac_flat']:.1f}')} % у несжатого и "
        f"{ru(f'{100 * S['raw']['frac_flat']:.1f}')} % у сырого видео): линейное сжатие срезает резкие края "
        f"независимо от числа координат. " + (
            f"Базис подогнан ПРЯМО НА БЛОКЕ на карте ({S['block_pca'].split('/')[-1]}, T4 34 с, 0,013 доллара), "
            f"доли дисперсии — отложенные; оценка из полного базиса давала те же r до третьего знака. "
            if S.get("block_pca") else
            f"ОГОВОРКА: базис блока взят из уже посчитанного PCA полного состояния, поэтому это оценка снизу. ")
        + f"Шесть отложенных клипов из 10 классов "
        f"UCF101, не виденных обучением; маска 13B объявляет T4c, T4d и все T5 отсутствующими; один z на клип; "
        f"показан образец {i + 1} из шести. Локально, {ru(f'{S['seconds']:.0f}')} с, даром. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(a.title or title, 165), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:24} {:>11} {:>9} {:>12}".format("латент", "r к сырому", "ровного", "дисперсии"))
    for k, v in G.items():
        kk = k.replace("k = ", "")
        e = exp.get(int(kk)) if kk.isdigit() else None
        print("{:24} {:11.3f} {:8.1f}% {:>12}".format(
            k, v["r_to_raw"], 100 * v["frac_flat"], f"{100 * e:.1f} %" if e else "—"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
