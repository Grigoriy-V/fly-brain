"""18.12: слышно ли метку вообще — один шум, одна модель, четыре силы.

    python tools/fig_labelclip18.py

Крайняя левая ячейка — то же самое поле с **нулевой меткой**, то есть модель,
которая метку не слышит вовсе; дальше та же модель с меткой при силе 0,5, 1
(как обучено) и 2. Шум один и тот же во всех четырёх, и 13B рисует с одним и
тем же z, так что между ячейками меняется ровно одно — насколько слушается
метка. Подпись под каждой — прогонка этого сэмпла и его корреляция с первой
ячейкой: если метка ничего не несёт, все четыре видео совпадут.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fig_gen13b_pick import row  # noqa: E402

CKPT = "corpus_dct16_w192_lr1e3_cls_g_c"
CELLS = [(0.001, "без метки\n(нулевой класс)"), (0.5, "метка, сила 0,5"),
         (1.0, "метка, сила 1\n(как обучено)"), (2.0, "метка, сила 2")]


def tag_of(g: float) -> str:
    return f"samples18_{CKPT}_g{g:g}".replace(".", "p")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--prefix", default="2026-09-20_malecns_labelclip18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    E = json.loads((run / "edges18_guidance.json").read_text(encoding="utf-8"))
    stat = {r["guidance"]: r for r in E["rows"]}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    key = f"prior_{a.sample}"
    vids, cells, rts = [], [], []
    for g, head in CELLS:
        S = json.loads((run / f"{tag_of(g)}.json").read_text(encoding="utf-8"))
        z = np.load(run / f"{tag_of(g)}.npz")
        v = z[f"video__{key}"]
        vids.append(v); rts.append(S["scores"][key]["round_trip"])
    base = vids[0].reshape(-1)
    for i, ((g, head), v) in enumerate(zip(CELLS, vids)):
        r = float(np.corrcoef(base, v.reshape(-1))[0, 1])
        foot = f"прогонка {ru(f'{rts[i]:.3f}')}"
        foot += "" if i == 0 else f", r к 1-й {ru(f'{r:+.3f}')}"
        cells.append((v, head, foot))

    rr = [float(np.corrcoef(base, v.reshape(-1))[0, 1]) for v in vids[1:]]
    frames = int(json.loads((run / f"{tag_of(1.0)}.json").read_text(encoding="utf-8"))["frames"])
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    s0, s1 = stat[0.001], stat[1]
    title = (f"Метку модель слышит, но она не несёт ничего: без метки и при полной силе видео совпадают "
             f"на r {ru(f'{rr[1]:+.3f}')}, а структура не отличается вовсе — "
             f"ровных переходов {ru(f'{100 * s0['frac_flat']:.1f}')} % против "
             f"{ru(f'{100 * s1['frac_flat']:.1f}')} %, эксцесс {ru(f'{s0['kurtosis']:.2f}')} против "
             f"{ru(f'{s1['kurtosis']:.2f}')}. Прайор K=16, ширина 192, 110 классов + нулевая метка. "
             f"Один шум, один z у 13B; 16 сэмплов на силу, ворота локально, $0. {slow}.")
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=80)
    print("  r к первой ячейке: " + ", ".join(f"{g:g} -> {r:+.4f}" for (g, _), r in zip(CELLS[1:], rr)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
