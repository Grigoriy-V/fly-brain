"""17.0 in one row: what 13B draws from z alone, beside the training video
nearest to it, with a real held-out clip and its nearest as the scale.

    python tools/fig_baseline17.py --run data/baseline17

Reads `<run>/{summary.json, baseline17.npz}`. Three unconditional samples are
chosen by their nearest-neighbour correlation (the highest — the worst case
for the novelty claim — and two typical), each shown beside the training video
it is closest to; the last pair is a real clip the generator never saw beside
*its* nearest training video, so the reader sees what "not a copy" looks like
for a genuine video.
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "baseline17"))
    p.add_argument("--prefix", default="2026-09-20_malecns_baseline17")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    S = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    z = np.load(run / "baseline17.npz", allow_pickle=True)
    frames = S["frames"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    def structured(v):                                # a flash or a blank frame makes an unreadable cell
        return float(np.asarray(v, np.float32).std(1).mean()) > 0.05

    samples = list(S["samples"].values())
    r = np.array([s["nn_r"] for s in samples])
    ok = [int(i) for i in np.argsort(r)[::-1] if structured(z["uncond_nn"][int(i)]) and structured(z["uncond"][int(i)])]
    pick = [ok[0], ok[len(ok) // 2]]                  # the worst case for the novelty claim, and a typical one
    reals = list(S["reals"].values())
    cand = [i for i, s in enumerate(reals) if s["label"].startswith("sintel") and structured(z["real_nn"][i])]
    rr = np.array([reals[i]["nn_r"] for i in cand])
    ir = cand[int(np.argmin(np.abs(rr - np.median(rr))))]

    cells = []
    for k in pick:
        s = samples[k]
        cells.append((z["uncond"][k], f"13B из одного z (#{k})", f"состояния на входе нет, mean {s['video_mean']:.2f}"))
        cells.append((z["uncond_nn"][k], f"ближайшее из обучения: {s['nn_label']}", f"r {s['nn_r']:+.2f}"))
    s = reals[ir]
    cells.append((z["real"][ir], f"реальный клип: {s['label'].split()[-1]}", "мерка: настоящее видео"))
    cells.append((z["real_nn"][ir], f"ближайшее из обучения: {s['nn_label']}", f"r {s['nn_r']:+.2f}"))

    slow = f"{frames} кадров по 20 мс (0.8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = ("17.0, базовая линия: 13B рисует видео из одного шума z, без состояния мозга. Рядом с каждым — самое похожее из "
             f"6 725 обучающих видео и корреляция с ним; справа та же пара для настоящего отложенного клипа. {slow}.")
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
