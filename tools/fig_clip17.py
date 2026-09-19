"""17.3: the new prior applied to one known clip — one row, full resolution.

    python tools/fig_clip17.py --run data/prior17 --name clip17_local

"What the eye saw" first, then 13B from the clip's real state, then the same
state taken to noise level t and finished by the prior (t = 0.8 → 0.3: the
further it is taken, the more the prior rewrites), and last a state the prior
drew from pure noise, which never saw this clip.
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
    p.add_argument("--run", default=str(ROOT / "data" / "prior17"))
    p.add_argument("--name", default="clip17_local")
    p.add_argument("--prefix", default="2026-09-20_malecns_prior17b_clipA")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    S = json.loads((run / f"{a.name}.json").read_text(encoding="utf-8"))
    z = np.load(run / f"{a.name}.npz")
    sc, frames = S["scores"], S["frames"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    def cell(name, head):
        s = sc[name]
        return (z[f"video__{name}"], head,
                f"прогонка {s['round_trip']:.3f}, r к клипу {s['r_video_to_clip']:+.2f}")

    cells = [(z["clip"], f"что видел глаз: клип А (sample {S['sample']})", "вход")]
    cells.append(cell("real", "13B ← настоящее состояние"))
    for t in S["levels"]:
        cells.append(cell(f"refined_t{t:g}", f"прайор дописал с t={t:g}"))
    cells.append(cell("prior_0", "прайор из чистого шума"))
    slow = f"{frames} кадров по 20 мс (0.8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = ("17.3: новый прайор (Sintel, DCT-16) на знакомом клипе. Состояние клипа зашумляется до уровня t и достраивается "
             f"прайором: t=0.8 — почти исходное, t=0.3 — от клипа остаётся только грубая структура. {slow}.")
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
