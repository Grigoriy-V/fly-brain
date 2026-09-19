"""17.3b: the flow run backwards — one row, full resolution.

    python tools/fig_noise17.py --run data/prior17 --name noise17_local

Clip A at the left, clip B at the right, and between them the path through
the prior's noise space: each clip's state taken to its own noise and back,
then the two noises mixed on the sphere at 0.25 / 0.5 / 0.75. Last cell is the
control the scale is read against — clip A's state with its columns permuted,
a state the brain cannot reach.
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
    p.add_argument("--name", default="noise17_local")
    p.add_argument("--prefix", default="2026-09-20_malecns_prior17b_noise")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    S = json.loads((run / f"{a.name}.json").read_text(encoding="utf-8"))
    z = np.load(run / f"{a.name}.npz")
    sc, frames = S["scores"], S["frames"]
    scene = {k: v["scene"].split("_", 1)[1].split(" ")[0] for k, v in S["clips"].items()}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    def cell(name, head, both=True):
        s = sc[name]
        foot = (f"прогонка {s['round_trip']:.3f}, состояние: А {s['r_state_to_A']:+.2f} / B {s['r_state_to_B']:+.2f}"
                if both else f"прогонка {s['round_trip']:.3f}, r к своему клипу {max(s['r_state_to_A'], s['r_state_to_B']):+.2f}")
        return (z[f"video__{name}"], head, foot)

    cells = [(z["clip_A"], f"вход: клип А ({scene['A']})", "что видел глаз"),
             cell("A_through_noise", "А → шум → состояние", both=False)]
    for al in S["alphas"]:
        cells.append(cell(f"mix_{al:g}", f"смесь шума {1 - al:.2f}·А + {al:.2f}·B"))
    cells.append(cell("B_through_noise", "B → шум → состояние", both=False))
    cells.append((z["clip_B"], f"вход: клип B ({scene['B']})", "что видел глаз"))
    cells.append(cell("shuffled_A", "контроль: колонки А перемешаны"))
    slow = f"{frames} кадров по 20 мс (0.8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = ("17.3b: поток прайора, прогнанный назад. Состояние клипа переводится в свой шум и обратно (2-я и 7-я ячейки), "
             f"а шумы двух клипов смешиваются на сфере — путь между сценами, которого нет ни в одном клипе. {slow}.")
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
