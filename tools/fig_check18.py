"""18.2: 13B on the new corpus beside 13B on Sintel — one row, full resolution.

    python tools/fig_check18.py --run data/corpus18 --name check18_local

Two clips of ordinary video and one Sintel clip, each beside what 13B draws
from that clip's own brain state, and last the unreachable control. The
question the row answers: does the generator trained on Sintel and procedural
stimuli render a state that came from ordinary video as well as it renders
its own?
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
    p.add_argument("--run", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--name", default="check18_local")
    p.add_argument("--prefix", default="2026-09-20_malecns_check18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--corpus-cells", type=int, default=2)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    S = json.loads((run / f"{a.name}.json").read_text(encoding="utf-8"))
    z = np.load(run / f"{a.name}.npz")
    frames = S["frames"]
    per = S["per_clip"]
    corpus = [i for i, r in enumerate(per) if r["kind"] == "corpus" and r.get("source", "video") == "video"]
    sintel = [i for i, r in enumerate(per) if r["kind"] == "sintel"]
    floor = np.median([per[i].get("contrast", 1.0) for i in sintel])
    legible = [i for i in sintel if per[i].get("contrast", 1.0) >= floor]          # a black scene shows nothing;
    sintel = legible or sintel                                                    # the score shown is still its own
    shuf = next(i for i, r in enumerate(per) if r["kind"] == "shuffled")
    ranked = sorted(corpus, key=lambda i: per[i]["round_trip"])                          # typical, not best:
    mid = len(ranked) // 2                                                               # the middle of the
    n = max(1, a.corpus_cells)                                                           # distribution, so the
    corpus = sorted(ranked[max(0, mid - n // 2):max(0, mid - n // 2) + n])               # row is not cherry-picked
    sin = sorted(sintel, key=lambda i: per[i]["round_trip"])[len(sintel) // 2]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    order = [k for k, r in enumerate(per) if r["kind"] == "corpus"]
    cells = []
    for i in corpus:
        j = order.index(i)
        cells.append((z["corpus_videos"][j].astype(np.float32), f"вход: {per[i]['label']} (UCF101)", "что видел глаз"))
        cells.append((z["corpus_rendered"][j].astype(np.float32), "13B ← состояние этого клипа",
                      f"прогонка {per[i]['round_trip']:.3f}"))
    js = [k for k, r in enumerate(per) if r["kind"] == "sintel"].index(sin)
    cells.append((z["sintel_videos"][js].astype(np.float32), f"вход: {per[sin]['label']}", "что видел глаз"))
    cells.append((z["sintel_rendered"][js].astype(np.float32), "13B ← состояние клипа Sintel",
                  f"прогонка {per[sin]['round_trip']:.3f}"))
    cells.append((z["shuffled_rendered"].astype(np.float32), "контроль: колонки перемешаны",
                  f"прогонка {per[shuf]['round_trip']:.3f}"))
    g = S["gates"]
    slow = f"{frames} кадров по 20 мс (0.8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (f"18.2: 13B на новом распределении. Медиана прогонки: корпус {g['corpus']['median']:.3f} "
             f"(n={g['corpus']['n']}), Sintel {g['sintel']['median']:.3f} (n={g['sintel']['n']}), "
             f"недостижимое состояние {g['shuffled']['median']:.3f}. {slow}.")
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
