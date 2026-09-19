"""17.2 in one row: states drawn from the prior, the video 13B makes of each,
and what the brain says about it — with a real clip's state as the reachable
reference and noise in the types as the unreachable control.

    python tools/fig_prior17.py --run data/prior17

Reads `<run>/<name>.{json,npz}` (the local run by default). Layout per `docs/ARTEFACTS.md`
for a state no video caused: the state itself (its T4a map) is the input
column, the generated video stands beside it, and under the video the round
trip and the direction the brain reads back.
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
    p.add_argument("--name", default="samples17_local")
    p.add_argument("--compare", default="", help="a second run tag: one row, both priors beside the same references")
    p.add_argument("--heads", default="прайор 17.1|прайор 17.1b")
    p.add_argument("--prefix", default="2026-09-20_malecns_prior17")
    p.add_argument("--title", default="")
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

    def state_map(name):                                   # T4a of the state, scaled to [0,1] for display
        m = z[f"T4a__{name}"][:frames]
        lo, hi = np.percentile(m, 1), np.percentile(m, 99)
        return np.clip((m - lo) / (hi - lo + 1e-6), 0, 1)

    def pair(name, head, note=""):
        s = sc[name]
        foot = (note + ", " if note else "") + f"направление {s['direction_state']}"
        return [(state_map(name), f"{head}: карта T4a", foot),
                (z[f"video__{name}"], f"{head}: видео 13B",
                 f"прогонка {s['round_trip']:.3f}, мозг читает {s['direction_video']}")]

    def best_of(scores):
        return sorted([n for n in scores if scores[n]["kind"] == "prior"], key=lambda n: scores[n]["round_trip"])

    clips = [n for n in sc if sc[n]["kind"] == "clip"]
    sintel = [n for n in clips if "Sintel" in sc[n].get("label", "")]
    ref = min(sintel or clips, key=lambda n: sc[n]["round_trip"])
    if a.compare:
        S2 = json.loads((run / f"{a.compare}.json").read_text(encoding="utf-8"))
        z2 = np.load(run / f"{a.compare}.npz")
        sc2 = S2["scores"]
        ha, hb = (a.heads.split("|") + ["", ""])[:2]

        def pair2(name, head, note=""):                                      # the same cell pair from the second run
            s2 = sc2[name]
            m = z2[f"T4a__{name}"][:frames]
            lo, hi = np.percentile(m, 1), np.percentile(m, 99)
            foot = (note + ", " if note else "") + f"направление {s2['direction_state']}"
            return [(np.clip((m - lo) / (hi - lo + 1e-6), 0, 1), f"{head}: карта T4a", foot),
                    (z2[f"video__{name}"], f"{head}: видео 13B",
                     f"прогонка {s2['round_trip']:.3f}, мозг читает {s2['direction_video']}")]

        cells = pair(best_of(sc)[0], ha) + pair2(best_of(sc2)[0], hb)
        ceil = [n for n in sc2 if sc2[n]["kind"] == "ceiling"]
        if ceil:
            cells += pair2(min(ceil, key=lambda n: sc2[n]["round_trip"]), "потолок DCT", "настоящее состояние, срезано")
        cells += pair(ref, "настоящее видео", sc[ref].get("label", "отложенный клип"))
    else:
        best = best_of(sc)
        mid = len(best) // 2                                                 # the middle of the distribution, not
        cells = pair(best[mid], "прайор, сэмпл 1") + pair(best[min(mid + 1, len(best) - 1)], "прайор, сэмпл 2") \
            + pair(ref, "настоящее видео", sc[ref].get("label", "отложенный клип")) \
            + pair("noise_white", "контроль: шум в типах")                   # ... the best two
    slow = f"{frames} кадров по 20 мс (0.8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    gates = S.get("gates", {})
    med = gates.get("prior", {}).get("round_trip", {}).get("median")
    ref_med = gates.get("clip", {}).get("round_trip", {}).get("median")
    head = a.title or ("17.2: состояние T4/T5, взятое из обученного прайора (видео на входе не было), и видео, которое 13B "
                       "из него делает. Прогонка — ошибка обратного прохода через мозг, меньше = совместимее")
    numbers = f" Медиана по 16 сэмплам {med:.3f}, у отложенного клипа {ref_med:.3f}." if med and ref_med else ""
    title = f"{head}.{numbers} Ячейки взяты из середины распределения, не лучшие. {slow}."
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
