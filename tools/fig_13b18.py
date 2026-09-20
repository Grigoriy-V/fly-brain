"""Вариант A: узкое место — 13B или прайор?

    python tools/fig_13b18.py

Три уровня в одном ряду, все в одном и том же пространстве фасеточного видео:

1. **оригинал** — то, что на самом деле видел глаз (клип корпуса);
2. **13B из настоящего состояния** — тот же клип, прогнанный через мозг в
   состояние и отрисованный 13B обратно: потолок отрисовщика;
3. **13B из сгенерированного состояния** — то, что делаем мы.

Если 2 — сцена, а 3 — текстура, виноват прайор. Если 2 тоже текстура, виноват
13B. Кадры состояния и видео выровнены один к одному (`simulate_states`
возвращает отклик кадр в кадр), так что сравнение поэлементное.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fig_cmp18 import pick  # noqa: E402
from tools.fig_gen13b_pick import row  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--name", default="samples18_corpus_dct16_w192_lr1e3_c")
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--prefix", default="2026-09-20_malecns_13b18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.name}.json").read_text(encoding="utf-8"))
    z = np.load(run / f"{a.name}.npz")
    sc, frames = S["scores"], S["frames"]
    clips = [n for n in sc if sc[n]["kind"] == "clip"][:2]
    cz = np.load(Path(a.corpus) / "videos.npz")
    bank = cz["videos"]
    labels = [json.loads(str(x)).get("label") or json.loads(str(x)).get("class") or "?" for x in cz["meta"]]

    cells = []
    for j, n in enumerate(clips):
        i = int(n.split("_")[1])
        rt = sc[n]["round_trip"]
        cells.append((np.asarray(bank[i][:frames], np.float32), f"оригинал {j + 1}", f"{labels[i]}, клип #{i}"))
        cells.append((z[f"video__{n}"], f"13B из настоящего состояния {j + 1}",
                      f"прогонка {ru(f'{rt:.3f}')}"))
    for j, n in enumerate(pick(S, [0.4, 0.6])):
        rt = sc[n]["round_trip"]
        cells.append((z[f"video__{n}"], f"13B из сгенерированного {j + 1}",
                      f"прогонка {ru(f'{rt:.3f}')}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    slow = f"{frames} кадров по 20 мс (0.8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = ("Где теряется сцена: оригинал → 13B из настоящего состояния → 13B из сгенерированного. "
             "Всё в одном пространстве фасеточного видео, кадры выровнены. "
             f"Если средние два — сцена, а правые два — нет, дело в прайоре, а не в отрисовщике. {slow}.")
    row(cells, title, Path(a.outdir) / a.prefix, frames, a.fps, plt, FuncAnimation, PillowWriter, dpi=80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
