"""18.20: закреплённая пара шум↔состояние против обычной случайной.

    python tools/fig_couple18.py

Человек, 2026-09-20: «проблема в сидах, ты их сделал неправильно… пока я не
могу вбить сид в нойз2стейт руками и не получу видео как на этом клипе — всё
хуйня». Критерий задан им же: судить по клипу против **сырого видео
корпуса**, не по метрике.

Слева — настоящее видео корпуса, то, чего мы хотим. Дальше — обычные
розыгрыши N(0, I) через приор и 13B: у обоих плеч K=16 / ширина 192, та же
конфигурация, те же 20 000 шагов, один и тот же z у 13B. Отличается ровно
одно: как на обучении собиралась пара шум↔состояние.

- «случайная пара» — свежий ε на каждый визит, как было всегда;
- «закреплённая пара» — у каждого клипа один ε на весь прогон (`fixed_noise`).
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

ARMS = [("reach18_corpus_dct16_w192_lr1e3_c", "случайная пара\n(как было)"),
        ("reach18_pair", "закреплённая пара\n(18.20)")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-20_malecns_couple18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = {t: json.loads((run / f"{t}.json").read_text(encoding="utf-8")) for t, _ in ARMS}
    Z = {t: np.load(run / f"{t}.npz") for t, _ in ARMS}
    base = S[ARMS[0][0]]
    frames = len(Z[ARMS[0][0]][f"video__обычный розыгрыш|{a.show}"])
    bank = np.load(Path(a.corpus) / "videos.npz")["videos"]
    ref = np.asarray(bank[base["clip_idx"]][:, :frames], np.float32)

    def flat(d, g):
        return 100 * d["groups"][g]["structure"]["frac_flat"]["mean"]

    cells = [(ref[a.show], f"сырое видео корпуса\nклип {base['clip_idx'][a.show]}", "то, чего мы хотим"),
             (Z[ARMS[0][0]][f"video__настоящее состояние|{a.show}"], "13B из настоящего\nсостояния",
              f"ровного {ru(f'{flat(base, 'настоящее состояние'):.1f}')} %")]
    for t, head in ARMS:
        d = S[t]
        cells.append((Z[t][f"video__обычный розыгрыш|{a.show}"], head,
                      f"ровного {ru(f'{flat(d, 'обычный розыгрыш'):.1f}')} %, "
                      f"sd прообраза {ru(f'{d['noise']['sd_per_dim']:.3f}')}"))

    old, new = (S[t] for t, _ in ARMS)
    A = json.loads((run / "assigned18.json").read_text(encoding="utf-8"))["groups"]
    hits = sum(w["hit"] for w in A["назначенный шум"])
    r_as = np.mean([w["r_expected"] for w in A["назначенный шум"]])
    r_dr = np.mean([w["r_expected"] for w in A["свежий розыгрыш"]])
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Закреплённая пара шум↔состояние на обучении: у каждого клипа один ε на весь прогон вместо свежего "
        f"на каждый визит. Оба плеча K=16 / ширина 192, 20 000 шагов, один z у 13B, один сид розыгрыша. "
        f"Разреженность розыгрыша {ru(f'{flat(old, 'обычный розыгрыш'):.1f}')} % → "
        f"{ru(f'{flat(new, 'обычный розыгрыш'):.1f}')} % при "
        f"{ru(f'{flat(old, 'настоящее состояние'):.1f}')} % у настоящего состояния; "
        f"sd прообраза {ru(f'{old['noise']['sd_per_dim']:.3f}')} → {ru(f'{new['noise']['sd_per_dim']:.3f}')}. "
        f"Сцепка не взялась: назначенный клипу шум даёт этот клип в {hits} случаях из "
        f"{len(A['назначенный шум'])}, а корреляция с ожидаемым клипом {ru(f'{r_as:+.3f}')} против "
        f"{ru(f'{r_dr:+.3f}')} у свежего розыгрыша — то есть неотличимо. "
        f"Судить по картинке: похож ли розыгрыш на левую клетку. "
        f"6 отложенных клипов, 100 шагов Эйлера; показан образец {a.show + 1} из шести. "
        f"Выборка, рендер и оценка локально, $0. {slow}.")
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 118), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print(f"{'сцепка':26} {'sd прообраза':>13} {'радиус':>8} {'ровного розыгрыша':>19}")
    for t, head in ARMS:
        d = S[t]
        print(f"{head.replace(chr(10), ' '):26} {d['noise']['sd_per_dim']:13.3f} "
              f"{d['noise']['radius_mean']:8.1f} {flat(d, 'обычный розыгрыш'):18.1f}%")
    print(f"{'настоящее состояние':26} {'—':>13} {'—':>8} {flat(base, 'настоящее состояние'):18.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
