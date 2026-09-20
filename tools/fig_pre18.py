"""18.23: заданный сид работает, разыгранный — нет.

    python tools/fig_pre18.py

Приор переобучен на парах «собственный прообраз состояния, поставленный на
оболочку» вместо свежего случайного шума на каждый визит. Слева — сырое видео
корпуса. Вторая клетка — что выходит, если подать **назначенный этому клипу**
шум: радиус 303,8, то есть законный розыгрыш по геометрии. Третья и четвёртая —
**свежий** розыгрыш из нового приора и из прежнего.

Вся разница между второй и третьей клеткой — какую точку оболочки взяли.
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
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--assigned", default="assigned18_pre")
    p.add_argument("--reach-new", default="reach18_pre")
    p.add_argument("--reach-old", default="reach18_corpus_dct16_w192_lr1e3_c")
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--show", type=int, default=0)
    p.add_argument("--prefix", default="2026-09-20_malecns_pre18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    A = json.loads((run / f"{a.assigned}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.assigned}.npz")
    N = json.loads((run / f"{a.reach_new}.json").read_text(encoding="utf-8"))
    O = json.loads((run / f"{a.reach_old}.json").read_text(encoding="utf-8"))
    ZN = np.load(run / f"{a.reach_new}.npz")
    ZO = np.load(run / f"{a.reach_old}.npz")

    i = a.show
    rec = A["groups"]["назначенный шум"][i]
    drw = A["groups"]["свежий розыгрыш"][i]
    bank = np.load(Path(a.corpus) / "videos.npz")["videos"]
    frames = len(Z[f"video__назначенный шум|{i}"])
    raw = np.asarray(bank[rec["expected_clip"]][:frames], np.float32)

    def flat(d, g):
        return 100 * d["groups"][g]["structure"]["frac_flat"]["mean"]

    hits = sum(w["hit"] for w in A["groups"]["назначенный шум"])
    n = len(A["groups"]["назначенный шум"])
    cells = [
        (raw, f"сырое видео корпуса\nклип {rec['expected_clip']}", "то, чего мы хотим"),
        (Z[f"video__назначенный шум|{i}"], "из НАЗНАЧЕННОГО\nему шума",
         f"r к сырому {ru(f'{rec['r_expected']:.3f}')}, попал {hits}/{n}"),
        (Z[f"video__свежий розыгрыш|{i}"], "свежий розыгрыш,\nпереставленные пары",
         f"ровного {ru(f'{flat(N, 'обычный розыгрыш'):.1f}')} %, "
         f"sd прообраза {ru(f'{N['noise']['sd_per_dim']:.3f}')}"),
        (ZO[f"video__обычный розыгрыш|{i}"], "свежий розыгрыш,\nпрежняя случайная пара",
         f"ровного {ru(f'{flat(O, 'обычный розыгрыш'):.1f}')} %, "
         f"sd прообраза {ru(f'{O['noise']['sd_per_dim']:.3f}')}"),
    ]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Пары наконец выучились — но только те, на которых учили. Назначенный клипу шум лежит на радиусе 303,8, "
        f"то есть законен для сэмплера, и возвращает свой клип {hits} раз из {n} при r = "
        f"{ru(f'{np.mean([w['r_expected'] for w in A['groups']['назначенный шум']]):.3f}')} "
        f"(второй ближайший из 13 555 — {ru(f'{np.mean([w['r_second'] for w in A['groups']['назначенный шум']]):.2f}')}); "
        f"у 18.20 с произвольными метками было 0 из 8. Тренировочный лосс 0,0289 против 0,2131 у случайной сцепки. "
        f"Но обобщения нет: на отложенных прообраз сдвинулся лишь с {ru(f'{O['noise']['sd_per_dim']:.3f}')} до "
        f"{ru(f'{N['noise']['sd_per_dim']:.3f}')} при цели 1,000, а свежий розыгрыш стал даже беднее — "
        f"{ru(f'{flat(N, 'обычный розыгрыш'):.1f}')} % против {ru(f'{flat(O, 'обычный розыгрыш'):.1f}')} % "
        f"и {ru(f'{flat(N, 'настоящее состояние'):.1f}')} % у настоящего состояния. "
        f"Модель выучила 13 555 конкретных пар, а не отображение. "
        f"Обучающие клипы, один z у 13B; показан образец {i + 1} из восьми. Локально, $0. {slow}.")
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 132), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print(f"попаданий назначенного шума: {hits}/{n}, r {np.mean([w['r_expected'] for w in A['groups']['назначенный шум']]):.3f}")
    print(f"свежий розыгрыш:             {sum(w['hit'] for w in A['groups']['свежий розыгрыш'])}/{n}, "
          f"r {np.mean([w['r_expected'] for w in A['groups']['свежий розыгрыш']]):.3f}")
    print(f"прообраз отложенных: sd {O['noise']['sd_per_dim']:.3f} -> {N['noise']['sd_per_dim']:.3f} (цель 1,000), "
          f"радиус {O['noise']['radius_mean']:.1f} -> {N['noise']['radius_mean']:.1f} (цель 303,8)")
    print(f"розыгрыш, ровного:   {flat(O, 'обычный розыгрыш'):.1f}% -> {flat(N, 'обычный розыгрыш'):.1f}% "
          f"(настоящее {flat(N, 'настоящее состояние'):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
