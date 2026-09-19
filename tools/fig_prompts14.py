"""Item 14 pictures: one row per test, full resolution.

    python tools/fig_prompts14.py --run data/prompts14

Reads `<run>/summary.json` and `<run>/prompts14.npz`. Rows:
`_random` (14.0): T4a map of the state / generator video for white and
structured noise, beside clip A's own state and the shuffled control;
`_edits` (14.1): the right-moving grating, its state reversed in time,
its direction channels swapped, clip A with left/right halves from two
clips; `_prompts` (14.2): prompts assembled from the brain's responses and
the hand-written stripe; `_loop` (14.3): the closed loop from the
hand-written start, iterations 0, 1, 2, 11. Under each video its round
trip and the direction the brain reads in it; under each state map the
direction it carries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flydream.decode.hexraster import to_raster  # noqa: E402
from tools.fig_gen13b_pick import row  # noqa: E402

DIR_RU = {"a": "a (вперёд-назад)", "b": "b (назад-вперёд)", "c": "c (вверх)", "d": "d (вниз)"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prompts14"))
    p.add_argument("--prefix", default="2026-09-20_malecns_prompts14")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    S = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    z = np.load(run / "prompts14.npz")
    sc, frames = S["scores"], S["frames"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    outdir = Path(a.outdir)
    slow = f"{frames} кадров по 20 мс, в {1 / (a.fps * 0.02):.0f}× медленнее"

    def t4a(name):                                                 # the state's T4a map, scaled to [0,1] for display
        m = z[f"T4a__{name}"]; lo, hi = np.percentile(m, 1), np.percentile(m, 99)
        return np.clip((m - lo) / (hi - lo + 1e-6), 0, 1)

    def vid(name, seed=0):
        k = f"{name}__seed{seed}"; s = sc[k]
        return z[f"video__{k}"], f"прогонка {s['round_trip']:.3f}, мозг читает {s['direction_video']['T4_argmax']}"

    def st(name):
        return t4a(name), f"состояние: T4a, направление {sc[f'{name}__seed0']['direction_state']['T4_argmax']}"

    def pair(name, head):
        m, mt = st(name); v, vt = vid(name)
        return [(m, f"{head}: карта T4a", mt), (v, f"{head}: видео 13B", vt)]

    row(pair("e_A", "клип A") + pair("r0_white", "белый шум в T4/T5") + pair("r0_structured", "структурный шум") + pair("c_shuffled", "контроль: перемешано"),
        f"14.0, случайное состояние: 8 типов T4/T5 = шум → 13B → видео. Под видео — ошибка прогонки и направление, которое мозг читает в нём. {slow}.",
        outdir / f"{a.prefix}_random", frames, a.fps, plt, FuncAnimation, PillowWriter)
    cells = [(z["stim__right"], "стимул: решётка вправо", "вход")] + pair("e_right", "его состояние") + \
        pair("e_right_timerev", "состояние ⟵ время") + pair("e_right_swap_ab", "каналы a↔b") + pair("e_A_left_B_right", "A слева, B справа")
    row(cells, f"14.1, правки состояния: ни обмен каналов направлений, ни обращение во времени не переворачивают движение (оба состояния недостижимы, прогонка 2–5); составные состояния по областям — достижимы. {slow}.",
        outdir / f"{a.prefix}_edits", frames, a.fps, plt, FuncAnimation, PillowWriter)
    cells = pair("p_left_right_conflict", "слева →, справа ←") + pair("p_right_expand", "слева →, справа расширение") + \
        pair("p_window_up", "↑ только в окне") + pair("p_hand_T4a_stripe", "рукой: полоса T4a")
    row(cells, f"14.2, промпты без видео: состояния собраны из ответов мозга на стимулы по областям поля, последнее написано рукой. {slow}.",
        outdir / f"{a.prefix}_prompts", frames, a.fps, plt, FuncAnimation, PillowWriter)
    cells = []
    for name, head in (("loop_hand", "старт: полоса T4a"), ("loop_white", "старт: белый шум")):
        h = S["loops"][name]
        for it in (0, 1, 2, S["loop_iters"] - 1):
            cells.append((z[f"video__{name}__it{it}"], f"{head}, шаг {it}", f"прогонка {h[it]['round_trip']:.3f}, изменение {h[it]['state_change']:.3f}"))
    row(cells, f"14.3, замкнутый цикл состояние → 13B → видео → мозг → состояние′: неподвижная точка за 1–2 шага. {slow}.",
        outdir / f"{a.prefix}_loop", frames, a.fps, plt, FuncAnimation, PillowWriter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
