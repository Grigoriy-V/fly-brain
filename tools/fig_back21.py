r"""21в: канал состояния, который выбросили, возвращается из видео.

    python tools/fig_back21.py

Первая клетка — вход (сырое видео корпуса). Вторая — видео, которое 13B
нарисовал из состояния, где четыре канала T5 заменены на среднее корпуса.
Дальше три карты ОДНОГО канала T5c: какой он у настоящего состояния, каким мы
его заказали (ровным) и каким его вычитал мозг из нарисованного видео.
Последняя клетка — та же карта, когда ничего не резали: собственный пол
цепочки.

Карты z-scored, показаны как 0,5 + z/4 с обрезкой, чтобы ноль был серым.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fig_gen13b_pick import row  # noqa: E402


def shade(m: np.ndarray) -> np.ndarray:
    """z-scored карта -> яркость 0..1, ноль серый."""
    return np.clip(0.5 + np.asarray(m, np.float32) / 4.0, 0.0, 1.0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="back21")
    p.add_argument("--arm", default="types=T4")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_back21")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.tag}.npz")
    i, arm, t = a.show, a.arm, S["show_type"]
    frames = len(Z[f"raw__{i}"])
    A, F = S["по армам"][arm], S["по армам"]["полное"]
    cut, back = A["asked_vs_real"], A["state_prime_vs_real"]
    askd = A["state_prime_vs_asked"]

    cells = [
        (Z[f"raw__{i}"], f"вход: сырое видео\nклип {S['clip_idx'][i]}", "то, что снято"),
        (Z[f"video__{arm}|{i}"], f"видео 13B из состояния\nБЕЗ T5 ({arm})",
         f"r к сырому {ru(f'{A['r_video_to_raw']:.3f}')}"),
        (shade(Z[f"mapreal__{i}"]), f"{t}: как было\nу настоящего состояния", "эталон"),
        (shade(Z[f"mapask__{arm}|{i}"]), f"{t}: как мы заказали\nРОВНО, среднее корпуса",
         f"ошибка к настоящему {ru(f'{cut['среднее']['err']:.3f}')}"),
        (shade(Z[f"mapback__{arm}|{i}"]), f"{t}: что мозг вычитал\nиз этого видео",
         f"ошибка {ru(f'{back[t]['err']:.3f}')}, r {ru(f'{back[t]['r']:.3f}')}"),
        (shade(Z[f"mapback__полное|{i}"]), f"{t}: то же, когда\nничего не резали",
         f"пол цепочки {ru(f'{F['state_prime_vs_real'][t]['err']:.3f}')}"),
    ]
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Цепочка восстанавливает выброшенную половину состояния. Мы обнулили все четыре канала T5 "
        f"(46 144 числа из 92 288), 13B нарисовал по такому состоянию видео, и замороженный мозг прочитал "
        f"из этого видео состояние заново. Ошибка к НАСТОЯЩЕМУ состоянию: срез сделал "
        f"{ru(f'{cut['среднее']['err']:.4f}')} при корреляции {ru(f'{cut['среднее']['r']:.3f}')}, после цепочки "
        f"{ru(f'{back['среднее']['err']:.4f}')} при {ru(f'{back['среднее']['r']:.3f}')} — в "
        f"{ru(f'{cut['среднее']['err'] / back['среднее']['err']:.1f}')} раза ближе, при собственном поле цепочки "
        f"{ru(f'{F['state_prime_vs_real']['среднее']['err']:.4f}')} (арма без среза). Именно поэтому ворота 21а "
        f"показывали {ru(f'{askd['среднее']['err']:.3f}')}: они мерили расстояние до ЗАКАЗАННОГО ровного T5 "
        f"(по каналам {ru(f'{askd['T5a']['err']:.2f}')}-{ru(f'{askd['T5c']['err']:.2f}')}), а до настоящего у тех же "
        f"каналов {ru(f'{back['T5a']['err']:.2f}')}-{ru(f'{back['T5c']['err']:.2f}')}. Ворота были верны как "
        f"измерение и неверны как судья: они наказывали невозможность ЗАКАЗА, а не качество видео. "
        f"Граница находки: возвращается то, что определено видео. Кольца периферии возвращаются наполовину "
        f"(0,634 -> 0,255), а срез PCA до 128 компонент не возвращается вовсе (0,110 -> 0,127) — он убрал детали "
        f"из самой сцены, и брать их неоткуда. Шесть отложенных клипов из 10 классов UCF101, не виденных "
        f"обучением; один z у 13B на клип во всех армах; показан образец {i + 1} из шести. Локально, "
        f"{ru(f'{S['seconds']:.0f}')} с, даром. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 165), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:20} {:>9} {:>11} {:>13} {:>11} {:>10}".format(
        "арма", "задано", "к заказу", "к настоящему", "r к наст.", "r видео"))
    for k, v in S["по армам"].items():
        print("{:20} {:9} {:11.4f} {:13.4f} {:11.3f} {:10.3f}".format(
            k, v["given"], v["state_prime_vs_asked"]["среднее"]["err"],
            v["state_prime_vs_real"]["среднее"]["err"], v["state_prime_vs_real"]["среднее"]["r"],
            v["r_video_to_raw"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
