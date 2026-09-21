r"""21г: половина типов, заявленная честно, обходит весь путь через PCA.

    python tools/fig_half21.py

У 13B есть штатный вход «дана только часть типов» — восемь бит маски, на
которых он учился (`MASK_MODES`: `t4` и `t5` по 10 % батчей). До сих пор во
всех прогонах подавалась полная маска, то есть модели сообщалось «T5
присутствует и он ровный» — вход, которого она не видела. Здесь то же самое
сказано правдой, и это лучший срез из всех измеренных.
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

SHOW = [("настоящее состояние", "13B из полного\nсостояния (потолок)"),
        ("types=T4+mask", "ПОЛОВИНА ТИПОВ (T4),\nмаска 13B: T5 не дан"),
        ("types=T4", "та же половина,\nно T5 ОБНУЛЁН молча"),
        ("types=T4+complete", "та же половина,\nT5 достроен по PCA"),
        ("pca=2048", "весь путь через PCA,\n2 048 координат")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="cut19_mask")
    p.add_argument("--back", default="back21_mask")
    p.add_argument("--show", type=int, default=1)
    p.add_argument("--prefix", default="2026-09-21_malecns_half21")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    B = json.loads((run / f"{a.back}.json").read_text(encoding="utf-8"))
    Z = np.load(run / f"{a.tag}.npz")
    i, G, D = a.show, S["groups"], S["dims"]
    frames = len(Z[f"raw__{i}"])
    bm, bc = B["по армам"]["types=T4+mask"], B["по армам"]["types=T4+complete"]
    bf = B["по армам"]["полное"]

    cells = [(Z[f"raw__{i}"], f"сырое видео корпуса\nклип {S['clip_idx'][i]}",
              f"ровного {ru(f'{100 * S['raw']['frac_flat']:.1f}')} %")]
    for key, head in SHOW:
        v = G[key]
        cells.append((Z[f"video__{key}|{i}"], head,
                      f"задано {v['given']}, r {ru(f'{v['r_to_raw']:.3f}')}\n"
                      f"ровного {ru(f'{100 * v['frac_flat']:.1f}')} %"))
    slow = f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее"
    title = (
        f"Достраивать не надо — надо сказать правду. У 13B есть штатный вход «часть типов не дана» (маска, на ней "
        f"10 % обучения), и до сих пор мы им не пользовались: подавали полную маску при обнулённом T5, то есть "
        f"сообщали «тип есть и он ровный». Сказанное честно, это лучший срез из всех: r к сырому видео "
        f"{ru(f'{G['types=T4+mask']['r_to_raw']:.3f}')} при потолке {ru(f'{G['настоящее состояние']['r_to_raw']:.3f}')} "
        f"у полного состояния, ровного поля {ru(f'{100 * G['types=T4+mask']['frac_flat']:.1f}')} % при "
        f"{ru(f'{100 * G['настоящее состояние']['frac_flat']:.1f}')} % у полного и "
        f"{ru(f'{100 * S['raw']['frac_flat']:.1f}')} % у сырого видео. Достройка по PCA теперь ХУЖЕ "
        f"({ru(f'{G['types=T4+complete']['r_to_raw']:.3f}')}), и весь путь через PCA-2048 тоже "
        f"({ru(f'{G['pca=2048']['r_to_raw']:.3f}')} при ровного {ru(f'{100 * G['pca=2048']['frac_flat']:.1f}')} %). "
        f"Выброшенная половина при этом возвращается физикой лучше, чем линейной достройкой: состояние, снятое "
        f"мозгом с этого видео, отстоит от настоящего на {ru(f'{bm['state_prime_vs_real']['среднее']['err']:.4f}')} "
        f"при r {ru(f'{bm['state_prime_vs_real']['среднее']['r']:.3f}')} — против "
        f"{ru(f'{bc['state_prime_vs_real']['среднее']['err']:.4f}')} у достройки и "
        f"{ru(f'{bf['state_prime_vs_real']['среднее']['err']:.4f}')} у собственного пола цепочки; срез отнёс на "
        f"{ru(f'{bm['asked_vs_real']['среднее']['err']:.4f}')}, то есть вернулось в "
        f"{ru(f'{bm['asked_vs_real']['среднее']['err'] / bm['state_prime_vs_real']['среднее']['err']:.0f}')} раз. "
        f"Ворота во всех срезах по типам бессмысленны по построению (они мерят расстояние до заказа, где T5 "
        f"объявлен ровным) и здесь не показаны: судья — r к сырому видео и доля ровного поля. Шесть отложенных "
        f"клипов из 10 классов UCF101, не виденных обучением; один z у 13B на клип; показан образец {i + 1} из "
        f"шести. Локально, даром. {slow}.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    row(cells, textwrap.fill(title, 165), Path(a.outdir) / a.prefix, frames, a.fps,
        plt, FuncAnimation, PillowWriter, dpi=110)
    print("{:24} {:>9} {:>11} {:>9} {:>10} {:>14}".format(
        "срез", "задано", "r к сырому", "ровного", "ближайшее", "сост' к наст."))
    for k, v in G.items():
        b = B["по армам"].get(k if k != "настоящее состояние" else "полное")
        print("{:24} {:9} {:11.3f} {:8.1f}% {:10.3f} {:>14}".format(
            k, v["given"], v["r_to_raw"], 100 * v["frac_flat"], v["nearest_r"],
            f"{b['state_prime_vs_real']['среднее']['err']:.4f}" if b else "—"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
