r"""21б: спадает ли ковариация состояния с гексагональным расстоянием.

    python tools/fig_hexcov21.py

Две панели: слева корреляция колонка-к-колонке по расстоянию в шагах решётки
вместе с контролем (номера колонок перемешаны — спектр тот же, пространство
разрушено); справа корреляция каналов внутри ОДНОЙ колонки между восемью
типами. Это не клип, поэтому картинка статическая: кривая — не сцена.
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

SURFACE = "#f5f5f0"
DEEP = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="hexcov19")
    p.add_argument("--prefix", default="2026-09-21_malecns_hexcov21")
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((Path(a.run) / f"{a.tag}.json").read_text(encoding="utf-8"))
    sp = S["space"]
    c = np.asarray(sp["corr"]); ctrl = np.asarray(sp["shuffled_control"])
    mass = np.asarray(sp["mass_fraction"]); d = np.arange(len(c))
    M = np.asarray(S["types"]["matrix"])
    per = S["space_per_dct"]
    dc0 = np.asarray(per["0"]["corr"]); dc8 = np.asarray(per["8"]["corr"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4), facecolor=SURFACE,
                           gridspec_kw={"width_ratios": [1.3, 1.0, 0.9]})
    for x in ax:
        x.set_facecolor(SURFACE)

    ax[0].plot(d, c, "o-", color="#1b3a6b", lw=2, label="состояние, все 16 x 8 каналов")
    ax[0].plot(d, dc0, "^--", color="#3f7d3f", lw=1.2, ms=4, label="только DCT 0 (среднее по времени)")
    ax[0].plot(d, dc8, "v--", color="#9a5b1e", lw=1.2, ms=4, label="только DCT 8 (быстрое движение)")
    ax[0].plot(d, ctrl, "s-", color="#a11", lw=1.5, ms=4, label="контроль: номера колонок перемешаны")
    ax[0].axhline(0, color="#888", lw=0.6)
    ax[0].set_xlabel("расстояние между колонками, шагов решётки")
    ax[0].set_ylabel("корреляция")
    ax[0].set_title("Ковариация локальна: 0,88 у соседа, 0,44 на четырёх шагах,\n"
                    "дальше полка 0,25-0,30 при контроле 0,27", fontsize=10)
    ax[0].set_ylim(-0.05, 1.12)
    ax[0].legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax[0].grid(alpha=0.25)

    ax[1].bar(d, 100 * mass, color="#1b3a6b", alpha=0.85)
    ax[1].set_xlabel("расстояние между колонками, шагов решётки")
    ax[1].set_ylabel("доля массы ковариации, %")
    ax[1].set_title(f"Но масса размазана: в пределах одного шага "
                    f"{ru(f'{100 * sp['mass_within_1']:.1f}')} %,\nдвух — "
                    f"{ru(f'{100 * sp['mass_within_2']:.1f}')} % (пар на расстоянии 12 в 36 раз больше,\n"
                    f"чем соседних)", fontsize=10)
    ax[1].grid(alpha=0.25, axis="y")

    im = ax[2].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[2].set_xticks(range(8)); ax[2].set_xticklabels(DEEP, fontsize=8, rotation=90)
    ax[2].set_yticks(range(8)); ax[2].set_yticklabels(DEEP, fontsize=8)
    ax[2].set_title(f"Восемь типов в ОДНОЙ колонке:\nвне диагонали "
                    f"{ru(f'{S['types']['mean_abs_off_diagonal']:.3f}')}, максимум "
                    f"{ru(f'{S['types']['max_abs_off_diagonal']:.3f}')}", fontsize=10)
    fig.colorbar(im, ax=ax[2], fraction=0.046)

    title = (
        f"Локальность есть в самих данных, а не только в архитектуре. Считано из PCA-базиса без скачивания "
        f"состояний: ковариация обучающих состояний в ранге {S['k']} это B diag(λ) B-транспонированное, а "
        f"сравнение идёт против перестановки номеров колонок — тот же спектр, разрушенное пространство. "
        f"Корреляция соседних колонок {ru(f'{c[1]:.3f}')} падает вдвое к {ru(f'{sp['locality_length']:.0f}')} шагам "
        f"и выходит на полку {ru(f'{c[8]:.3f}')}, тогда как перемешанный контроль плоский на "
        f"{ru(f'{ctrl[8]:.3f}')} — то есть у данных есть локальная часть ПЛЮС общий фон, который локальным окном "
        f"не описывается. Это ответ на возражение arXiv 2509.09672 (локальность — свойство ковариации данных): "
        f"здесь она в данных есть, значит гекс-локальное внимание опирается на что-то реальное, но одного окна "
        f"мало — нужен и глобальный канал. Третья панель — про риск «128 каналов на сайт» из отчёта-ресёрча: "
        f"восемь типов в одной колонке скоррелированы в среднем на {ru(f'{S['types']['mean_abs_off_diagonal']:.3f}')}, "
        f"а значит сжимать надо по каналу, сохраняя гекс-раскладку. Локально, {ru(f'{S['seconds']:.0f}')} с, даром.")
    fig.suptitle(textwrap.fill(title, 200), fontsize=10, x=0.01, y=0.99, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = Path(a.outdir) / a.prefix
    fig.savefig(out.with_suffix(".png"), dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}.png")
    print("{:>4} {:>10} {:>12} {:>12}".format("d", "корр.", "перемешано", "доля массы"))
    for i in range(len(c)):
        print("{:4} {:10.3f} {:12.3f} {:12.3f}".format(i, c[i], ctrl[i], mass[i]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
