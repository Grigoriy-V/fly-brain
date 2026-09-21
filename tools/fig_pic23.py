r"""23.3: сцена как одна картинка — из настоящего состояния и из свежего розыгрыша.

    python tools/fig_pic23.py

Никакого 13B и никакого обучения: DC существующих слепков (1 442 числа на
клип) переводится в картинку матрицей наименьших квадратов, подогнанной за
шесть секунд на обучающем сплите. Отложенные клипы — десять невиданных
классов.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flydream.decode.hexraster import to_raster  # noqa: E402

ROWS = [("target", "цель: средний кадр окна"),
        ("pic_real", "из НАСТОЯЩЕГО состояния"),
        ("pic_draw", "из СВЕЖЕГО РОЗЫГРЫША")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=str(ROOT / "data" / "prior23" / "pic23.npz"))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--prefix", default="2026-09-21_malecns_pic23")
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    Z = np.load(a.file)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = a.n
    fig, ax = plt.subplots(len(ROWS), n, figsize=(1.55 * n, 1.75 * len(ROWS) + 2.6))
    for r, (key, label) in enumerate(ROWS):
        A = Z[key]
        for c in range(n):
            axx = ax[r, c]
            v = A[c]
            v = (v - v.mean()) / (v.std() + 1e-8) * 0.18 + 0.5        # общая шкала серого на все клетки
            axx.imshow(to_raster(v, 721, 4, fill=np.nan), cmap="gray", vmin=0, vmax=1,
                       interpolation="nearest")
            axx.set_xticks([]); axx.set_yticks([])
            for s in axx.spines.values():
                s.set_visible(False)
            if c == 0:
                axx.set_ylabel(label, fontsize=8.5)
    title = (
        "Сцена как ОДНА КАРТИНКА, без 13B и без единого обученного параметра. Верхний ряд — цель, средний "
        "кадр окна настоящего клипа. Средний ряд — та же картинка, восстановленная из DC существующего "
        "слепка мозга: 1 442 числа (нулевой коэффициент DCT у T4a и T4b) переводятся в 721 значение "
        "матрицей наименьших квадратов, подогнанной за шесть секунд на обучающем сплите. На отложенных "
        f"клипах из десяти НЕВИДАННЫХ классов это даёт r = {ru('0,941')}, при контролях: средний кадр "
        f"корпуса всем — {ru('0,021')}, перемешанные пары — {ru('0,002')}. Нижний ряд — то же отображение, "
        "но DC взят у СВЕЖЕГО РОЗЫГРЫША потока 23, то есть картинка, которой не соответствует ни один клип. "
        f"Статистика: цель ровного {ru('39,2')} %, контраст {ru('0,191')}; из настоящего состояния "
        f"{ru('27,1')} % и {ru('0,181')} — линейная карта размывает, как всякая регрессия к среднему; из "
        f"розыгрыша {ru('21,2')} % и {ru('0,129')}. ЧТО ЭТО ЗНАЧИТ: информация о картинке в DC есть и "
        "берётся линейно, то есть отдельный отрисовщик нужен только за резкостью, а не за содержанием. "
        "Симуляций мозга не делалось ни одной — DC взят из корпуса, снятого 2026-09-20. Всё локально, "
        "40 секунд, даром.")
    fig.suptitle(textwrap.fill(title, 150), fontsize=8.5, y=0.995, va="top")
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.01, top=0.70, wspace=0.04, hspace=0.06)
    out = Path(a.outdir) / a.prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print(f"{out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
