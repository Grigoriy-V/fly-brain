r"""23.3b: картинка из полного состояния (K = 16) матрицей МНК, без 13B.

    python tools/fig_pic23k16.py

Отличие от `fig_pic23.py`: вход не нулевой коэффициент, а всё состояние блока,
и цель не средний кадр окна, а один кадр (двадцатый). 13B не участвует — он
гаснет на состоянии, постоянном во времени, и весь этот путь его обходит.
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

ROWS = [("target", "цель: кадр 20"),
        ("pic_real", "из НАСТОЯЩЕГО состояния"),
        ("pic_draw", "из СВЕЖЕГО РОЗЫГРЫША")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=str(ROOT / "data" / "prior23" / "pic23k16.npz"))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--prefix", default="2026-09-21_malecns_pic23k16")
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    Z = np.load(a.file)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(a.n, len(Z["target"]))
    fig, ax = plt.subplots(len(ROWS), n, figsize=(1.55 * n, 1.75 * len(ROWS) + 2.4))
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
        "Одна картинка из полного состояния, без 13B. Вход — весь блок T4a+T4b (K = 16, все 16 "
        "коэффициентов времени), цель — ОДИН кадр, двадцатый, а не усреднение по окну. Отображение — "
        "матрица наименьших квадратов, подогнанная на обучающем сплите; отложенные клипы из десяти "
        f"невиданных классов дают r = {ru('0,902')} против {ru('0,825')} у того же метода от одного "
        f"нулевого коэффициента, то есть мгновенная часть состояния для одного кадра нужна. Контраст почти "
        f"сошёлся: {ru('0,197')} при {ru('0,213')} у цели. Не сошлась резкость — доля ровного поля "
        f"{ru('24,3')} % при {ru('45,1')} % у цели, то есть энергия правильная, но размазана ровно вместо "
        "того чтобы собраться на границах. Это свойство наименьших квадратов: они рисуют условное среднее "
        "по всем картинкам, совместимым с этим состоянием, а не одну из них. 13B здесь не участвует "
        "намеренно — он гаснет на состоянии, постоянном во времени (измерено: срез, повторённый 40 раз, "
        f"даёт r {ru('0,129')} и контраст {ru('0,029')}, и дело именно в постоянстве, а не в величине — "
        f"|состояние| у среза {ru('0,678')} против {ru('0,689')} у полного). Нижний ряд — то же "
        "отображение, но состояние взято у свежего розыгрыша потока 23, обученного на движущихся клипах: "
        f"ровного {ru('20,8')} %, контраст {ru('0,143')}. Ни одной симуляции мозга, ни одного обученного "
        "параметра, 14 секунд локально.")
    fig.suptitle(textwrap.fill(title, 150), fontsize=8.5, y=0.995, va="top")
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.01, top=0.70, wspace=0.04, hspace=0.06)
    out = Path(a.outdir) / a.prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print(f"{out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
