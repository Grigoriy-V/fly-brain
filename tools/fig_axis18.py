"""18.10: все плечи на новой оси — согласуется ли она с воротами?

    python tools/fig_axis18.py

Слева: ворота против разреженности границ для каждого плеча item 18. Если
оси согласны, точки лягут на монотонную кривую; расхождения — там, где ворота
нас обманывали. Справа: контраст против той же разреженности — проверка, не
является ли новая ось переодетым контрастом.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHORT = {"контроль: w128, 3e-4, K16": "контроль w128",
         "w128, lr 6e-4": "w128 6e-4", "w128, lr 1e-3": "w128 1e-3",
         "w128, 60k шагов": "w128 60k", "w128, классы": "классы",
         "w192, 3e-4": "w192 3e-4", "w192, lr 1e-3 — лучшее по воротам": "w192 1e-3",
         "w192, lr 1e-3, 100 шагов сэмплера": "w192 1e-3, 100 шагов",
         "K=32, w128": "K32 w128", "K=32, w192, равный вес": "K32 w192",
         "K=32, w192, вес sd¹": "K32 w192 + вес"}
COL = {"w128": "#1f77b4", "w192": "#2ca02c", "K32": "#8c564b", "классы": "#e377c2"}


def kind(label: str) -> str:
    if "классы" in label:
        return "классы"
    if "K=32" in label:
        return "K32"
    return "w192" if "w192" in label else "w128"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--name", default="edges18_all")
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_axis18.png"))
    a = p.parse_args(argv)
    G = json.loads((Path(a.run) / f"{a.name}.json").read_text(encoding="utf-8"))["groups"]
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    ceil = G["13B из настоящего состояния"]; raw = G["сырое видео корпуса"]
    arms = [(l, v) for l, v in G.items() if "gate" in v]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    lo, hi = 100 * min(raw["frac_flat"]["mean"], ceil["frac_flat"]["mean"]), 100 * max(
        raw["frac_flat"]["mean"], ceil["frac_flat"]["mean"])

    for ax, xk, xlab in ((ax1, "gate", "прогонка через мозг, лог. шкала"),
                         (ax2, "sd", "контраст: ст. отклонение по клипу")):
        ax.axhspan(lo, hi, color="#2ca02c", alpha=0.13)
        ax.text(0.015, hi + 0.8, "настоящее: сырой клип и отрисовка из настоящего состояния",
                transform=ax.get_yaxis_transform(), fontsize=8.6, color="#2ca02c")
        pts = sorted(((v[xk] if xk == "gate" else v[xk]["mean"]), 100 * v["frac_flat"]["mean"], lab)
                     for lab, v in arms)
        span = np.log10(pts[-1][0] / pts[0][0]) if xk == "gate" else pts[-1][0] - pts[0][0]
        placed = []                                                  # подписи в кластерах разводятся по вертикали
        for x, y, lab in pts:
            near = sum(1 for px, py in placed
                       if abs((np.log10(x / px) if xk == "gate" else x - px)) < 0.11 * span
                       and abs(y - py) < 2.2)
            ax.plot(x, y, "o", ms=10, color=COL[kind(lab)], markeredgecolor="white", markeredgewidth=0.8)
            ax.annotate(SHORT.get(lab, lab), (x, y), textcoords="offset points",
                        xytext=(0, 10 + 12 * near),                   # только вверх: вниз некуда
                        ha="center", fontsize=8.1)
            placed.append((x, y))
        if xk == "gate":
            ax.set_xscale("log")
        ax.set_xlabel(xlab); ax.set_ylabel("доля ровных переходов, %")
        ax.set_ylim(18.2, hi + 6)
        ax.grid(alpha=0.3, which="both")

    cl = next(v for l, v in arms if "классы" in l)
    ax1.annotate("классы: худшие ворота в своей группе,\nно лучшая структура в ней",
                 xy=(cl["gate"], 100 * cl["frac_flat"]["mean"]),
                 xytext=(0.16, 33.0), fontsize=9, color="#c2185b", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#c2185b", lw=1.5))
    g = [v["gate"] for _, v in arms]; f = [100 * v["frac_flat"]["mean"] for _, v in arms]
    r = float(np.corrcoef(np.log10(g), f)[0, 1])
    ax1.set_title(f"Оси в основном согласны (r = {ru(f'{r:.2f}')} по логарифму ворот),\n"
                  f"но никто и близко не подошёл", fontsize=10.5)
    sd = [v["sd"]["mean"] for _, v in arms]
    r2 = float(np.corrcoef(sd, f)[0, 1])
    ax2.set_title(f"И новая ось — не переодетый контраст:\nсвязь обратная, r = {ru(f'{r2:.2f}')}", fontsize=10.5)

    fig.suptitle("18.10: одиннадцать плеч на оси «сцена или пятно» — ёмкость двигает и её, "
                 "но разрыв до настоящего вдвое", fontsize=12.3, y=0.985)
    nl = "\n"
    best = max(f)
    ceil_flat = 100 * ceil["frac_flat"]["mean"]
    foot = (f"Лучшее плечо даёт {ru(f'{best:.1f}')} % ровного против {ru(f'{ceil_flat:.1f}')} % "
            f"у отрисовки из настоящего состояния." + nl +
            "По 16 сэмплов на плечо, 3 отрисовки клипов, 32 сырых клипа. Локально на CPU, $0. "
            "Скрипты flydream/generate/edges18.py и tools/fig_axis18.py.")
    fig.text(0.5, 0.012, foot, ha="center", fontsize=8.2, color="#444")
    fig.tight_layout(rect=(0, 0.06, 1, 0.935))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
