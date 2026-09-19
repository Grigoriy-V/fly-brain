"""The whole chain on one sheet: what is frozen, what is trained, where item 18 sits.

    python tools/fig_pipeline18.py

Not a measurement — an explanation. Three rows: how the training data is made
(everything frozen), the one module that learns (the prior), and how a state
becomes video and is verified (frozen again).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SURFACE = "#f7f6f3"
FROZEN, FROZEN_EDGE = "#dfe3e8", "#8a94a0"
TRAIN, TRAIN_EDGE = "#cfe9d4", "#2f7d43"
DATA, DATA_EDGE = "#ffffff", "#b9b3a8"

NL = chr(10)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_pipeline18.png"))
    p.add_argument("--dpi", type=int, default=110)
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(17.5, 9.0), facecolor=SURFACE)
    ax.set_xlim(0, 122); ax.set_ylim(0, 58); ax.axis("off"); ax.set_facecolor(SURFACE)

    def box(x, y, w, h, lines, kind="frozen", fs=10.0):
        face, edge = {"frozen": (FROZEN, FROZEN_EDGE), "train": (TRAIN, TRAIN_EDGE), "data": (DATA, DATA_EDGE)}[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    linewidth=2.2 if kind == "train" else 1.2, facecolor=face, edgecolor=edge))
        ax.text(x + w / 2, y + h / 2, NL.join(lines), ha="center", va="center", fontsize=fs, linespacing=1.5)

    def arrow(x1, y1, x2, y2, text="", fs=9, dy=1.0, colour="#4a4a4a"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.5, color=colour, shrinkA=1, shrinkB=1))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", va="bottom", fontsize=fs, color="#555")

    ax.text(1, 55.6, "Что снится мухе: где что учится", fontsize=16, weight="bold")
    ax.text(1, 53.2, "Серое — заморожено.    Зелёное — единственный обучаемый модуль.    Белое — данные.",
            fontsize=10.5, color="#444")

    ax.text(1, 48.8, "1.  Как делаются данные для обучения", fontsize=12, weight="bold", color="#333")
    row1 = [(1, 21, ["15 514 клипов", "12 411 обычного видео", "+ 3 103 процедурных"], "data"),
            (26, 21, ["замороженная модель", "MaleCNS"], "frozen"),
            (51, 21, ["состояние T4/T5", "40 кадров × 8 типов", "× 721 колонка"], "data"),
            (76, 20, ["DCT-16 по времени", "фиксированный базис"], "frozen"),
            (100, 21, ["компактное состояние", "16 × 8 × 721", "обучающий набор"], "data")]
    for x, w, lines, kind in row1:
        box(x, 39.5, w, 7.4, lines, kind)
    for (x, w, _, _), (nx, *_rest) in zip(row1, row1[1:]):
        arrow(x + w, 43.2, nx, 43.2)

    ax.text(1, 35.0, "2.  Что обучается: один модуль, 1.4 М параметров", fontsize=12, weight="bold", color="#333")
    box(6, 24.5, 23, 8.0, ["шум ε ~ N(0, I)", "16 × 8 × 721", "той же размерности"], "data")
    box(45, 24.0, 32, 9.0, ["ПРАЙОР  (учится сейчас)", "SiT 128 × 4, поток по состояниям",
                            "цель: поле скоростей  v = x₁ − ε"], "train", fs=11)
    box(93, 24.5, 24, 8.0, ["компактное состояние", "16 × 8 × 721"], "data")
    arrow(29, 30.0, 45, 30.0, "генерация: 20 шагов Эйлера", dy=0.8)
    arrow(93, 26.0, 77, 26.0, "", colour=TRAIN_EDGE)
    ax.text(85, 26.7, NL.join(["инверсия:", "то же поле назад"]), ha="center", va="bottom",
            fontsize=9, color=TRAIN_EDGE, linespacing=1.4)
    ax.text(1, 19.2, NL.join([
        "Пар «состояние ↔ шум» в обучении нет: каждый шаг берёт свежий случайный ε и случайное t, строит точку "
        "x_t = (1−t)·ε + t·x₁",
        "и учит предсказывать направление к данным. Соответствие «это состояние ← этот шум» появляется после "
        "обучения, как свойство потока."]), fontsize=9.8, color="#555", linespacing=1.7)

    ax.text(1, 15.0, "3.  Как состояние становится видео и как это проверяется", fontsize=12, weight="bold", color="#333")
    row3 = [(1, 17, ["состояние", "из прайора"], "data"),
            (22, 16, ["обратный DCT", "→ 40 кадров"], "frozen"),
            (42, 21, ["13B: состояние → видео", "(обучен, заморожен)"], "frozen"),
            (67, 13, ["видео", "40 × 721"], "data"),
            (84, 17, ["замороженная", "модель MaleCNS"], "frozen"),
            (105, 16, ["состояние′", "→ прогонка"], "data")]
    for x, w, lines, kind in row3:
        box(x, 6.6, w, 7.0, lines, kind)
    for (x, w, _, _), (nx, *_rest) in zip(row3, row3[1:]):
        arrow(x + w, 10.1, nx, 10.1)
    ax.text(1, 3.0, NL.join([
        "Гейт — прогонка: расстояние между состоянием, которое просили, и тем, что мозг прочитал обратно.",
        "Реальный клип 0.006–0.038,    потолок представления 0.009,    прайор 17.1b 0.142,    "
        "недостижимое состояние 1.0–1.8."]), fontsize=10, color="#333", linespacing=1.7)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
