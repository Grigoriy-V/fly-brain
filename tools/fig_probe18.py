"""18.15: есть ли вообще в состоянии то, на что условливаться.

    python tools/fig_probe18.py

Слева: доля дисперсии состояния, которую держат средние по классам, против
того же на перемешанных метках. Справа: можно ли отложенному состоянию
назначить его класс по ближайшему среднему — против перемешанных меток и
против случайного угадывания. Девять процедурных классов (bar, dots, flash,
grating…) стоят положительным контролем: это радикально разные стимулы, и
если бы они не разделились, сломано было бы измерение, а не метки.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLES = {"actions_ucf101": "действия UCF101\n(91 класс, то, чем мы условливали)",
          "procedural_control": "процедурные классы\n(9, положительный контроль)"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "data" / "prior18" / "class_probe18.json"))
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_probe18.png"))
    a = p.parse_args(argv)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads(Path(a.src).read_text(encoding="utf-8"))
    G = S["groups"]
    keys = list(TITLES)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.3), gridspec_kw={"width_ratios": [1, 1.25]})

    xs = np.arange(len(keys))
    real = [100 * G[k]["eta2"] for k in keys]
    null = [100 * G[k]["eta2_shuffled"] for k in keys]
    ax1.bar(xs - 0.19, real, width=0.36, color="#1f77b4", label="настоящие метки")
    ax1.bar(xs + 0.19, null, width=0.36, color="#bbb", label="метки перемешаны")
    for x, r, n, k in zip(xs, real, null, keys):
        ax1.text(x - 0.19, r * 1.04, ru(f"{r:.2f}") + " %", ha="center", fontsize=9.5)
        ax1.text(x + 0.19, n * 1.06, ru(f"{n:.2f}") + " %", ha="center", fontsize=9.5, color="#666")
        ax1.text(x, max(r, n) * 1.22, f"×{ru(f'{G[k]['eta2_ratio']:.1f}')}", ha="center", fontsize=13,
                 color="#c2185b", fontweight="bold")
    ax1.set_xticks(xs); ax1.set_xticklabels([TITLES[k] for k in keys], fontsize=9)
    ax1.set_ylim(0, max(real) * 1.45)
    ax1.set_ylabel("доля дисперсии состояния при классе, %")
    ax1.set_title("Класс держит долю дисперсии заметно выше нулевой", fontsize=10.5)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, axis="y")

    w, off = 0.26, np.arange(len(keys)) * 1.0
    for j, (lab, fld, col) in enumerate((("настоящие метки", "top1", "#1f77b4"),
                                         ("метки перемешаны", "top1_shuffled", "#bbb"),
                                         ("случайное угадывание", "chance", "#e0e0e0"))):
        v = [100 * G[k][fld] for k in keys]
        ax2.bar(off + (j - 1) * w, v, width=w, color=col, label=lab,
                edgecolor="#999" if j == 2 else "none")
        for x, y in zip(off + (j - 1) * w, v):
            ax2.text(x, y + 0.7, ru(f"{y:.1f}"), ha="center", fontsize=9)
    for x, k in zip(off, keys):
        ax2.text(x, 100 * G[k]["top1"] + 4.2, f"×{ru(f'{G[k]['top1_over_shuffled']:.1f}')} над перемешанными",
                 ha="center", fontsize=10, color="#c2185b", fontweight="bold")
    ax2.set_xticks(off); ax2.set_xticklabels([TITLES[k] for k in keys], fontsize=9)
    ax2.set_ylim(0, 40)
    ax2.set_ylabel("класс угадан по ближайшему среднему, % (top-1)")
    ax2.set_title("И читается обратно из отложенного состояния — вдвое выше случайного", fontsize=10.5)
    ax2.legend(fontsize=9, loc="upper left"); ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("18.15: метки не пустые — сигнал в состоянии есть, но слабый; "
                 "значит условие проваливалось не из-за данных", fontsize=12.5, y=0.975)
    nl = "\n"
    ga, gp = G["actions_ucf101"], G["procedural_control"]
    foot = (f"Ни модели, ни генератора — статистика на самих состояниях (DCT-16, z-скоринг по корпусу): "
            f"{ga['n_train']} обучающих клипов, {ga['n_val']} отложенных," + nl +
            f"классификатор — ближайшее среднее класса в собственном пространстве состояния." + nl +
            f"top-5 у действий {ru(f'{100 * ga['top5']:.1f}')} % против "
            f"{ru(f'{100 * ga['top5_shuffled']:.1f}')} % на перемешанных; у контроля "
            f"{ru(f'{100 * gp['top5']:.1f}')} % против {ru(f'{100 * gp['top5_shuffled']:.1f}')} %." + nl +
            f"Modal, CPU 2 ядра / 12 ГБ, без GPU, {S['seconds']:.0f} с, $0,10. Скрипт tools/fig_probe18.py.")
    fig.text(0.5, 0.008, foot, ha="center", fontsize=8.2, color="#444")
    fig.tight_layout(rect=(0, 0.115, 1, 0.925))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
