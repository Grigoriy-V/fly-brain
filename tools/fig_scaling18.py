"""18.4, the scaling view: do our arms fall on one diagonal, as an LLM's do?

    python tools/fig_scaling18.py

The human, 2026-09-20, on seeing the sweep: the shape recalls an LLM training
curve, where a different interpolation puts everything on one straight line.
This plots ours the way those are plotted — loss against compute,
C = 6·N·B·S, on log-log axes — and answers the question with the data we
already paid for. Both panels carry the same seven arms: the left one the
validation loss the arms were trained on, the right one the gate the project
is actually judged by.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARMS = [
    ("corpus_dct16_c", "3e-4, 20k", "w128"),
    ("corpus_dct16_lr6e4_c", "6e-4, 20k", "w128"),
    ("corpus_dct16_lr1e3_c", "1e-3, 20k", "w128"),
    ("corpus_dct16_60k_c", "3e-4, 60k", "w128"),
    ("corpus_dct16_w192_c", "3e-4, 20k, ширина 192", "w192"),
    ("corpus_dct16_cls_c", "классы", "classes"),
    ("corpus_dct32_c", "K=32", "K32"),
    ("corpus_dct16_w192_lr1e3_c", "1e-3, 20k, ширина 192", "w192lr"),
    ("corpus_dct32_w192_c", "K=32, ширина 192", "K32w"),
]
COLOUR = {"w128": "#1f77b4", "w192": "#2ca02c", "classes": "#e377c2", "K32": "#8c564b",
          "w192lr": "#17becf", "K32w": "#bcbd22"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_scaling18.png"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    rows = []
    for tag, lab, kind in ARMS:
        t = run / f"{tag}_train.json"
        s = run / f"samples18_{tag}.json"
        if not (t.exists() and s.exists()):
            continue
        tr = json.loads(t.read_text(encoding="utf-8"))
        g = json.loads(s.read_text(encoding="utf-8"))["gates"]["prior"]["round_trip"]["median"]
        val = [h["val_loss"] for h in tr["history"] if "val_loss" in h][-1]
        rows.append({"tag": tag, "label": lab, "kind": kind, "val": val, "gate": g,
                     "C": 6.0 * tr["parameters"] * tr["batch"] * tr["steps"]})
    base = [r for r in rows if r["kind"] == "w128"]                  # the fit is on one width only
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for ax, key, name in ((axes[0], "val", "валидационный лосс"), (axes[1], "gate", "прогонка через мозг")):
        x = np.log10([r["C"] for r in base]); y = np.log10([r[key] for r in base])
        b, c = np.polyfit(x, y, 1)
        xs = np.logspace(np.log10(min(r["C"] for r in rows)) - 0.08, np.log10(max(r["C"] for r in rows)) + 0.08, 50)
        ax.plot(xs, 10 ** (c + b * np.log10(xs)), "--", color="#1f77b4", lw=1.4,
                label=f"степенная подгонка по ширине 128, наклон {ru(f'{b:+.3f}')}")
        seen = {}
        for r in rows:                                               # points that share C are spread a little
            n = seen.get(round(np.log10(r["C"]), 3), 0)
            seen[round(np.log10(r["C"]), 3)] = n + 1
            xr = r["C"] * (1 + 0.075 * n)
            r[f"x_{key}"] = xr
            ax.plot(xr, r[key], "o", color=COLOUR[r["kind"]], ms=9,
                    markeredgecolor="white", markeredgewidth=0.8)
            ax.annotate(r["label"], (xr, r[key] * (1.07 + 0.19 * n)), fontsize=8,
                        ha="left" if n else "center", va="bottom")
        w = next((r for r in rows if r["kind"] == "w192"), None)
        if w:
            pred = 10 ** (c + b * np.log10(w["C"]))
            ax.annotate("", xy=(w[f"x_{key}"], w[key]), xytext=(w[f"x_{key}"], pred),
                        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.8))
            ax.text(w[f"x_{key}"] * 1.06, (w[key] * pred) ** 0.5,
                    f"−{ru(f'{100 * (1 - w[key] / pred):.0f}')} % от предсказанного",
                    color="#2ca02c", fontsize=9.5, va="center")
        ax.set_xscale("log"); ax.set_yscale("log")
        lo, hi = min(r[key] for r in rows), max(r[key] for r in rows)
        ax.set_ylim(lo * 0.62, hi * 1.9)
        ax.set_xlabel("вычисления C = 6·N·B·S (FLOP-эквивалент)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8.5, loc="lower left")
    same = [r for r in base if abs(r["C"] / base[0]["C"] - 1) < 0.01]
    spread = max(r["val"] for r in same) / min(r["val"] for r in same) - 1
    axes[0].set_title(f"три плеча при одинаковых вычислениях расходятся на "
                      f"{ru(f'{100 * spread:.0f}')} % — их различает только темп", fontsize=9.5)
    axes[1].set_title("ёмкость уводит точку с прямой, длина — нет", fontsize=9.5)
    fig.suptitle("18.4 в координатах закона масштабирования: наши точки на одну диагональ не ложатся — "
                 "ширина уходит вниз, вычисления её не объясняют", fontsize=11.5, y=0.985)
    foot = ("Семь плеч, по одному seed; диапазон вычислений 3×, законы масштабирования строятся на 4–6 порядках. "
            "Подгонка по четырём плечам ширины 128 — линейка, не закон.\n"
            "Точки с равными вычислениями разведены по горизонтали. Скрипт tools/fig_scaling18.py.")
    fig.text(0.5, 0.012, foot, ha="center", fontsize=8.2, color="#444")
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out} ({len(rows)} arms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
