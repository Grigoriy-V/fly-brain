"""18.4 in one sheet: every arm of the sweep against one control.

    python tools/fig_arms18.py

Left: the validation curve of each arm. Right: the main gate — the round trip
through the frozen brain for 16 samples of each prior — with the references
that bound it. Arms that change the representation (K) or the input (classes)
are drawn in a different colour, because their validation loss is not on the
same scale as the rest and only the gate compares them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# tag, label, whether its validation loss is comparable to the control's
ARMS = [
    ("corpus_dct16_c", "контроль: 3e-4, w128, K16", True, "#1f77b4"),
    ("corpus_dct16_lr6e4_c", "lr 6e-4", True, "#ff7f0e"),
    ("corpus_dct16_lr1e3_c", "lr 1e-3", True, "#d62728"),
    ("corpus_dct16_w192_c", "ширина 192", True, "#2ca02c"),
    ("corpus_dct16_60k_c", "60 000 шагов", True, "#9467bd"),
    ("corpus_dct32_c", "K=32", False, "#8c564b"),
    ("corpus_dct16_cls_c", "классы", False, "#e377c2"),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_arms18.png"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    rows = []
    for tag, lab, same, col in ARMS:
        t = run / f"{tag}_train.json"
        s = run / f"samples18_{tag}.json"
        if not t.exists():
            continue
        tr = json.loads(t.read_text(encoding="utf-8"))
        sm = json.loads(s.read_text(encoding="utf-8")) if s.exists() else None
        rows.append({"tag": tag, "label": lab, "same": same, "colour": col, "train": tr, "samples": sm})
    if not rows:
        print("no arm has finished yet")
        return 1
    base = next((r for r in rows if r["tag"] == "corpus_dct16_c"), rows[0])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    for r in rows:
        h = [x for x in r["train"]["history"] if "val_loss" in x]
        st, vl = [x["step"] for x in h], [x["val_loss"] for x in h]
        ax1.plot(st, vl, "-" if r["same"] else "--", color=r["colour"], lw=2,
                 label=f"{r['label']}: {ru(f'{vl[-1]:.4f}')}" + ("" if r["same"] else " *"))
    ax1.set_xlabel("шаг обучения"); ax1.set_ylabel("валидационный лосс")
    ax1.set_title("* пунктир — другая цель, лосс с остальными несравним", fontsize=9.5)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8.5); ax1.set_ylim(0.25, 1.4)

    gated = [r for r in rows if r["samples"]]
    bars, refs = [], None
    for r in gated:
        g = r["samples"]["gates"]
        bars.append((r["label"], g["prior"]["round_trip"]["median"], r["colour"]))
        refs = refs or (g["clip"]["round_trip"]["median"], r["samples"]["scores"]["shuffled_clip"]["round_trip"],
                        r["samples"]["scores"]["noise_white"]["round_trip"],
                        g.get("ceiling", {}).get("round_trip", {}).get("median", 0.021))
    if bars:
        bars.sort(key=lambda b: -b[1])
        if refs:
            bars = [("контроль: шум в типах", refs[2], "#999"),
                    ("контроль: колонки переставлены", refs[1], "#999")] + bars + \
                   [("пол представления (DCT-16)", refs[3], "#7f7f7f"),
                    ("настоящий отложенный клип", refs[0], "#2ca02c")]
        ax2.barh(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], height=0.6)
        for i, b in enumerate(bars):
            ax2.text(b[1] * 1.12, i, ru(f"{b[1]:.3f}"), va="center", fontsize=9)
        ax2.set_yticks(range(len(bars))); ax2.set_yticklabels([b[0] for b in bars], fontsize=8.5)
        ax2.set_xscale("log"); ax2.set_xlim(0.006, 6)
        best = min(b[1] for b in bars if b[2] not in ("#999", "#7f7f7f") and "клип" not in b[0])
        ax2.set_title(f"лучшее плечо {ru(f'{best:.3f}')} против 0,095 у 18.3, при поле {ru(f'{refs[3]:.3f}')}", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "ворота ещё не считались", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("прогонка через мозг, лог. шкала (меньше = совместимее)", fontsize=9)
    ax2.grid(alpha=0.3, axis="x")

    n_done = len(rows)
    sec = sum(r["train"]["seconds"] for r in rows)
    base_val = [x["val_loss"] for x in base["train"]["history"] if "val_loss" in x][-1]
    fig.suptitle(f"18.4: {n_done} плеч на одном контроле — что на самом деле двигает прайор "
                 f"(валидация контроля {ru(f'{base_val:.4f}')})", fontsize=11.5, y=0.985)
    fig.text(0.5, 0.008, f"Каждое плечо — один T4, cpu 1 / 12 ГБ, компиляция включена; суммарно {sec / 60:.0f} GPU-минут. "
                         f"Ворота считались локально на CPU, $0. Скрипт tools/fig_arms18.py.",
             ha="center", fontsize=8.2, color="#444")
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out} ({n_done} arms, {len(gated)} gated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
