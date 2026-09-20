"""18.6: what the coefficient weight did — and what it did not.

    python tools/fig_weight18.py

Left: the gate for the three width-192 arms that differ in representation and
in loss weighting, each against the floor of its own K, with the reading
registered before the run (what the learning rate alone would have bought)
marked. Right: the same change seen on the two yardsticks at once — the
unweighted validation loss barely moves while the gate moves 6.5x, which is
the point of the intervention and the reason validation was kept unweighted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(run: Path, tag: str) -> dict:
    tr = json.loads((run / f"{tag}_train.json").read_text(encoding="utf-8"))
    g = json.loads((run / f"samples18_{tag}.json").read_text(encoding="utf-8"))["gates"]
    return {"val": [h["val_loss"] for h in tr["history"] if "val_loss" in h][-1],
            "gate": g["prior"]["round_trip"]["median"], "lo": g["prior"]["round_trip"]["min"],
            "hi": g["prior"]["round_trip"]["max"], "floor": g["ceiling"]["round_trip"]["median"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_weight18.png"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    k16 = load(run, "corpus_dct16_w192_lr1e3_c")
    raw = load(run, "corpus_dct32_w192_c")
    wgt = load(run, "corpus_dct32_w192_lr1e3_w1_c")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.3), gridspec_kw={"width_ratios": [1.35, 1]})

    bars = [("K=16, lr 1e-3\n(лучшее плечо)", k16, "#1f77b4"),
            ("K=32, равный вес\n3e-4", raw, "#8c564b"),
            ("K=32, вес sd¹\nlr 1e-3", wgt, "#e377c2")]
    xs = np.arange(len(bars))
    ax1.bar(xs, [b[1]["gate"] for b in bars], color=[b[2] for b in bars], width=0.6)
    ax1.errorbar(xs, [b[1]["gate"] for b in bars],
                 yerr=[[b[1]["gate"] - b[1]["lo"] for b in bars], [b[1]["hi"] - b[1]["gate"] for b in bars]],
                 fmt="none", ecolor="#333", elinewidth=1.1, capsize=4)
    for x, (_, b, _) in zip(xs, bars):
        ax1.text(x, b["hi"] * 1.14, ru(f"{b['gate']:.4f}"), ha="center", fontsize=10)
    ax1.annotate("", xy=(2, wgt["gate"] * 1.55), xytext=(1, raw["gate"] * 0.92),
                 arrowprops=dict(arrowstyle="->", color="#e377c2", lw=2.4,
                                 connectionstyle="arc3,rad=-0.25"))
    ratio = raw["gate"] / wgt["gate"]
    ax1.text(1.5, 0.47, f"×{ru(f'{ratio:.1f}')} от одного веса", ha="center",
             fontsize=13, color="#c2185b", fontweight="bold")
    pred = 0.147
    ax1.plot([2], [pred], "v", color="#333", ms=10, zorder=5)
    ax1.text(2.42, pred, f"только темп дал бы\n≈{ru(f'{pred:.3f}')} (предсказано\nдо прогона)",
             fontsize=8.6, color="#333", ha="left", va="center")
    f32, f16 = raw["floor"], k16["floor"]
    ax1.axhline(f32, color="#8c564b", ls="--", lw=1.4)
    ax1.text(2.42, f32 * 1.05, f"пол K=32: {ru(f'{f32:.4f}')}", fontsize=8.6, color="#8c564b",
             ha="left", va="bottom")
    ax1.axhline(f16, color="#1f77b4", ls=":", lw=1.4)
    ax1.text(2.42, f16 * 1.05, f"пол K=16: {ru(f'{f16:.4f}')}", fontsize=8.6, color="#1f77b4",
             ha="left", va="bottom")
    ax1.set_yscale("log"); ax1.set_ylim(0.008, 0.75); ax1.set_xlim(-0.55, 4.5)
    ax1.set_xticks(xs); ax1.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax1.set_ylabel("прогонка через мозг (меньше = совместимее)")
    ax1.set_title("Гипотеза 18.5b подтвердилась: равный вес и правда травил K=32.\n"
                  "Но пол K=32 всё равно не достаётся — K=16 остаётся лучше", fontsize=10)
    ax1.grid(alpha=0.3, axis="y", which="both")

    names = ["валидация\n(невзвешенная, линейка)", "прогонка\n(то, что мерим)"]
    rel = [wgt["val"] / raw["val"], wgt["gate"] / raw["gate"]]
    cols = ["#9e9e9e", "#e377c2"]
    for i, r in enumerate(rel):                                      # lollipops: a bar from 0 lies on a log axis
        ax2.plot([1.0, r], [i, i], "-", color=cols[i], lw=3.5, solid_capstyle="round")
        ax2.plot([r], [i], "o", color=cols[i], ms=13)
    ax2.axvline(1.0, color="#333", lw=1.4)
    ax2.text(1.04, -0.5, "без изменений", fontsize=8.5, color="#333", va="center")
    for i, r in enumerate(rel):
        ax2.text(r * 1.16, i, f"×{ru(f'{r:.3f}')}" + ("  (−2,4 %)" if i == 0 else f"  (в {ru(f'{1 / r:.1f}')} раза лучше)"),
                 va="center", fontsize=10)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(names, fontsize=9.5)
    ax2.set_xscale("log"); ax2.set_xlim(0.1, 2.6); ax2.set_ylim(-0.75, 1.5)
    ax2.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0])                        # the default log ticks collide here
    ax2.set_xticklabels([f"×{ru(v)}" for v in ("0,1", "0,2", "0,5", "1", "2")], fontsize=9)
    ax2.minorticks_off()
    ax2.set_xlabel("во сколько раз изменилось от веса (лог. шкала)")
    ax2.set_title("Две линейки разошлись: по лоссу почти ничего,\nпо воротам — в 6,5 раза", fontsize=10)
    ax2.grid(alpha=0.3, axis="x", which="both")

    fig.suptitle("18.6: вес лосса по коэффициентам — ворота в 6,5 раза лучше при неизменной валидации",
                 fontsize=12.5, y=0.985)
    nl = "\n"
    foot = ("Оба плеча K=32: ширина 192, 20 000 шагов, batch 32, seed 0, один T4 (cpu 1 / 12 ГБ)." + nl +
            "Взвешенное плечо меняет две вещи разом — вес и темп; предсказание «только темп» получено из "
            "мультипликативности 18.5a и записано до прогона." + nl +
            "Ворота — медиана прогонки 16 сэмплов через замороженный мозг, локально на CPU, $0. "
            "Скрипт tools/fig_weight18.py.")
    fig.text(0.5, 0.008, foot, ha="center", fontsize=8.1, color="#444")
    fig.tight_layout(rect=(0, 0.085, 1, 0.935))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
