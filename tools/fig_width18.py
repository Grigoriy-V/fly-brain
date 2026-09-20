"""18.5: the two arms at width 192 — do the levers add, and does capacity save K=32?

    python tools/fig_width18.py

Left: the four K=16 arms as a composition test. Each is one change from the
control, and the rightmost is both at once; the reference lines are the
representation's own floor and a real held-out clip. Middle: K=16 against
K=32 at both widths, all four at the same 3e-4, so the pair ratio is a clean
read on whether capacity is what K=32 lacked. Right: what the coefficients
are actually worth, which is where the middle panel's answer comes from.
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
    sm = json.loads((run / f"samples18_{tag}.json").read_text(encoding="utf-8"))
    g = sm["gates"]
    return {"val": [h["val_loss"] for h in tr["history"] if "val_loss" in h][-1],
            "gate": g["prior"]["round_trip"]["median"],
            "lo": g["prior"]["round_trip"]["min"], "hi": g["prior"]["round_trip"]["max"],
            "floor": g["ceiling"]["round_trip"]["median"], "clip": g["clip"]["round_trip"]["median"],
            "coef_std": np.array(tr["coef_std"])}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_width18.png"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    d = {t: load(run, t) for t in ("corpus_dct16_c", "corpus_dct16_lr1e3_c", "corpus_dct16_w192_c",
                                   "corpus_dct16_w192_lr1e3_c", "corpus_dct32_c", "corpus_dct32_w192_c")}
    floor16, clip = d["corpus_dct16_c"]["floor"], d["corpus_dct16_c"]["clip"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.6, 5.2), gridspec_kw={"width_ratios": [1.15, 1, 1.1]})

    # --- 1. the composition test, K = 16 -------------------------------------
    comp = [("контроль\nw128, 3e-4", "corpus_dct16_c", "#1f77b4"),
            ("только\nlr 1e-3", "corpus_dct16_lr1e3_c", "#d62728"),
            ("только\nширина 192", "corpus_dct16_w192_c", "#2ca02c"),
            ("оба\nвместе", "corpus_dct16_w192_lr1e3_c", "#17becf")]
    xs = np.arange(len(comp))
    vals = [d[t]["gate"] for _, t, _ in comp]
    err = np.array([[v - d[t]["lo"] for v, (_, t, _) in zip(vals, comp)],
                    [d[t]["hi"] - v for v, (_, t, _) in zip(vals, comp)]])
    ax1.bar(xs, vals, color=[c for _, _, c in comp], width=0.62)
    ax1.errorbar(xs, vals, yerr=err, fmt="none", ecolor="#333", elinewidth=1.1, capsize=4)
    for x, v, (_, t, _) in zip(xs, vals, comp):                      # above the whisker, never under it
        ax1.text(x, d[t]["hi"] * 1.12, ru(f"{v:.4f}"), ha="center", fontsize=9.5)
    pred = vals[1] * vals[2] / vals[0]                               # if the two factors were independent
    ax1.plot([xs[3]], [pred], "v", color="#333", ms=10, zorder=5)
    ax1.annotate(f"перемножение осей\nпредсказывало {ru(f'{pred:.4f}')}",
                 xy=(xs[3] + 0.14, pred), xytext=(3.66, 0.070), fontsize=9, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax1.axhline(floor16, color="#7f7f7f", ls="--", lw=1.5)
    ax1.text(3.66, floor16, f"пол представления\nDCT-16: {ru(f'{floor16:.4f}')}",
             fontsize=8.8, color="#555", ha="left", va="bottom")
    ax1.axhline(clip, color="#2ca02c", ls=":", lw=1.5)
    ax1.text(3.66, clip, f"настоящий отложенный\nклип: {ru(f'{clip:.4f}')}",
             fontsize=8.8, color="#2ca02c", ha="left", va="top")
    ax1.set_yscale("log"); ax1.set_ylim(0.008, 0.30); ax1.set_xlim(-0.55, 5.75)
    ax1.set_xticks(xs); ax1.set_xticklabels([c[0] for c in comp], fontsize=9)
    ax1.set_ylabel("прогонка через мозг (меньше = совместимее)")
    ax1.set_title(f"Оси складываются — и упираются в пол:\nсобственная доля прайора "
                  f"{ru(f'{vals[2] - floor16:.4f}')} → {ru(f'{vals[3] - floor16:.4f}')}", fontsize=10)
    ax1.grid(alpha=0.3, axis="y", which="both")

    # --- 2. K = 16 against K = 32 at both widths, all at 3e-4 ----------------
    pairs = [("ширина 128", "corpus_dct16_c", "corpus_dct32_c"),
             ("ширина 192", "corpus_dct16_w192_c", "corpus_dct32_w192_c")]
    xp = np.arange(len(pairs))
    k16 = [d[b]["gate"] for _, b, _ in pairs]
    k32 = [d[b]["gate"] for _, _, b in pairs]
    ax2.bar(xp - 0.19, k16, width=0.36, color="#1f77b4", label="K = 16")
    ax2.bar(xp + 0.19, k32, width=0.36, color="#8c564b", label="K = 32")
    for x, v16, v32 in zip(xp, k16, k32):
        ax2.text(x - 0.19, v16 * 1.22, ru(f"{v16:.3f}"), ha="center", fontsize=9)
        ax2.text(x + 0.19, v32 * 1.22, ru(f"{v32:.3f}"), ha="center", fontsize=9)
        ax2.text(x, 1.30, f"×{ru(f'{v32 / v16:.1f}')}", ha="center", fontsize=12,
                 color="#8c564b", fontweight="bold")
    floor32 = d["corpus_dct32_c"]["floor"]
    ax2.axhline(floor32, color="#8c564b", ls="--", lw=1.2)
    ax2.text(0.5, floor32 * 1.12, f"пол при K=32: {ru(f'{floor32:.4f}')}", fontsize=8.5, color="#8c564b",
             ha="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax2.set_yscale("log"); ax2.set_ylim(0.008, 8.0); ax2.set_xlim(-0.55, 1.55)
    ax2.set_xticks(xp); ax2.set_xticklabels([p[0] for p in pairs], fontsize=10)
    ax2.set_title("Ёмкость не закрывает разрыв K=32:\nотношение не сжалось, а выросло", fontsize=10)
    ax2.legend(fontsize=9, loc="upper left"); ax2.grid(alpha=0.3, axis="y", which="both")

    # --- 3. what the coefficients are worth ----------------------------------
    sd = d["corpus_dct32_w192_c"]["coef_std"].mean(1)
    e = sd ** 2; e = e / e.sum()
    tail = 100 * e[16:].sum()
    ax3.semilogy(np.arange(len(sd)), sd, "o-", color="#8c564b", ms=4.5, lw=1.6)
    ax3.axvspan(15.5, 31.5, color="#8c564b", alpha=0.12)
    ax3.axvline(15.5, color="#1f77b4", lw=1.5, ls="--")
    ax3.text(7.6, sd[0] * 0.62, "K = 16\nкончается здесь", fontsize=9, color="#1f77b4", ha="center")
    over = 50.0 / tail
    ax3.text(0.97, 0.80, f"верхние 16 коэффициентов:\n{ru(f'{tail:.2f}')} % энергии состояния,\n"
                         f"но 50 % веса в лоссе\n(переоценка в {ru(f'{over:.0f}')}×)",
             transform=ax3.transAxes, ha="right", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.45", fc="#fff6f0", ec="#8c564b", lw=1.1))
    ax3.set_xlabel("индекс коэффициента DCT"); ax3.set_ylabel("ст. отклонение по корпусу")
    ax3.set_title("Причина: z-скоринг уравнивает коэффициенты,\nкоторые несут разную энергию", fontsize=10)
    ax3.grid(alpha=0.3, which="both")

    fig.suptitle("18.5: ширина 192 + lr 1e-3 доводит прайор до пола представления; "
                 "K=32 на той же ширине остаётся в 6,6 раза хуже", fontsize=12.5, y=0.985)
    foot = ("Каждое плечо — 20 000 шагов, batch 32, seed 0, один T4 (cpu 1 / 12 ГБ), компиляция включена. "
            "Ворота — медиана прогонки 16 сэмплов через замороженный мозг, локально на CPU, $0. "
            "Усы — минимум и максимум по 16 сэмплам. Скрипт tools/fig_width18.py.")
    fig.text(0.5, 0.013, foot, ha="center", fontsize=8.2, color="#444")
    fig.tight_layout(rect=(0, 0.05, 1, 0.935))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
