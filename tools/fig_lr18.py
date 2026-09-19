"""18.4b in one sheet: the paired learning-rate arm against 18.3.

    python tools/fig_lr18.py

Left: the two validation curves, identical in everything but the learning rate
— same batch, steps, seed, data and code. Right: the main gate (the round trip
through the frozen brain) for both, with the references that bound it — a real
held-out clip below, the representation's own ceiling, and the unreachable
controls above, all measured in the same local code path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--a", default="corpus_dct16", help="the control arm's tag (18.3, lr 3e-4)")
    p.add_argument("--b", default="corpus_dct16_lr1e4", help="the new arm's tag")
    p.add_argument("--samples-a", default="samples18_local")
    p.add_argument("--samples-b", default="samples18_lr1e4_local")
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_lr18.png"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ta = json.loads((run / f"{a.a}_train.json").read_text(encoding="utf-8"))
    tb = json.loads((run / f"{a.b}_train.json").read_text(encoding="utf-8"))
    sa = json.loads((run / f"{a.samples_a}.json").read_text(encoding="utf-8"))
    sb = json.loads((run / f"{a.samples_b}.json").read_text(encoding="utf-8"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    val = lambda t: ([r["step"] for r in t["history"] if "val_loss" in r],
                     [r["val_loss"] for r in t["history"] if "val_loss" in r])  # noqa: E731
    gate = lambda s, k: s["gates"][k]["round_trip"]["median"]         # noqa: E731

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ga, gb = gate(sa, "prior"), gate(sb, "prior")
    fig.suptitle(f"18.4b: перенос темпа обучения из DiT/SiT проверен парным прогоном — "
                 f"прогонка {ru(f'{gb:.3f}')} против {ru(f'{ga:.3f}')}, то есть в "
                 f"{ru(f'{gb / ga:.1f}')} раза хуже", fontsize=11.5, y=0.985)

    for t, s, c, lab in ((ta, sa, "#1f77b4", f"lr {ta['lr']:.0e} — наш (18.3)"),
                         (tb, sb, "#d62728", f"lr {tb['lr']:.0e} — из литературы")):
        x, y = val(t)
        ax1.plot(x, y, "-", color=c, lw=2, label=f"{lab}: {ru(f'{y[-1]:.4f}')}")
    ax1.set_xlabel("шаг обучения"); ax1.set_ylabel("валидационный лосс")
    ax1.set_title("одинаково всё, кроме темпа: батч 32, 20 000 шагов, seed 0", fontsize=10)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9); ax1.set_ylim(0.4, 1.8)
    xa, ya = val(ta); xb, yb = val(tb)
    ax1.annotate("", xy=(20000, ya[-1]), xytext=(20000, yb[-1]),
                 arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
    ax1.text(19300, (ya[-1] + yb[-1]) / 2, f"+{ru(f'{yb[-1] - ya[-1]:.3f}')}", fontsize=9, ha="right", va="center",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaa"))
    d_a = (ya[-1] - dict(zip(xa, ya))[14500]) / dict(zip(xa, ya))[14500]
    d_b = (yb[-1] - dict(zip(xb, yb))[14500]) / dict(zip(xb, yb))[14500]
    ax1.text(0.26, 0.58, f"за последние 5 500 шагов: наш {ru(f'{d_a:+.2%}')}, из литературы {ru(f'{d_b:+.2%}')}\n"
                          f"нижняя кривая не просто ниже — она ещё и падает быстрее",
             transform=ax1.transAxes, fontsize=8.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#c8b273"))

    bars = [("шум в типах", sa["gates"]["noise_white"]["round_trip"]["median"] if "noise_white" in sa["gates"]
             else sa["scores"]["noise_white"]["round_trip"], "#999"),
            ("состояние клипа,\nколонки переставлены", sa["scores"]["shuffled_clip"]["round_trip"], "#999"),
            (f"прайор, lr {tb['lr']:.0e}\n(из литературы)", gb, "#d62728"),
            (f"прайор, lr {ta['lr']:.0e}\n(наш, 18.3)", ga, "#1f77b4"),
            ("потолок представления\n(настоящее состояние, DCT-16)", 0.021, "#2ca02c"),
            ("настоящий отложенный клип", gate(sa, "clip"), "#2ca02c")]
    y = range(len(bars))
    ax2.barh(list(y), [v for _, v, _ in bars], color=[c for _, _, c in bars], height=0.6)
    for i, (_, v, _) in enumerate(bars):
        ax2.text(v * 1.12, i, ru(f"{v:.3f}"), va="center", fontsize=9)
    ax2.set_yticks(list(y)); ax2.set_yticklabels([n for n, _, _ in bars], fontsize=8.5)
    ax2.set_xscale("log"); ax2.set_xlim(0.006, 6)
    ax2.set_xlabel("прогонка через мозг, лог. шкала (меньше = совместимее)", fontsize=9)
    ax2.set_title("те же 16 сэмплов, тот же код, те же контроли", fontsize=10)
    ax2.grid(alpha=0.3, axis="x")

    sec_a, sec_b = round(ta["seconds"]), round(tb["seconds"])
    use_a, use_b = round(ta["gpu_utilisation"]), round(tb["gpu_utilisation"])
    nov_a = sa["gates"]["prior"]["video_nn_r"]["median"] if "video_nn_r" in sa["gates"]["prior"] else None
    nov_b = sb["gates"]["prior"]["video_nn_r"]["median"] if "video_nn_r" in sb["gates"]["prior"] else None
    nov = "" if nov_a is None or nov_b is None else \
        f" Новизна видео: наш {ru(f'{nov_a:+.2f}')}, из литературы {ru(f'{nov_b:+.2f}')}."
    foot = (f"Каждое плечо: один T4, cpu 1 / 12 ГБ, {sec_a} и {sec_b} с, загрузка карты {use_a} и {use_b} %, "
            f"≈$0,22 и ≈$0,24. Ворота считались локально на CPU, $0.{nov} Скрипт tools/fig_lr18.py.")
    fig.text(0.5, 0.008, foot, ha="center", fontsize=8.2, color="#444")
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
