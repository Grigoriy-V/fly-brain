"""18.17: где в пространстве шума лежит сцена — и почему сэмплер туда не попадает.

    python tools/fig_reach18.py

Слева: радиус. Гауссов шум в 92 288 измерениях садится на сферу радиуса
√D = 303,8 шириной всего 0,71, а шум, из которого прайор выдаёт настоящее
состояние, лежит на 252–257 — это **65–73 стандартных отклонения** внутрь.
Обратимость потока делает сцену достижимой тривиально; нетривиально то, что
эта точка недостижима розыгрышем. Справа: две попытки туда попасть длиной —
нормировать шум сцены обратно на √D и просто тянуть короткий шум — и обе
проваливаются, причём вторая роняет контраст втрое, а **ворота при этом
улучшаются вдвое**, что окончательно показывает, чему они рады.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARMS = [("reach18_s100", "K=16, w384", "#1f77b4"),
        ("reach18_corpus_dct16_w192_lr1e3_c", "K=16, w192", "#2ca02c"),
        ("reach18_corpus_dct16_c", "K=16, w128", "#ff7f0e")]
NS = [("samples18_w384_ns1p0", 1.0), ("samples18_w384_ns0p9", 0.9),
      ("samples18_w384_ns0p83", 0.83), ("samples18_w384_ns0p75", 0.75)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_reach18.png"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = {t: json.loads((run / f"{t}.json").read_text(encoding="utf-8")) for t, _, _ in ARMS}
    E = json.loads((run / "edges18_noise.json").read_text(encoding="utf-8"))["groups"]
    main_s = S["reach18_s100"]
    D, tgt = main_s["dims"], main_s["typical_radius"]
    sd_shell = 1 / np.sqrt(2)                                        # ст. отклонение радиуса χ_D

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.6, 5.3), gridspec_kw={"width_ratios": [1.1, 1.15, 1]})

    # --- 1. радиус -------------------------------------------------------------
    ax1.axvspan(tgt - 3 * sd_shell, tgt + 3 * sd_shell, color="#d62728", alpha=0.25)
    ax1.axvline(tgt, color="#d62728", lw=2)
    ax1.text(tgt + 1.5, 2.55, f"сюда садится ЛЮБОЙ розыгрыш:\n√D = {ru(f'{tgt:.1f}')} ± "
                              f"{ru(f'{sd_shell:.2f}')}", fontsize=9.5, color="#d62728", va="center")
    for i, (t, lab, col) in enumerate(ARMS):
        r = S[t]["noise"]["radius_per_clip"]
        ax1.plot(r, [1.6 - 0.45 * i] * len(r), "o", color=col, ms=9, alpha=0.85)
        m = float(np.mean(r))
        ax1.text(m, 1.6 - 0.45 * i + 0.16, f"{lab}: {ru(f'{m:.0f}')}  "
                 f"({ru(f'{(tgt - m) / sd_shell:.0f}')} σ внутрь)", fontsize=9.3, color=col, ha="center")
    ax1.set_ylim(0, 3.0); ax1.set_xlim(240, 312)
    ax1.set_yticks([])
    ax1.set_xlabel("радиус шума ‖ε‖ в пространстве размерности 92 288")
    ax1.set_title("Шум, дающий сцену, лежит на 65–73 σ\nвнутрь от оболочки, где живёт сэмплер", fontsize=10.5)
    ax1.grid(alpha=0.3, axis="x")

    # --- 2. структура: что получается из чего ----------------------------------
    keys = ["настоящее состояние", "из его шума", "тот же шум на радиусе √D", "обычный розыгрыш"]
    short = ["настоящее\nсостояние", "из его\nшума", "шум сцены,\nнормирован √D", "обычный\nрозыгрыш"]
    cols = ["#2ca02c", "#1f77b4", "#9467bd", "#999"]
    flat = [100 * main_s["groups"][k]["structure"]["frac_flat"]["mean"] for k in keys]
    kurt = [main_s["groups"][k]["structure"]["grad_kurtosis"]["mean"] for k in keys]
    xs = np.arange(len(keys))
    ax2.bar(xs, flat, color=cols, width=0.62)
    for x, f, k in zip(xs, flat, kurt):
        ax2.text(x, f + 0.6, ru(f"{f:.1f}") + " %", ha="center", fontsize=10)
        ax2.text(x, f / 2, f"эксцесс\n{ru(f'{k:.2f}')}", ha="center", fontsize=8.8, color="white",
                 fontweight="bold")
    ax2.set_xticks(xs); ax2.set_xticklabels(short, fontsize=8.2)
    ax2.set_ylim(0, 46)
    ax2.set_ylabel("ровных переходов, % (разреженность)")
    ax2.set_title("Обращение потока тривиально возвращает состояние;\n"
                  "а вот исправить один только радиус — уже нет", fontsize=10.5)
    ax2.grid(alpha=0.3, axis="y")

    # --- 3. короткий шум: ворота хвалят, картинка гаснет ------------------------
    sig = [s for _, s in NS]
    lab = {0.83: "sigma 0.83", 0.9: "sigma 0.9", 0.75: "sigma 0.75", 1.0: "sigma 1.0"}
    fl = [100 * E[lab[s]]["frac_flat"]["mean"] for s in sig]
    ct = [E[lab[s]]["sd"]["mean"] for s in sig]
    gt = [json.loads((run / f"{t}.json").read_text(encoding="utf-8"))["gates"]["prior"]["round_trip"]["median"]
          for t, _ in NS]
    ax3.plot(sig, fl, "o-", color="#1f77b4", lw=2.2, ms=7, label="разреженность, %")
    ax3.plot(sig, [1000 * c for c in ct], "s-", color="#2ca02c", lw=2.0, ms=6,
             label="контраст × 1000")
    ax3.set_xlabel("длина стартового шума σ")
    ax3.set_ylabel("разреженность, % и контраст × 1000")
    ax3.set_ylim(0, 150)
    ax3b = ax3.twinx()
    ax3b.plot(sig, gt, "^--", color="#d62728", lw=2.0, ms=8)
    ax3b.set_ylabel("прогонка через мозг", color="#d62728")
    ax3b.tick_params(axis="y", labelcolor="#d62728")
    ax3b.set_ylim(0, 0.021)
    ax3b.annotate("ворота УЛУЧШАЮТСЯ вдвое,\nпока картинка гаснет", xy=(0.83, gt[2]),
                  xytext=(0.80, 0.0155), fontsize=9.2, color="#d62728",
                  arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3))
    ax3.invert_xaxis()
    ax3.set_title("Просто укоротить шум нельзя:\nсэмпл гаснет в серое", fontsize=10.5)
    ax3.legend(fontsize=9, loc="upper left"); ax3.grid(alpha=0.3)

    fig.suptitle("18.17: сцена в пространстве шума достижима, но лежит на 73 σ внутрь от оболочки — "
                 "и длиной туда не попасть", fontsize=12.5, y=0.985)
    nl = "\n"
    n = main_s["noise"]
    foot = (f"Обращение потока (`prior17.to_noise`, 100 шагов Эйлера, 3 итерации неподвижной точки) на "
            f"{main_s['n_clips']} отложенных клипах корпуса; восстановление состояния "
            f"{ru(f'{100 * n['recon_rel_error']:.1f}')} % относительной ошибки." + nl +
            f"Обратимый поток достигает ЛЮБОГО состояния по построению, поэтому «прайор умеет сцену» само по "
            f"себе ничего не значит; измеряется здесь геометрия — где эта точка лежит." + nl +
            f"Ст. отклонение обращённого шума по осям {ru(f'{n['sd_per_dim']:.3f}')} вместо 1,000 у всех трёх "
            f"плеч (0,815 / 0,848 / 0,833): образ данных под обратным потоком уже стандартного гауссова." + nl +
            f"Всё локально на CPU, $0. Скрипт tools/fig_reach18.py.")
    fig.text(0.5, 0.008, foot, ha="center", fontsize=8.1, color="#444")
    fig.tight_layout(rect=(0, 0.105, 1, 0.935))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
