"""18.18: область сцен широкая по пути, но направление у каждой своё.

    python tools/fig_walk18.py

Слева: сферическая интерполяция от обычного розыгрыша к шуму настоящего
состояния. Структура растёт **плавно и рано** — к 40 % пути закрыта почти
треть разрыва, к 80 % почти девять десятых. Область сцен не иголка.
В середине: решающее сравнение при почти одинаковом радиусе — пройти 60 %
пути в нужную **сторону** против того, чтобы просто укоротить случайный шум
(18.17). Радиусы 267 и 273, а исходы противоположны. Справа: прообразы
разных сцен между собой почти ортогональны, то есть общего подпространства,
из которого можно было бы тянуть, нет.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 18.17, развёртка по длине шума на том же плече: (радиус, ровного, контраст)
SHRINK = {0.9: (273.4, 0.214, 0.047), 0.83: (252.2, 0.206, 0.037)}
# измерено /tmp/subspace.py, 16 отложенных клипов, тот же приор
COS = {"mean": 0.0123, "sd": 0.0109, "max": 0.0484,
       "rnd_mean": 0.0001, "rnd_sd": 0.0034, "rnd_max": 0.0102,
       "energy_shared": 0.0735, "top_eig": 1.213}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--src", default="walk18_local")
    p.add_argument("--shared", default="walk18_shared")
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_walk18.png"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.src}.json").read_text(encoding="utf-8"))
    rows = [r for r in S["rows"] if r["alpha"] is not None]
    real = next(r for r in S["rows"] if r["alpha"] is None)
    H = json.loads((run / f"{a.shared}.json").read_text(encoding="utf-8"))
    sh = [r for r in H["rows"] if r["alpha"] is not None]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.8, 5.4), gridspec_kw={"width_ratios": [1.25, 1, 1.1]})

    al = [r["alpha"] for r in rows]
    fl = [100 * r["frac_flat"] for r in rows]
    ku = [r["kurtosis"] for r in rows]
    ax1.axhline(100 * real["frac_flat"], color="#2ca02c", ls="--", lw=1.8)
    ax1.text(0.02, 100 * real["frac_flat"] + 0.5, f"настоящее состояние: {ru(f'{100 * real['frac_flat']:.1f}')} %",
             fontsize=9.5, color="#2ca02c")
    ax1.plot(al, fl, "o-", color="#1f77b4", lw=2.4, ms=7.5, label="к прообразу своей сцены")
    ax1.plot([r["alpha"] for r in sh], [100 * r["frac_flat"] for r in sh], "o-", color="#8c564b",
             lw=2.2, ms=6, label="к общей компоненте")
    ax1.legend(fontsize=9, loc="lower right")
    for r, f in zip(rows, fl):
        if r["alpha"] in (0.0, 0.4, 0.6, 0.8, 1.0):
            ax1.annotate(f"‖ε‖ {r['radius']:.0f}", (r["alpha"], f), textcoords="offset points",
                         xytext=(2, -15), fontsize=8.6, color="#555")
    ax1.set_ylabel("ровных переходов, % (разреженность)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(20, 49)
    ax1b = ax1.twinx()
    ax1b.plot(al, ku, "s--", color="#9467bd", lw=1.8, ms=5.5)
    ax1b.plot([r["alpha"] for r in sh], [r["kurtosis"] for r in sh], "s--", color="#c49a8a", lw=1.5, ms=4.5)
    ax1b.axhline(real["kurtosis"], color="#9467bd", ls=":", lw=1.2)
    ax1b.set_ylabel("эксцесс перепадов", color="#9467bd")
    ax1b.tick_params(axis="y", labelcolor="#9467bd")
    ax1b.set_ylim(3.5, 12)
    ax1.set_xlabel("доля пути от обычного розыгрыша к шуму сцены")
    ax1.set_title("Своё направление работает, общее вредит:\nструктура растёт плавно, а по общему падает", fontsize=10.5)
    ax1.grid(alpha=0.3)

    r06 = next(r for r in rows if r["alpha"] == 0.6)
    pairs = [(f"60 % пути\nв нужную сторону\n‖ε‖ {r06['radius']:.0f}", 100 * r06["frac_flat"], r06["sd"], "#1f77b4"),
             (f"просто короче,\nσ = 0,9\n‖ε‖ {SHRINK[0.9][0]:.0f}", 100 * SHRINK[0.9][1], SHRINK[0.9][2], "#d62728")]
    xs = np.arange(len(pairs))
    ax2.bar(xs - 0.18, [p[1] for p in pairs], width=0.34, color=[p[3] for p in pairs], label="разреженность, %")
    ax2.bar(xs + 0.18, [1000 * p[2] for p in pairs], width=0.34, color=[p[3] for p in pairs], alpha=0.42,
            label="контраст × 1000")
    for x, p_ in zip(xs, pairs):
        ax2.text(x - 0.18, p_[1] + 1.0, ru(f"{p_[1]:.1f}"), ha="center", fontsize=10)
        ax2.text(x + 0.18, 1000 * p_[2] + 1.0, ru(f"{1000 * p_[2]:.0f}"), ha="center", fontsize=10)
    ax2.set_xticks(xs); ax2.set_xticklabels([p[0] for p in pairs], fontsize=9)
    ax2.set_ylim(0, 155)
    ax2.set_title("Радиус почти один и тот же — исходы\nпротивоположные: решает направление", fontsize=10.5)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")

    lo, hi = -0.02, 0.06
    xg = np.linspace(lo, hi, 400)
    for m, s, col, lab in ((COS["rnd_mean"], COS["rnd_sd"], "#999", "случайные векторы"),
                           (COS["mean"], COS["sd"], "#1f77b4", "прообразы сцен")):
        ax3.plot(xg, np.exp(-0.5 * ((xg - m) / s) ** 2), "-", color=col, lw=2.4, label=lab)
        ax3.fill_between(xg, np.exp(-0.5 * ((xg - m) / s) ** 2), color=col, alpha=0.18)
    ax3.axvline(0, color="#333", lw=1.0)
    ax3.text(COS["mean"] + 0.004, 0.55, f"среднее +{ru(f'{COS['mean']:.4f}')}\n"
             f"(у случайных +{ru(f'{COS['rnd_mean']:.4f}')})", fontsize=9.2, color="#1f77b4")
    ax3.text(0.5, 0.965, f"общая компонента держит всего "
             f"{ru(f'{100 * COS['energy_shared']:.1f}')} % энергии;\n"
             f"главное собственное число {ru(f'{COS['top_eig']:.2f}')} против 1 у случайных",
             transform=ax3.transAxes, ha="center", va="top", fontsize=9.2,
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff8f0", ec="#8c564b", lw=1.0))
    ax3.set_xlim(lo, hi); ax3.set_ylim(0, 1.18)
    ax3.set_yticks([])
    ax3.set_xlabel("косинус между прообразами двух разных сцен")
    ax3.set_title("Но направление у каждой сцены своё:\nобщего подпространства нет", fontsize=10.5)
    ax3.legend(fontsize=9, loc="center left"); ax3.grid(alpha=0.3, axis="x")

    fig.suptitle("18.18: до сцены можно дойти направленно, и область широкая — но у каждой сцены "
                 "своё направление, почти ортогональное прочим", fontsize=12.5, y=0.985)
    nl = "\n"
    r04, r08 = (next(r for r in rows if r["alpha"] == v) for v in (0.4, 0.8))
    foot = (f"Сферическая интерполяция между обычным розыгрышем и прообразом настоящего состояния "
            f"(`prior17.slerp`), {S['n_clips']} отложенных клипа, лучшее плечо K=16 / w384, "
            f"100 шагов Эйлера." + nl +
            f"Закрыто разрыва по разреженности: к 40 % пути "
            f"{ru(f'{100 * (r04['frac_flat'] - rows[0]['frac_flat']) / (real['frac_flat'] - rows[0]['frac_flat']):.0f}')} %, "
            f"к 60 % {ru(f'{100 * (r06['frac_flat'] - rows[0]['frac_flat']) / (real['frac_flat'] - rows[0]['frac_flat']):.0f}')} %, "
            f"к 80 % {ru(f'{100 * (r08['frac_flat'] - rows[0]['frac_flat']) / (real['frac_flat'] - rows[0]['frac_flat']):.0f}')} %." + nl +
            f"Ортогональность измерена на 16 отложенных клипах того же приора; ожидаемое sd косинуса для "
            f"случайных векторов в 92 288 измерениях — 0,0033, измеренное у случайных 0,0034." + nl +
            f"Всё локально на CPU, $0. Скрипты flydream/generate/walk18.py, tools/fig_walk18.py.")
    fig.text(0.5, 0.008, foot, ha="center", fontsize=8.1, color="#444")
    fig.tight_layout(rect=(0, 0.105, 1, 0.935))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
