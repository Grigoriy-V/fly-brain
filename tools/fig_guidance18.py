"""18.12: что делает направление — и почему одного числа для «сцены» мало.

    python tools/fig_guidance18.py

Слева: одна ручка, две оси. Доля ровных переходов (18.9) растёт с силой
направления, прогонка через мозг портится; обе от одной и той же модели,
вся кривая стоит $0. Посередине: то же самое в плоскости двух мер —
разреженность перепадов против связности с соседями (18.12). Настоящее видео
стоит в правом нижнем углу: редкие резкие границы при крупных ровных
областях. Зернистость на колонку набирает разреженность, теряя связность, и
именно так сила 15 ставит рекорд проекта по одной мере, будучи солью с
перцем. Справа: кадр 20 от одного и того же шума при 1, 10 и 15.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fig_gen13b_pick import to_raster  # noqa: E402

BEST_GATE = 0.0224                            # 18.5: лучшее плечо проекта по воротам
SHOTS = (1, 10, 15)


def tag_of(ckpt: str, g: float) -> str:
    return f"samples18_{ckpt}_g{g:g}".replace(".", "p")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--src", default="edges18_guidance")
    p.add_argument("--arms", default="edges18_all_v2")
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_guidance18.png"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.src}.json").read_text(encoding="utf-8"))
    A = json.loads((run / f"{a.arms}.json").read_text(encoding="utf-8"))["groups"]
    rows = S["rows"]
    rend = A["13B из настоящего состояния"]
    raw = A["сырое видео корпуса"]
    arms = {k: v for k, v in A.items() if k not in ("13B из настоящего состояния", "сырое видео корпуса")}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17.2, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 1.15])
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    xs = np.arange(len(rows))
    lab = [("≈0" if r["guidance"] < 0.01 else f"{r['guidance']:g}") for r in rows]
    flat = np.array([100 * r["frac_flat"] for r in rows])
    gate = np.array([r["gate"] for r in rows])
    neigh = np.array([r["neigh_r"] for r in rows])
    f_rend, f_raw = 100 * rend["frac_flat"]["mean"], 100 * raw["frac_flat"]["mean"]
    n_rend, n_raw = rend["neigh_r"]["mean"], raw["neigh_r"]["mean"]

    # --- 1. одна ручка, две оси ------------------------------------------------
    ax1.axhspan(f_rend, f_raw, color="#2ca02c", alpha=0.14)
    ax1.text(len(xs) - 0.4, f_rend - 0.7, f"настоящее: {ru(f'{f_rend:.1f}')}–{ru(f'{f_raw:.1f}')} %",
             fontsize=9, color="#2ca02c", ha="right", va="top")
    ax1.plot(xs, flat, "o-", color="#1f77b4", lw=2.2, ms=6.5)
    ax1.set_ylabel("ровных переходов, % (разреженность)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(20, 50)
    ax1b = ax1.twinx()
    ax1b.plot(xs, gate, "s--", color="#d62728", lw=2.0, ms=5.5)
    ax1b.axhline(BEST_GATE, color="#d62728", ls=":", lw=1.3)
    ax1b.text(len(xs) - 0.4, BEST_GATE * 0.86, f"лучшие ворота проекта: {ru(f'{BEST_GATE:.4f}')}",
              fontsize=8.6, color="#d62728", ha="right", va="top")
    ax1b.set_yscale("log"); ax1b.set_ylim(0.012, 1.2)
    ax1b.set_ylabel("прогонка (меньше = совместимее)", color="#d62728")
    ax1b.tick_params(axis="y", labelcolor="#d62728")
    i1 = next(i for i, r in enumerate(rows) if r["guidance"] == 1)
    ax1.annotate("метка сама по себе\nне даёт ничего:\n≈0 и 1 совпадают",
                 xy=(i1, flat[i1]), xytext=(1.15, 35.0), fontsize=9, ha="center", color="#333",
                 arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax1.set_xticks(xs); ax1.set_xticklabels(lab, fontsize=9)
    ax1.set_xlabel("сила направления s   (0 — метку игнорируем, 1 — как обучено)")
    ax1.set_title("Ручка двигает ось содержания, ворота платят", fontsize=10.5)
    ax1.grid(alpha=0.25)

    # --- 2. плоскость двух мер -------------------------------------------------
    for k, v in arms.items():
        ax2.plot(v["neigh_r"]["mean"], 100 * v["frac_flat"]["mean"], "o", color="#bbb", ms=6, zorder=1)
    best = arms["w192, lr 1e-3 — лучшее по воротам"]
    ax2.plot(best["neigh_r"]["mean"], 100 * best["frac_flat"]["mean"], "o", color="#1f77b4", ms=10, zorder=3)
    ax2.annotate("прежнее лучшее\nплечо проекта", xy=(best["neigh_r"]["mean"], 100 * best["frac_flat"]["mean"]),
                 xytext=(0.60, 27.5), fontsize=8.8, color="#1f77b4", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.1))
    worst = arms["K=32, w128"]
    ax2.annotate("K=32 w128: связность 0,004 —\nполе без пространственной структуры,\n"
                 "а по одной мере неотличимо от соседей",
                 xy=(worst["neigh_r"]["mean"], 100 * worst["frac_flat"]["mean"]), xytext=(0.30, 38.0),
                 fontsize=8.5, color="#777", ha="left",
                 arrowprops=dict(arrowstyle="->", color="#777", lw=1.0))
    ax2.plot(neigh, flat, "o-", color="#e377c2", lw=2.0, ms=6, zorder=4)
    for i, r in enumerate(rows):
        if r["guidance"] in (1, 3, 5, 7, 10, 15):
            ax2.annotate(f"{r['guidance']:g}", (neigh[i], flat[i]), textcoords="offset points",
                         xytext=(7, -3), fontsize=9, color="#c2185b", fontweight="bold")
    ax2.axhspan(f_rend, f_raw, color="#2ca02c", alpha=0.10)
    ax2.axvspan(min(n_rend, n_raw), max(n_rend, n_raw), color="#2ca02c", alpha=0.10)
    ax2.plot([n_rend, n_raw], [f_rend, f_raw], "*", color="#2ca02c", ms=17, zorder=5)
    ax2.text(n_raw + 0.012, f_raw, "настоящее\nвидео", fontsize=9.5, color="#2ca02c", va="center")
    ax2.set_xlim(-0.02, 0.95); ax2.set_ylim(18, 52)
    ax2.set_xlabel("связность с соседями (крупные области → выше)")
    ax2.set_ylabel("ровных переходов, % (разреженность)")
    ax2.set_title("Одного числа мало: направление покупает разреженность\n"
                  "за связность и в угол не попадает", fontsize=10.5)
    ax2.grid(alpha=0.25)

    # --- 3. кадры --------------------------------------------------------------
    inner = gs[0, 2].subgridspec(1, 4, wspace=0.05)
    ax3.axis("off")
    zs = {g: np.load(run / f"{tag_of(S['ckpt'], g)}.npz") for g in SHOTS}
    zref = zs[SHOTS[0]]
    ref_key = next(k for k in zref.files if k.startswith("video__clip_"))
    shots = [(zs[g]["video__prior_0"], f"s = {g}") for g in SHOTS]
    shots.append((zref[ref_key], "рендерер из\nнастоящего состояния"))
    for j, (v, t) in enumerate(shots):
        b = fig.add_subplot(inner[0, j])
        b.set_xticks([]); b.set_yticks([])
        b.imshow(to_raster(v[20], 721, 4, fill=np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        b.set_title(t, fontsize=9.5)
        b.set_xlabel(f"sd {ru(f'{v.std():.3f}')}", fontsize=8.6)
    ax3.set_title("Кадр 20: один и тот же шум, разная сила", fontsize=10.5, pad=20)

    fig.suptitle("18.12: сама метка не даёт ничего (≈0 = 1); направление двигает структуру, "
                 "но покупает разреженность за связность и до настоящего не доходит", fontsize=12.5, y=0.985)
    nl = "\n"
    r10 = next(r for r in rows if r["guidance"] == 10)
    foot = ("Прайор K=16, ширина 192, lr 1e-3, 110 классов + нулевая метка (label_drop 0,1), 2 655 488 "
            "параметров; обучение 1 202 с на T4 при 94,7 %, cpu 1 / 12 ГБ, $0,26." + nl +
            "Сила направления — ручка на этапе сэмплирования, поэтому вся кривая бесплатна: 11 прогонов "
            "ворот по 45 с локально на CPU, по 16 сэмплов, seed 0; серые точки — все 13 плеч пункта 18." + nl +
            f"Ближе всего к настоящему по связности сила 10 ({ru(f'{r10['neigh_r']:.3f}')} против "
            f"{ru(f'{n_rend:.3f}')}–{ru(f'{n_raw:.3f}')}), но разреженности там "
            f"{ru(f'{100 * r10['frac_flat']:.1f}')} % из {ru(f'{f_rend:.1f}')}, а ворота "
            f"{ru(f'{r10['gate']:.4f}')} против {ru(f'{BEST_GATE:.4f}')}. Скрипт tools/fig_guidance18.py.")
    fig.text(0.5, 0.008, foot, ha="center", fontsize=8.1, color="#444")
    fig.tight_layout(rect=(0, 0.085, 1, 0.935))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
