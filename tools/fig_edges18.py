"""18.9: контраст совпадает по-честному, а различает разреженность границ.

    python tools/fig_edges18.py

Слева: устойчивые меры разброса сэмпла при 100 шагах в долях от того же у
отрисовки 13B из настоящего состояния. Если бы совпадение по ст. отклонению
держалось на выбросах, межквартильный размах и MAD его не подтвердили бы.
В середине: доля ровных переходов между соседними колонками — у сцены
большие ровные области с редкими резкими границами. Справа: те же два кадра
с этим числом под ними.
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

CEIL = "13B из настоящего состояния"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--name", default="edges18_local")
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_edges18.png"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    G = json.loads((run / f"{a.name}.json").read_text(encoding="utf-8"))["groups"]
    ceil, s100 = G[CEIL], G["прайор, 100 шагов"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.8, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.3, 1.0])
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    meas = [("ст. откл.", "sd"), ("межквартильный\nразмах", "iqr"),
            ("медианное абс.\nотклонение", "mad"), ("размах\n5–95 %", "p5_95")]
    xs = np.arange(len(meas))
    rel = [s100[k]["mean"] / ceil[k]["mean"] for _, k in meas]
    ax1.bar(xs, rel, color="#3949ab", width=0.6)
    ax1.axhline(1.0, color="#333", lw=1.6)
    for x, r in zip(xs, rel):
        ax1.text(x, r + 0.025, f"×{ru(f'{r:.2f}')}", ha="center", fontsize=10.5)
    ax1.set_xticks(xs); ax1.set_xticklabels(["ст. откл.", "IQR", "MAD", "5–95 %"], fontsize=9.5)
    ax1.set_ylim(0, 1.35)
    ax1.set_ylabel("сэмпл ÷ отрисовка из настоящего", fontsize=9.5)
    ax1.set_title("Совпадение по амплитуде не держится на выбросах:\nустойчивые меры дают то же самое",
                  fontsize=10.5)
    ax1.grid(alpha=0.3, axis="y")

    order = ["прайор, 20 шагов", "прайор, 100 шагов", CEIL, "сырое видео корпуса"]
    cols = ["#7986cb", "#3949ab", "#8c8c8c", "#2ca02c"]
    ys = np.arange(len(order))
    flat = [100 * G[g]["frac_flat"]["mean"] for g in order]
    ax2.barh(ys, flat, color=cols, height=0.55)
    raw = G["сырое видео корпуса"]
    ax2.axvspan(100 * raw["frac_flat"]["p10"], 100 * raw["frac_flat"]["p90"], color="#2ca02c", alpha=0.12)
    for y, g, v in zip(ys, order, flat):
        k = G[g]["grad_kurtosis"]["mean"]
        ax2.text(v + 0.9, y, f"{ru(f'{v:.1f}')} %   (эксцесс перепада {ru(f'{k:.1f}')})", va="center", fontsize=9.5)
    ax2.set_yticks(ys); ax2.set_yticklabels([g.replace(", ", ",\n") for g in order], fontsize=9)
    ax2.set_xlim(0, 78)
    ax2.set_xlabel("доля ровных переходов между соседними колонками, %")
    ax2.set_title("А различает разреженность границ:\nу сцены половина поля ровная, у сэмпла четверть", fontsize=10.5)
    ax2.grid(alpha=0.3, axis="x")

    S = json.loads((run / "samples18_corpus_dct16_w192_lr1e3_c.json").read_text(encoding="utf-8"))
    z20 = np.load(run / "samples18_corpus_dct16_w192_lr1e3_c.npz")
    z100 = np.load(run / "samples18_corpus_dct16_w192_lr1e3_c_s100.npz")
    S100 = json.loads((run / "samples18_corpus_dct16_w192_lr1e3_c_s100.json").read_text(encoding="utf-8"))
    cli = [k for k in S["scores"] if S["scores"][k]["kind"] == "clip"][0]
    pri = [k for k in S100["scores"] if S100["scores"][k]["kind"] == "prior"]
    sds = np.array([z100[f"video__{k}"].std() for k in pri])
    j = int(np.argmin(np.abs(sds - z20[f"video__{cli}"].std())))
    mid = S["frames"] // 2
    shots = [(z20[f"video__{cli}"], "13B из настоящего состояния", ceil),
             (z100[f"video__{pri[j]}"], "прайор, 100 шагов", s100)]
    inner = gs[0, 2].subgridspec(1, 2, wspace=0.06)
    ax3.axis("off")
    for k2, (v, t, st) in enumerate(shots):
        b = fig.add_subplot(inner[0, k2])
        b.set_xticks([]); b.set_yticks([])
        b.imshow(to_raster(v[mid], 721, 4, fill=np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        b.set_title(t.replace(" состояния", ""), fontsize=8.8)
        fl = 100 * st["frac_flat"]["mean"]
        b.set_xlabel(f"ст. откл. {ru(f'{v.std():.3f}')}\nровного {ru(f'{fl:.0f}')} %", fontsize=9)
    ax3.set_title("Одна амплитуда, разное устройство", fontsize=10.5, pad=18)

    fig.suptitle("18.9: амплитуда совпадает по всем устойчивым мерам — различает то, "
                 "как эта амплитуда разложена по полю", fontsize=12.3, y=0.985)
    nl = "\n"
    k_s, k_c, k_r = (G[g]["grad_kurtosis"]["mean"] for g in ("прайор, 100 шагов", CEIL, "сырое видео корпуса"))
    foot = ("Перепад — разность между колонкой и каждым из шести соседей; «ровно» — модуль перепада меньше "
            "0,25 его ст. отклонения. IQR — межквартильный размах, MAD — медианное абсолютное отклонение." + nl +
            f"Эксцесс распределения перепада: {ru(f'{k_s:.1f}')} у сэмпла против {ru(f'{k_c:.1f}')} у отрисовки "
            f"из настоящего состояния и {ru(f'{k_r:.1f}')} у сырого видео корпуса (n=32, полоса — 10–90 %)." + nl +
            "Локально на CPU, $0. Скрипты flydream/generate/edges18.py и tools/fig_edges18.py.")
    fig.text(0.5, 0.008, foot, ha="center", fontsize=8.2, color="#444")
    fig.tight_layout(rect=(0, 0.085, 1, 0.935))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
