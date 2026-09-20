"""18.7: почему сцены нет — три проверки в одном листе.

    python tools/fig_why18.py

Слева: контраст (ст. отклонение по клипу) у сэмплов при 20 / 100 / 250 шагах
прайора против настоящих клипов корпуса. В середине: насколько сэмпл похож на
среднее видео корпуса — проверка грубой версии «усреднения мод». Справа:
средний кадр одной и той же начальной точки, проинтегрированной за 20 и за
250 шагов, и настоящий клип рядом.
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--name", default="why18_local")
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_why18.png"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.name}.json").read_text(encoding="utf-8"))
    z = np.load(run / f"{a.name}.npz")
    gr, steps = S["groups"], S["steps_list"]
    clip = gr["clip"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.6, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.35])
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    keys = [f"prior_{s}" for s in steps]
    cols = ["#c5cae9", "#7986cb", "#3949ab"]
    xs = np.arange(len(keys))
    sds = [gr[k]["sd"] for k in keys]
    ax1.bar(xs, sds, color=cols[:len(keys)], width=0.6)
    for x, k, v in zip(xs, keys, sds):
        ax1.errorbar(x, v, yerr=[[v - min(gr[k]["sd_per_clip"])], [max(gr[k]["sd_per_clip"]) - v]],
                     fmt="none", ecolor="#333", elinewidth=1.1, capsize=4)
        ax1.text(x, max(gr[k]["sd_per_clip"]) * 1.03, ru(f"{v:.3f}"), ha="center", fontsize=10)
        ax1.text(x, v * 0.5, f"{100 * v / clip['sd']:.0f} %", ha="center", fontsize=11,
                 color="white", fontweight="bold")
    ax1.axhline(clip["sd"], color="#2ca02c", ls="--", lw=1.8)
    clip_sd = clip["sd"]
    ax1.text(len(keys) - 0.5, clip_sd * 1.02, f"настоящий клип корпуса: {ru(f'{clip_sd:.3f}')}",
             fontsize=9.5, color="#2ca02c", ha="right", va="bottom")
    ax1.set_xticks(xs); ax1.set_xticklabels([f"{s} шагов" for s in steps], fontsize=10)
    ax1.set_ylim(0, clip["sd"] * 1.22)
    ax1.set_ylabel("контраст: ст. отклонение по клипу")
    ax1.set_title("Контраста вдвое меньше, и шаги сэмплера\nзакрывают только часть разрыва", fontsize=10.5)
    ax1.grid(alpha=0.3, axis="y")

    rm = [float(np.mean(gr[k]["r_to_corpus_mean"])) for k in keys] + [float(np.mean(clip["r_to_corpus_mean"]))]
    lab = [f"{s} шагов" for s in steps] + ["настоящий клип"]
    ax2.barh(np.arange(len(rm)), rm, color=cols[:len(keys)] + ["#2ca02c"], height=0.55)
    for i, v in enumerate(rm):
        ax2.text(v + 0.0012, i, ru(f"{v:+.3f}"), va="center", fontsize=10)
    ax2.set_yticks(np.arange(len(rm))); ax2.set_yticklabels(lab, fontsize=10)
    ax2.set_xlim(-0.004, 0.030)
    ax2.set_xlabel("корреляция со средним видео корпуса")
    ax2.set_title("Грубого схлопывания в среднее нет:\nсэмплы от среднего почти так же далеки", fontsize=10.5)
    ax2.grid(alpha=0.3, axis="x")

    mid = S["frames"] // 2
    shots = [(z[f"video__prior_{steps[0]}"][0], f"сэмпл, {steps[0]} шагов"),
             (z[f"video__prior_{steps[-1]}"][0], f"тот же шум, {steps[-1]} шагов"),
             (z["video__clip"][0], "настоящий клип")]
    inner = gs[0, 2].subgridspec(1, 3, wspace=0.05)
    ax3.axis("off")
    for j, (v, t) in enumerate(shots):
        b = fig.add_subplot(inner[0, j])
        b.set_xticks([]); b.set_yticks([])
        b.imshow(to_raster(v[mid], 721, 4, fill=np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        b.set_title(t, fontsize=9.5)
        b.set_xlabel(f"sd {ru(f'{v.std():.3f}')}", fontsize=9)
    ax3.set_title("Кадр 20: резче, но всё ещё не сцена", fontsize=10.5, pad=18)

    fig.suptitle("18.7: сцена теряется в прайоре — контраста 53 % от настоящего, и это не схлопывание в среднее",
                 fontsize=12.5, y=0.985)
    nl = "\n"
    ring_p, ring_c = gr[keys[0]]["ring1_spatial"], clip["ring1_spatial"]
    lag_p, lag_c = gr[keys[0]]["temporal_lag1"], clip["temporal_lag1"]
    par = f"{S['prior_parameters']:,}"
    foot = ("Одна и та же начальная точка, разное число шагов Эйлера у прайора; 13B рисует везде своими 20 шагами "
            "и с одним и тем же z, так что меняется только интегрирование прайора." + nl +
            f"Прайор — лучшее плечо по воротам (K=16, ширина 192, lr 1e-3, {par} параметров). "
            f"{S['n']} сэмплов, {S['n_ref']} отложенных клипов, среднее по {S['n_mean']} обучающим." + nl +
            f"Одно кольцо {ru(f'{ring_p:.3f}')} против {ru(f'{ring_c:.3f}')} и лаг-1 {ru(f'{lag_p:.3f}')} "
            f"против {ru(f'{lag_c:.3f}')} — обе статистики у потолка и групп не различают. "
            "Локально на CPU, $0. Скрипт tools/fig_why18.py.")
    fig.text(0.5, 0.008, foot, ha="center", fontsize=8.1, color="#444")
    fig.tight_layout(rect=(0, 0.085, 1, 0.935))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
