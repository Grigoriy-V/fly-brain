"""18.8: шаги сэмплера — контраст вырос, ворота стали хуже.

    python tools/fig_steps18.py

Слева: контраст с правильными эталонами. 18.7 сравнивал сгенерированное видео
с сырым видео корпуса, но между ними стоит 13B, который сам теряет амплитуду;
честный потолок для прайора — это отрисовка 13B из настоящего состояния.
В середине: ворота против контраста по всем 32 сэмплам — метрика штрафует
контраст и внутри групп, и между ними. Справа: две отрисовки с почти
одинаковым контрастом, одна из настоящего состояния, другая из
сгенерированного.
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

RUNS = [("samples18_corpus_dct16_w192_lr1e3_c", 20, "#7986cb"),
        ("samples18_corpus_dct16_w192_lr1e3_c_s100", 100, "#3949ab")]


def load(run: Path, tag: str):
    S = json.loads((run / f"{tag}.json").read_text(encoding="utf-8"))
    z = np.load(run / f"{tag}.npz")
    sc = S["scores"]
    pri = [k for k in sc if sc[k]["kind"] == "prior"]
    cli = [k for k in sc if sc[k]["kind"] == "clip"]
    return S, z, sc, pri, cli


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--why", default=str(ROOT / "data" / "prior18" / "why18_local.json"))
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_steps18.png"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only

    raw_sd = json.loads(Path(a.why).read_text(encoding="utf-8"))["groups"]["clip"]["sd"]
    G = {}
    for tag, st, col in RUNS:
        S, z, sc, pri, cli = load(run, tag)
        G[st] = {"sd": np.array([z[f"video__{k}"].std() for k in pri]),
                 "rt": np.array([sc[k]["round_trip"] for k in pri]),
                 "med": S["gates"]["prior"]["round_trip"]["median"], "colour": col,
                 "clip_sd": float(np.mean([z[f"video__{k}"].std() for k in cli])),
                 "clip_rt": S["gates"]["clip"]["round_trip"]["median"],
                 "z": z, "pri": pri, "cli": cli, "frames": S["frames"]}
    g20, g100 = G[20], G[100]
    ceil_sd = g20["clip_sd"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.8, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.15, 1.0])
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    bars = [("настоящее видео\nкорпуса", raw_sd, "#2ca02c"),
            ("13B из настоящего\nсостояния", ceil_sd, "#8c8c8c"),
            ("прайор,\n20 шагов", float(g20["sd"].mean()), g20["colour"]),
            ("прайор,\n100 шагов", float(g100["sd"].mean()), g100["colour"])]
    xs = np.arange(len(bars))
    ax1.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
    for x, b in zip(xs, bars):
        ax1.text(x, b[1] * 1.03, ru(f"{b[1]:.3f}"), ha="center", fontsize=10)
    ax1.axhline(ceil_sd, color="#8c8c8c", ls="--", lw=1.5)
    ax1.text(0.62, ceil_sd * 0.90, "потолок отрисовщика", fontsize=9, color="#555", ha="left", va="top")
    for x, b in list(zip(xs, bars))[2:]:
        ax1.text(x, b[1] * 0.5, f"{100 * b[1] / ceil_sd:.0f} %", ha="center", fontsize=11,
                 color="white", fontweight="bold")
    ax1.set_xticks(xs); ax1.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax1.set_ylabel("контраст: ст. отклонение по клипу")
    ax1.set_ylim(0, raw_sd * 1.18)
    ax1.set_title("Эталон был выбран неверно: 13B сам теряет 45 %\nамплитуды, и прайор уже у его потолка",
                  fontsize=10.5)
    ax1.grid(alpha=0.3, axis="y")

    for st in (20, 100):
        g = G[st]
        med = g["med"]
        ax2.scatter(g["sd"], g["rt"], s=46, color=g["colour"], edgecolor="white", linewidth=0.7,
                    label=f"{st} шагов, медиана {ru(f'{med:.4f}')}")
    ax2.set_xlabel("контраст сэмпла"); ax2.set_ylabel("прогонка через мозг")
    ax2.set_title("Ворота штрафуют контраст", fontsize=10.5)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9)

    target = g100["sd"]
    j = int(np.argmin(np.abs(target - ceil_sd)))
    mid = g20["frames"] // 2
    shots = [(g20["z"][f"video__{g20['cli'][0]}"], "13B из настоящего состояния"),
             (g100["z"][f"video__{g100['pri'][j]}"], "прайор, 100 шагов")]
    inner = gs[0, 2].subgridspec(1, 2, wspace=0.06)
    ax3.axis("off")
    for k, (v, t) in enumerate(shots):
        b = fig.add_subplot(inner[0, k])
        b.set_xticks([]); b.set_yticks([])
        b.imshow(to_raster(v[mid], 721, 4, fill=np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        b.set_title(t, fontsize=9.5)
        b.set_xlabel(f"контраст {ru(f'{v.std():.3f}')}", fontsize=9)
    ax3.set_title("При одинаковом контрасте одно — сцена,\nдругое — нет", fontsize=10.5, pad=18)

    fig.suptitle("18.8: больше шагов сэмплера — контраста на 18 % больше, а ворота на 34 % хуже",
                 fontsize=12.5, y=0.985)
    nl = "\n"
    r20 = float(np.corrcoef(g20["sd"], g20["rt"])[0, 1])
    r100 = float(np.corrcoef(g100["sd"], g100["rt"])[0, 1])
    foot = ("Меняется только интегрирование прайора: 13B рисует своими 20 шагами в обоих прогонах "
            "(`samples17 --prior-steps`, добавлено в 18.8; 0 воспроизводит все прежние ворота)." + nl +
            f"По шестнадцати сэмплам внутри группы контраст и прогонка связаны положительно: "
            f"r = {ru(f'{r20:+.2f}')} при 20 шагах и {ru(f'{r100:+.2f}')} при 100 — больше контраста, хуже ворота." + nl +
            "Локально на CPU, $0. Скрипт tools/fig_steps18.py.")
    fig.text(0.5, 0.008, foot, ha="center", fontsize=8.2, color="#444")
    fig.tight_layout(rect=(0, 0.085, 1, 0.935))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
