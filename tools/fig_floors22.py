r"""22.8: полы судей и калиброванная шкала.

    python tools/fig_floors22.py

Ни одного нового прогона: читает `floors22.json` и `seed22_ab1536.json`.
Смысл картинки в том, что без пола ни одно число из правой таблицы прочитать
нельзя, а с полом две строки читаются иначе, чем читались.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GROUPS = [("настоящее состояние", "13B из настоящего\nсостояния (потолок)"),
          ("сид от клипа через поток", "сид клипа\nчерез поток"),
          ("только PCA (19.1)", "только PCA-1536"),
          ("свежий розыгрыш", "СВЕЖИЙ РОЗЫГРЫШ"),
          ("розыгрыш + sdproj", "розыгрыш + поправка\nсэмплера (20)")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--prefix", default="2026-09-21_malecns_floors22")
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    F = json.loads((run / "floors22.json").read_text(encoding="utf-8"))
    S = json.loads((run / "seed22_ab1536.json").read_text(encoding="utf-8"))
    lo, hi = F["arms"]["белый шум"], F["arms"]["настоящий клип"]
    cal = lambda v, k: 100 * (v - lo[k]) / (hi[k] - lo[k])            # noqa: E731

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    C = {"flat": "#1b5e20", "kurt": "#7b1fa2", "draw": "#b71c1c", "grey": "#777777"}
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # 1. калиброванная лестница
    names = [h for _, h in GROUPS]
    flat = [cal(S["groups"][k]["frac_flat"], "frac_flat") for k, _ in GROUPS]
    kurt = [cal(S["groups"][k]["kurtosis"], "kurtosis") for k, _ in GROUPS]
    y = np.arange(len(names))
    ax[0].barh(y + 0.19, flat, 0.36, color=C["flat"], label="резкость (доля ровного поля)")
    ax[0].barh(y - 0.19, kurt, 0.36, color=C["kurt"], label="разреженность градиента")
    for i, (f, k) in enumerate(zip(flat, kurt)):
        ax[0].text(f + 1.5, i + 0.19, f"{f:.0f}", va="center", fontsize=8.5, color=C["flat"])
        ax[0].text(k + 1.5, i - 0.19, f"{k:.0f}", va="center", fontsize=8.5, color=C["kurt"])
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=8)
    ax[0].invert_yaxis(); ax[0].set_xlim(0, 100)
    ax[0].axvline(100, color="k", lw=1.2); ax[0].axvline(0, color="k", lw=1.2)
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[0].set_title("0 = шум, 100 = сырое видео корпуса", fontsize=10)

    # 2. ближайшее обучающее видео
    pts = [("белый шум", F["arms"]["белый шум"]["nearest_r"], C["grey"]),
           ("шум по статистике клипа", F["arms"]["шум по статистике клипа"]["nearest_r"], C["grey"]),
           ("НАСТОЯЩИЙ клип\n(банк без него самого)", F["arms"]["настоящий клип"]["nearest_r"], C["flat"]),
           ("СВЕЖИЙ РОЗЫГРЫШ", S["groups"]["свежий розыгрыш"]["nearest_r"], C["draw"]),
           ("сид клипа через поток", S["groups"]["сид от клипа через поток"]["nearest_r"], C["grey"]),
           ("13B из настоящего состояния", S["groups"]["настоящее состояние"]["nearest_r"], C["grey"])]
    yy = np.arange(len(pts))
    ax[1].barh(yy, [v for _, v, _ in pts], 0.55, color=[c for _, _, c in pts])
    for i, (_, v, _) in enumerate(pts):
        ax[1].text(v + 0.012, i, ru(f"{v:.3f}"), va="center", fontsize=8.5)
    ax[1].set_yticks(yy); ax[1].set_yticklabels([n for n, _, _ in pts], fontsize=8)
    ax[1].invert_yaxis(); ax[1].set_xlim(0, 1.0)
    ax[1].set_title("ближайшее из 15 514 обучающих", fontsize=10)

    # 3. r к сырому со своим полом и своим контролем
    fl = F["r_to_raw"]["два разных клипа"]
    tsh = F["r_to_raw"]["свой клип, перемешанный во времени"]["mean"]
    bars = [("два разных настоящих клипа\n(ПОЛ колонки)", fl["mean"], C["grey"]),
            ("СВЕЖИЙ РОЗЫГРЫШ", S["groups"]["свежий розыгрыш"]["r_to_raw"], C["draw"]),
            ("свой клип, перемешанный\nво времени (контроль)", tsh, C["grey"]),
            ("только PCA-1536", S["groups"]["только PCA (19.1)"]["r_to_raw"], C["flat"]),
            ("13B из настоящего\nсостояния (потолок)", S["groups"]["настоящее состояние"]["r_to_raw"], C["grey"])]
    yy = np.arange(len(bars))
    ax[2].axvspan(fl["mean"] - fl["sd"], fl["mean"] + fl["sd"], color=C["grey"], alpha=0.18)
    ax[2].barh(yy, [v for _, v, _ in bars], 0.55, color=[c for _, _, c in bars])
    for i, (_, v, _) in enumerate(bars):
        ax[2].text(v + 0.012, i, ru(f"{v:.3f}"), va="center", fontsize=8.5)
    ax[2].set_yticks(yy); ax[2].set_yticklabels([n for n, _, _ in bars], fontsize=8)
    ax[2].invert_yaxis(); ax[2].set_xlim(-0.12, 1.0)
    ax[2].axvline(0, color="k", lw=0.8)
    ax[2].set_title("r к сырому видео: серая полоса — пол ± ст. откл.", fontsize=10)

    d = S["groups"]["свежий розыгрыш"]
    title = (
        f"Полы судей измерены впервые, и две строки прежней таблицы читаются иначе. СЛЕВА, шкала от шума до "
        f"сырого видео: свежий розыгрыш даёт {ru(f'{cal(d['frac_flat'], 'frac_flat'):.0f}')} из 100 по резкости "
        f"и {ru(f'{cal(d['kurtosis'], 'kurtosis'):.0f}')} по разреженности градиента; потолок самой цепочки — не "
        f"100, а {ru(f'{cal(S['groups']['настоящее состояние']['frac_flat'], 'frac_flat'):.0f}')}, то есть 13B из "
        f"настоящего состояния теряет треть резкости, и этого не было видно, пока не появился пол. Поправка "
        f"сэмплера из пункта 20 по видео — ПРОИГРЫШ ({ru(f'{cal(d['frac_flat'], 'frac_flat'):.0f}')} -> "
        f"{ru(f'{cal(S['groups']['розыгрыш + sdproj']['frac_flat'], 'frac_flat'):.0f}')}), хотя по геометрии "
        f"латента она была выигрышем. В ЦЕНТРЕ: «ближайшее {ru(f'{d['nearest_r']:.3f}')} — значит не копия» — "
        f"верно, но слабо: настоящий отложенный клип против того же банка без себя самого даёт "
        f"{ru(f'{F['arms']['настоящий клип']['nearest_r']:.3f}')}, то есть ровно столько же. Зато розыгрыш и не "
        f"шум — у белого шума {ru(f'{F['arms']['белый шум']['nearest_r']:.3f}')}. СПРАВА: r к сырому у розыгрыша "
        f"{ru(f'{d['r_to_raw']:.3f}')} — внутри пола {ru(f'{fl['mean']:.3f}')} ± {ru(f'{fl['sd']:.3f}')}, "
        f"измеренного между двумя разными настоящими клипами, то есть уликой никогда не было: для розыгрыша, "
        f"не опирающегося ни на какой клип, эта величина не определена по построению. Контроль «перемешан во "
        f"времени» ({ru(f'{tsh:.3f}')}) ступень PCA проходит. Все судьи взяты теми же функциями, что в замерах. "
        f"Шесть отложенных клипов из 10 классов UCF101. Ни карты, ни денег: 82 с локально.")
    fig.suptitle(textwrap.fill(title, 170), fontsize=8.5, y=0.985, va="top")
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.06, top=0.70, wspace=0.62)
    out = Path(a.outdir) / a.prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=110)
    print(f"{out.with_suffix('.png')}")
    for k, h in GROUPS:
        g = S["groups"][k]
        print("%-30s резкость %3.0f  разреженность %3.0f  ближайшее %.3f  r %.3f" % (
            h.replace("\n", " "), cal(g["frac_flat"], "frac_flat"), cal(g["kurtosis"], "kurtosis"),
            g["nearest_r"], g["r_to_raw"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
