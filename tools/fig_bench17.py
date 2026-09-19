"""18.4a in one sheet: where the prior's training step actually goes.

    python tools/fig_bench17.py --run data/prior18/bench17.json

Left: throughput against batch size — the line is flat, so the batch is not a
lever and the step is paid per sample, not per launch. Right: milliseconds per
step for every variant measured in the same container, with 18.3's own loop as
the reference line; only `torch.compile` moves it. The affine fit under the
left panel is the number that settles the question the research note left open.
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
    p.add_argument("--run", default=str(ROOT / "data" / "prior18" / "bench17.json"))
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-20_malecns_bench17.png"))
    a = p.parse_args(argv)
    d = json.loads(Path(a.run).read_text(encoding="utf-8"))
    rows = [r for r in d["rows"] if r.get("ms_per_step")]
    fit = d["fit"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = [r for r in rows if r["variant"].startswith("both") and not r["compile"]]
    sweep.sort(key=lambda r: r["batch"])
    comp = next((r for r in rows if r["compile"]), None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1, 1.25]})
    b = [r["batch"] for r in sweep]
    s = [r["samples_per_s"] for r in sweep]
    base_ms = next(r["ms_per_step"] for r in rows if r["variant"] == "base (18.3's loop)")
    gain_b = s[-1] / s[0] - 1
    gain_c = base_ms / comp["ms_per_step"] if comp else 0
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    fig.suptitle(f"18.4a: шаг обучения прайора упирается в работу на сэмпл, а не в запуск ядер — "
                 f"батч даёт {ru(f'{gain_b:+.1%}')}, компиляция {ru(f'{gain_c:.2f}')}×",
                 fontsize=11.5, y=0.985)

    ax1.plot(b, s, "o-", color="#1f77b4", lw=2, ms=7, label="измерено, без компиляции")
    if comp:
        ax1.plot([comp["batch"]], [comp["samples_per_s"]], "*", color="#d62728", ms=18,
                 label=f"torch.compile, батч {comp['batch']}")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(b); ax1.set_xticklabels([str(x) for x in b])
    ax1.set_ylim(0, max(s + ([comp["samples_per_s"]] if comp else [])) * 1.22)
    ax1.set_xlabel("размер батча"); ax1.set_ylabel("сэмплов в секунду")
    ax1.set_title(f"рост батча в 8 раз даёт {ru(f'{gain_b:+.1%}')}", fontsize=10)
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax1.text(0.03, 0.05, (f"время шага = {fit['fixed_ms']} мс + {fit['per_sample_ms']:.3f} мс x батч\n"
                          f"постоянная часть при батче 32: {fit['fixed_share_at_32']:.1%}").replace(".", ","),
             transform=ax1.transAxes, fontsize=8.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#c8b273"))

    at32 = [r for r in rows if r["batch"] == 32]
    names = {"base (18.3's loop)": "петля 18.3, как есть", "no loss.item()": "без loss.item()\n(снят хостовый sync)",
             "foreach EMA": "foreach EMA\n(вместо цикла по параметрам)", "both": "оба изменения",
             f"compile {comp['compile']}" if comp else "": f"torch.compile\n({comp['compile'] if comp else ''})"}
    lab = [names.get(r["variant"], r["variant"]) for r in at32]
    val = [r["ms_per_step"] for r in at32]
    col = ["#d62728" if r["compile"] else ("#7f7f7f" if r["variant"] == "base (18.3's loop)" else "#1f77b4")
           for r in at32]
    bars = ax2.barh(range(len(val)), val, color=col, height=0.62)
    base = next(r["ms_per_step"] for r in at32 if r["variant"] == "base (18.3's loop)")
    ax2.axvline(base, color="#7f7f7f", ls="--", lw=1.2)
    for i, (bar, v) in enumerate(zip(bars, val)):
        ax2.text(v + 1.0, i, ru(f"{v:.2f}") + " мс" + (f"  ({ru(f'{base / v:.2f}')}x)" if abs(v - base) > 1 else "  (без изменений)"),
                 va="center", fontsize=9)
    ax2.set_yticks(range(len(lab))); ax2.set_yticklabels(lab, fontsize=8.5)
    ax2.invert_yaxis(); ax2.set_xlim(0, max(val) * 1.42)
    ax2.set_xlabel("миллисекунд на шаг, батч 32 (60 шагов после 15 прогревочных)")
    ax2.set_title("два подозреваемых в коде не подтвердились, компиляция — да", fontsize=10)
    ax2.grid(alpha=0.3, axis="x")
    warm = next((r.get("compile_warmup_s") for r in rows if r.get("compile_warmup_s")), None)
    foot = (f"Один T4, cpu 1 / 12 ГБ, контейнер {d['seconds_worker']} с, ≈$0.03. Обучающий набор {d['n_train']} "
            f"состояний {tuple(d['shape'])}. Разогрев компиляции {warm} с разово. "
            f"Прогон 18.3 для сверки: {d['baseline_18_3_ms']} мс/шаг. Скрипт tools/fig_bench17.py.")
    fig.text(0.5, 0.008, foot, ha="center", fontsize=8.2, color="#444")
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
