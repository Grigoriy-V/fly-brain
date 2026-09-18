"""13A pictures: per input condition, held-out clips beside what each model
recovers from their state.

    python tools/fig_train13.py --run data/train13

Reads `<run>/summary.json` and `<run>/<cond>_{linear,cnn,inversion}_roundtrip.npz`
(fetched from the Modal volume). Per condition one GIF and one PNG: rows =
held-out clips (their class or scene in the row label), columns = the true
video ("what the eye saw"), the linear hex-temporal decoder, the CNN, the
Adam inversion; under each column the mean r to the video over the rows and
the round-trip error (the generated video re-simulated through the frozen
brain, per-type normalised distance of its state to the target state; the
inversion's is the reachable floor). A bar figure sums up r and round trip
per condition and model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flydream.decode.hexraster import to_raster  # noqa: E402

SURFACE = "#fcfcfb"
COND_RU = {"early": "вход: L1 + L3 (ранние)", "deep": "вход: T4a-d + T5a-d (глубокие)", "all": "вход: все 14 типов лестницы"}
MODEL_RU = {"linear": "линейный\nhex-temporal", "cnn": "CNN\nhex+temporal", "inversion": "Adam-инверсия\n(планка)"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "train13"))
    p.add_argument("--prefix", default="2026-09-19_malecns_train13")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    meta_path = run / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    def rs(v, f):
        return to_raster(v[f], 721, 4, fill=np.nan)

    def style(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def label_of(i: int) -> str:
        if meta is None:
            return f"клип {i}"
        m = meta[i]
        return f"Sintel {m['scene']}" if m["source"] == "sintel" else f"{m['class']}"

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for cond, res in summary["conditions"].items():
        files = {k: run / f"{cond}_{k}_roundtrip.npz" for k in ("linear", "cnn", "inversion")}
        if not all(f.exists() for f in files.values()):
            print(f"{cond}: round-trip files missing, skipped"); continue
        z = {k: np.load(f) for k, f in files.items()}
        ids, true = z["linear"]["ids"], z["linear"]["true"]
        rows = list(range(min(a.rows, len(ids))))
        n, dt = true.shape[1], summary.get("dt", 0.02)
        cols = [("что видел глаз", true, None, None)]
        for k in ("linear", "cnn", "inversion"):
            r = float(np.mean(z[k]["r"])); rt = float(np.mean(z[k]["rt_per_clip"]))
            cols.append((MODEL_RU[k], z[k]["pred"], r, rt))
        title = (f"13A, амортизированная инверсия, MaleCNS модель нуль. {COND_RU[cond]}. Отложенные клипы (сцены и классы, "
                 f"которых модель не видела). Под колонкой: r к видео / ошибка обратной прогонки (видео -> мозг -> состояние, "
                 f"относительно цели; у инверсии — достижимый пол). {n} кадров по {dt * 1000:.0f} мс, в {1 / (a.fps * dt):.0f}x медленнее.")
        for kind in ("png", "gif"):
            fig, ax = plt.subplots(len(rows), len(cols), figsize=(1.7 * len(cols) + 1.2, 1.75 * len(rows) + 1.1),
                                   facecolor=SURFACE, squeeze=False)
            ims = []
            f0 = n // 2
            for i, r_ in enumerate(rows):
                for j, (name, vid, r, rt) in enumerate(cols):
                    im = ax[i, j].imshow(rs(vid[r_], f0 if kind == "png" else 0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    ims.append((im, vid[r_]))
                    style(ax[i, j])
                    if i == 0:
                        ax[i, j].set_title(name + (f"\nr {r:+.2f} / {rt:.3f}" if r is not None else ""), fontsize=8)
                ax[i, 0].set_ylabel(label_of(int(ids[r_])), fontsize=8)
            fig.suptitle(title + (f" Кадр {f0}." if kind == "png" else ""), fontsize=8.5, x=0.01, ha="left", wrap=True)
            fig.tight_layout(rect=(0, 0, 1, 0.86))
            out = outdir / f"{a.prefix}_{cond}.{kind}"
            if kind == "png":
                fig.savefig(out, dpi=170, facecolor=SURFACE)
            else:
                def upd(f, ims=ims):
                    for im, v in ims:
                        im.set_data(rs(v, f))
                    return [im for im, _ in ims]
                FuncAnimation(fig, upd, frames=n, blit=True).save(out, writer=PillowWriter(fps=a.fps))
            plt.close(fig)
        print(f"{cond}: wrote {outdir / (a.prefix + '_' + cond)}.png and .gif")

    # the summary bars
    conds = list(summary["conditions"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), facecolor=SURFACE)
    x = np.arange(len(conds)); wdt = 0.26
    for k, kind in enumerate(("linear", "cnn", "inversion")):
        r = [summary["conditions"][c]["models"][kind]["test_r"] if kind != "inversion" else summary["conditions"][c]["inversion"]["test_r"] for c in conds]
        rt = [np.mean(summary["conditions"][c]["models"][kind]["roundtrip_per_clip"]) if kind != "inversion"
              else np.mean(summary["conditions"][c]["inversion"]["roundtrip_per_clip"]) for c in conds]
        ax[0].bar(x + (k - 1) * wdt, r, wdt, label=MODEL_RU[kind].replace("\n", " "))
        ax[1].bar(x + (k - 1) * wdt, rt, wdt)
    ax[0].set_xticks(x); ax[0].set_xticklabels(conds); ax[0].set_ylabel("r к видео, отложенный тест"); ax[0].set_ylim(0, 1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(conds); ax[1].set_ylabel("ошибка обратной прогонки"); ax[1].set_yscale("log")
    ax[0].legend(fontsize=7, frameon=False)
    for a_ in ax:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.suptitle("13A: точность к видео и совместимость с состоянием по условиям входа и моделям", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outdir / f"{a.prefix}_summary.png", dpi=170, facecolor=SURFACE); plt.close(fig)
    print(f"wrote {outdir / (a.prefix + '_summary.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
