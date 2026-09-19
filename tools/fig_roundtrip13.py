"""13A, last part: the states of items 11-12 read out by the decoders of
train13, beside the inversion and the shuffled-state control.

    python tools/fig_roundtrip13.py --run data/roundtrip13

Reads `<run>/summary.json` and `<run>/<cond>.npz` (fetched from the Modal
volume). Per condition and group (dreams of item 11, mixes of item 12) one
GIF and one PNG: rows = states, columns = the reference video (the eye's input
for a dream, clip A / B / the averaged video for a mix; grey where none
exists), the linear decoder, the CNN, the Adam inversion on the same types,
the linear decoder on the state with its cells permuted. Under each cell the
round-trip error (its video re-simulated through the frozen brain, per-type
normalised distance to the target state; lower is more compatible) and, where
a reference exists, r to it.
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
COLS = [("linear", "линейный\nhex-temporal"), ("cnn", "CNN\nhex+temporal"), ("inversion", "Adam-инверсия\n(те же типы)"),
        ("shuffled", "контроль: линейный,\nклетки перемешаны")]
GROUPS = {"dream": ["eye_noise", "flash", "dark_after", "neuron_noise"],
          "mix": ["clip_A", "clip_B", "C_T4a_x0.5", "C_T4a_x2", "C_T5_x0", "C_Mi4_x1.5", "A_a0.25", "A_a0.5", "A_a0.75",
                  "A_avgvideo", "B_earlyA_motionB", "B_earlyB_motionA"]}
ROW_RU = {"eye_noise": "шум в глаз", "flash": "вспышка", "dark_after": "темнота после клипа", "neuron_noise": "шум в нейронах",
          "clip_A": "клип A (3, лицо)", "clip_B": "клип B (10, лес)", "C_T4a_x0.5": "A, T4a × 0.5", "C_T4a_x2": "A, T4a × 2",
          "C_T5_x0": "A, T5 × 0", "C_Mi4_x1.5": "A, Mi4 × 1.5", "A_a0.25": "0.25·A + 0.75·B", "A_a0.5": "0.5·A + 0.5·B",
          "A_a0.75": "0.75·A + 0.25·B", "A_avgvideo": "состояние ½(A+B) видео", "B_earlyA_motionB": "L1…Tm9 ← A, T4/T5 ← B",
          "B_earlyB_motionA": "L1…Tm9 ← B, T4/T5 ← A"}
GROUP_RU = {"dream": "состояния снов (п. 11)", "mix": "правки и смеси состояний (п. 12)"}


def reference(name: str, z) -> tuple[np.ndarray | None, str, str]:
    """The reference column for a state: (video, its label, the r key)."""
    if name == "eye_noise" or name == "flash":
        return z[f"{name}__input"][:40], "вход в глаз", "r_input"
    if name == "dark_after":
        return z[f"{name}__ref_last_frame"], "последний кадр клипа", "r_last_frame"
    if name == "neuron_noise":
        return None, "входа нет (серый)", None
    if name == "clip_B":
        return z[f"{name}__ref_clip_b"], "клип B", "r_clip_b"
    if name.startswith("A_"):
        return z[f"{name}__ref_clip_avg"], "½(A+B) видео", "r_clip_avg"
    return z[f"{name}__ref_clip_a"], "клип A", "r_clip_a"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "roundtrip13"))
    p.add_argument("--prefix", default="2026-09-19_malecns_roundtrip13")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--conditions", default="")
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    frames, dt = summary["frames"], 0.02

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

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    conds = a.conditions.split(",") if a.conditions else list(summary["conditions"])
    for cond in conds:
        z = np.load(run / f"{cond}.npz")
        sc = summary["conditions"][cond]["states"]
        for group, names in GROUPS.items():
            names = [n for n in names if n in sc]
            ncol = 1 + len(COLS)
            title = (f"13A, декодеры train13 на состояниях, которые не вызвал ни один клип. MaleCNS модель нуль, {COND_RU[cond]}, "
                     f"{GROUP_RU[group]}. Под клеткой: ошибка обратной прогонки (видео -> мозг -> состояние, относительно цели; "
                     f"меньше = совместимее) / r к опорному видео. {frames} кадров по {dt * 1000:.0f} мс, в {1 / (a.fps * dt):.0f}x медленнее.")
            for kind in ("png", "gif"):
                fig, ax = plt.subplots(len(names), ncol, figsize=(1.75 * ncol + 1.3, 1.9 * len(names) + 1.2),
                                       facecolor=SURFACE, squeeze=False)
                ims = []
                f0 = frames // 2
                for i, name in enumerate(names):
                    ref, ref_label, rkey = reference(name, z)
                    cells = [(ref, ref_label, "")]
                    for k, lab in COLS:
                        s = sc[name][k]
                        r = s.get(rkey) if rkey else None
                        cells.append((z[f"{name}__{k}"], lab, f"{s['round_trip']:.3f}" + (f" / r {r:+.2f}" if r is not None else "")))
                    for j, (vid, lab, txt) in enumerate(cells):
                        if vid is None:
                            vid = np.full((frames, 721), 0.5, np.float32)
                        im = ax[i, j].imshow(rs(vid, f0 if kind == "png" else 0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                        ims.append((im, vid))
                        style(ax[i, j])
                        if i == 0:
                            ax[i, j].set_title(lab if j else "опорное видео", fontsize=8)
                        ax[i, j].set_xlabel(txt if j else ref_label, fontsize=7)
                    ax[i, 0].set_ylabel(ROW_RU.get(name, name), fontsize=7.5)
                fig.suptitle(title + (f" Кадр {f0}." if kind == "png" else ""), fontsize=8.5, x=0.01, ha="left", wrap=True)
                fig.tight_layout(rect=(0, 0, 1, 1 - 1.15 / fig.get_figheight()))
                out = outdir / f"{a.prefix}_{cond}_{group}.{kind}"
                if kind == "png":
                    fig.savefig(out, dpi=160, facecolor=SURFACE)
                else:
                    def upd(f, ims=ims):
                        for im, v in ims:
                            im.set_data(rs(v, f))
                        return [im for im, _ in ims]
                    FuncAnimation(fig, upd, frames=frames, blit=True).save(out, writer=PillowWriter(fps=a.fps), dpi=55)
                plt.close(fig)
            print(f"{cond}/{group}: wrote {outdir / (a.prefix + '_' + cond + '_' + group)}.png and .gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
