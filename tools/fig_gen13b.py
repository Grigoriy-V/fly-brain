"""13B pictures: samples of the generator beside their state's reference video.

    python tools/fig_gen13b.py --run data/gen13b

Reads `<run>/samples.json` and `<run>/samples.npz` (fetched from the Modal
volume). Four GIF/PNG pairs: held-out test clips; the item 11-12 states;
the knobs (masks and guidance on clip A's state); the conditioning-strength
test (one z: true / shuffled / zero state) with the shuffled-state control.
Under each sample its round-trip error (its video re-simulated through the
frozen brain, per-type normalised distance to the state on T4/T5; lower is
more compatible); the row label carries the median over seeds and r to the
reference where one exists.
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
ROW_RU = {"clip_A": "клип A (3, лицо)", "clip_B": "клип B (10, лес)", "C_T4a_x0.5": "A, T4a × 0.5", "C_T4a_x2": "A, T4a × 2",
          "C_T5_x0": "A, T5 × 0", "A_a0.5": "0.5·A + 0.5·B", "A_avgvideo": "состояние ½(A+B) видео",
          "B_earlyA_motionB": "L1…Tm9 ← A, T4/T5 ← B", "eye_noise": "шум в глаз", "neuron_noise": "шум в нейронах",
          "control_shuffled": "контроль: клетки перемешаны"}
MASK_RU = {"full": "все T4/T5", "T4a": "только T4a", "t4": "только T4a-d", "t5": "только T5a-d", "dir_a": "T4a + T5a"}
REF_PREF = ["clip", "clip_a", "clip_b", "clip_avg", "input"]


def ref_of(state: str, z) -> tuple[np.ndarray | None, str]:
    if state.startswith("A_"):
        pref = ["clip_avg"]
    elif state == "clip_B":
        pref = ["clip_b"]
    elif state == "B_earlyA_motionB":
        pref = ["clip_b"]
    else:
        pref = REF_PREF
    for r in pref:
        k = f"ref__{state}__{r}"
        if k in z:
            return z[k], {"clip": "клип", "clip_a": "клип A", "clip_b": "клип B", "clip_avg": "½(A+B) видео", "input": "вход в глаз"}[r]
    return None, "входа нет (серый)"


def draw(rows: list, title: str, out: Path, frames: int, fps: int, plt, FuncAnimation, PillowWriter):
    """rows: (label, [(video, header, text)…])."""
    ncol = max(len(c) for _, c in rows)
    for kind in ("png", "gif"):
        fig, ax = plt.subplots(len(rows), ncol, figsize=(1.75 * ncol + 1.3, 1.9 * len(rows) + 1.2), facecolor=SURFACE, squeeze=False)
        ims, f0 = [], frames // 2
        for i, (label, cells) in enumerate(rows):
            for j in range(ncol):
                a = ax[i, j]
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
                if j >= len(cells):
                    a.set_visible(False); continue
                vid, head, txt = cells[j]
                if vid is None:
                    vid = np.full((frames, 721), 0.5, np.float32)
                im = a.imshow(to_raster(vid[f0 if kind == "png" else 0], 721, 4, fill=np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ims.append((im, vid))
                if i == 0:
                    a.set_title(head, fontsize=8)
                a.set_xlabel(txt, fontsize=7)
            ax[i, 0].set_ylabel(label, fontsize=7)
        fig.suptitle(title + (f" Кадр {f0}." if kind == "png" else ""), fontsize=8.5, x=0.01, ha="left", wrap=True)
        fig.tight_layout(rect=(0, 0, 1, 1 - 1.15 / fig.get_figheight()))
        if kind == "png":
            fig.savefig(out.with_suffix(".png"), dpi=160, facecolor=SURFACE)
        else:
            def upd(f, ims=ims):
                for im, v in ims:
                    im.set_data(to_raster(v[f], 721, 4, fill=np.nan))
                return [im for im, _ in ims]
            FuncAnimation(fig, upd, frames=frames, blit=True).save(out.with_suffix(".gif"), writer=PillowWriter(fps=fps), dpi=55)
        plt.close(fig)
    print(f"wrote {out}.png and .gif")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "gen13b"))
    p.add_argument("--prefix", default="2026-09-20_malecns_gen13b")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    S = json.loads((run / "samples.json").read_text(encoding="utf-8"))
    z = np.load(run / "samples.npz")
    frames, sc, n_seeds = S["frames"], S["scores"], S["n_seeds"]
    rt = S["roundtrip_per_sample"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    base = (f"13B, генератор «deep-состояние + маска + z → видео» (SiT-интерполянт), MaleCNS модель нуль. Под сэмплом: ошибка "
            f"обратной прогонки (видео → мозг → состояние T4/T5; меньше = совместимее). {frames} кадров по 20 мс, в {1 / (a.fps * 0.02):.0f}x медленнее. ")

    def samples_row(state, mask, g, label):
        key = f"{state}__{mask}__s{g:g}"
        s = sc[key]
        ref, ref_label = ref_of(state, z)
        cells = [(ref, "опорное видео", ref_label)]
        for k in range(s["n"]):
            kk = f"{state}__{mask}__s{g:g}__seed{k}"
            cells.append((z[kk], f"сэмпл z{k}", f"{rt[kk]:.3f}"))
        rk = [q for q in s if q.startswith("r_") and q != "r_between_samples"]
        lab = label + f"\nмедиана {s['roundtrip_median']:.3f}" + (f", r {s[rk[0]]:+.2f}" if rk else "") + \
            (f"\nразброс r {s['r_between_samples']:.2f}" if "r_between_samples" in s else "")
        return (lab, cells)

    tests = sorted([k.split("__")[0] for k in sc if k.startswith("test_") and k.endswith("__full__s1")], key=lambda q: int(q[5:]))
    draw([samples_row(t, "full", 1, f"отложенный {t[5:]}") for t in tests],
         base + "Отложенные клипы (сцены и классы, которых модель не видела), полная маска, s = 1.",
         outdir / f"{a.prefix}_test", frames, a.fps, plt, FuncAnimation, PillowWriter)
    st = [s_ for s_ in ("clip_A", "clip_B", "C_T4a_x0.5", "C_T4a_x2", "C_T5_x0", "A_a0.5", "A_avgvideo", "B_earlyA_motionB", "eye_noise", "neuron_noise") if f"{s_}__full__s1" in sc]
    draw([samples_row(s_, "full", 1, ROW_RU.get(s_, s_)) for s_ in st],
         base + "Состояния пунктов 11–12 (i2i, правки, смеси, сны), полная маска, s = 1.",
         outdir / f"{a.prefix}_states", frames, a.fps, plt, FuncAnimation, PillowWriter)
    knobs = [samples_row("clip_A", m, 1, f"клип A, {MASK_RU[m]}") for m in ("full", "T4a", "t4", "t5", "dir_a") if f"clip_A__{m}__s1" in sc]
    knobs += [samples_row("clip_A", "full", g, f"клип A, все, s = {g:g}") for g in S["guidances"][1:] if f"clip_A__full__s{g:g}" in sc]
    draw(knobs, base + "Крутилки: маска типов (промпт частью состояния) и сила условия s на состоянии клипа A.",
         outdir / f"{a.prefix}_knobs", frames, a.fps, plt, FuncAnimation, PillowWriter)
    ref = z["ref__clip_A__clip_a"]
    strength = [("один z:\nверное / перемешанное /\nнулевое состояние",
                 [(ref, "опорное видео", "клип A")] + [(z[f"strength_{q}__full__s1__seed0"], lab, f"{rt[f'strength_{q}__full__s1__seed0']:.3f}")
                                                     for q, lab in (("true", "верное"), ("shuffled", "перемешанное"), ("zero", "нулевое"))]),
                samples_row("control_shuffled", "full", 1, ROW_RU["control_shuffled"])]
    S_ = S["strength"]
    draw(strength, base + f"Тест силы условия: r верное~перемешанное {S_['r_true_shuffled']:+.2f}, верное~нулевое {S_['r_true_zero']:+.2f} "
         "(близко к 1 = прайор игнорирует мозг). Вторая строка — контроль на перемешанном состоянии.",
         outdir / f"{a.prefix}_strength", frames, a.fps, plt, FuncAnimation, PillowWriter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
