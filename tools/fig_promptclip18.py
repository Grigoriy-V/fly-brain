"""Условный прайор: сэмпл, его промпт и его собственное ближайшее настоящее видео.

    python tools/fig_promptclip18.py --tag samples18_corpus_dct16_w192_lr1e3_cls_fix_c

Верхний ряд — сгенерированное из шума, под каждой ячейкой её промпт (класс, на
который сэмпл условлен). Нижний ряд — **своё** для каждой ячейки ближайшее
обучающее видео по корреляции, а не один общий случайный клип: человек,
2026-09-20, «мы же обучаем видео → состояние → шум, потом генерим шум →
состояние → видео, так что находить самое ближайшее видео не так то и трудно».
Сэмплы берутся подряд с нулевого, без отбора.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fig_gen13b_pick import SURFACE, to_raster  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--tag", default="samples18_corpus_dct16_w192_lr1e3_cls_fix_c")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--prefix", default="2026-09-20_malecns_promptclip18")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    p.add_argument("--note", default="")
    a = p.parse_args(argv)
    run = Path(a.run)
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only
    S = json.loads((run / f"{a.tag}.json").read_text(encoding="utf-8"))
    z = np.load(run / f"{a.tag}.npz")
    frames = int(S["frames"])
    cls = S.get("sampled_classes") or []
    idx = list(range(a.first, a.first + a.n))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    top, bot, heads, foots = [], [], [], []
    for i in idx:
        k = f"prior_{i}"
        sc = S["scores"][k]
        top.append(z[f"video__{k}"]); bot.append(z[f"nnvideo__{k}"])
        heads.append(f"промпт: «{cls[i] if i < len(cls) else '?'}»")
        foots.append((f"прогонка {ru(f'{sc['round_trip']:.3f}')}",
                      f"ближайшее обучающее, r {ru(f'{sc['video_nn_r']:+.2f}')}"))

    g = S["gates"]
    pool = S.get("label_pool") or {}
    line = (f"Условный прайор из шума, {a.n} сэмпла подряд без отбора. "
            f"Ворота {ru(f'{g['prior']['round_trip']['median']:.4f}')} "
            f"(настоящий клип {ru(f'{g['clip']['round_trip']['median']:.4f}')}, "
            f"пол представления {ru(f'{g['ceiling']['round_trip']['median']:.4f}')}). "
            + (f"Метка тянется из {pool.get('n')} обученных классов из {pool.get('of')}. "
               if pool.get("restricted") else "")
            + f"{frames} кадров по 20 мс (0,8 с мухи), в {1 / (a.fps * 0.02):.1f}× медленнее. "
            + (a.note or "Нижний ряд — ближайшее к своей ячейке настоящее обучающее видео по корреляции, "
                         "то есть то, что видел глаз, БЕЗ прохода через 13B; верхний ряд через 13B прошёл, "
                         "а он сам теряет около четверти амплитуды (18.9), так что разрыв между рядами "
                         "шире, чем вина прайора."))

    for kind in ("png", "gif"):
        fig, ax = plt.subplots(2, a.n, figsize=(3.0 * a.n, 7.4), facecolor=SURFACE, squeeze=False)
        ims, f0 = [], frames // 2
        for j in range(a.n):
            for r, (vids, lab) in enumerate(((top, "сгенерировано"), (bot, "ближайшее настоящее"))):
                b = ax[r, j]
                b.set_xticks([]); b.set_yticks([])
                for sp in b.spines.values():
                    sp.set_visible(False)
                v = vids[j]
                im = b.imshow(to_raster(v[f0 if kind == "png" else 0], 721, 4, fill=np.nan),
                              cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ims.append((im, v))
                b.set_title(heads[j] if r == 0 else "", fontsize=10.5, pad=8)
                b.set_xlabel(foots[j][r], fontsize=9.5)
                if j == 0:
                    b.set_ylabel(lab, fontsize=11)
        fig.suptitle(line, fontsize=10.5, x=0.01, ha="left", wrap=True)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = Path(a.outdir) / a.prefix
        if kind == "png":
            fig.savefig(out.with_suffix(".png"), dpi=80, facecolor=SURFACE)
        else:
            def upd(f, ims=ims):
                for im, v in ims:
                    im.set_data(to_raster(v[f], 721, 4, fill=np.nan))
                return [im for im, _ in ims]
            FuncAnimation(fig, upd, frames=frames, blit=True).save(
                out.with_suffix(".gif"), writer=PillowWriter(fps=a.fps), dpi=80)
        plt.close(fig)
    print(f"wrote {Path(a.outdir) / a.prefix}.png / .gif")
    print("  промпты: " + ", ".join(heads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
