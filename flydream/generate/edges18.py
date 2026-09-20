"""18.9: чем сцена отличается от пятна, если контраст одинаковый.

    python -m flydream.generate.edges18          # локально, CPU, $0

Человек, 2026-09-20, на картинку 18.8: «где якобы один контраст, но визуально
же вообще не так… если у нас одна тёмная точка на сером поле, она же не
показывает диапазон всей картинки, это просто выброс».

Проверка в две стороны. Первая — устойчивые меры разброса (межквартильный
размах, медианное абсолютное отклонение, размах 5–95 %): если совпадение по
ст. отклонению держится на выбросах, они его не подтвердят. Вторая — то, чем
натуральная сцена на самом деле отличается: **разреженность перепадов между
соседними колонками**. У сцены большие ровные области, разделённые редкими
резкими границами, поэтому распределение перепада тяжелохвостое; у гладкого
случайного поля перепад размазан ровно и почти гауссов.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from flydream.generate import learned as L

# Каждое плечо item 18, у которого на диске есть сэмплы; порядок — как в отчёте.
ARMS = [
    ("samples18_corpus_dct16_c", "контроль: w128, 3e-4, K16"),
    ("samples18_corpus_dct16_lr6e4_c", "w128, lr 6e-4"),
    ("samples18_corpus_dct16_lr1e3_c", "w128, lr 1e-3"),
    ("samples18_corpus_dct16_60k_c", "w128, 60k шагов"),
    ("samples18_corpus_dct16_cls_c", "w128, классы"),
    ("samples18_corpus_dct16_w192_c", "w192, 3e-4"),
    ("samples18_corpus_dct16_w192_lr1e3_c", "w192, lr 1e-3 — лучшее по воротам"),
    ("samples18_corpus_dct16_w192_lr1e3_c_s100", "w192, lr 1e-3, 100 шагов сэмплера"),
    ("samples18_corpus_dct32_c", "K=32, w128"),
    ("samples18_corpus_dct32_w192_c", "K=32, w192, равный вес"),
    ("samples18_corpus_dct32_w192_lr1e3_w1_c", "K=32, w192, вес sd¹"),
]


def spread(v: np.ndarray) -> dict:
    """Разброс значений: обычный и устойчивый к выбросам."""
    f = np.asarray(v, np.float64).reshape(-1)
    return {"sd": float(f.std()), "iqr": float(np.subtract(*np.percentile(f, [75, 25]))),
            "mad": float(np.median(np.abs(f - np.median(f)))),
            "p5_95": float(np.subtract(*np.percentile(f, [95, 5])))}


def edges(v: np.ndarray, nb: np.ndarray) -> dict:
    """Распределение перепада между соседними колонками: хвост и доля ровного."""
    x = np.asarray(v, np.float64)
    d = [(x[:, nb[:, j] >= 0] - x[:, np.clip(nb[nb[:, j] >= 0, j], 0, None)]).reshape(-1)
         for j in range(nb.shape[1])]
    d = np.concatenate(d)
    d = d - d.mean(); s = float(d.std())
    return {"grad_sd": s, "grad_kurtosis": float((d ** 4).mean() / (s ** 4 + 1e-24)),
            "frac_flat": float((np.abs(d) < 0.25 * s).mean()),
            "frac_strong": float((np.abs(d) > 3.0 * s).mean())}


def describe(vids, nb) -> dict:
    per = [{**spread(v), **edges(v, nb)} for v in vids]
    keys = list(per[0])
    out = {"n": len(per), "per_clip": per}
    for k in keys:
        a = np.array([p[k] for p in per])
        out[k] = {"mean": float(a.mean()), "median": float(np.median(a)),
                  "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}
    return out


def run(run_dir: Path, corpus: Path, tags: list[tuple[str, str]], *, n_raw: int = 32,
        frames: int = 40, seed: int = 0, log=print) -> dict:
    nb = np.asarray(L.neighbour_index(721))
    groups = {}
    for tag, label in tags:
        if not (run_dir / f"{tag}.json").exists():
            log(f"  {label}: нет на диске, пропущено")
            continue
        S = json.loads((run_dir / f"{tag}.json").read_text(encoding="utf-8"))
        z = np.load(run_dir / f"{tag}.npz")
        sc = S["scores"]
        pri = [k for k in sc if sc[k]["kind"] == "prior"]
        groups[label] = describe([z[f"video__{k}"] for k in pri], nb)
        groups[label]["gate"] = S["gates"]["prior"]["round_trip"]["median"]
        groups[label]["tag"] = tag
        if "13B из настоящего состояния" not in groups:               # потолок отрисовщика, берём один раз
            cli = [k for k in sc if sc[k]["kind"] == "clip"]
            groups["13B из настоящего состояния"] = describe([z[f"video__{k}"] for k in cli], nb)
        log(f"  {label}: n={groups[label]['n']}")
    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.asarray(cm["split"]["test"]), n_raw, replace=False))
    groups["сырое видео корпуса"] = describe([np.asarray(cz["videos"][i][:frames], np.float32) for i in idx], nb)
    log(f"  сырое видео корпуса: n={n_raw}")
    return {"groups": groups, "n_raw": n_raw, "frames": frames, "seed": seed}


def main(argv=None) -> int:
    from flydream.model import ROOT

    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--tag", default="edges18_local")
    p.add_argument("--n-raw", type=int, default=32)
    p.add_argument("--arms", default="", help='"all" — все плечи из ARMS; пусто — два плеча 18.9')
    a = p.parse_args(argv)
    run_dir = Path(a.run)
    tags = ARMS if a.arms == "all" else [("samples18_corpus_dct16_w192_lr1e3_c", "прайор, 20 шагов"),
                                         ("samples18_corpus_dct16_w192_lr1e3_c_s100", "прайор, 100 шагов")]
    r = run(run_dir, Path(a.corpus), tags, n_raw=a.n_raw)
    (run_dir / f"{a.tag}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'группа':30s} {'sd':>6} {'IQR':>6} {'MAD':>6} {'p5-95':>6} | "
          f"{'эксцесс':>8} {'ровного':>8} {'границ':>7}")
    for g, v in r["groups"].items():
        print(f"{g:30s} {v['sd']['mean']:6.3f} {v['iqr']['mean']:6.3f} {v['mad']['mean']:6.3f} "
              f"{v['p5_95']['mean']:6.3f} | {v['grad_kurtosis']['mean']:8.2f} "
              f"{100 * v['frac_flat']['mean']:7.1f}% {100 * v['frac_strong']['mean']:6.2f}%")
    print(f"wrote {run_dir / a.tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
