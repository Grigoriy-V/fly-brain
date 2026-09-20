"""18.12: направление (classifier-free guidance) — вся кривая за один проход.

    python tools/sweep_guidance18.py --ckpt corpus_dct16_w192_lr1e3_cls_g_c

Обучение с выбрасыванием метки даёт одну модель, а сила направления — ручка
на этапе сэмплирования, так что кривая стоит $0: для каждой силы гоняются те
же ворота (`samples17.run`), потом по всем плечам разом считается статистика
края 18.9 (`edges18`). Ворота локально ~45 с на плечо.

Масштаб 0 — направление выключено, это ровно условный путь 18.4e за один
проход вместо двух; 1 воспроизводит его же через формулу (проверка, что
арифметика на месте); дальше сэмпл тянут к его классу сильнее.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def tag_of(ckpt: str, g: float) -> str:
    """Имя плеча на диске. 0 остаётся без суффикса — это база сравнения."""
    return f"samples18_{ckpt}" if not g else f"samples18_{ckpt}_g{g:g}".replace(".", "p")


def main(argv=None) -> int:
    from flydream.generate import edges18 as E
    from flydream.generate import samples17 as S
    from flydream.generate.invert import settings

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="corpus_dct16_w192_lr1e3_cls_g_c")
    p.add_argument("--run", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--scales", default="0,1,1.5,2,3,5")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="edges18_guidance")
    p.add_argument("--skip-done", action="store_true", help="не пересчитывать плечи, уже лежащие на диске")
    a = p.parse_args(argv)

    run_dir = Path(a.run); run_dir.mkdir(parents=True, exist_ok=True)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    proc = sorted(pdir.glob("procedural_*.npz"))[0]
    cfg = settings()
    scales = [float(x) for x in a.scales.split(",")]

    rows, t0 = [], time.time()
    for g in scales:
        tag = tag_of(a.ckpt, g)
        if a.skip_done and (run_dir / f"{tag}.json").exists():
            print(f"[{g:g}] {tag} уже есть, пропускаю")
        else:
            print(f"[{g:g}] ворота -> {tag}", flush=True)
            r = S.run("malecns", run_dir / f"{a.ckpt}.pt", Path(a.gen), manifest, columns, proc,
                      corpus=Path(a.corpus), n_samples=a.samples, n_clips=3,
                      frames=cfg.get("frames", 40), margin=cfg.get("margin", 5), dt=cfg.get("dt", 0.02),
                      t_pre=cfg.get("t_pre", 1.0), guidance=g, seed=a.seed, log=lambda s: None)
            (run_dir / f"{tag}.json").write_text(json.dumps(r["summary"], indent=1), encoding="utf-8")
            np.savez_compressed(run_dir / f"{tag}.npz", **r["arrays"])
        d = json.loads((run_dir / f"{tag}.json").read_text(encoding="utf-8"))
        gt = d["gates"]
        rows.append({"guidance": g, "tag": tag,
                     "gate": gt["prior"]["round_trip"]["median"],
                     "gate_lo": gt["prior"]["round_trip"]["min"], "gate_hi": gt["prior"]["round_trip"]["max"],
                     "floor": gt["ceiling"]["round_trip"]["median"],
                     "clip": gt["clip"]["round_trip"]["median"],
                     "novelty": gt["prior"]["video_nn_distance"]["median"],
                     "nn_r": gt["prior"]["video_nn_r"]["median"]})
        print(f"    ворота {rows[-1]['gate']:.4f}  новизна {rows[-1]['novelty']:+.3f}", flush=True)

    tags = [(r["tag"], f"направление {r['guidance']:g}") for r in rows]
    e = E.run(run_dir, Path(a.corpus), tags, n_raw=32)
    for r in rows:                                                            # структура рядом с воротами
        v = e["groups"][f"направление {r['guidance']:g}"]
        r.update({"frac_flat": v["frac_flat"]["mean"], "kurtosis": v["grad_kurtosis"]["mean"],
                  "frac_strong": v["frac_strong"]["mean"], "sd": v["sd"]["mean"],
                  "neigh_r": v["neigh_r"]["mean"]})

    ref = {k: e["groups"][k] for k in e["groups"] if k not in {t[1] for t in tags}}
    out = {"ckpt": a.ckpt, "scales": scales, "rows": rows, "reference": ref,
           "seconds": round(time.time() - t0, 1), "seed": a.seed, "n_samples": a.samples}
    (run_dir / f"{a.out}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'сила':>6} {'ворота':>8} {'новизна':>8} {'ровного':>9} {'эксцесс':>8} {'границ':>7} "
          f"{'сосед r':>8} {'контраст':>9}")
    for r in rows:
        print(f"{r['guidance']:6g} {r['gate']:8.4f} {r['novelty']:+8.3f} {100 * r['frac_flat']:8.1f}% "
              f"{r['kurtosis']:8.2f} {100 * r['frac_strong']:6.2f}% {r['neigh_r']:8.3f} {r['sd']:9.3f}")
    for k, v in ref.items():
        print(f"{k:>6.6s} {'—':>8} {'—':>8} {100 * v['frac_flat']['mean']:8.1f}% "
              f"{v['grad_kurtosis']['mean']:8.2f} {100 * v['frac_strong']['mean']:6.2f}% "
              f"{v['neigh_r']['mean']:8.3f} {v['sd']['mean']:9.3f}   <- {k}")
    print(f"\nwrote {run_dir / a.out}.json  ({out['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
