"""Контроль к 18.17: смещено ли само обращение потока.

    python -m flydream.generate.selfinv18        # локально, CPU, $0

18.17 измерил, что прообразы настоящих состояний лежат на радиусе 246–257 при
типичном √D = 303,8 и имеют ст. отклонение по осям 0,82–0,85 вместо 1,0, и
сделал из этого вывод про обученное распределение. Но обращение ODE не
точное (7,4 % относительной ошибки), поэтому часть дефицита может быть
**смещением прибора**, а не геометрией данных.

Проверка не требует ни мозга, ни 13B. Берём **заведомо стандартный** шум
ε ~ N(0, I), гоним его вперёд через приор в состояние, которое приор сделал
сам, и обращаем это состояние назад тем же `to_noise` с теми же настройками.

- вернулись радиус ≈ 303,8 и ст. откл. ≈ 1,0 → прибор не смещён, и 0,83 на
  настоящих данных — настоящая геометрия;
- вернулись ≈ 250 и ≈ 0,83 → обращение само сжимает, и вывод 18.17 о том,
  что образ N(0, I) шире данных, надо снимать.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flydream.generate import prior17 as R
from flydream.generate.invert import device_of


def one(prior_ckpt: Path, *, n: int = 6, steps: int = 100, fixed_point: int = 3,
        seed: int = 0, device=None, log=print) -> dict:
    t0 = time.time()
    dev = device or torch.device("cpu")
    prior, pmeta = R.load(prior_ckpt, dev)
    g = torch.Generator(device=dev).manual_seed(3000 + seed)
    shape = (n, pmeta["frames"], pmeta.get("k", 8), 721)              # ровно то, из чего рисует `sample`
    eps = torch.randn(shape, device=dev, generator=g).cpu().numpy().astype(np.float32)
    D = int(np.prod(shape[1:]))
    x = R.from_noise(prior, pmeta, eps, steps=steps, device=dev)      # состояние, сделанное самим приором
    back = R.to_noise(prior, pmeta, x, steps=steps, fixed_point=fixed_point, device=dev)
    f0, f1 = eps.reshape(n, -1), back.reshape(n, -1)
    out = {"prior_ckpt": str(prior_ckpt), "width": pmeta["width"], "dct_k": pmeta.get("dct_k"),
           "n": n, "steps": steps, "fixed_point": fixed_point, "seed": seed, "dims": D,
           "typical_radius": float(np.sqrt(D)),
           "eps_radius": float(np.linalg.norm(f0, axis=1).mean()), "eps_sd": float(f0.std()),
           "back_radius": float(np.linalg.norm(f1, axis=1).mean()), "back_sd": float(f1.std()),
           "back_rel_error": float(np.linalg.norm(f1 - f0, axis=1).mean() / np.linalg.norm(f0, axis=1).mean()),
           "cos_back_eps": float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                                          for a, b in zip(f0, f1)])),
           "seconds": round(time.time() - t0, 1)}
    log(f"{prior_ckpt.name}: вперёд из N(0,I) и назад за {out['seconds']:.0f} с")
    return out


def main(argv=None) -> int:
    from flydream.model import ROOT

    p = argparse.ArgumentParser()
    p.add_argument("--priors", nargs="+", default=[
        str(ROOT / "data" / "prior18" / "corpus_dct16_c.pt"),
        str(ROOT / "data" / "prior18" / "corpus_dct16_w192_lr1e3_c.pt"),
        str(ROOT / "data" / "prior18" / "corpus_dct16_w384_lr1e3_c.pt")])
    p.add_argument("--out", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--tag", default="selfinv18")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--fixed-point", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    dev = torch.device("cpu")
    rows = [one(Path(x), n=a.n, steps=a.steps, fixed_point=a.fixed_point, seed=a.seed, device=dev)
            for x in a.priors]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.tag}.json").write_text(json.dumps({"rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nтипичный радиус √D = {rows[0]['typical_radius']:.1f};  вход ε ~ N(0, I)")
    print(f"{'плечо':34} {'радиус ε':>9} {'ст.откл ε':>10} {'радиус назад':>13} {'ст.откл назад':>14} {'ошибка':>8}")
    for r in rows:
        print(f"{Path(r['prior_ckpt']).stem:34} {r['eps_radius']:9.1f} {r['eps_sd']:10.3f} "
              f"{r['back_radius']:13.1f} {r['back_sd']:14.3f} {100 * r['back_rel_error']:7.1f}%")
    print(f"\nwrote {out / a.tag}.json  ($0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
