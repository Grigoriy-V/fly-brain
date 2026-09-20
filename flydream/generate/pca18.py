"""Сколько измерений на самом деле занимает состояние.

    python -m flydream.generate.pca18            # локально, CPU, $0

Последняя дешёвая проверка перед сборкой VAE. 18.21 измерил, что по **времени**
сжимать больше нечего: DCT ниже K = 16 роняет корреляцию с сырым клипом с 0,912
до 0,517. Если и по остальным осям запаса нет, обучаемый энкодер выдаст латент
почти той же размерности — ту же болезнь в новой обёртке, только дороже.

Меряется то самое пространство, в котором работает поток: z-scored DCT-16
коэффициенты, D = 92 288 (`prior17.to_model_space` с метой рабочего плеча).

**Доля дисперсии считается на отложенных состояниях.** Это принципиально: у
выборки из N образцов ковариация имеет ранг ≤ N−1, поэтому «объяснено 100 %
своими же компонентами» верно всегда и не значит ничего. Компоненты строятся на
обучающей части, а объяснённая доля меряется на тех, которых PCA не видела, —
такая оценка занижена, но честна.

Так как N < D, собственные числа берутся из матрицы Грама N × N, а не из
ковариации D × D: это то же самое разложение и в тысячи раз дешевле.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flydream.generate import learned as L
from flydream.generate import prior17 as R
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep


def states_of(net, d, bank, idx, mean, std, *, frames, dt, t_pre, batch, log=print) -> np.ndarray:
    out, t0 = [], time.time()
    for i in range(0, len(idx), batch):
        chunk = np.asarray(bank[idx[i:i + batch]], np.float16)
        st = simulate_states(net, chunk, d.cells_all, dt, t_pre, batch).astype(np.float32)
        m = L.to_maps(st.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
        out.append((m - mean[None, None, :, None]) / std[None, None, :, None])
        if (i // batch) % 10 == 0:
            log(f"  {i + len(chunk)}/{len(idx)} состояний, {time.time() - t0:.0f} с")
    return np.concatenate(out)


def run(model: str, prior_ckpt, gen_ckpt, manifest: dict, columns: dict, corpus: Path, *,
        n_fit: int = 2400, n_test: int = 600, frames: int = 40, dt: float = 0.02, t_pre: float = 1.0,
        batch: int = 16, seed: int = 0, log=print) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    net = load_network(model); dev = device_of(net)
    d = Deep(manifest, columns)
    from flydream.generate import gen13b as G
    _, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)
    _, pmeta = R.load(prior_ckpt, dev)

    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    bank = np.load(Path(corpus) / "videos.npz")["videos"]
    fit_idx = rng.choice(np.asarray(cm["split"]["train"]), n_fit, replace=False)
    test_idx = rng.choice(np.asarray(cm["split"]["test"]), n_test, replace=False)
    log(f"{n_fit} обучающих и {n_test} отложенных клипов; симуляция через мозг")

    kw = dict(frames=frames, dt=dt, t_pre=t_pre, batch=batch, log=log)
    A = states_of(net, d, bank, fit_idx, mean, std, **kw)
    B = states_of(net, d, bank, test_idx, mean, std, **kw)
    X = R.to_model_space(pmeta, A, dev).reshape(len(A), -1).cpu().numpy().astype(np.float64)
    Y = R.to_model_space(pmeta, B, dev).reshape(len(B), -1).cpu().numpy().astype(np.float64)
    D = X.shape[1]
    log(f"состояния готовы: {X.shape} и {Y.shape}, D = {D}, {time.time() - t0:.0f} с")

    mu = X.mean(0)
    Xc, Yc = X - mu, Y - mu
    gram = Xc @ Xc.T                                                   # N x N, ранг тот же
    w, U = np.linalg.eigh(gram)
    order = np.argsort(-w); w = np.clip(w[order], 0, None); U = U[:, order]
    keep = int((w > 1e-9 * w[0]).sum())
    V = (Xc.T @ U[:, :keep]) / np.sqrt(w[:keep])[None, :]              # (D, keep) главные направления
    log(f"{keep} компонент из матрицы Грама за {time.time() - t0:.0f} с")

    P = Yc @ V                                                         # проекции отложенных
    tot = float((Yc ** 2).sum())
    cum = np.cumsum((P ** 2).sum(0)) / tot
    cum_fit = np.cumsum(w[:keep]) / float((Xc ** 2).sum())

    def need(frac: float) -> int | None:
        j = np.searchsorted(cum, frac) + 1
        return int(j) if j <= keep else None

    out = {"prior_ckpt": str(prior_ckpt), "n_fit": n_fit, "n_test": n_test, "dims": int(D),
           "components": keep, "seed": seed,
           "held_out_at": {str(k): float(cum[min(k, keep) - 1]) for k in (1, 8, 32, 128, 512, 1024, 2048, keep)},
           "fit_at": {str(k): float(cum_fit[min(k, keep) - 1]) for k in (1, 8, 32, 128, 512, 1024, 2048, keep)},
           "need_held_out": {str(f): need(f) for f in (0.5, 0.8, 0.9, 0.95, 0.99)},
           "seconds": round(time.time() - t0, 1)}
    return {"summary": out, "arrays": {"eigenvalues": w[:keep].astype(np.float32),
                                       "cum_held_out": cum.astype(np.float32),
                                       "cum_fit": cum_fit.astype(np.float32)}}


def main(argv=None) -> int:
    from flydream.generate.invert import settings
    from flydream.model import ROOT

    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--prior", default=str(ROOT / "data" / "prior18" / "corpus_dct16_w192_lr1e3_c.pt"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--tag", default="pca18")
    p.add_argument("--fit", type=int, default=2400)
    p.add_argument("--test", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.prior), Path(a.gen), manifest, columns, Path(a.corpus),
            n_fit=a.fit, n_test=a.test, frames=g.get("frames", 40), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), batch=a.batch, seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    print(f"\nD = {S['dims']}, компонент построено {S['components']} (предел — размер обучающей выборки)")
    print(f"\n{'компонент':>10} {'на обучающих':>14} {'НА ОТЛОЖЕННЫХ':>16}")
    for k, v in S["held_out_at"].items():
        print(f"{int(k):10d} {100 * S['fit_at'][k]:13.1f}% {100 * v:15.1f}%")
    print(f"\nсколько компонент нужно на отложенных:")
    for f, k in S["need_held_out"].items():
        print(f"  {100 * float(f):5.0f}% дисперсии: {'не достигнуто до ' + str(S['components']) if k is None else k}")
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
