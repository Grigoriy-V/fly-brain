"""Новое видео, которого нет ни в одном клипе — путь между двумя сценами.

    python -m flydream.generate.blend18          # локально, CPU, $0

Безусловный розыгрыш сцены не даёт (18.17–18.21). Но генерация видео этим не
исчерпывается: два настоящих отложенных клипа обращаются в свои прообразы, и
**сферический путь между ними** проходит через состояния, которых не вызывал ни
один клип. 13B делает из них видео. Обучения не требуется — всё на уже
обученных чекпойнтах.

Что меряется на каждой точке пути:

- **прогонка** (видео → замороженный мозг → состояние против того, из которого
  рендерили): достижимо ли это состояние для мозга вообще;
- **ближайшее видео корпуса** — максимум корреляции по всем 15 514 клипам.
  Без этого числа «нового видео» заявлять нельзя: оно отделяет «сделано без
  исходного клипа» от «не копия ни одного клипа». Граница честности: тест на
  ближайшего соседа не ловит запоминание отдельного образца, поэтому низкая
  корреляция значит «не копия», а не «не запомнено» (AGENTS, 18.3).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flydream.decode import pairs as P
from flydream.generate import gen13b as G
from flydream.generate import learned as L
from flydream.generate import prior17 as R
from flydream.generate.edges18 import describe
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep
from flydream.generate.roundtrip13 import build_states, round_trip


def nearest(v: np.ndarray, bank: np.ndarray, chunk: int = 2048) -> tuple[int, float]:
    """(индекс, корреляция) ближайшего клипа банка — по всему клипу целиком."""
    a = np.asarray(v, np.float32).reshape(-1)
    a = (a - a.mean()) / (a.std() + 1e-8)
    best, who = -2.0, -1
    for i in range(0, len(bank), chunk):
        B = np.asarray(bank[i:i + chunk], np.float32).reshape(len(bank[i:i + chunk]), -1)
        B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-8)
        r = (B @ a) / len(a)
        j = int(np.argmax(r))
        if float(r[j]) > best:
            best, who = float(r[j]), i + j
    return who, best


def run(model: str, prior_ckpt, gen_ckpt, manifest: dict, columns: dict, corpus: Path, *,
        pairs: int = 2, alphas=(0.0, 0.25, 0.5, 0.75, 1.0), frames: int = 40, margin: int = 5,
        dt: float = 0.02, t_pre: float = 1.0, steps: int = 100, fixed_point: int = 3,
        seed: int = 0, log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = load_network(model); dev = device_of(net)
    _, index = P.type_index(net.connectome)
    d = Deep(manifest, columns)
    prior, pmeta = R.load(prior_ckpt, dev)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)

    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]
    ta = sa["target"][:, w0:w1, :][:, :, d.cells_all].cpu().numpy().astype(np.float32)
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}

    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    bank_all = np.asarray(cz["videos"][:, :frames], np.float16)
    ends = rng.choice(np.asarray(cm["split"]["test"]), 2 * pairs, replace=False).reshape(pairs, 2)
    log(f"{pairs} пар отложенных клипов: {ends.tolist()}")

    flat = ends.reshape(-1)
    st = simulate_states(net, np.asarray(cz["videos"], np.float16)[flat], d.cells_all,
                         dt, t_pre, 8).astype(np.float32)
    maps = L.to_maps(st.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
    real = (maps - mean[None, None, :, None]) / std[None, None, :, None]
    eps = R.to_noise(prior, pmeta, real, steps=steps, fixed_point=fixed_point, device=dev)
    log(f"оба конца обращены в прообразы за {time.time() - t0:.0f} с")

    a_eps, b_eps = eps[0::2], eps[1::2]
    jobs = {}
    for al in alphas:
        for p_ in range(pairs):
            jobs[f"{al}|{p_}"] = R.from_noise(prior, pmeta, R.slerp(a_eps[p_:p_ + 1], b_eps[p_:p_ + 1], float(al)),
                                              steps=steps, device=dev)[0]
    names = list(jobs)
    cond = torch.as_tensor(np.stack([jobs[n] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), 8):
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)       # один z у 13B везде
            vids.append(G.sample(gen, cond[i:i + 8], mask[i:i + 8], steps=20, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    target_raw = np.stack([R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0]
                           for n in names])
    rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
    log(f"{len(videos)} видео отрисовано и прогнано через мозг за {time.time() - t0:.0f} с")

    nb = np.asarray(L.neighbour_index(721))
    rows = []
    for al in alphas:
        sel = [i for i, n in enumerate(names) if n.startswith(f"{al}|")]
        s = describe([videos[i] for i in sel], nb)
        near = [nearest(videos[i], bank_all) for i in sel]
        rows.append({"alpha": float(al), "gate": float(np.median(rts[sel])),
                     "frac_flat": s["frac_flat"]["mean"], "kurtosis": s["grad_kurtosis"]["mean"],
                     "neigh_r": s["neigh_r"]["mean"], "sd": s["sd"]["mean"],
                     "nearest_r": float(np.mean([r for _, r in near])),
                     "nearest_idx": [int(j) for j, _ in near]})
        log(f"  α = {al}: ворота {rows[-1]['gate']:.4f}, ближайшее видео корпуса r = {rows[-1]['nearest_r']:.3f}")
    raw = describe([np.asarray(bank_all[i], np.float32) for i in flat], nb)
    out = {"prior_ckpt": str(prior_ckpt), "pairs": pairs, "ends": ends.tolist(), "alphas": list(alphas),
           "steps": steps, "seed": seed, "rows": rows, "n_bank": int(len(bank_all)),
           "raw": {"frac_flat": raw["frac_flat"]["mean"], "sd": raw["sd"]["mean"]},
           "seconds": round(time.time() - t0, 1)}
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    arrays.update({f"raw__{p_}__{e}": np.asarray(bank_all[ends[p_][e]], np.float32)
                   for p_ in range(pairs) for e in (0, 1)})
    return {"summary": out, "arrays": arrays}


def main(argv=None) -> int:
    from flydream.generate.invert import settings
    from flydream.model import ROOT

    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--prior", default=str(ROOT / "data" / "prior18" / "corpus_dct16_w384_lr1e3_c.pt"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--tag", default="blend18")
    p.add_argument("--pairs", type=int, default=2)
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.prior), Path(a.gen), manifest, columns, Path(a.corpus), pairs=a.pairs,
            alphas=tuple(float(x) for x in a.alphas.split(",")), frames=g.get("frames", 40),
            margin=g.get("margin", 5), dt=g.get("dt", 0.02), t_pre=g.get("t_pre", 1.0),
            steps=a.steps, seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    print(f"\nсырые концы: ровного {100 * S['raw']['frac_flat']:.1f} %, контраст {S['raw']['sd']:.3f}")
    print("\n{:>10} {:>8} {:>9} {:>8} {:>8} {:>9} {:>22}".format(
        "доля пути", "ворота", "ровного", "эксцесс", "сосед r", "контраст", "ближайшее видео корпуса"))
    for w in S["rows"]:
        print("{:10.2f} {:8.4f} {:8.1f}% {:8.2f} {:8.3f} {:9.3f} {:22.3f}".format(
            w["alpha"], w["gate"], 100 * w["frac_flat"], w["kurtosis"], w["neigh_r"], w["sd"], w["nearest_r"]))
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
