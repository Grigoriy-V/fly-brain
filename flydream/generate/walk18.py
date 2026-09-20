"""18.18: насколько толста область сцен в пространстве шума.

    python -m flydream.generate.walk18          # локально, CPU, $0

18.17 нашёл, где лежит шум, дающий сцену: на 66–81 σ внутрь от оболочки, на
которую садится любой розыгрыш. Остался вопрос, достижима ли эта область
направленно — или она исчезающе тонкая.

Проверка: сферическая интерполяция от **обычного розыгрыша** к **шуму
настоящего состояния** и структура на каждой доле пути. Если сцена появляется
уже к середине — область широкая, и направленный поиск осмыслен. Если только
на последних процентах — она тонкая, и никакой сэмплер её не нащупает.

Радиус по пути меняется вместе с направлением, поэтому он записывается на
каждом шаге: без него нельзя отличить «двинулись в нужную сторону» от «просто
укоротили», а 18.17 показал, что одно укорачивание гасит сэмпл в серое.
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


def run(model: str, prior_ckpt, gen_ckpt, manifest: dict, columns: dict, corpus: Path, *,
        n_clips: int = 4, alphas=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0), frames: int = 40,
        margin: int = 5, dt: float = 0.02, t_pre: float = 1.0, steps: int = 100, fixed_point: int = 3,
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
    bank = np.asarray(cz["videos"], np.float16)
    clip_idx = rng.choice(np.asarray(cm["split"]["test"]), n_clips, replace=False)
    clip_states = simulate_states(net, bank[clip_idx], d.cells_all, dt, t_pre, 8).astype(np.float32)
    maps = L.to_maps(clip_states.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
    real = (maps - mean[None, None, :, None]) / std[None, None, :, None]
    log(f"{n_clips} real states, clips {clip_idx.tolist()}, {time.time() - t0:.0f} s")

    eps_star = R.to_noise(prior, pmeta, real, steps=steps, fixed_point=fixed_point, device=dev)
    g = torch.Generator(device=dev).manual_seed(4000 + seed)
    eps0 = torch.randn(eps_star.shape, device=dev, generator=g).cpu().numpy().astype(np.float32)
    log(f"inverted; ‖ε*‖ {np.linalg.norm(eps_star.reshape(n_clips, -1), axis=1).mean():.1f}, "
        f"‖ε0‖ {np.linalg.norm(eps0.reshape(n_clips, -1), axis=1).mean():.1f}, {time.time() - t0:.0f} s")

    jobs, radii = {}, {}
    for al in alphas:
        e = R.slerp(eps0, eps_star, float(al))
        radii[al] = float(np.linalg.norm(e.reshape(n_clips, -1), axis=1).mean())
        st = R.from_noise(prior, pmeta, e, steps=steps, device=dev)
        for i in range(n_clips):
            jobs[f"{al}|{i}"] = st[i]
    for i in range(n_clips):                                                  # опора: само настоящее состояние
        jobs[f"real|{i}"] = real[i]
    names = list(jobs)
    log(f"{len(names)} states along the walk in {time.time() - t0:.0f} s")

    cond = torch.as_tensor(np.stack([jobs[n] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), 8):
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)          # один z у 13B везде
            vids.append(G.sample(gen, cond[i:i + 8], mask[i:i + 8], steps=20, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    target_raw = np.stack([R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0]
                           for n in names])
    rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
    log(f"rendered and scored in {time.time() - t0:.0f} s")

    nb = np.asarray(L.neighbour_index(721))
    rows = []
    for al in list(alphas) + ["real"]:
        sel = [i for i, n in enumerate(names) if n.startswith(f"{al}|")]
        st = describe([videos[i] for i in sel], nb)
        rows.append({"alpha": None if al == "real" else float(al),
                     "radius": radii.get(al), "gate": float(np.median(rts[sel])),
                     "frac_flat": st["frac_flat"]["mean"], "kurtosis": st["grad_kurtosis"]["mean"],
                     "neigh_r": st["neigh_r"]["mean"], "sd": st["sd"]["mean"]})
    out = {"prior_ckpt": str(prior_ckpt), "n_clips": n_clips, "alphas": list(alphas), "steps": steps,
           "seed": seed, "clip_idx": clip_idx.tolist(), "dims": int(np.prod(eps_star.shape[1:])),
           "typical_radius": float(np.sqrt(np.prod(eps_star.shape[1:]))), "rows": rows,
           "seconds": round(time.time() - t0, 1)}
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
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
    p.add_argument("--tag", default="walk18_local")
    p.add_argument("--clips", type=int, default=4)
    p.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,0.9,0.95,1.0")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.prior), Path(a.gen), manifest, columns, Path(a.corpus), n_clips=a.clips,
            alphas=tuple(float(x) for x in a.alphas.split(",")), frames=g.get("frames", 40),
            margin=g.get("margin", 5), dt=g.get("dt", 0.02), t_pre=g.get("t_pre", 1.0),
            steps=a.steps, seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    print(f"\nтипичный радиус √D = {S['typical_radius']:.1f}")
    print(f"{'доля пути':>10} {'радиус':>8} {'ворота':>8} {'ровного':>9} {'эксцесс':>8} {'сосед r':>8} {'контраст':>9}")
    for w in S["rows"]:
        al = "настоящее" if w["alpha"] is None else f"{w['alpha']:.2f}"
        rad = "—" if w["radius"] is None else f"{w['radius']:.1f}"
        print(f"{al:>10} {rad:>8} {w['gate']:8.4f} {100 * w['frac_flat']:8.1f}% {w['kurtosis']:8.2f} "
              f"{w['neigh_r']:8.3f} {w['sd']:9.3f}")
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
