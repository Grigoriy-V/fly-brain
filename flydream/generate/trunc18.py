"""Потолок усечения DCT: до какого K настоящее состояние ещё остаётся сценой.

    python -m flydream.generate.trunc18          # локально, CPU, $0

Вся ставка на VAE держится на одном: **меньше размерность — задача сэмплера
выполнимее** (при D = 92 288 оболочка занимает полосу 0,23 %, и всё от всего в
73 σ; при 45× меньшей размерности тот же относительный дефект даёт 9 σ).
Прежде чем строить энкодер, размерность проверяется бесплатно тем энкодером,
который у нас уже есть, — усечением DCT по времени.

Это верхняя граница, а не результат обучения: настоящее состояние ужимается до
K коэффициентов, разворачивается обратно в 40 кадров и рендерится через 13B.
Если при K = 8 сцена на клипе выживает, представление выдерживает половинную
размерность и учить на ней осмысленно. Если разваливается — DCT до такого K
опускать нельзя, и нужен обучаемый энкодер, а не усечение.

Сравнение всегда с **сырым видео корпуса**, как требует критерий приёмки
(`reports/2026-09-20_the_seed_problem.md`).
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
from flydream.generate.invert import device_of, load_network, pixcorr_per_frame
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep
from flydream.generate.roundtrip13 import build_states, round_trip


def run(model: str, gen_ckpt, manifest: dict, columns: dict, corpus: Path, *,
        ks=(16, 8, 4, 2), n_clips: int = 6, frames: int = 40, margin: int = 5,
        dt: float = 0.02, t_pre: float = 1.0, seed: int = 0, log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = load_network(model); dev = device_of(net)
    _, index = P.type_index(net.connectome)
    d = Deep(manifest, columns)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)

    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]
    ta = sa["target"][:, w0:w1, :][:, :, d.cells_all].cpu().numpy().astype(np.float32)
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}

    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    clip_idx = rng.choice(np.asarray(cm["split"]["test"]), n_clips, replace=False)
    ref = np.asarray(cz["videos"][clip_idx][:, :frames], np.float32)
    st = simulate_states(net, np.asarray(cz["videos"], np.float16)[clip_idx], d.cells_all,
                         dt, t_pre, 8).astype(np.float32)
    maps = L.to_maps(st.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
    real = (maps - mean[None, None, :, None]) / std[None, None, :, None]
    log(f"{n_clips} настоящих состояний, клипы {clip_idx.tolist()}, {time.time() - t0:.0f} с")

    full = "K = 40 (без усечения)"
    groups = {full: real}
    err = {full: 0.0}
    for k in ks:
        dm = R.dct_matrix(frames, k)
        back = R.from_dct(R.to_dct(real, dm), dm).astype(np.float32)
        groups[f"K = {k}"] = back
        e = float(np.linalg.norm((back - real).reshape(n_clips, -1), axis=1).mean()
                  / np.linalg.norm(real.reshape(n_clips, -1), axis=1).mean())
        err[f"K = {k}"] = e
        log(f"K = {k:2d}: D = {k * 8 * 721:6d}, относительная ошибка усечения {100 * e:.2f} %")

    jobs = {f"{g}|{i}": v[i] for g, v in groups.items() for i in range(n_clips)}
    names = list(jobs)
    cond = torch.as_tensor(np.stack([jobs[n] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), 8):
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)      # один z у 13B везде
            vids.append(G.sample(gen, cond[i:i + 8], mask[i:i + 8], steps=20, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    target_raw = np.stack([R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0]
                           for n in names])
    rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
    log(f"{len(videos)} видео отрисовано и оценено за {time.time() - t0:.0f} с")

    nb = np.asarray(L.neighbour_index(721))
    out = {"clip_idx": clip_idx.tolist(), "n_clips": n_clips, "frames": frames, "ks": list(ks), "groups": {}}
    for g in groups:
        sel = [i for i, n in enumerate(names) if n.startswith(g + "|")]
        s = describe([videos[i] for i in sel], nb)
        r = float(np.mean([pixcorr_per_frame(videos[i], ref[j]).mean() for j, i in enumerate(sel)]))
        k_of = frames if g == full else int(g.split("=")[1])
        out["groups"][g] = {"dims": int(k_of * 8 * 721), "trunc_error": err[g], "r_to_raw": r,
                            "gate": float(np.median(rts[sel])), "frac_flat": s["frac_flat"]["mean"],
                            "kurtosis": s["grad_kurtosis"]["mean"], "neigh_r": s["neigh_r"]["mean"],
                            "sd": s["sd"]["mean"]}
    raw = describe(list(ref), nb)
    out["raw"] = {"frac_flat": raw["frac_flat"]["mean"], "kurtosis": raw["grad_kurtosis"]["mean"],
                  "neigh_r": raw["neigh_r"]["mean"], "sd": raw["sd"]["mean"]}
    out["seconds"] = round(time.time() - t0, 1)
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    arrays.update({f"raw__{i}": ref[i] for i in range(n_clips)})
    return {"summary": out, "arrays": arrays}


def main(argv=None) -> int:
    from flydream.generate.invert import settings
    from flydream.model import ROOT

    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--tag", default="trunc18")
    p.add_argument("--ks", default="16,8,4,2")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.gen), manifest, columns, Path(a.corpus),
            ks=tuple(int(x) for x in a.ks.split(",")), n_clips=a.clips, frames=g.get("frames", 40),
            margin=g.get("margin", 5), dt=g.get("dt", 0.02), t_pre=g.get("t_pre", 1.0), seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    w = S["raw"]
    print(f"\nсырое видео корпуса: ровного {100 * w['frac_flat']:.1f} %, контраст {w['sd']:.3f}")
    head = ("представление", "D", "ошибка", "r к сырому", "ворота", "ровного", "эксцесс", "контраст")
    print("\n{:24} {:>7} {:>8} {:>11} {:>8} {:>9} {:>8} {:>9}".format(*head))
    for name, v in S["groups"].items():
        print("{:24} {:7d} {:7.2f}% {:11.3f} {:8.4f} {:8.1f}% {:8.2f} {:9.3f}".format(
            name, v["dims"], 100 * v["trunc_error"], v["r_to_raw"], v["gate"],
            100 * v["frac_flat"], v["kurtosis"], v["sd"]))
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
