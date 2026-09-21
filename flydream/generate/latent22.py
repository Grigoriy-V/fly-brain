r"""22.1: лестница k для латента над блоком типов. Локально, $0, ~2 мин.

    python -m flydream.generate.latent22

Блок — это часть состояния, которую мы решили задавать (по умолчанию T4a+T4b,
все 16 коэффициентов, все 721 колонка = 23 072 числа). Вопрос: сколько
координат нужно, чтобы клип ещё держался.

Базис блока берётся **из уже посчитанного PCA полного состояния**, без
скачивания состояний и без Modal. Ковариация блока в ранге k₀ равна
`A Aᵀ`, где `A = B[блок]·√λ`; собственные векторы `AᵀA` (матрица k₀ × k₀)
дают направления блока, а `U = A W / √ev` — сам базис. Ограничение отсюда:
всё, что не попало в исходные 2 048 компонент полного состояния, здесь
невидимо, и собственные числа обучающие. Честная кривая — после подгонки PCA
прямо на блоке; это оценка, которая ничего не стоит.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flydream.decode import pairs as P13
from flydream.generate import gen13b as G
from flydream.generate import learned as L
from flydream.generate import prior17 as R
from flydream.generate.edges18 import describe
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network, pixcorr_per_frame
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep
from flydream.generate.roundtrip13 import build_states


def block_index(types, n_dct: int = 16, n_types: int = 8, n_cols: int = 721) -> np.ndarray:
    """Плоские индексы блока в состоянии (DCT, тип, колонка), порядок строк C."""
    out = []
    for t in types:
        c = DEEP.index(t)
        for k in range(n_dct):
            i0 = (k * n_types + c) * n_cols
            out.append(np.arange(i0, i0 + n_cols))
    return np.concatenate(out)


def block_basis(basis: torch.Tensor, lam: torch.Tensor, idx: np.ndarray, kmax: int) -> tuple:
    """Главные направления блока внутри уже посчитанного подпространства."""
    A = basis[idx] * lam.sqrt()
    ev, W = torch.linalg.eigh(A.T @ A)
    ev, W = ev.flip(0).clamp_min(0), W.flip(1)
    kmax = min(kmax, int((ev > 1e-8).sum()))
    U = (A @ W[:, :kmax]) / ev[:kmax].sqrt()
    return U, ev[:kmax]


def run(model: str, pca_path: Path, gen_ckpt: Path, manifest: dict, columns: dict, corpus: Path, *,
        types=("T4a", "T4b"), ks=(256, 512, 1024, 1536), n_clips: int = 6, frames: int = 40,
        margin: int = 5, dt: float = 0.02, t_pre: float = 1.0, seed: int = 0, log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = load_network(model); dev = device_of(net)
    _, index = P13.type_index(net.connectome)
    d = Deep(manifest, columns)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)

    z = np.load(pca_path)
    pmeta = json.loads(str(z["meta"]))
    basis = torch.as_tensor(np.asarray(z["basis"], np.float32), device=dev)
    lam = torch.as_tensor(np.asarray(z["lam"], np.float32), device=dev)
    mu_full = torch.as_tensor(np.asarray(z["mean"], np.float32), device=dev)
    idx = block_index(types)
    U, ev = block_basis(basis, lam, idx, max(ks))
    cum = (torch.cumsum(ev, 0) / ev.sum()).cpu().numpy()
    log(f"блок {'+'.join(types)}: {len(idx)} чисел, базис {U.shape[1]} направлений, "
        f"{time.time() - t0:.0f} с")

    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    clip_idx = rng.choice(np.asarray(cm["split"]["test"]), n_clips, replace=False)
    raw = np.asarray(cz["videos"][clip_idx][:, :frames], np.float32)
    st = simulate_states(net, np.asarray(cz["videos"], np.float16)[clip_idx], d.cells_all,
                         dt, t_pre, 8).astype(np.float32)[:, :frames]
    maps = L.to_maps(st.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
    real = (maps - mean[None, None, :, None]) / std[None, None, :, None]
    x = R.to_model_space(pmeta, real, dev)
    shape = tuple(x.shape[1:])
    flat = x.reshape(len(x), -1)
    blk = flat[:, idx] - mu_full[idx]
    coef = blk @ U
    log(f"{n_clips} состояний, клипы {clip_idx.tolist()}, {time.time() - t0:.0f} с")

    def state_from(c: torch.Tensor, k: int) -> np.ndarray:
        out = torch.zeros_like(flat)
        out[:, idx] = mu_full[idx] + c[:, :k] @ U[:, :k].T
        return R.from_model_space(pmeta, out.reshape(len(out), *shape))

    groups = {}
    full = torch.zeros_like(flat); full[:, idx] = flat[:, idx]
    groups[f"{'+'.join(types)} без сжатия"] = R.from_model_space(pmeta, full.reshape(len(full), *shape))
    for k in ks:
        groups[f"k = {k}"] = state_from(coef, min(k, U.shape[1]))

    names = [f"{g}|{i}" for g in groups for i in range(n_clips)]
    cond = torch.as_tensor(np.stack([groups[n.rsplit('|', 1)[0]][int(n.rsplit('|', 1)[1])]
                                     for n in names]), device=dev)
    bits = np.array([1.0 if t in types else 0.0 for t in DEEP], np.float32)
    mask = torch.as_tensor(np.tile(bits, (len(names), 1)), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), n_clips):                       # пачка = группа (ISS-0009)
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)
            vids.append(G.sample(gen, cond[i:i + n_clips], mask[i:i + n_clips],
                                 steps=20, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    log(f"{len(videos)} видео отрисовано, {time.time() - t0:.0f} с")

    nb = np.asarray(L.neighbour_index(721))
    out = {"types": list(types), "ks": list(ks), "block_numbers": int(len(idx)),
           "clip_idx": clip_idx.tolist(), "n_clips": n_clips,
           "explained": {int(k): float(cum[min(k, len(cum)) - 1]) for k in ks},
           "groups": {}}
    for g in groups:
        sel = [i for i, n in enumerate(names) if n.startswith(g + "|")]
        s = describe([videos[i] for i in sel], nb)
        r = float(np.mean([pixcorr_per_frame(videos[i], raw[j]).mean() for j, i in enumerate(sel)]))
        out["groups"][g] = {"r_to_raw": r, "frac_flat": s["frac_flat"]["mean"], "sd": s["sd"]["mean"]}
        log(f"  {g:22} r {r:.3f}, ровного {100 * s['frac_flat']['mean']:.1f} %")
    rw = describe(list(raw), nb)
    out["raw"] = {"frac_flat": rw["frac_flat"]["mean"], "sd": rw["sd"]["mean"]}
    out["seconds"] = round(time.time() - t0, 1)
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    arrays.update({f"raw__{i}": raw[i] for i in range(n_clips)})
    return {"summary": out, "arrays": arrays}


def main(argv=None) -> int:
    from flydream.generate.invert import settings
    from flydream.model import ROOT

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--pca", default=str(ROOT / "data" / "prior19" / "pca2048.npz"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="latent22")
    p.add_argument("--types", default="T4a,T4b")
    p.add_argument("--ks", default="256,512,1024,1536")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.pca), Path(a.gen), manifest, columns, Path(a.corpus),
            types=tuple(x.strip() for x in a.types.split(",") if x.strip()),
            ks=tuple(int(x) for x in a.ks.split(",")), n_clips=a.clips,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    print("\n{:24} {:>11} {:>9} {:>12}".format("латент", "r к сырому", "ровного", "дисперсии"))
    for k, v in S["groups"].items():
        kk = k.replace("k = ", "")
        exp = S["explained"].get(kk) or S["explained"].get(int(kk)) if kk.isdigit() else None
        print("{:24} {:11.3f} {:8.1f}% {:>12}".format(
            k, v["r_to_raw"], 100 * v["frac_flat"], f"{100 * exp:.1f} %" if exp else "—"))
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
