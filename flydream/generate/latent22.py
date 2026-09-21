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
from flydream.generate import pca19 as P
from flydream.generate import prior17 as R
from flydream.generate.edges18 import describe
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network, pixcorr_per_frame
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep
from flydream.generate.roundtrip13 import build_states


def block_index(types, n_dct: int = 16, n_types: int = 8, n_cols: int = 721,
                maps_order: bool = False) -> np.ndarray:
    """Плоские индексы блока в состоянии (DCT, тип, колонка), порядок строк C.

    `maps_order=True` перечисляет их так, как их видит `maps[:, :, ch]` на
    Modal — коэффициент снаружи, тип внутри. Базис, подогнанный там, разложен
    именно в этом порядке, и перепутать их значит перемешать каналы."""
    ch = [DEEP.index(t) for t in types]
    if maps_order:
        return np.concatenate([np.arange((k * n_types + c) * n_cols, (k * n_types + c) * n_cols + n_cols)
                               for k in range(n_dct) for c in ch])
    return np.concatenate([np.arange((k * n_types + c) * n_cols, (k * n_types + c) * n_cols + n_cols)
                           for c in ch for k in range(n_dct)])


def channel_basis(basis: torch.Tensor, lam: torch.Tensor, idx: np.ndarray, n_chan: int,
                  n_cols: int = 721) -> tuple:
    """PCA по КАНАЛАМ в колонке: одно отображение 32 -> m, общее для всех колонок.

    Глобальная PCA мешает вместе каналы и пространство, и падение резкости
    (доля ровного поля 24-28 % при любом k, 22.1) приходит от пространственной
    части. Здесь пространство не трогается вовсе: ковариация каналов
    усредняется по колонкам, а сжатие применяется к каждой колонке отдельно.
    Избыточность там измерена (21б): восемь типов в колонке скоррелированы на
    0,373 в среднем, шестнадцать коэффициентов DCT — на 0,183.

    Ковариация берётся из уже посчитанного базиса: `A = B[блок]·√λ`, тогда
    `C = mean_колонки A Aᵀ` по оси каналов."""
    A = (basis[idx] * lam.sqrt()).reshape(n_chan, n_cols, -1)
    C = torch.einsum("acj,bcj->ab", A, A) / n_cols
    ev, V = torch.linalg.eigh(C)
    return V.flip(1), ev.flip(0).clamp_min(0)


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
        margin: int = 5, dt: float = 0.02, t_pre: float = 1.0, seed: int = 0,
        block_pca: str = "", channels=(), log=print) -> dict:
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
    p_block = None
    if block_pca:                                                     # настоящая PCA по блоку, подогнанная на карте
        zb = np.load(block_pca)
        p_block = P.from_numpy(zb, dev)
        fitted = json.loads(str(zb["summary"])) if "summary" in zb.files else {}
        idx = block_index(types, maps_order=True)
        if p_block["dims"] != len(idx):
            raise ValueError(f"базис на {p_block['dims']} чисел, блок на {len(idx)}")
        cum = None
        log(f"блок {'+'.join(types)}: {len(idx)} чисел, НАСТОЯЩИЙ базис {p_block['k']} из {block_pca}, "
            f"{time.time() - t0:.0f} с")
    else:
        idx = block_index(types)
        U, ev = block_basis(basis, lam, idx, max(ks))
        cum = (torch.cumsum(ev, 0) / ev.sum()).cpu().numpy()
        fitted = {}
        log(f"блок {'+'.join(types)}: {len(idx)} чисел, ОЦЕНОЧНЫЙ базис {U.shape[1]} направлений, "
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
    coef = None if p_block is not None else (flat[:, idx] - mu_full[idx]) @ U
    log(f"{n_clips} состояний, клипы {clip_idx.tolist()}, {time.time() - t0:.0f} с")

    def state_from(k: int) -> np.ndarray:
        out = torch.zeros_like(flat)
        if p_block is not None:
            pk = P.truncate(p_block, k)
            out[:, idx] = P.decode(P.encode(flat[:, idx], pk), pk)
        else:
            out[:, idx] = mu_full[idx] + coef[:, :k] @ U[:, :k].T
        return R.from_model_space(pmeta, out.reshape(len(out), *shape))

    groups = {}
    full = torch.zeros_like(flat); full[:, idx] = flat[:, idx]
    groups[f"{'+'.join(types)} без сжатия"] = R.from_model_space(pmeta, full.reshape(len(full), *shape))
    chan_ev = None
    if channels:
        # Ось каналов — (коэффициент DCT, тип), 16 x 2 = 32. Индексы обязаны идти
        # в порядке карт: коэффициент снаружи, тип внутри, колонка последней.
        cidx = block_index(types, maps_order=True)
        n_chan = len(types) * 16
        V, chan_ev = channel_basis(basis, lam, cidx, n_chan)
        mu_c = mu_full[cidx].reshape(n_chan, 721)
        xc = flat[:, cidx].reshape(len(flat), n_chan, 721) - mu_c
        for m in channels:
            Vm = V[:, :m]
            back = mu_c + torch.einsum("cm,nmk->nck", Vm, torch.einsum("cm,nck->nmk", Vm, xc))
            out = torch.zeros_like(flat)
            out[:, cidx] = back.reshape(len(flat), -1)
            groups[f"каналов {m}"] = R.from_model_space(pmeta, out.reshape(len(out), *shape))
            log(f"  каналов {m:2}: {m * 721} чисел, {time.time() - t0:.0f} с")
    for k in ks:
        groups[f"k = {k}"] = state_from(min(k, p_block["k"] if p_block is not None else U.shape[1]))

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
    out = {"types": list(types), "ks": list(ks), "channels": list(channels),
           "channel_variance": None if chan_ev is None else (chan_ev / chan_ev.sum()).tolist(),
           "block_numbers": int(len(idx)),
           "clip_idx": clip_idx.tolist(), "n_clips": n_clips,
           "block_pca": str(block_pca or ""),
           "explained": ({int(k): float(cum[min(k, len(cum)) - 1]) for k in ks} if cum is not None else
                         {int(k): float(fitted.get("explained", {}).get("test", {}).get(str(k), {})
                                        .get("explained", float("nan"))) for k in ks}),
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
    p.add_argument("--block-pca", default="", help="npz настоящей PCA по блоку вместо оценки из полного базиса")
    p.add_argument("--channels", default="", help="лестница сжатия КАНАЛОВ в колонке: 1,2,4,8,16,32")
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.pca), Path(a.gen), manifest, columns, Path(a.corpus),
            types=tuple(x.strip() for x in a.types.split(",") if x.strip()),
            ks=tuple(int(x) for x in a.ks.split(",")), n_clips=a.clips,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), seed=a.seed, block_pca=a.block_pca,
            channels=tuple(int(x) for x in a.channels.split(",") if x.strip()))
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    nums = {f"каналов {m}": m * 721 for m in S["channels"]} | {f"k = {k}": k for k in S["ks"]}
    print()
    print("{:24} {:>9} {:>11} {:>9}".format("сжатие", "чисел", "r к сырому", "ровного"))
    for k, v in S["groups"].items():
        print("{:24} {:9} {:11.3f} {:8.1f}%".format(
            k, nums.get(k, S["block_numbers"]), v["r_to_raw"], 100 * v["frac_flat"]))
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
