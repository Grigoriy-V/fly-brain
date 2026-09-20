"""18.24: приёмка VAE — заданный латент и разыгранный, оба против сырого видео.

    python -m flydream.generate.vaeval18         # локально, CPU, $0

Критерий человека (2026-09-20, `reports/2026-09-20_the_seed_problem.md` § 5)
состоит из двух половин, и смысл только в том, чтобы они сошлись вместе:

**A. Пара работает.** Отложенный клип → мозг → состояние → энкодер → z →
декодер → 13B → видео, сравнение с **сырым видео корпуса**. Считается дважды:
из разыгранного z = μ + σ·ε (так пара выглядит в генеративном режиме) и из
самого μ (так пара выглядит как пара). При сильном KL σ сравнима с μ, и первое
число меряет уже не пару, а шум апостериора, поэтому нужны оба.

**B. Сид разыгрываемый.** z от отложенных состояний обязан быть стандартным
нормальным (радиус √D ± 1,4 при ст. откл. 0,99–1,01), и тогда свежий розыгрыш
z неотличим от закодированного настоящего — то есть обязан давать сцену.

Поэтому здесь считаются обе: энкодер-декодер на отложенных, и чистый
`torch.randn` через тот же декодер. Плюс две опоры — 13B прямо из настоящего
состояния (потолок цепочки) и, для розыгрыша, ближайшее видео корпуса из всех
15 514 (без этого числа «новое видео» заявлять нельзя).
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
from flydream.generate import vae18 as V
from flydream.generate.edges18 import describe
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network, pixcorr_per_frame
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep
from flydream.generate.roundtrip13 import build_states, round_trip


def nearest(v: np.ndarray, bank: np.ndarray, chunk: int = 2048) -> tuple[int, float]:
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


def run(model: str, vae_ckpt, gen_ckpt, manifest: dict, columns: dict, corpus: Path, *,
        n_clips: int = 6, frames: int = 40, margin: int = 5, dt: float = 0.02, t_pre: float = 1.0,
        seed: int = 0, log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = load_network(model); dev = device_of(net)
    _, index = P.type_index(net.connectome)
    d = Deep(manifest, columns)
    vae, vmeta = V.load(vae_ckpt, dev)
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
    clip_idx = rng.choice(np.asarray(cm["split"]["test"]), n_clips, replace=False)
    raw = np.asarray(cz["videos"][clip_idx][:, :frames], np.float32)
    st = simulate_states(net, np.asarray(cz["videos"], np.float16)[clip_idx], d.cells_all,
                         dt, t_pre, 8).astype(np.float32)
    maps = L.to_maps(st.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
    real = (maps - mean[None, None, :, None]) / std[None, None, :, None]
    log(f"{n_clips} отложенных состояний, клипы {clip_idx.tolist()}, {time.time() - t0:.0f} с")

    x = R.to_model_space(vmeta, real, dev)                            # то, что видит энкодер
    with torch.no_grad():
        mu, logvar = vae.encode(x)
        g = torch.Generator(device=dev).manual_seed(7000 + seed)
        z_enc = mu + torch.exp(0.5 * logvar) * torch.randn(mu.shape, device=dev, generator=g)
        z_drw = V.sample_latent(vae, n_clips, device=dev,
                                generator=torch.Generator(device=dev).manual_seed(8000 + seed))
        rec_enc = R.from_model_space(vmeta, vae.decode(z_enc))
        rec_mu = R.from_model_space(vmeta, vae.decode(mu))            # A без шума апостериора
        rec_drw = R.from_model_space(vmeta, vae.decode(z_drw))
    D = int(np.prod(z_enc.shape[1:]))
    zf, zd = (t_.reshape(n_clips, -1).float().cpu().numpy() for t_ in (z_enc, z_drw))
    latent = {"dims": D, "typical_radius": float(np.sqrt(D)),
              "enc_sd": float(zf.std()), "enc_radius": float(np.linalg.norm(zf, axis=1).mean()),
              "enc_radius_sd": float(np.linalg.norm(zf, axis=1).std()),
              "draw_sd": float(zd.std()), "draw_radius": float(np.linalg.norm(zd, axis=1).mean()),
              "mu_sd": float(mu.float().std()), "sigma": float(torch.exp(0.5 * logvar).float().mean())}
    log(f"латент: закодированный {latent['enc_radius']:.1f} (sd {latent['enc_sd']:.3f}), "
        f"розыгрыш {latent['draw_radius']:.1f}, √D = {latent['typical_radius']:.1f}")

    groups = {"настоящее состояние": real, "через VAE (A)": rec_enc,
              "через VAE, из среднего (A)": rec_mu, "розыгрыш латента (B)": rec_drw}
    jobs = {f"{k}|{i}": v[i] for k, v in groups.items() for i in range(n_clips)}
    names = list(jobs)
    cond = torch.as_tensor(np.stack([jobs[n] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), 8):
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)  # один z у 13B везде
            vids.append(G.sample(gen, cond[i:i + 8], mask[i:i + 8], steps=20, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    target_raw = np.stack([R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0]
                           for n in names])
    rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
    log(f"{len(videos)} видео отрисовано и прогнано через мозг за {time.time() - t0:.0f} с")

    nb = np.asarray(L.neighbour_index(721))
    out = {"vae_ckpt": str(vae_ckpt), "clip_idx": clip_idx.tolist(), "n_clips": n_clips,
           "latent": latent, "groups": {}}
    for k in groups:
        sel = [i for i, n in enumerate(names) if n.startswith(k + "|")]
        s = describe([videos[i] for i in sel], nb)
        r = float(np.mean([pixcorr_per_frame(videos[i], raw[j]).mean() for j, i in enumerate(sel)]))
        near = [nearest(videos[i], bank_all) for i in sel]
        out["groups"][k] = {"r_to_raw": r, "gate": float(np.median(rts[sel])),
                            "frac_flat": s["frac_flat"]["mean"], "kurtosis": s["grad_kurtosis"]["mean"],
                            "neigh_r": s["neigh_r"]["mean"], "sd": s["sd"]["mean"],
                            "nearest_r": float(np.mean([r_ for _, r_ in near])),
                            "nearest_idx": [int(j) for j, _ in near]}
    rw = describe(list(raw), nb)
    out["raw"] = {"frac_flat": rw["frac_flat"]["mean"], "kurtosis": rw["grad_kurtosis"]["mean"],
                  "neigh_r": rw["neigh_r"]["mean"], "sd": rw["sd"]["mean"]}
    out["seconds"] = round(time.time() - t0, 1)
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    arrays.update({f"raw__{i}": raw[i] for i in range(n_clips)})
    return {"summary": out, "arrays": arrays}


def main(argv=None) -> int:
    from flydream.generate.invert import settings
    from flydream.model import ROOT

    try:                                                              # вывод в файл под cp1251 иначе
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")    # падает на знаке корня
    except (AttributeError, OSError):
        pass
    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--vae", default=str(ROOT / "data" / "prior18" / "vae_z3_b1e4.pt"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--tag", default="vaeval18")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.vae), Path(a.gen), manifest, columns, Path(a.corpus), n_clips=a.clips,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]; lt = S["latent"]
    print(f"\nлатент {lt['dims']}, √D = {lt['typical_radius']:.1f}")
    print(f"  закодированный отложенный: радиус {lt['enc_radius']:.1f} +- {lt['enc_radius_sd']:.2f}, "
          f"ст. откл. {lt['enc_sd']:.3f}  (mu {lt['mu_sd']:.3f}, sigma {lt['sigma']:.3f})")
    print(f"  свежий розыгрыш:           радиус {lt['draw_radius']:.1f}, ст. откл. {lt['draw_sd']:.3f}")
    w = S["raw"]
    print(f"\nсырое видео корпуса: ровного {100 * w['frac_flat']:.1f} %, контраст {w['sd']:.3f}")
    print("\n{:24} {:>11} {:>8} {:>9} {:>8} {:>9} {:>10}".format(
        "группа", "r к сырому", "ворота", "ровного", "эксцесс", "контраст", "ближайшее"))
    for k, v in S["groups"].items():
        print("{:24} {:11.3f} {:8.4f} {:8.1f}% {:8.2f} {:9.3f} {:10.3f}".format(
            k, v["r_to_raw"], v["gate"], 100 * v["frac_flat"], v["kurtosis"], v["sd"], v["nearest_r"]))
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
