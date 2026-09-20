"""18.17: умеет ли прайор сцену вообще — если шум задать заведомо.

    python -m flydream.generate.reach18          # локально, CPU, $0

Человек, 2026-09-20: «меня больше волнует, что мы вообще пока не получили ни
одного состояния, которое было бы сценой. Можем ли заведомо задать шум,
который даст нам сцену из видео? Так мы хотя бы поймём, что сам прайор это
умеет».

Прайор — детерминированное обратимое отображение шума в состояние, поэтому
вопрос проверяется напрямую. Берём настоящий отложенный клип, прогоняем через
замороженный мозг в состояние, **обращаем** поток прайора (`to_noise`) и
получаем тот единственный шум, из которого этот прайор выдал бы именно это
состояние. Дальше три вещи:

1. **Достижимость.** Пройти шум вперёд и сравнить с исходным состоянием.
   Если сошлось — сцена лежит в области значений прайора, и вопрос «умеет
   ли» закрыт положительно.
2. **Где этот шум лежит.** Гауссов шум в D измерениях концентрируется на
   радиусе √D. Если радиус сценного шума далеко от √D, прайор сцену
   **представляет**, но случайным розыгрышем практически никогда не выдаст.
   Это и есть количественная форма «умеет, но не делает».
3. **Выживает ли сцена на типичном радиусе.** Тот же шум, нормированный на
   √D, и снова вперёд. Если сцена выживает — дело только в направлении, а не
   в длине, и значит сцены есть и внутри типичного множества.

Всё считается на уже обученном плече и уже собранном корпусе: ни одного
платного работника.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flydream.generate import gen13b as G
from flydream.generate import learned as L
from flydream.generate import prior17 as R
from flydream.generate.edges18 import describe
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep
from flydream.generate.roundtrip13 import build_states, round_trip


def radius(x: np.ndarray) -> float:
    """Евклидова длина одного образца, развёрнутого в вектор."""
    return float(np.linalg.norm(np.asarray(x, np.float64).reshape(-1)))


def run(model: str, prior_ckpt, gen_ckpt, manifest: dict, columns: dict, corpus: Path, *,
        n_clips: int = 6, frames: int = 40, margin: int = 5, dt: float = 0.02, t_pre: float = 1.0,
        steps: int = 20, fixed_point: int = 3, seed: int = 0, log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = load_network(model); dev = device_of(net)
    from flydream.decode import pairs as P
    _, index = P.type_index(net.connectome)
    d = Deep(manifest, columns)
    prior, pmeta = R.load(prior_ckpt, dev)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)
    log(f"prior {pmeta['parameters']} par (K={pmeta.get('dct_k')}, w={pmeta['width']}), 13B loaded, "
        f"{time.time() - t0:.0f} s")

    # --- нормализация ворот: дисперсии по типам из клипа A, как везде в 13/14/18 ---
    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]
    ta = sa["target"][:, w0:w1, :][:, :, d.cells_all].cpu().numpy().astype(np.float32)
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}

    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    bank = np.asarray(cz["videos"], np.float16)
    test_idx = np.asarray(cm["split"]["test"])
    clip_idx = rng.choice(test_idx, n_clips, replace=False)
    log(f"corpus {bank.shape}, {len(test_idx)} held out; taking {clip_idx.tolist()}")

    clip_states = simulate_states(net, bank[clip_idx], d.cells_all, dt, t_pre, 8).astype(np.float32)
    maps = L.to_maps(clip_states.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
    real = (maps - mean[None, None, :, None]) / std[None, None, :, None]        # 13B's units
    log(f"{len(real)} real states {real.shape} in {time.time() - t0:.0f} s")

    # --- 1. обращение: тот самый шум ---
    eps = R.to_noise(prior, pmeta, real, steps=steps, fixed_point=fixed_point, device=dev)
    back = R.from_noise(prior, pmeta, eps, steps=steps, device=dev)
    # --- 3. тот же шум на типичном радиусе ---
    D = int(np.prod(eps.shape[1:]))
    tgt = float(np.sqrt(D))
    scaled = (eps.reshape(len(eps), -1) * (tgt / np.linalg.norm(eps.reshape(len(eps), -1), axis=1, keepdims=True))
              ).reshape(eps.shape).astype(np.float32)
    from_scaled = R.from_noise(prior, pmeta, scaled, steps=steps, device=dev)
    # --- контроль: обычный розыгрыш ---
    g = torch.Generator(device=dev).manual_seed(2000 + seed)
    y = None
    n_cls = int(pmeta.get("n_classes") or 0)
    if n_cls:
        pool = pmeta.get("trained_classes") or list(range(n_cls))
        pt = torch.as_tensor(pool, device=dev)
        y = pt[torch.randint(0, len(pool), (n_clips,), device=dev, generator=g)]
    drawn = R.sample_states(prior, pmeta, n_clips, steps=steps, device=dev, generator=g, y=y)
    log(f"inverted, re-integrated and drawn in {time.time() - t0:.0f} s")

    # --- рендер всего одним проходом 13B и прогонка через мозг ---
    groups = {"настоящее состояние": real, "из его шума": back,
              "тот же шум на радиусе √D": from_scaled, "обычный розыгрыш": drawn}
    jobs = {f"{k}|{i}": v[i] for k, v in groups.items() for i in range(len(v))}
    names = list(jobs)
    cond = torch.as_tensor(np.stack([jobs[n] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), 8):
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)      # один и тот же z у 13B везде
            vids.append(G.sample(gen, cond[i:i + 8], mask[i:i + 8], steps=steps, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    target_raw = np.stack([R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0]
                           for n in names])
    rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
    log(f"{len(videos)} videos rendered and scored in {time.time() - t0:.0f} s")

    out = {"prior_ckpt": str(prior_ckpt), "frames": frames, "steps": steps, "fixed_point": fixed_point,
           "seed": seed, "n_clips": int(n_clips), "clip_idx": clip_idx.tolist(),
           "dims": D, "typical_radius": tgt, "groups": {}}
    nb = np.asarray(L.neighbour_index(721))
    for k, v in groups.items():
        sel = [i for i, n in enumerate(names) if n.startswith(k + "|")]
        out["groups"][k] = {"structure": describe([videos[i] for i in sel], nb),
                            "round_trip": {"median": float(np.median(rts[sel])),
                                           "min": float(np.min(rts[sel])), "max": float(np.max(rts[sel]))}}
    # --- числа про сам шум ---
    flat_eps = eps.reshape(len(eps), -1)
    out["noise"] = {
        "radius_per_clip": [radius(e) for e in eps],
        "radius_mean": float(np.mean([radius(e) for e in eps])),
        "radius_typical": tgt,
        "radius_ratio": float(np.mean([radius(e) for e in eps]) / tgt),
        "sd_per_dim": float(flat_eps.std()),
        "recon_rel_error": float(np.linalg.norm((back - real).reshape(len(real), -1), axis=1).mean()
                                 / np.linalg.norm(real.reshape(len(real), -1), axis=1).mean()),
        "scaled_rel_error": float(np.linalg.norm((from_scaled - real).reshape(len(real), -1), axis=1).mean()
                                  / np.linalg.norm(real.reshape(len(real), -1), axis=1).mean()),
    }
    out["seconds"] = round(time.time() - t0, 1)
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    arrays["noise_radius"] = np.array([radius(e) for e in eps], np.float32)
    return {"summary": out, "arrays": arrays, "names": names}


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
    p.add_argument("--tag", default="reach18_local")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--fixed-point", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.prior), Path(a.gen), manifest, columns, Path(a.corpus), n_clips=a.clips,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), steps=a.steps, fixed_point=a.fixed_point, seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    n = S["noise"]
    print(f"\nразмерность {S['dims']}, типичный радиус √D = {n['radius_typical']:.1f}")
    print(f"радиус шума сцены        {n['radius_mean']:.1f}  ({n['radius_ratio']:.3f} от типичного)")
    print(f"ст. отклонение по осям   {n['sd_per_dim']:.3f}  (у гауссова шума 1,000)")
    print(f"восстановление состояния {100 * n['recon_rel_error']:.2f} % относительной ошибки")
    print(f"после нормировки на √D   {100 * n['scaled_rel_error']:.2f} %")
    print(f"\n{'группа':28s} {'ворота':>8} {'ровного':>9} {'эксцесс':>8} {'сосед r':>8} {'контраст':>9}")
    for k, v in S["groups"].items():
        st = v["structure"]
        print(f"{k:28s} {v['round_trip']['median']:8.4f} {100 * st['frac_flat']['mean']:8.1f}% "
              f"{st['grad_kurtosis']['mean']:8.2f} {st['neigh_r']['mean']:8.3f} {st['sd']['mean']:9.3f}")
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
