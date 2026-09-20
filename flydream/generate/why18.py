"""Почему сцены нет в состоянии: три бесплатные проверки.

    python -m flydream.generate.why18            # локально, CPU, без Modal, $0

Вариант A показал, что 13B из настоящего состояния рисует сцену (r = +0,979),
а из сгенерированного — гладкое пятно. Наиболее вероятная механика —
усреднение мод: безусловная модель, которой не хватает ёмкости или разрешения
траектории, выдаёт не сэмпл из моды, а нечто ближе к среднему. Три проверки
отделяют «модель мала» от «сэмплер груб»:

1. **энергия** — сколько контраста в сгенерированном видео против настоящего;
2. **пространственная гладкость** — корреляция колонки с её одним кольцом, и
   насколько сэмпл похож на среднее видео корпуса;
3. **шаги сэмплера** — одна и та же начальная точка, проинтегрированная за
   20, 100 и 250 шагов Эйлера. Если резче — дело в интегрировании, а не в
   модели.

13B везде рисует с одним и тем же z и своими обычными 20 шагами: меняется
только интегрирование прайора.
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
from flydream.generate.check18 import structure
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network
from flydream.generate.prompts14 import Deep


def stats(vids: np.ndarray, ring: np.ndarray, mean_video: np.ndarray) -> dict:
    """(N, T, 721) видео -> энергия, структура, близость к среднему корпуса."""
    x = np.asarray(vids, np.float32)
    s = structure(x[:, :, None, :], ring)                            # та же функция, что в 18.2
    m = mean_video.reshape(-1) - mean_video.mean()

    def r_to_mean(v):
        a = v.reshape(-1) - v.mean()
        return float(a @ m / (np.linalg.norm(a) * np.linalg.norm(m) + 1e-12))

    return {"n": int(len(x)), "sd": float(x.std()), "sd_per_clip": [float(v.std()) for v in x],
            "ring1_spatial": s["ring1_spatial"], "temporal_lag1": s["temporal_lag1"],
            "r_to_corpus_mean": [r_to_mean(v) for v in x]}


def run(model: str, prior_ckpt, gen_ckpt, manifest: dict, columns: dict, corpus: Path, *,
        steps_list=(20, 100, 250), n: int = 6, n_ref: int = 32, n_mean: int = 512,
        frames: int = 40, seed: int = 0, log=print) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    net = load_network(model); dev = device_of(net)
    _, index = P.type_index(net.connectome)
    d = Deep(manifest, columns)
    prior, pmeta = R.load(prior_ckpt, dev)
    gen, gmeta = G.load(gen_ckpt, dev)
    ring = L.ring_index(1)
    log(f"prior ({pmeta['parameters']} par) and 13B loaded, {time.time() - t0:.0f} s")

    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    bank = cz["videos"]
    split = {k: np.asarray(v) for k, v in cm["split"].items()}
    rng = np.random.default_rng(seed)
    mean_video = np.asarray(bank[split["train"][:n_mean]], np.float32)[:, :frames].mean(0)
    ref_idx = rng.choice(split["test"], n_ref, replace=False)
    ref = np.asarray(bank[np.sort(ref_idx)], np.float32)[:, :frames]
    log(f"corpus: mean over {n_mean} clips, {n_ref} held-out clips as the reference, {time.time() - t0:.0f} s")

    # --- одна и та же начальная точка, разное число шагов ---
    g = torch.Generator(device=dev).manual_seed(4000 + seed)
    eps = torch.randn(n, pmeta["frames"], pmeta.get("k", 8), 721, device=dev, generator=g)
    groups, arrays = {}, {}
    with torch.no_grad():
        for st in steps_list:
            x = R.integrate(prior, eps.clone(), steps=int(st))
            states = R.from_model_space(pmeta, x).astype(np.float32)
            cnd = torch.as_tensor(states, device=dev)
            mask = torch.ones(len(states), len(DEEP), device=dev)
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)   # тот же z для каждой группы
            vids = G.sample(gen, cnd, mask, steps=20, generator=gg).cpu().numpy().astype(np.float32)
            groups[f"prior_{st}"] = stats(vids, ring, mean_video)
            arrays[f"video__prior_{st}"] = vids
            log(f"  {st:3d} шагов прайора: sd {groups[f'prior_{st}']['sd']:.4f}, "
                f"ring1 {groups[f'prior_{st}']['ring1_spatial']:.3f}, {time.time() - t0:.0f} s")
    groups["clip"] = stats(ref, ring, mean_video)
    arrays["video__clip"] = ref[:n]
    summary = {"model": model, "prior_ckpt": str(prior_ckpt), "prior_parameters": pmeta["parameters"],
               "prior_width": pmeta.get("width"), "dct_k": pmeta.get("dct_k"), "seed": seed,
               "frames": frames, "steps_list": list(steps_list), "n": n, "n_ref": n_ref, "n_mean": n_mean,
               "groups": groups, "seconds": round(time.time() - t0, 1)}
    return {"summary": summary, "arrays": arrays}


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
    p.add_argument("--tag", default="why18_local")
    p.add_argument("--steps", default="20,100,250")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.prior), Path(a.gen), manifest, columns, Path(a.corpus),
            steps_list=[int(x) for x in a.steps.split(",") if x.strip()], n=a.n,
            frames=g.get("frames", 40), seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    print("\nгруппа            sd      ring1   lag1    r к среднему корпуса")
    for k, v in r["summary"]["groups"].items():
        print(f"  {k:14s} {v['sd']:.4f}  {v['ring1_spatial']:.3f}  {v['temporal_lag1']:.3f}  "
              f"{np.mean(v['r_to_corpus_mean']):+.3f}")
    print(f"wrote {out / a.tag}.json/.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
