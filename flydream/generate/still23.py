r"""23.1: есть ли в нашем состоянии статическая картинка вообще.

    python -m flydream.generate.still23          # локально, CPU, $0

Человек, 2026-09-21: «можно с этого получить статическую картинку? я бы тогда
вообще сделал это направлением 1, добиться сцены не с видео а просто
картинкой, а движение меня пока не парят».

Возражение, которое надо снять измерением, а не рассуждением: T4 и T5 —
детекторы движения, и нулевой коэффициент DCT это не «как выглядит сцена», а
«сколько в среднем было движения». Поэтому вопрос ставится как потолок: берём
НАСТОЯЩИЕ блоки отложенных клипов, оставляем первые K коэффициентов времени и
обнуляем остальные (латент выбелен, так что ноль — это среднее корпуса), и
смотрим, что 13B из этого рисует.

Если при K = 1 сцена есть — направление живое, и объект падает с 23 072 чисел
до 1 442. Если нет — статическая картинка требует другого представления
(ранние слои, несущие яркость), то есть переобучения 13B.

22.0 уже мерил соседнее и мерил ДРУГОЕ: там вопрос был «насколько точно
восстановится ЭТОТ клип» (T4/2/15 дал r 0,191 при доле ровного 50,2 %).
Здесь вопрос другой — появляется ли сцена вообще, — и судить его надо глазом
и статистикой кадра, а не корреляцией с исходником.
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
from flydream.generate.vaeval18 import nearest


def lag1(v: np.ndarray) -> float:
    """Связность кадр-к-кадру: у по-настоящему статической картинки ~1."""
    x = np.asarray(v, np.float64)
    a, b = x[:-1] - x[:-1].mean(1, keepdims=True), x[1:] - x[1:].mean(1, keepdims=True)
    return float(((a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-24)).mean())


def run(model: str, pca_path: Path, gen_ckpt: Path, manifest: dict, columns: dict, corpus: Path, *,
        types: str = "T4a,T4b", keeps=(1, 2, 4, 16), n_clips: int = 6, frames: int = 40,
        margin: int = 5, dt: float = 0.02, t_pre: float = 1.0, seed: int = 0, log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = load_network(model); dev = device_of(net)
    _, index = P13.type_index(net.connectome)
    d = Deep(manifest, columns)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)
    pmeta = json.loads(str(np.load(pca_path)["meta"]))
    btypes = [x.strip() for x in types.split(",") if x.strip()]
    ch = [DEEP.index(x) for x in btypes]

    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    bank_all = np.asarray(cz["videos"][:, :frames], np.float16)
    clip_idx = rng.choice(np.asarray(cm["split"]["test"]), n_clips, replace=False)
    raw = np.asarray(cz["videos"][clip_idx][:, :frames], np.float32)
    st_real = simulate_states(net, np.asarray(cz["videos"], np.float16)[clip_idx], d.cells_all,
                              dt, t_pre, 8).astype(np.float32)
    maps = L.to_maps(st_real.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
    real = (maps - mean[None, None, :, None]) / std[None, None, :, None]
    block = R.to_model_space(pmeta, real[:, :, ch], dev)               # (n, 16, 2, 721)
    log(f"блоки {tuple(block.shape)}, {time.time() - t0:.0f} с")

    groups = {}
    for k in keeps:
        b = torch.zeros_like(block)
        b[:, :k] = block[:, :k]                                        # ноль = среднее корпуса, латент выбелен
        m2 = R.from_model_space(pmeta, b)
        full = np.zeros((len(m2), *real.shape[1:]), np.float32)
        full[:, :, ch] = m2
        groups[f"коэффициентов времени {k} из 16"] = full

    names = [f"{g}|{i}" for g in groups for i in range(n_clips)]
    cond = torch.as_tensor(np.stack([groups[n.rsplit("|", 1)[0]][int(n.rsplit("|", 1)[1])]
                                     for n in names]), device=dev)
    bits = np.array([1.0 if t in btypes else 0.0 for t in DEEP], np.float32)
    mask = torch.as_tensor(np.tile(bits, (len(names), 1)), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), n_clips):                        # пачка = группа (ISS-0009)
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)
            vids.append(G.sample(gen, cond[i:i + n_clips], mask[i:i + n_clips],
                                 steps=20, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    log(f"{len(videos)} видео отрисовано, {time.time() - t0:.0f} с")

    nb = np.asarray(L.neighbour_index(721))
    out = {"clip_idx": clip_idx.tolist(), "types": btypes, "keeps": list(keeps),
           "numbers": {f"коэффициентов времени {k} из 16": int(k * len(ch) * 721) for k in keeps},
           "groups": {}}
    for g in groups:
        sel = [i for i, n in enumerate(names) if n.startswith(g + "|")]
        s = describe([videos[i] for i in sel], nb)
        out["groups"][g] = {
            "r_to_raw": float(np.mean([pixcorr_per_frame(videos[i], raw[j]).mean()
                                       for j, i in enumerate(sel)])),
            "frac_flat": s["frac_flat"]["mean"], "neigh_r": s["neigh_r"]["mean"],
            "sd": s["sd"]["mean"], "kurtosis": s["grad_kurtosis"]["mean"],
            "lag1": float(np.mean([lag1(videos[i]) for i in sel])),
            "nearest_r": float(np.mean([r for _, r in (nearest(videos[i], bank_all) for i in sel)]))}
        v = out["groups"][g]
        log(f"  {g}: r {v['r_to_raw']:.3f}, ровного {100 * v['frac_flat']:.1f} %, контраст "
            f"{v['sd']:.3f}, соседи {v['neigh_r']:.3f}, кадр-к-кадру {v['lag1']:.3f}")
    rw = describe(list(raw), nb)
    out["raw"] = {"frac_flat": rw["frac_flat"]["mean"], "neigh_r": rw["neigh_r"]["mean"],
                  "sd": rw["sd"]["mean"], "kurtosis": rw["grad_kurtosis"]["mean"],
                  "lag1": float(np.mean([lag1(v) for v in raw]))}
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
    g = settings().get("gen13b", {})
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--pca", default=str(ROOT / "data" / "prior19" / "pca_ab2048.npz"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior23"))
    p.add_argument("--tag", default="still23")
    p.add_argument("--types", default="T4a,T4b")
    p.add_argument("--keeps", default="1,2,4,16")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    r = run(a.model, Path(a.pca), Path(a.gen), manifest, columns, Path(a.corpus), types=a.types,
            keeps=tuple(int(x) for x in a.keeps.split(",")), n_clips=a.clips,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), seed=a.seed)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.tag}.json").write_text(json.dumps(r["summary"], ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    np.savez_compressed(outdir / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    print("\n{:30} {:>8} {:>11} {:>9} {:>9} {:>9} {:>12}".format(
        "оставлено", "чисел", "r к сырому", "ровного", "контраст", "соседи", "кадр-к-кадру"))
    for k, v in S["groups"].items():
        print("{:30} {:8} {:11.3f} {:8.1f}% {:9.3f} {:9.3f} {:12.3f}".format(
            k, S["numbers"][k], v["r_to_raw"], 100 * v["frac_flat"], v["sd"], v["neigh_r"], v["lag1"]))
    w = S["raw"]
    print("{:30} {:>8} {:>11} {:8.1f}% {:9.3f} {:9.3f} {:12.3f}".format(
        "сырое видео корпуса", "—", "—", 100 * w["frac_flat"], w["sd"], w["neigh_r"], w["lag1"]))
    print(f"\n-> {outdir / a.tag}.{{json,npz}} за {S['seconds']:.0f} с, $0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
