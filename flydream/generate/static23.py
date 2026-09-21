r"""23.2: может ли эта цепочка нести ОДНУ статическую картинку. Валидный тест.

    python -m flydream.generate.static23          # локально, CPU, $0

23.1 спрашивал то же и спрашивал неверно: там у настоящего состояния обнулялись
15 коэффициентов DCT из 16, то есть 13B получал состояние, какого не вызывал ни
один клип. Человек, 2026-09-21: «это невалидный тест, ты пытаешься отрендерить
с 1 кадра через 13б которая обучалась на 16». Он прав — тот тест не мог
отделить «в состоянии нет картинки» от «13B не умеет читать такое состояние».

Здесь состояние не калечится. Мозгу подаётся стимул, и снимается то, что он на
самом деле выдаёт:

- **неподвижный** — один кадр настоящего клипа, удержанный всё окно. Ровно то,
  что человек и просит: одна статическая картинка;
- **плывущий** — тот же кадр, равномерно сдвигаемый по решётке на один шаг в
  кадр. Сцена статична по содержанию, но по глазу движется, а детекторы
  движения именно на этом и работают: муха видит мир через параллакс. 22.0b
  намерил, что горизонтальная пара a+b несёт почти всё (0,884 из 0,904), а
  сдвиг здесь как раз горизонтальный;
- **настоящий клип** — контроль сверху.

Все три идут одним и тем же путём: стимул -> замороженный мозг -> состояние ->
13B -> видео. Состояние в каждом случае настоящее, вне распределения ничего
нет, и отвечает тест ровно на вопрос «есть ли в T4/T5 статическая картинка».
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
from flydream.decode.hexraster import neighbour_index
from flydream.generate import gen13b as G
from flydream.generate import learned as L
from flydream.generate import prior17 as R
from flydream.generate.edges18 import describe
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network, pixcorr_per_frame
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep


def lag1(v: np.ndarray) -> float:
    x = np.asarray(v, np.float64)
    a, b = x[:-1] - x[:-1].mean(1, keepdims=True), x[1:] - x[1:].mean(1, keepdims=True)
    return float(((a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-24)).mean())


def drift(frame: np.ndarray, n_frames: int, direction: int = 0) -> np.ndarray:
    """Кадр, сдвигаемый по решётке на один шаг в кадр. За краем поля берётся
    среднее кадра: это край зрения, а не дыра в данных."""
    nb = neighbour_index(721)[:, direction]
    out = np.empty((n_frames, 721), np.float32)
    cur = np.asarray(frame, np.float32).copy()
    for t in range(n_frames):
        out[t] = cur
        nxt = np.full(721, float(cur.mean()), np.float32)
        ok = nb >= 0
        nxt[ok] = cur[nb[ok]]
        cur = nxt
    return out


def run(model: str, gen_ckpt: Path, manifest: dict, columns: dict, corpus: Path, *,
        types: str = "T4a,T4b", n_clips: int = 6, frames: int = 40, margin: int = 5,
        dt: float = 0.02, t_pre: float = 1.0, seed: int = 0, log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = load_network(model); dev = device_of(net)
    _, index = P13.type_index(net.connectome)
    d = Deep(manifest, columns)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)
    btypes = [x.strip() for x in types.split(",") if x.strip()]

    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    clip_idx = rng.choice(np.asarray(cm["split"]["test"]), n_clips, replace=False)
    full = np.asarray(cz["videos"][clip_idx], np.float32)              # (n, 45, 721)
    n_t = full.shape[1]
    stim = {"настоящий клип": full,
            "неподвижный кадр": np.repeat(full[:, :1], n_t, axis=1),
            "тот же кадр, плывущий": np.stack([drift(full[i, 0], n_t) for i in range(n_clips)])}
    log(f"стимулы построены: {n_clips} клипов по {n_t} кадров, {time.time() - t0:.0f} с")

    groups, target = {}, {}
    for name, vid in stim.items():
        st = simulate_states(net, np.asarray(vid, np.float16), d.cells_all, dt, t_pre, 8)
        m = L.to_maps(st.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
        groups[name] = (m - mean[None, None, :, None]) / std[None, None, :, None]
        target[name] = np.asarray(vid[:, :frames], np.float32)
        # Сколько энергии вообще выдали детекторы движения на этот стимул.
        e = {t: float(np.abs(groups[name][:, :, DEEP.index(t)]).mean()) for t in btypes}
        log(f"  {name}: |состояние| по типам {({k: round(v, 3) for k, v in e.items()})}, "
            f"{time.time() - t0:.0f} с")
        groups[name] = (groups[name], e)

    names = [f"{g}|{i}" for g in groups for i in range(n_clips)]
    cond = torch.as_tensor(np.stack([groups[n.rsplit("|", 1)[0]][0][int(n.rsplit("|", 1)[1])]
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
    out = {"clip_idx": clip_idx.tolist(), "types": btypes, "groups": {}, "stimulus": {}}
    for g in groups:
        sel = [i for i, n in enumerate(names) if n.startswith(g + "|")]
        s = describe([videos[i] for i in sel], nb)
        tg = target[g]
        out["groups"][g] = {
            "r_to_stimulus": float(np.mean([pixcorr_per_frame(videos[i], tg[j]).mean()
                                            for j, i in enumerate(sel)])),
            "frac_flat": s["frac_flat"]["mean"], "neigh_r": s["neigh_r"]["mean"],
            "sd": s["sd"]["mean"], "lag1": float(np.mean([lag1(videos[i]) for i in sel])),
            "state_energy": groups[g][1]}
        sw = describe(list(tg), nb)
        out["stimulus"][g] = {"frac_flat": sw["frac_flat"]["mean"], "sd": sw["sd"]["mean"],
                              "neigh_r": sw["neigh_r"]["mean"],
                              "lag1": float(np.mean([lag1(v) for v in tg]))}
        v, w = out["groups"][g], out["stimulus"][g]
        log(f"  {g}: r к своему стимулу {v['r_to_stimulus']:.3f}, контраст {v['sd']:.3f} "
            f"(у стимула {w['sd']:.3f}), кадр-к-кадру {v['lag1']:.3f} (у стимула {w['lag1']:.3f})")
    out["seconds"] = round(time.time() - t0, 1)
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    for g in stim:
        for i in range(n_clips):
            arrays[f"stim__{g}|{i}"] = target[g][i]
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
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior23"))
    p.add_argument("--tag", default="static23")
    p.add_argument("--types", default="T4a,T4b")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    r = run(a.model, Path(a.gen), manifest, columns, Path(a.corpus), types=a.types,
            n_clips=a.clips, frames=g.get("frames", 40), margin=g.get("margin", 5),
            dt=g.get("dt", 0.02), t_pre=g.get("t_pre", 1.0), seed=a.seed)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.tag}.json").write_text(json.dumps(r["summary"], ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    np.savez_compressed(outdir / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    print("\n{:26} {:>14} {:>10} {:>10} {:>13} {:>13}".format(
        "стимул", "r к стимулу", "контраст", "у стимула", "кадр-к-кадру", "у стимула"))
    for k, v in S["groups"].items():
        w = S["stimulus"][k]
        print("{:26} {:14.3f} {:10.3f} {:10.3f} {:13.3f} {:13.3f}".format(
            k, v["r_to_stimulus"], v["sd"], w["sd"], v["lag1"], w["lag1"]))
    print("\nэнергия состояния, |значение| по типам блока:")
    for k, v in S["groups"].items():
        print("  {:26} {}".format(k, {kk: round(vv, 3) for kk, vv in v["state_energy"].items()}))
    print(f"\n-> {outdir / a.tag}.{{json,npz}} за {S['seconds']:.0f} с, $0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
