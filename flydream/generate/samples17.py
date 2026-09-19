"""17.2: states sampled from the 17.1 prior, rendered by 13B, checked by the brain.

    python -m flydream.generate.samples17            # local CPU, no Modal, $0

`prior → T4/T5 state → 13B → video → frozen brain → state′`. Scores per sample
(ROADMAP 17.2):

- **round trip**, the main gate: the state the brain reads back against the
  state that was asked for, per type over clip A's variances — the metric of
  13A/13B/14, in the same code path as the controls below;
- **video novelty**: the nearest of the 6,725 training videos (correlation and
  a normalised squared distance), the metric 17.0 validated on the generator
  itself;
- **diversity**: pairwise correlation between samples, of the states and of
  the videos, beside the same for real clips;
- **the direction the brain reads** in the video against the direction the
  sampled state carries.

Controls rebuilt here, so the whole table shares one scale: a held-out clip's
own state (the reachable reference), white noise and structured noise in the
types (item 14.0's constructions), and a clip state whose columns are
permuted. **State novelty** — the nearest *training state* — is not here: the
9,468 training states are the 5.5 GB `maps_deep.npz` on the Modal volume, and
that gate is measured where the file lives (`sample17` in the Modal app).
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
from flydream.generate.baseline17 import flat, nearest, pairwise_corr, triu_mean, video_bank
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep, direction_energy
from flydream.generate.roundtrip13 import build_states, round_trip


def structured_noise(rng, frames: int, k: int, ring: np.ndarray, rounds: int = 5) -> np.ndarray:
    """White noise smoothed in time (~100 ms) and over one hex ring, unit sd per type."""
    x = rng.standard_normal((frames, k, 721)).astype(np.float32)
    for _ in range(rounds):
        x = 0.5 * x + 0.5 * np.concatenate([x[:1], x[:-1]], 0)
        x = np.nanmean(np.where(ring[None, None] >= 0, x[:, :, np.clip(ring, 0, None)], np.nan), -1)
    return ((x - x.mean((0, 2), keepdims=True)) / (x.std((0, 2), keepdims=True) + 1e-6)).astype(np.float32)


def run(model: str, prior_ckpt, gen_ckpt, manifest: dict, columns: dict, procedural_file: Path, *,
        corpus: Path | None = None, n_samples: int = 16, n_clips: int = 3, frames: int = 40, margin: int = 5, dt: float = 0.02, t_pre: float = 1.0,
        sample_steps: int = 20, seed: int = 0, log=print) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    net = load_network(model); dev = device_of(net)
    _, index = P.type_index(net.connectome)
    d = Deep(manifest, columns)
    prior, pmeta = R.load(prior_ckpt, dev)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)
    T = frames + margin
    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]
    ta = sa["target"][:, w0:w1, :][:, :, d.cells_all].cpu().numpy().astype(np.float32)
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}      # 13B's and item 14's normalisation
    log(f"model, prior ({pmeta['parameters']} par, {pmeta['steps']} steps) and 13B loaded, {time.time() - t0:.0f} s")

    if corpus is not None:                                                   # item 18: the prior's own training set
        cz = np.load(Path(corpus) / "videos.npz")
        cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
        bank = np.asarray(cz["videos"], np.float16)
        meta_c = [json.loads(str(x)) for x in cz["meta"]]
        labels = [f"{m.get('label') or m.get('class')}" for m in meta_c]
        split = {k: np.asarray(v) for k, v in cm["split"].items()}
        manifest = {**manifest, "meta": meta_c, "n_sintel": 0}
        log(f"bank from the corpus: {bank.shape}, train {len(split['train'])}, test {len(split['test'])}")
    else:
        bank, labels = video_bank(manifest, procedural_file, T, dt, log=log)
        split = {k: np.asarray(v) for k, v in manifest["split"].items()}
    train_idx, test_idx = split["train"], split["test"]
    bank_train = flat(bank[train_idx], frames)

    def to_z(states: np.ndarray) -> np.ndarray:                              # (N,T,cells) -> z-scored maps
        m = L.to_maps(states.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
        return (m - mean[None, None, :, None]) / std[None, None, :, None]

    # --- the states to render ---
    g = torch.Generator(device=dev).manual_seed(2000 + seed)
    prior_states = R.sample_states(prior, pmeta, n_samples, steps=sample_steps, device=dev, generator=g)
    log(f"{n_samples} states from the prior ({pmeta.get('sources', 'all')} data, "
        f"DCT {pmeta.get('dct_k') or 'off'}) in {time.time() - t0:.0f} s")
    n_s = int(manifest["n_sintel"])
    if n_s:                                                                  # 13A's set: take clips from both sources
        half = max(1, n_clips // 2)
        clip_idx = np.concatenate([rng.choice(test_idx[test_idx < n_s], half, replace=False),
                                   rng.choice(test_idx[test_idx >= n_s], n_clips - half, replace=False)])
    else:                                                                    # the corpus: held-out classes only
        clip_idx = rng.choice(test_idx, n_clips, replace=False)
    clip_states = simulate_states(net, bank[clip_idx], d.cells_all, dt, t_pre, 8).astype(np.float32)
    clip_maps = to_z(clip_states)
    jobs = {f"prior_{k}": prior_states[k] for k in range(n_samples)}
    jobs.update({f"clip_{int(i)}": clip_maps[j] for j, i in enumerate(clip_idx)})
    jobs["noise_white"] = rng.standard_normal((frames, len(DEEP), 721)).astype(np.float32)
    jobs["noise_structured"] = structured_noise(rng, frames, len(DEEP), L.ring_index(1))
    jobs["shuffled_clip"] = clip_maps[0][:, :, rng.permutation(721)]
    if pmeta.get("dct_k"):                                                   # the ceiling of the representation itself
        dmat = R.dct_matrix(frames, int(pmeta["dct_k"]))
        for j, i in enumerate(clip_idx):
            jobs[f"dct_ceiling_{int(i)}"] = R.from_dct(R.to_dct(clip_maps[j][None], dmat), dmat)[0].astype(np.float32)
    names = list(jobs)
    kinds = {n: ("prior" if n.startswith("prior") else "clip" if n.startswith("clip")
                 else "ceiling" if n.startswith("dct_ceiling") else "control") for n in names}
    meta = manifest["meta"]
    clip_label = {f"clip_{int(i)}": (f"сцена Sintel: {meta[int(i)]['scene']}" if meta[int(i)]["source"] == "sintel"
                                     else f"обычное видео: {meta[int(i)].get('label')}" if meta[int(i)]["source"] == "video"
                                     else f"процедурный стимул: {meta[int(i)]['class']}") for i in clip_idx}

    # --- 13B renders them, one z shared across the states of a chunk ---
    cond = torch.as_tensor(np.stack([jobs[n] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), 8):
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)
            vids.append(G.sample(gen, cond[i:i + 8], mask[i:i + 8], steps=sample_steps, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    log(f"{len(videos)} videos rendered in {time.time() - t0:.0f} s")

    # --- the frozen brain ---
    target_raw = np.stack([R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0] for n in names])
    rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
    vids_m = np.concatenate([videos, np.repeat(videos[:, -1:], margin, 1)], 1).astype(np.float16)
    st_back = simulate_states(net, vids_m, d.cells_all, dt, t_pre, 8).astype(np.float32)
    st_grey = simulate_states(net, np.full((1, T, 721), 0.5, np.float16), d.cells_all, dt, t_pre, 1).astype(np.float32)[0]
    log(f"round trips and states back in {time.time() - t0:.0f} s")

    # --- novelty (video) and diversity ---
    q_vid = flat(videos, frames)
    i_v, r_v, d_v = nearest(q_vid, bank_train)
    q_state = np.asarray(np.stack([jobs[n] for n in names]), np.float32).reshape(len(names), -1)
    q_state -= q_state.mean(1, keepdims=True)
    sel = {k: [i for i, n in enumerate(names) if kinds[n] == k] for k in ("prior", "clip", "ceiling")}
    diversity = {}
    for k, v in sel.items():
        if len(v) > 1:
            diversity[f"videos_{k}"] = triu_mean(pairwise_corr(q_vid[v]))
            diversity[f"states_{k}"] = triu_mean(pairwise_corr(q_state[v]))

    scores = {}
    for i, n in enumerate(names):
        scores[n] = {
            "kind": kinds[n], "round_trip": float(rts[i]),
            "video_nn_r": float(r_v[i]), "video_nn_distance": float(d_v[i]),
            "video_nn_label": labels[int(train_idx[i_v[i]])], "video_nn_index": int(train_idx[i_v[i]]),
            "direction_state": direction_energy(target_raw[i], st_grey, d)["T4_argmax"],
            "direction_video": direction_energy(st_back[i], st_grey, d)["T4_argmax"],
            "video_mean": float(videos[i].mean()), "video_sd": float(videos[i].std()),
            "frame_to_frame": float(np.abs(np.diff(videos[i], axis=0)).mean()),
            "state_sd_per_type": {t: float(jobs[n][:, k].std()) for k, t in enumerate(DEEP)},
        }
        if n in clip_label:
            scores[n]["label"] = clip_label[n]
        log(f"  {n:<16} {kinds[n]:<7} rt {scores[n]['round_trip']:7.3f}  video nn r {scores[n]['video_nn_r']:+.2f}"
            f"  dir {scores[n]['direction_state']}->{scores[n]['direction_video']}  ({scores[n]['video_nn_label']})")

    def group(k, q):
        v = [scores[n][q] for n in names if kinds[n] == k]
        return {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}

    summary = {"model": model, "prior_ckpt": str(prior_ckpt), "gen_ckpt": str(gen_ckpt), "frames": frames,
               "sample_steps": sample_steps, "seed": seed, "n_samples": n_samples,
               "prior": {k: pmeta[k] for k in ("width", "depth", "steps", "parameters")},
               "bank": {"train": int(len(train_idx)), "test": int(len(test_idx))},
               "prior_meta": {k: pmeta.get(k) for k in ("sources", "dct_k", "n_train", "steps", "width", "depth")},
               "gates": {k: {q: group(k, q) for q in ("round_trip", "video_nn_r", "video_nn_distance")}
                         for k in ("prior", "clip", "ceiling") if any(kinds[n] == k for n in names)},
               "diversity": diversity, "scores": scores, "seconds": round(time.time() - t0, 1)}
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    arrays.update({f"T4a__{n}": target_raw[i][:, d.pos["T4a"]] for i, n in enumerate(names)})
    arrays.update({f"nnvideo__{n}": bank[train_idx[i_v[i]]][:frames].astype(np.float32) for i, n in enumerate(names)})
    arrays["prior_states_z"] = np.stack([jobs[f"prior_{k}"] for k in range(n_samples)]).astype(np.float16)
    return {"summary": summary, "arrays": arrays}


def main(argv=None) -> int:
    from flydream.generate.invert import settings
    from flydream.model import ROOT

    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--prior", default=str(ROOT / "data" / "prior17" / "state_flow.pt"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior17"))
    p.add_argument("--tag", default="samples17_local")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--clips", type=int, default=3)
    p.add_argument("--corpus", default="")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    proc = sorted(pdir.glob("procedural_*.npz"))[0]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.prior), Path(a.gen), manifest, columns, proc, corpus=Path(a.corpus) if a.corpus else None, n_samples=a.samples, n_clips=a.clips,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02), t_pre=g.get("t_pre", 1.0),
            seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    gt = r["summary"]["gates"]
    print(f"\nround trip: prior {gt['prior']['round_trip']['median']:.3f} "
          f"({gt['prior']['round_trip']['min']:.3f}-{gt['prior']['round_trip']['max']:.3f}), "
          f"real clip {gt['clip']['round_trip']['median']:.3f}")
    for n in ("noise_white", "noise_structured", "shuffled_clip"):
        print(f"  control {n:<18} {r['summary']['scores'][n]['round_trip']:.3f}")
    if "ceiling" in gt:
        print(f"  representation ceiling (a real state, same DCT) {gt['ceiling']['round_trip']['median']:.3f}")
    print(f"video novelty: prior nearest training video r {gt['prior']['video_nn_r']['median']:+.2f} "
          f"(max {gt['prior']['video_nn_r']['max']:+.2f}), real clip {gt['clip']['video_nn_r']['median']:+.2f}")
    print(f"diversity: {json.dumps(r['summary']['diversity'])}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
