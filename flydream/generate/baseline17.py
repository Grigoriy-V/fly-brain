"""17.0: what the unconditional 13B generator already does — video from z alone.

    python -m flydream.generate.baseline17            # local CPU, no Modal, $0

`z -> 13B with the mask "none" -> video -> frozen brain -> state`. 13B saw the
unconditional mask on 5 % of its training steps, so it can draw without being
given a state; this measures how far that alone gets us **before** a state
prior is trained (ROADMAP 17.0).

No round trip is scored here: no state was given, so the state read back from
the generated video would be compared with itself. The questions are novelty
and diversity, and both are read against the same numbers for real videos:

- **novelty**: for every generated video, the nearest video of 13B's own
  training split (correlation over the 40 x 721 hexals and a normalised
  squared distance), beside the nearest training video of a *held-out real*
  clip — the scale for "not a copy";
- **diversity**: pairwise correlation among the generated videos and among
  their brain states, beside the same among real held-out clips;
- **what the brain does with them**: per-type activity above the grey
  baseline and the direction T4/T5 read (`prompts14.direction_energy`).

The nearest-neighbour metric is checked on the generator itself: 13B
conditioned on a training clip's own state must come back nearest to that
clip. Without that check a novelty number proves nothing.

The video bank is rebuilt locally in the order of the 13A/13B manifest
(`data/pairs13/manifest.json`): indices 0..n_sintel-1 are the augmented Sintel
clips of `pairs13.sintel_videos`, the rest the procedural clips saved in
`data/pairs13/procedural_*.npz`; `manifest["split"]` then says which of them
13B trained on.
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
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep, direction_energy


# ----------------------------------------------------------------- metrics


def flat(v: np.ndarray, frames: int) -> np.ndarray:
    """(N, T, 721) -> (N, frames*721) float32, centred per sample."""
    x = np.asarray(v[:, :frames], np.float32).reshape(len(v), -1)
    return x - x.mean(1, keepdims=True)


def pairwise_corr(x: np.ndarray) -> np.ndarray:
    """Correlation matrix of centred rows (N, D); diagonal is 1."""
    n = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return n @ n.T


def nearest(query: np.ndarray, bank: np.ndarray, *, skip: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each centred query row, the bank row with the highest correlation.

    Returns (index, r, d) where `d` is the squared difference of the two
    centred videos over the query's own variance: 0 is an exact copy, 1 is as
    far as the query's own mean image. `skip[i]` is a bank index excluded for
    query i (leave-one-out for a video that is itself in the bank)."""
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    bn = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12)
    r = qn @ bn.T
    if skip is not None:
        r[np.arange(len(query)), skip] = -np.inf
    idx = r.argmax(1)
    best = r[np.arange(len(query)), idx]
    d = ((query - bank[idx]) ** 2).sum(1) / ((query ** 2).sum(1) + 1e-12)
    return idx, best, d


def triu_mean(c: np.ndarray) -> dict:
    """Mean, median and max of a correlation matrix's off-diagonal upper part."""
    iu = np.triu_indices(len(c), 1)
    v = c[iu]
    return {"mean": float(v.mean()), "median": float(np.median(v)), "max": float(v.max())}


# ----------------------------------------------------------------- the video bank


def video_bank(manifest: dict, procedural_file: Path, frames: int, dt: float, log=print) -> tuple[np.ndarray, list[str]]:
    """(n, frames, 721) float16 videos in the manifest's index order, and a label per video."""
    from flydream.generate.pairs13 import sintel_videos

    n_s = int(manifest["n_sintel"])
    t0 = time.time()
    vids, scenes, names = sintel_videos(frames, dt)
    if len(vids) != n_s:
        raise SystemExit(f"rebuilt {len(vids)} Sintel clips, the manifest has {n_s}: the rendering settings differ")
    log(f"sintel rebuilt: {len(vids)} clips in {time.time() - t0:.0f} s")
    z = np.load(procedural_file)
    proc = z["videos"][:, :frames]
    params = [json.loads(str(p)) for p in z["params"]]
    bank = np.concatenate([np.asarray(vids, np.float16)[:, :frames], proc.astype(np.float16)], 0)
    labels = [f"sintel {s}" for s in scenes] + [f"proc {p['class']}" for p in params]
    if len(bank) != int(manifest["n"]):
        raise SystemExit(f"rebuilt {len(bank)} videos, the manifest has {manifest['n']}")
    for i in (0, n_s - 1, n_s, len(bank) - 1):                       # the order must match the manifest
        m = manifest["meta"][i]
        want = f"sintel {m['scene']}" if m["source"] == "sintel" else f"proc {m['class']}"
        if labels[i] != want:
            raise SystemExit(f"video {i} is {labels[i]}, the manifest says {want}")
    log(f"bank {bank.shape} ({bank.nbytes / 1e6:.0f} MB), {n_s} sintel + {len(proc)} procedural")
    return bank, labels


# ----------------------------------------------------------------- the run


def run(model: str, ckpt, manifest: dict, columns: dict, procedural_file: Path, *, n_samples: int = 16,
        n_real: int = 16, n_check: int = 4, frames: int = 40, margin: int = 5, dt: float = 0.02, t_pre: float = 1.0,
        sample_steps: int = 20, seed: int = 0, log=print) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    net = load_network(model); dev = device_of(net)
    _, index = P.type_index(net.connectome)
    d = Deep(manifest, columns)
    gen, meta = G.load(ckpt, dev)
    mean = torch.as_tensor(np.array(meta["mean"], np.float32), device=dev)[None, None, :, None]
    std = torch.as_tensor(np.array(meta["std"], np.float32), device=dev)[None, None, :, None]
    T = frames + margin
    log(f"13B loaded ({meta['kind']} {meta['width']}x{meta['depth']}, {meta['steps']} steps), {time.time() - t0:.0f} s")

    bank, labels = video_bank(manifest, procedural_file, T, dt, log=log)
    split = {k: np.asarray(v) for k, v in manifest["split"].items()}
    train_idx, test_idx = split["train"], split["test"]
    bank_train = flat(bank[train_idx], frames)
    log(f"train bank {bank_train.shape} ({bank_train.nbytes / 1e6:.0f} MB), test {len(test_idx)}")

    # --- 1. unconditional samples: z only, the mask of no type at all ---
    cond = torch.zeros(n_samples, frames, len(DEEP), 721, device=dev)
    none = torch.zeros(n_samples, len(DEEP), device=dev)
    g = torch.Generator(device=dev).manual_seed(1000 + seed)
    with torch.no_grad():
        uncond = G.sample(gen, cond, none, steps=sample_steps, generator=g).cpu().numpy().astype(np.float32)
    log(f"{n_samples} unconditional samples in {time.time() - t0:.0f} s")

    # --- 2. the metric's own check: 13B conditioned on a training clip's state ---
    check_idx = np.concatenate([rng.choice(train_idx[train_idx < manifest["n_sintel"]], n_check // 2, replace=False),
                                rng.choice(train_idx[train_idx >= manifest["n_sintel"]], n_check - n_check // 2, replace=False)])
    src = bank[check_idx].astype(np.float16)
    src_states = simulate_states(net, src, d.cells_all, dt, t_pre, 16).astype(np.float32)
    full = torch.ones(1, len(DEEP), device=dev)
    cond_videos = []
    with torch.no_grad():
        for st in src_states:
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)
            cond_videos.append(G.sample(gen, d.maps(st[:frames], mean, std, dev), full, steps=sample_steps, generator=gg)[0].cpu().numpy())
    cond_videos = np.stack(cond_videos).astype(np.float32)
    log(f"{len(cond_videos)} state-conditioned checks in {time.time() - t0:.0f} s")

    # --- 3. real held-out clips, the scale for everything below ---
    real_idx = rng.choice(test_idx, n_real, replace=False)
    real = bank[real_idx].astype(np.float32)

    # --- 4. the frozen brain on all of them ---
    def with_margin(v):
        return np.concatenate([v[:, :frames], np.repeat(v[:, frames - 1:frames], margin, 1)], 1).astype(np.float16)

    grey = np.full((1, T, 721), 0.5, np.float16)
    st_grey = simulate_states(net, grey, d.cells_all, dt, t_pre, 1).astype(np.float32)[0]
    st_uncond = simulate_states(net, with_margin(uncond), d.cells_all, dt, t_pre, 16).astype(np.float32)
    st_real = simulate_states(net, real[:, :T].astype(np.float16), d.cells_all, dt, t_pre, 16).astype(np.float32)
    st_cond = simulate_states(net, with_margin(cond_videos), d.cells_all, dt, t_pre, 16).astype(np.float32)
    log(f"brain pass on {n_samples + n_real + len(cond_videos) + 1} videos in {time.time() - t0:.0f} s")

    # --- 5. novelty ---
    q_uncond, q_real, q_cond = flat(uncond, frames), flat(real, frames), flat(cond_videos, frames)
    pos_in_train = {int(g_): i for i, g_ in enumerate(train_idx)}
    i_u, r_u, d_u = nearest(q_uncond, bank_train)
    i_r, r_r, d_r = nearest(q_real, bank_train)                       # a held-out real clip against the training set
    i_c, r_c, d_c = nearest(q_cond, bank_train)
    hit = [bool(train_idx[i_c[k]] == check_idx[k]) for k in range(len(check_idx))]
    r_self = [float(np.corrcoef(q_cond[k], flat(bank[check_idx[k]][None], frames)[0])[0, 1]) for k in range(len(check_idx))]
    log(f"metric check: conditioned samples whose nearest training video is their own source: {sum(hit)}/{len(hit)}"
        f" (r to the source {np.mean(r_self):+.2f})")

    # --- 6. diversity, video and state ---
    deep_pos = np.concatenate([d.pos[t] for t in DEEP])

    def states_flat(st):
        x = np.asarray(st[:, :frames, deep_pos], np.float32).reshape(len(st), -1)
        return x - x.mean(1, keepdims=True)

    div = {"videos_generated": triu_mean(pairwise_corr(q_uncond)), "videos_real": triu_mean(pairwise_corr(q_real)),
           "states_generated": triu_mean(pairwise_corr(states_flat(st_uncond))),
           "states_real": triu_mean(pairwise_corr(states_flat(st_real)))}

    # --- 7. per sample ---
    def describe(v, st, i_nn, r_nn, d_nn, k):
        de = direction_energy(st, st_grey, d)
        return {"nn_label": labels[int(train_idx[i_nn])], "nn_index": int(train_idx[i_nn]), "nn_r": float(r_nn),
                "nn_distance": float(d_nn), "video_mean": float(v[:frames].mean()), "video_sd": float(v[:frames].std()),
                "frame_to_frame_sd": float(np.abs(np.diff(v[:frames], axis=0)).mean()),
                "T4_argmax": de["T4_argmax"], "T5_argmax": de["T5_argmax"],
                "deep_energy": {t: round(de["per_type"][t], 4) for t in DEEP}}

    samples = {f"uncond_{k}": describe(uncond[k], st_uncond[k], i_u[k], r_u[k], d_u[k], k) for k in range(n_samples)}
    reals = {f"real_{int(real_idx[k])}": dict(describe(real[k], st_real[k], i_r[k], r_r[k], d_r[k], k),
                                              label=labels[int(real_idx[k])]) for k in range(n_real)}
    checks = {f"cond_{int(check_idx[k])}": dict(describe(cond_videos[k], st_cond[k], i_c[k], r_c[k], d_c[k], k),
                                                source_label=labels[int(check_idx[k])], nn_is_source=hit[k],
                                                r_to_source=r_self[k]) for k in range(len(check_idx))}
    for k, s in samples.items():
        log(f"  {k:<12} nn r {s['nn_r']:+.2f} d {s['nn_distance']:.2f}  ({s['nn_label']})  "
            f"mean {s['video_mean']:.2f} sd {s['video_sd']:.2f}  brain reads T4 {s['T4_argmax']}")
    summ = {
        "model": model, "ckpt": str(ckpt), "frames": frames, "margin": margin, "sample_steps": sample_steps,
        "n_samples": n_samples, "n_real": n_real, "seed": seed,
        "bank": {"n": int(manifest["n"]), "train": int(len(train_idx)), "test": int(len(test_idx))},
        "novelty": {"generated": {"nn_r_median": float(np.median(r_u)), "nn_r_max": float(r_u.max()),
                                  "nn_distance_median": float(np.median(d_u))},
                    "real_heldout": {"nn_r_median": float(np.median(r_r)), "nn_r_max": float(r_r.max()),
                                     "nn_distance_median": float(np.median(d_r))},
                    "conditioned_check": {"nn_is_source": f"{sum(hit)}/{len(hit)}", "r_to_source_mean": float(np.mean(r_self)),
                                          "nn_r_median": float(np.median(r_c))}},
        "diversity": div, "samples": samples, "reals": reals, "checks": checks,
        "seconds": round(time.time() - t0, 1),
    }
    arrays = {"uncond": uncond[:, :frames], "uncond_nn": bank[train_idx[i_u]][:, :frames].astype(np.float32),
              "real": real[:, :frames], "real_nn": bank[train_idx[i_r]][:, :frames].astype(np.float32),
              "cond": cond_videos[:, :frames], "cond_src": bank[check_idx][:, :frames].astype(np.float32),
              "uncond_T4a": np.stack([d.get(st, "T4a")[:frames] for st in st_uncond]),
              "real_T4a": np.stack([d.get(st, "T4a")[:frames] for st in st_real]),
              "uncond_nn_labels": np.array([labels[int(train_idx[i])] for i in i_u]),
              "real_labels": np.array([labels[int(i)] for i in real_idx]),
              "real_nn_labels": np.array([labels[int(train_idx[i])] for i in i_r])}
    return {"summary": summ, "arrays": arrays}


def main(argv=None) -> int:
    from flydream.generate.invert import settings
    from flydream.model import ROOT

    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--ckpt", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--procedural", default="")
    p.add_argument("--out", default=str(ROOT / "data" / "baseline17"))
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--reals", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    proc = Path(a.procedural) if a.procedural else sorted(pdir.glob("procedural_*.npz"))[0]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.ckpt), manifest, columns, proc, n_samples=a.samples, n_real=a.reals,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02), t_pre=g.get("t_pre", 1.0),
            seed=a.seed)
    (out / "summary.json").write_text(json.dumps(r["summary"], indent=1), encoding="utf-8")
    np.savez_compressed(out / "baseline17.npz", **r["arrays"])
    n = r["summary"]["novelty"]
    print(f"\ngenerated: nearest training video r {n['generated']['nn_r_median']:+.2f} (max {n['generated']['nn_r_max']:+.2f}), "
          f"d {n['generated']['nn_distance_median']:.2f}")
    print(f"real held out: nearest training video r {n['real_heldout']['nn_r_median']:+.2f} "
          f"(max {n['real_heldout']['nn_r_max']:+.2f}), d {n['real_heldout']['nn_distance_median']:.2f}")
    print(f"metric check: {n['conditioned_check']['nn_is_source']} conditioned samples land on their own source")
    print(f"diversity (pairwise r): generated videos {r['summary']['diversity']['videos_generated']['mean']:+.2f}, "
          f"real {r['summary']['diversity']['videos_real']['mean']:+.2f}; "
          f"states {r['summary']['diversity']['states_generated']['mean']:+.2f} vs {r['summary']['diversity']['states_real']['mean']:+.2f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
