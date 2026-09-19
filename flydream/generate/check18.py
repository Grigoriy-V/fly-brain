"""ROADMAP 18.2: the two checks before anything is retrained. Local CPU, $0.

    python -m flydream.generate.check18 --corpus data/corpus18            # after the corpus is built

(a) **The corpus against Sintel.** Motion and contrast of the videos, and the
structure of the states they produce — temporal lag-1, one-ring spatial,
cross-type coupling — beside the same numbers for the Sintel clips 13B was
trained on. If the new states are a different kind of object, a later change
in the prior is not attributable to "more data".

(b) **13B on the new distribution.** A corpus clip's own state → 13B → video →
frozen brain → round trip, against Sintel clips in the same code path
(0.023-0.038, items 17.3b/13B). 13B was trained on Sintel and procedural
stimuli only; if it renders the corpus states as well as the old ones, it is
not retrained, which is the order the human set (2026-09-20).

A shuffled state is carried through both as the unreachable control.
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
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import clip_from_sintel, device_of, load_network
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep, direction_energy
from flydream.generate.roundtrip13 import build_states, round_trip


def structure(maps: np.ndarray, ring: np.ndarray) -> dict:
    """The three statistics 17.1b read a sampled state against.

    `maps` (N, T, K, 721) in 13B's units: temporal lag-1 correlation, the
    correlation of a column with the mean of its one-ring neighbours, and the
    mean correlation between the type channels."""
    x = np.asarray(maps, np.float32)

    def r(a, b):
        a = a.reshape(-1) - a.mean(); b = b.reshape(-1) - b.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    lag1 = float(np.mean([r(v[:-1], v[1:]) for v in x]))
    ring = np.asarray(ring)[:, 1:]                       # column 0 is the column itself: with it, white noise reads 0.40
    nb = np.where(ring[None, None, None] >= 0, x[..., np.clip(ring, 0, None)], np.nan)
    ring1 = float(np.mean([r(v, np.nanmean(n, -1)) for v, n in zip(x, nb)]))
    cross = []
    for v in x:
        f = v.reshape(v.shape[0], v.shape[1], -1).transpose(1, 0, 2).reshape(v.shape[1], -1)
        c = np.corrcoef(f)
        cross.append(float(np.abs(c[np.triu_indices(len(c), 1)]).mean()))
    return {"temporal_lag1": lag1, "ring1_spatial": ring1, "cross_type": float(np.mean(cross))}


def run(model: str, gen_ckpt, manifest: dict, columns: dict, corpus: Path, *, n: int = 16, n_sintel: int = 8,
        frames: int = 40, margin: int = 5, dt: float = 0.02, t_pre: float = 1.0, sample_steps: int = 20,
        seed: int = 0, log=print) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    net = load_network(model); dev = device_of(net)
    _, index = P.type_index(net.connectome)
    d = Deep(manifest, columns)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)
    T = frames + margin
    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]
    ta = sa["target"][:, w0:w1, :][:, :, d.cells_all].cpu().numpy().astype(np.float32)
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}      # the scale of every table on this track
    log(f"model and 13B loaded, {time.time() - t0:.0f} s")

    def to_z(states: np.ndarray) -> np.ndarray:
        m = L.to_maps(states.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
        return (m - mean[None, None, :, None]) / std[None, None, :, None]

    # --- the videos: the corpus, and Sintel clips 13B was trained on ---
    cz = np.load(corpus / "videos.npz")
    cmeta = [json.loads(str(x)) for x in cz["meta"]]
    pick = rng.choice(len(cz["videos"]), min(n, len(cz["videos"])), replace=False)
    corpus_v = np.asarray(cz["videos"][np.sort(pick)], np.float32)
    labels = [cmeta[int(i)]["label"] for i in np.sort(pick)]
    sin_idx = rng.choice(189, n_sintel, replace=False)
    sintel_v = np.stack([clip_from_sintel(int(i), frames, dt, margin) for i in sin_idx])
    log(f"corpus {corpus_v.shape} ({len(set(labels))} labels), sintel {sintel_v.shape}")

    def pad(v):
        v = np.asarray(v, np.float32)
        return (v[:, :T] if v.shape[1] >= T
                else np.concatenate([v, np.repeat(v[:, -1:], T - v.shape[1], 1)], 1)).astype(np.float16)

    st_corpus = simulate_states(net, pad(corpus_v), d.cells_all, dt, t_pre, 8).astype(np.float32)
    st_sintel = simulate_states(net, pad(sintel_v), d.cells_all, dt, t_pre, 8).astype(np.float32)
    m_corpus, m_sintel = to_z(st_corpus), to_z(st_sintel)
    shuffled = m_sintel[:1][:, :, :, rng.permutation(721)]
    log(f"states in {time.time() - t0:.0f} s")

    # --- (a) the corpus against Sintel ---
    ring = L.ring_index(1)
    stats = {
        "corpus": {"video_motion": float(np.abs(np.diff(corpus_v[:, :frames], axis=1)).mean()),
                   "video_contrast": float(corpus_v[:, :frames].std()), "video_mean": float(corpus_v[:, :frames].mean()),
                   **structure(m_corpus, ring)},
        "sintel": {"video_motion": float(np.abs(np.diff(sintel_v[:, :frames], axis=1)).mean()),
                   "video_contrast": float(sintel_v[:, :frames].std()), "video_mean": float(sintel_v[:, :frames].mean()),
                   **structure(m_sintel, ring)}}
    for k, v in stats.items():
        log(f"  {k:<7} motion {v['video_motion']:.4f} contrast {v['video_contrast']:.3f} mean {v['video_mean']:.3f}"
            f" | lag1 {v['temporal_lag1']:.3f} ring1 {v['ring1_spatial']:.3f} cross {v['cross_type']:.3f}")

    # --- (b) 13B on both, and the unreachable control ---
    jobs = np.concatenate([m_corpus, m_sintel, shuffled])
    kinds = ["corpus"] * len(m_corpus) + ["sintel"] * len(m_sintel) + ["shuffled"]
    cond = torch.as_tensor(jobs, device=dev)
    mask = torch.ones(len(jobs), len(DEEP), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(jobs), 8):
            g = torch.Generator(device=dev).manual_seed(1000 + seed)
            vids.append(G.sample(gen, cond[i:i + 8], mask[i:i + 8], steps=sample_steps, generator=g).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    target_raw = np.stack([R.from_maps(R.unscale(j[None], mean, std), d.layout, len(d.cells_all))[0] for j in jobs])
    rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
    vids_m = np.concatenate([videos, np.repeat(videos[:, -1:], margin, 1)], 1).astype(np.float16)
    st_back = simulate_states(net, vids_m, d.cells_all, dt, t_pre, 8).astype(np.float32)
    st_grey = simulate_states(net, np.full((1, T, 721), 0.5, np.float16), d.cells_all, dt, t_pre, 1).astype(np.float32)[0]
    log(f"13B and the round trip in {time.time() - t0:.0f} s")

    per = []
    for i, k in enumerate(kinds):
        per.append({"kind": k, "round_trip": float(rts[i]),
                    "label": labels[i] if k == "corpus" else (f"sintel {int(sin_idx[i - len(m_corpus)])}" if k == "sintel" else "shuffled"),
                    "direction_state": direction_energy(target_raw[i], st_grey, d)["T4_argmax"],
                    "direction_video": direction_energy(st_back[i], st_grey, d)["T4_argmax"]})
    gates = {k: {"median": float(np.median([p["round_trip"] for p in per if p["kind"] == k])),
                 "min": float(np.min([p["round_trip"] for p in per if p["kind"] == k])),
                 "max": float(np.max([p["round_trip"] for p in per if p["kind"] == k])),
                 "n": sum(p["kind"] == k for p in per)} for k in ("corpus", "sintel", "shuffled")}
    for k, v in gates.items():
        log(f"  13B round trip on {k:<9} median {v['median']:.3f} ({v['min']:.3f}-{v['max']:.3f}, n={v['n']})")

    summary = {"model": model, "gen_ckpt": str(gen_ckpt), "corpus": str(corpus), "n_corpus": len(m_corpus),
               "n_sintel": len(m_sintel), "frames": frames, "sample_steps": sample_steps, "seed": seed,
               "statistics": stats, "gates": gates, "per_clip": per, "labels": labels,
               "seconds": round(time.time() - t0, 1)}
    arrays = {"corpus_videos": corpus_v[:, :frames].astype(np.float16),
              "corpus_rendered": videos[:len(m_corpus)].astype(np.float16),
              "sintel_videos": sintel_v[:, :frames].astype(np.float16),
              "sintel_rendered": videos[len(m_corpus):len(m_corpus) + len(m_sintel)].astype(np.float16),
              "shuffled_rendered": videos[-1].astype(np.float16),
              "T4a_corpus": np.stack([t[:, d.pos["T4a"]] for t in target_raw[:len(m_corpus)]]).astype(np.float16)}
    return {"summary": summary, "arrays": arrays}


def main(argv=None) -> int:
    from flydream.generate.invert import settings
    from flydream.model import ROOT

    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--tag", default="check18_local")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--n-sintel", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.gen), manifest, columns, Path(a.corpus), n=a.n, n_sintel=a.n_sintel,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    print(f"wrote {out / a.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
