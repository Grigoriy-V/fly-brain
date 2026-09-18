"""The training pairs of 13A: (state of the ladder's types, video), made by the
frozen brain from Sintel and procedural videos.

    python -m flydream.generate.pairs13 --make-procedural --n-per-class 800   # local, CPU
    modal run deploy/modal/generate_app.py --pairs13                          # simulate on a T4

Videos: the augmented Sintel set (flips and the six lattice rotations of the
189 clips) and `flydream.generate.stimuli` clips, all `frames + margin`
long. States: the activity of the 14 ladder types (L1, L3, Mi1, Mi4, Tm5a,
Tm9, T4a-d, T5a-d; 10,094 cells of model zero) from the same grey steady
state the generator uses, stored float16 in shards on the Modal volume
(`/runs/pairs13/shard_*.npz`) — about 0.9 MB per clip, which is why the
models of 13A train where the shards are.

Split (`split_indices`): whole Sintel scenes and whole stimulus classes are
held out, so the test asks about videos of a kind the model has not seen,
not about a new crop of a seen scene. `config.toml [generate.pairs13]`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LADDER_TYPES = ["L1", "L3", "Mi1", "Mi4", "Tm5a", "Tm9", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]


def pairs13_settings() -> dict:
    from flydream.generate.invert import settings

    return settings().get("pairs13", {})


# ----------------------------------------------------------------- videos


def _one(args):
    cls, k, frames, seed = args
    from flydream.generate.stimuli import clip, hex_xy

    rng = np.random.default_rng([seed, hash(cls) % (2 ** 31), k])
    v, p = clip(cls, frames, rng, hex_xy())
    p["index_in_class"] = k
    return v.astype(np.float16), json.dumps(p)


def make_procedural(n_per_class: int, frames: int, seed: int, workers: int = 8, classes=None):
    """(N, frames, 721) float16 videos and their parameter strings, in parallel."""
    from multiprocessing import Pool

    from flydream.generate.stimuli import CLASSES

    classes = list(classes or CLASSES)
    jobs = [(c, k, frames, seed) for c in classes for k in range(n_per_class)]
    with Pool(workers) as pool:
        out = pool.map(_one, jobs, chunksize=8)
    videos = np.stack([v for v, _ in out])
    params = [p for _, p in out]
    return videos, params


def sintel_videos(frames: int, dt: float, flips=(0, 1), rotations=(0, 1, 2, 3, 4, 5)):
    """The augmented Sintel clips (N, frames, 721) float16 with scene labels;
    renders on first use (slow, cached by flyvis under FLYVIS_ROOT_DIR)."""
    import torch
    from flyvis.datasets.sintel import AugmentedSintel

    ds = AugmentedSintel(tasks=["lum"], n_frames=19, dt=dt, temporal_split=True, interpolate=True,
                         flip_axes=list(flips), n_rotations=list(rotations), augment=True, vertical_splits=3,
                         center_crop_fraction=0.7, boxfilter=dict(extent=15, kernel_size=13))
    vids, scenes, names = [], [], []
    df = ds.arg_df
    for i in range(len(ds)):
        lum = ds[i]["lum"]
        lum = lum.detach().cpu().numpy() if torch.is_tensor(lum) else np.asarray(lum)
        v = lum.astype(np.float32).reshape(lum.shape[0], -1)
        if len(v) < frames:                     # the margin: the next chunk of the same scene, else held
            nxt = None
            if i + 1 < len(df) and df.iloc[i]["name"] == df.iloc[i + 1]["name"]:
                n2 = ds[i + 1]["lum"]
                n2 = (n2.detach().cpu().numpy() if torch.is_tensor(n2) else np.asarray(n2)).astype(np.float32)
                nxt = n2.reshape(n2.shape[0], -1)
            from flydream.generate.invert import extend_clip
            v, _ = extend_clip(v, nxt, frames - len(v))
        vids.append(v[:frames].astype(np.float16))
        name = str(df.iloc[i]["name"])
        scenes.append(name.split("_split_")[0].split("_", 2)[-1])
        names.append(name)
    return np.stack(vids), scenes, names


# ----------------------------------------------------------------- split


def split_indices(meta: list[dict], held_scenes: list[str], held_classes: list[str], val_fraction: float, seed: int):
    """train / val / test index arrays. Test = whole held-out scenes and
    classes; val = a random fraction of the rest, by clip."""
    rng = np.random.default_rng(seed)
    test = [i for i, m in enumerate(meta)
            if (m["source"] == "sintel" and m["scene"] in held_scenes) or (m["source"] == "procedural" and m["class"] in held_classes)]
    rest = np.array([i for i in range(len(meta)) if i not in set(test)])
    rng.shuffle(rest)
    n_val = int(round(val_fraction * len(rest)))
    return {"train": np.sort(rest[n_val:]).tolist(), "val": np.sort(rest[:n_val]).tolist(), "test": sorted(test)}


# ----------------------------------------------------------------- states


def simulate_states(net, videos: np.ndarray, cells: np.ndarray, dt: float, t_pre: float, batch: int = 32):
    """(N, T, len(cells)) float16 activity of `cells` for `videos` (N, T, 721),
    each clip from the same grey steady state (as the generator's target)."""
    import torch
    from flydream.generate.invert import device_of, simulate

    dev = device_of(net)
    idx = torch.as_tensor(cells, dtype=torch.long, device=dev)
    out = np.empty((len(videos), videos.shape[1], len(cells)), np.float16)
    with torch.no_grad():
        for i in range(0, len(videos), batch):
            v = torch.as_tensor(videos[i:i + batch].astype(np.float32), device=dev)
            state = net.steady_state(t_pre, dt, batch_size=len(v), value=0.5)
            out[i:i + batch] = simulate(net, v, dt, state)[:, :, idx].cpu().numpy().astype(np.float16)
    return out


def main(argv=None) -> int:
    from flydream.model import ROOT
    from flydream.generate.invert import settings

    g, s = settings(), pairs13_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--make-procedural", action="store_true")
    p.add_argument("--n-per-class", type=int, default=s.get("n_per_class", 800))
    p.add_argument("--seed", type=int, default=s.get("seed", 0))
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args(argv)
    frames = g.get("frames", 40) + g.get("margin", 5)
    if a.make_procedural:
        videos, params = make_procedural(a.n_per_class, frames, a.seed, a.workers)
        out = ROOT / "data" / "pairs13" / f"procedural_{a.n_per_class}_s{a.seed}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, videos=videos, params=np.array(params))
        print(f"wrote {out}: {videos.shape} float16, {out.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
