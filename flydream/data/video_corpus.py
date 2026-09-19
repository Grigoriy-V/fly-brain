"""ROADMAP 18: a corpus of ordinary video, seen through the fly's eye.

    python -m flydream.data.video_corpus --src data/ucf101/UCF-101 --probe 200     # free, the thresholds
    python -m flydream.data.video_corpus --src data/ucf101/UCF-101 --out data/corpus18 --videos 6000

Item 17's prior learned 1,695 states from 19 Sintel scenes and its samples
land at a round trip of 0.142 where a real clip reaches 0.006. The set is the
first suspect, so this builds the replacement: ordinary video, many scenes,
each clip rendered by `video_hex` through the same eye and the same
processing as the Sintel clips 13A/13B already use.

**Selection, not collection.** Ordinary video carries three things Sintel does
not: hard cuts (a flash across the whole eye, which T4/T5 answer with a
transient no motion produces), tripod shots (no motion at all — a prior
trained on them learns that nothing moves) and flat or black frames. Every
window is scored by `motion`, `contrast` and `cut_score` and kept only inside
the band the real Sintel clips occupy, which `--probe` measures first and the
manifest records.

**Augmentation is for direction balance, not for volume.** One rotation of the
hex lattice per window, cycled, so the T4/T5 direction types see the corpus's
motion from every heading; the rest of the diversity comes from taking more
distinct videos, not from repeating one.

The output is `videos_XXX.npz` shards of (n, frames, 721) float16 plus
`manifest.json` in the shape `pairs13` uses, so the states are built by the
same Modal function and the same split rules apply.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from flydream.data import video_hex as V

SINTEL_BAND = {"motion": (0.008, 0.20), "contrast": (0.05, 0.45), "cut": (0.0, 12.0)}


def score_window(w: np.ndarray) -> dict:
    return {"motion": V.motion(w), "contrast": V.contrast(w), "cut": V.cut_score(w),
            "mean": float(w.mean()), "min": float(w.min()), "max": float(w.max())}


def keep(s: dict, band: dict) -> bool:
    return (band["motion"][0] <= s["motion"] <= band["motion"][1]
            and band["contrast"][0] <= s["contrast"] <= band["contrast"][1]
            and s["cut"] <= band["cut"][1] and s["max"] - s["min"] > 0.05)


def from_file(path: Path, *, frames: int, dt: float, raw_frames: int, per_video: int, band: dict,
              splits: int, box=None) -> list[dict]:
    """The windows of one video file that pass the band, best motion first."""
    gray, fps = V.read_gray(path, max_frames=raw_frames)
    if len(gray) < 4:
        return []
    hexals = V.to_hexals(gray, fps, dt, box, splits=splits)
    out = []
    for si, view in enumerate(hexals):
        for wi, w in enumerate(V.windows(view, frames)):
            s = score_window(w)
            if keep(s, band):
                out.append({"video": w.astype(np.float16), "split": si, "window": wi, "scores": s})
    out.sort(key=lambda d: -d["scores"]["motion"])
    return out[:per_video]


def probe(files: list[Path], n: int, *, frames: int, dt: float, raw_frames: int, splits: int, log=print) -> dict:
    """Measure the three scores on `n` files before any thresholds are set."""
    box = V.eye()
    rows = []
    t0 = time.time()
    for i, p in enumerate(files[:n]):
        try:
            gray, fps = V.read_gray(p, max_frames=raw_frames)
            for view in V.to_hexals(gray, fps, dt, box, splits=splits):
                for w in V.windows(view, frames):
                    rows.append(score_window(w))
        except Exception as e:                                           # a corrupt file is data, not a crash
            log(f"  skipped {p.name}: {type(e).__name__} {e}")
        if (i + 1) % 25 == 0:
            log(f"  probed {i + 1}/{n} files, {len(rows)} windows, {time.time() - t0:.0f} s")
    q = {k: {f"p{p}": float(np.percentile([r[k] for r in rows], p)) for p in (1, 5, 25, 50, 75, 95, 99)}
         for k in ("motion", "contrast", "cut", "mean")}
    passed = sum(keep(r, SINTEL_BAND) for r in rows)
    return {"files": min(n, len(files)), "windows": len(rows), "percentiles": q,
            "pass_rate_sintel_band": passed / max(1, len(rows)), "seconds": round(time.time() - t0, 1)}


def label_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "video"


def main(argv=None) -> int:
    from flydream.model import ROOT

    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--videos", type=int, default=6000)
    p.add_argument("--per-video", type=int, default=2)
    p.add_argument("--frames", type=int, default=45)
    p.add_argument("--raw-frames", type=int, default=120)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--rotations", type=int, default=6)
    p.add_argument("--shard", type=int, default=4000)
    p.add_argument("--probe", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    src = Path(a.src)
    files = V.video_files(src)
    print(f"{len(files)} video files under {src}")
    if a.probe:
        rng = np.random.default_rng(a.seed)
        sample = [files[i] for i in rng.choice(len(files), min(a.probe, len(files)), replace=False)]
        r = probe(sample, len(sample), frames=a.frames, dt=a.dt, raw_frames=a.raw_frames, splits=a.splits)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out / "probe.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps(r["percentiles"], indent=1))
        print(f"windows {r['windows']}, pass rate in the Sintel band {r['pass_rate_sintel_band']:.1%}, {r['seconds']} s")
        return 0
    raise SystemExit("building the corpus is a separate run; --probe first")


if __name__ == "__main__":
    sys.exit(main())
