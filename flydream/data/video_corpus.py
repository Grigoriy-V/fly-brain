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
import os
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


def sintel_band(n_clips: int = 189, frames: int = 45, dt: float = 0.02, lo: int = 5, hi: int = 95, log=print) -> dict:
    """The band, derived: the same three scores on the Sintel clips 13A/13B
    were built from (`AGENTS.md`: a limit is derived, not written). A corpus
    window is kept when its motion and contrast fall between the `lo` and `hi`
    percentiles of those clips; the cut threshold is separate, because a cut
    is not a property Sintel has at all — it is set at the 99th percentile of
    those same clean clips and read against what a *spliced* clip gives
    (measured below). The two distributions overlap in the tail: at that
    threshold about 1 % of clean windows are dropped and a soft cut can still
    pass, which the report states rather than hides."""
    from flydream.generate.invert import clip_from_sintel

    rows, clips = [], []
    for i in range(n_clips):
        v = clip_from_sintel(i, frames, dt, 0)
        clips.append(v)
        rows.append(score_window(v))
    q = {k: {f"p{p}": float(np.percentile([r[k] for r in rows], p)) for p in (1, 5, 25, 50, 75, 95, 99)}
         for k in ("motion", "contrast", "cut", "mean")}
    spliced = [score_window(np.concatenate([clips[i][:frames // 2], clips[(i + 7) % n_clips][frames // 2:]]))["cut"]
               for i in range(0, n_clips, 7)]                         # what a real cut reads on this scale
    band = {"motion": [q["motion"][f"p{lo}"], q["motion"][f"p{hi}"]],
            "contrast": [q["contrast"][f"p{lo}"], q["contrast"][f"p{hi}"]],
            "cut": [0.0, round(float(np.percentile([r["cut"] for r in rows], 99)), 2)]}
    log(f"sintel band from {n_clips} clips: motion {band['motion'][0]:.4f}-{band['motion'][1]:.4f}, "
        f"contrast {band['contrast'][0]:.3f}-{band['contrast'][1]:.3f}, cut <= {band['cut'][1]} "
        f"(clean clips p99 {q['cut']['p99']:.1f}, spliced median {float(np.median(spliced)):.1f})")
    return {"n_clips": n_clips, "percentiles": q, "spliced_cut": {"min": float(np.min(spliced)),
            "median": float(np.median(spliced)), "n": len(spliced)}, "band": band}


def label_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "video"


_CFG: dict = {}


def _init(cfg: dict) -> None:
    import torch

    torch.set_num_threads(1)                                             # 16 processes, one thread each
    _CFG.update(cfg)
    _CFG["box"] = V.eye()


def _one(arg) -> dict:
    """One file in a worker: decode, render, score, keep, rotate."""
    i, path = arg
    c = _CFG
    try:
        got = from_file(Path(path), frames=c["frames"], dt=c["dt"], raw_frames=c["raw_frames"],
                        per_video=c["per_video"], band=c["band"], splits=c["splits"], box=c["box"])
    except Exception as e:
        return {"path": str(path), "error": f"{type(e).__name__}: {e}", "clips": []}
    clips = []
    for j, d in enumerate(got):
        n_rot = (i + j) % max(1, c["rotations"])                         # cycled: every heading equally often
        v = V.augment(d["video"].astype(np.float32), n_rot=n_rot) if n_rot else d["video"].astype(np.float32)
        clips.append({"video": v.astype(np.float16), "n_rot": int(n_rot), "split": d["split"],
                      "window": d["window"], "scores": d["scores"]})
    return {"path": str(path), "clips": clips}


def build(files: list[Path], root: Path, out: Path, *, videos: int, per_video: int, frames: int, dt: float,
          raw_frames: int, splits: int, band: dict, rotations: int, shard: int, workers: int, seed: int,
          log=print) -> dict:
    """Render, score and shard the corpus. Local CPU, one process per core."""
    from concurrent.futures import ProcessPoolExecutor

    rng = np.random.default_rng(seed)
    pick = files if videos >= len(files) else [files[i] for i in sorted(rng.choice(len(files), videos, replace=False))]
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"frames": frames, "dt": dt, "raw_frames": raw_frames, "per_video": per_video, "band": band,
           "splits": splits, "rotations": rotations}
    t0, meta, buf, shards, errors = time.time(), [], [], [], []
    def flush():
        if not buf:
            return
        name = f"videos_{len(shards):03d}.npz"
        np.savez(out / name, videos=np.stack([b["video"] for b in buf]))
        shards.append({"file": name, "n": len(buf)})
        log(f"  wrote {name} ({len(buf)} clips, {time.time() - t0:.0f} s)")
        buf.clear()

    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(cfg,)) as pool:
        for k, r in enumerate(pool.map(_one, list(enumerate(pick)), chunksize=8)):
            if r.get("error"):
                errors.append(r)
            for c in r["clips"]:
                buf.append(c)
                meta.append({"source": "video", "path": str(Path(r["path"]).relative_to(root)),
                             "label": label_of(Path(r["path"]), root), "n_rot": c["n_rot"],
                             "view": c["split"], "window": c["window"], "scores": c["scores"]})
            if len(buf) >= shard:
                flush()
            if (k + 1) % 500 == 0:
                log(f"  {k + 1}/{len(pick)} files, {len(meta)} clips kept, {len(errors)} failed, {time.time() - t0:.0f} s")
    flush()
    manifest = {"source_root": str(root), "files_read": len(pick), "n": len(meta), "frames": frames, "dt": dt,
                "splits": splits, "per_video": per_video, "rotations": rotations, "band": band,
                "shards": shards, "failed": len(errors), "seconds": round(time.time() - t0, 1), "meta": meta}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if errors:
        (out / "errors.json").write_text(json.dumps(errors[:200], indent=1), encoding="utf-8")
    log(f"corpus {len(meta)} clips from {len(pick)} files ({len(errors)} failed) in {time.time() - t0:.0f} s")
    return manifest


def pack(out: Path, *, procedural: Path | None = None, fraction: float = 0.0, frames: int = 45,
         seed: int = 0, log=print) -> dict:
    """The shards plus the manifest as one `videos.npz` in the shape
    `pairs13` reads: `videos` (n, frames, 721) float16 and `meta`, one JSON
    string per clip. That is what goes to the Modal volume.

    `procedural` mixes 13A's stimuli back in as a **minority** (the human,
    2026-09-20: ordinary video mostly, procedural only a small part for
    motion-space coverage). They are appended here, at the video level, rather
    than merged later as states: one pass through the brain, one per-type
    scale, one split, and the manifest keeps them marked as their own source.
    `fraction` is the share of the *final* set, sampled evenly over the
    stimulus classes."""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    vids = [np.load(out / sh["file"])["videos"] for sh in manifest["shards"]]
    videos = np.concatenate(vids).astype(np.float16)
    if len(videos) != manifest["n"]:
        raise SystemExit(f"{len(videos)} clips in the shards, {manifest['n']} in the manifest")
    meta = [{"source": "video", "label": m["label"], "path": m["path"], "n_rot": m["n_rot"],
             "view": m["view"], "window": m["window"]} for m in manifest["meta"]]
    n_video = len(videos)
    n_proc = 0
    if procedural is not None and fraction > 0:
        z = np.load(procedural)
        params = [json.loads(str(x)) for x in z["params"]]
        classes = sorted({p["class"] for p in params})
        want = int(round(fraction / (1 - fraction) * n_video))
        rng = np.random.default_rng(seed)
        pick = []
        for c in classes:                                            # evenly over the classes, not by file order
            ids = [i for i, p in enumerate(params) if p["class"] == c]
            k = min(len(ids), int(np.ceil(want / len(classes))))
            pick += list(rng.choice(ids, k, replace=False))
        pick = np.sort(np.array(pick[:want]))
        pv = z["videos"][pick][:, :frames].astype(np.float16)
        if pv.shape[1] < frames:
            pv = np.concatenate([pv, np.repeat(pv[:, -1:], frames - pv.shape[1], 1)], 1)
        videos = np.concatenate([videos, pv])
        meta += [{"source": "procedural", "class": params[int(i)]["class"], "params": params[int(i)]} for i in pick]
        n_proc = len(pick)
        log(f"mixed in {n_proc} procedural clips over {len(classes)} classes "
            f"({n_proc / len(videos):.0%} of the final set)")
    np.savez(out / "videos.npz", videos=videos, meta=np.array([json.dumps(m) for m in meta]))
    size = (out / "videos.npz").stat().st_size / 1e9
    log(f"packed {videos.shape} into {out / 'videos.npz'} ({size:.2f} GB)")
    return {"n": len(videos), "n_video": n_video, "n_procedural": n_proc, "shape": list(videos.shape),
            "gigabytes": round(size, 3)}


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
    p.add_argument("--sintel-band", action="store_true")
    p.add_argument("--band", default="")
    p.add_argument("--pack", action="store_true")
    p.add_argument("--procedural", default="")
    p.add_argument("--procedural-fraction", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    src = Path(a.src)
    files = V.video_files(src)
    print(f"{len(files)} video files under {src}")
    if a.sintel_band:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        r = sintel_band(frames=a.frames, dt=a.dt)
        (out / "sintel_band.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps(r["band"], indent=1))
        return 0
    if a.probe:
        rng = np.random.default_rng(a.seed)
        sample = [files[i] for i in rng.choice(len(files), min(a.probe, len(files)), replace=False)]
        r = probe(sample, len(sample), frames=a.frames, dt=a.dt, raw_frames=a.raw_frames, splits=a.splits)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out / "probe.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps(r["percentiles"], indent=1))
        print(f"windows {r['windows']}, pass rate in the Sintel band {r['pass_rate_sintel_band']:.1%}, {r['seconds']} s")
        return 0
    if a.pack:
        pack(Path(a.out), procedural=Path(a.procedural) if a.procedural else None,
             fraction=a.procedural_fraction, frames=a.frames, seed=a.seed)
        return 0
    band = json.loads(a.band) if a.band else json.loads((Path(a.out) / "sintel_band.json").read_text(encoding="utf-8"))["band"]         if (Path(a.out) / "sintel_band.json").exists() else SINTEL_BAND
    out = Path(a.out)
    build(files, src, out, videos=a.videos, per_video=a.per_video, frames=a.frames, dt=a.dt,
          raw_frames=a.raw_frames, splits=a.splits, band=band, rotations=a.rotations, shard=a.shard,
          workers=a.workers or (os.cpu_count() or 8) // 2, seed=a.seed)
    pack(out, procedural=Path(a.procedural) if a.procedural else None,
         fraction=a.procedural_fraction, frames=a.frames, seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
