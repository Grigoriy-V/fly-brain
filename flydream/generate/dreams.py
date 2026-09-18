"""Dreams-lite: the generator run on states that no clip caused (ROADMAP item 11).

    python -m flydream.generate.dreams --model malecns --source neuron_noise

The item-8 generator inverts a state into the video most compatible with it.
Here the state does not come from a natural clip. Four sources, each a
setting in `config.toml [generate.dreams]`:

- `eye_noise`: white noise into the eye (mean grey, sd `eye_noise_sd`, a
  new draw every frame).
- `flash`: grey, then `flash_frames` frames of white, then grey again.
- `dark_after`: clip `dark_after_sample` for `frames` frames, then
  `dark_frames` frames of black; the fit reads the dark frames only, so the
  recovered video says what the state after the clip still carries.
- `neuron_noise`: the eye sees grey throughout and Gaussian noise of sd
  `neuron_noise_sd` is added to every cell's activity at every Euler step of
  the *target* simulation (`simulate_noisy`); the inversion then runs
  through the noiseless network, so whatever picture comes back is what the
  wiring makes of internally generated activity.

Every source is inverted from every stage of the ladder (the stages of
`deploy/modal/generate_app.py`) in one batch, each stage beside its
**shuffled-state control**: the same target activity with the stage's cells
permuted by one fixed permutation (the numbers are kept, the map from cell
to place is destroyed). What the control recovers is what any state with
these statistics gives; the difference to the inversion is what the state's
structure adds. Scores per stage: PixCorr against the input video where the
input is a picture (all but `neuron_noise`; in the dark after a clip the
reference is the clip's last frame), and the recovered video's
spatial contrast (sd over hexals, mean over frames) for every source.

Output under `data/generate/<tag>/recovered.npz` + `meta.json`, one tag per
(source, stage), the shape `tools/fig_dreams.py` draws. Nothing here is what
the fly sees: a recovered video is the stimulus most compatible with the
state under this encoder (AGENTS "Primary principle").
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flydream.model import ROOT
from flydream.generate.invert import (GpuSampler, clip_from_sintel, device_of, invert_batch, load_network,
                                      pixcorr_per_frame, settings, simulate)
from flydream.decode import pairs as P

SOURCES = ("eye_noise", "flash", "dark_after", "neuron_noise")
# the ladder of deploy/modal/generate_app.py (kept there too: that module must not import flydream at load)
LADDER = [["R1"], ["L1"], ["L3"], ["Mi1"], ["Mi4"], ["Tm5a"], ["Tm9"], ["T4a"], ["T5a"],
          ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]]


def dream_settings() -> dict:
    return settings().get("dreams", {})


def input_video(source: str, frames: int, margin: int, dt: float, d: dict, rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray]:
    """The eye's video (T, 721) for a source and the time weight (T,) the fit reads."""
    T = frames + margin
    if source == "eye_noise":
        v = np.clip(0.5 + d.get("eye_noise_sd", 0.15) * rng.standard_normal((T, 721)), 0, 1).astype(np.float32)
        return v, np.ones(T, np.float32)
    if source == "flash":
        v = np.full((T, 721), 0.5, np.float32)
        on, n = int(d.get("flash_start", 10)), int(d.get("flash_frames", 5))
        v[on:on + n] = 1.0
        return v, np.ones(T, np.float32)
    if source == "dark_after":
        clip = clip_from_sintel(int(d.get("dark_after_sample", 3)), frames, dt)
        dark = int(d.get("dark_frames", 40))
        v = np.concatenate([clip, np.zeros((dark + margin, 721), np.float32)])
        tw = np.zeros(len(v), np.float32)
        tw[frames:frames + dark] = 1.0
        return v, tw
    if source == "neuron_noise":
        return np.full((T, 721), 0.5, np.float32), np.ones(T, np.float32)
    raise ValueError(f"unknown source {source}; one of {SOURCES}")


def simulate_noisy(net, video: torch.Tensor, dt: float, state, noise_sd: float, seed: int) -> torch.Tensor:
    """`simulate` with Gaussian noise of sd `noise_sd` added to every cell's
    activity after every Euler step (the network's own forward, unrolled)."""
    from flyvis.network.stimulus import Stimulus
    from flyvis.utils.tensor_utils import AutoDeref

    stim = Stimulus(net.connectome, video.shape[0], video.shape[1], init_buffer=False)
    if hasattr(stim, "buffer"):
        del stim.buffer
    stim.add_input(video[:, :, None, :])
    x = stim()
    net.clamp()
    params = net._param_api()
    g = torch.Generator(device=x.device).manual_seed(seed)
    out = []
    for i in range(x.shape[1]):
        state = net._next_state(params, state, x[:, i], dt)
        act = state.nodes.activity + noise_sd * torch.randn(state.nodes.activity.shape, generator=g, device=x.device)
        state = net._state_api(AutoDeref(nodes=AutoDeref(activity=act), edges=AutoDeref()))
        out.append(act)
    return torch.stack(out, 1)


def shuffled(target: torch.Tensor, cells: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
    """The target with the stage's cells permuted by one fixed permutation."""
    t = target.clone()
    perm = rng.permutation(len(cells))
    idx = torch.as_tensor(cells, dtype=torch.long, device=t.device)
    t[:, :, idx] = target[:, :, idx[torch.as_tensor(perm, device=t.device)]]
    return t


def contrast(video: np.ndarray) -> float:
    return float(video.std(1).mean())


def run_dreams(model: str, source: str, stages: list[list[str]], *, frames: int, margin: int, steps: int, lr: float,
               tv: float, dt: float, t_pre: float, batch: int = 0, plateau_steps: int = 0, plateau_tol: float = 0.0,
               plateau_floor: float = 1e-3, seed: int = 0, out_root: Path | None = None, tag_prefix: str = "",
               dreams: dict | None = None) -> list[dict]:
    d = dreams if dreams is not None else dream_settings()
    rng = np.random.default_rng(seed)
    net = load_network(model)
    dev = device_of(net)
    _, index = P.type_index(net.connectome)
    vid_np, tw_np = input_video(source, frames, margin, dt, d, rng)
    shown = len(vid_np) - margin
    video = torch.as_tensor(vid_np[None], device=dev)
    state1 = net.steady_state(t_pre, dt, batch_size=1, value=0.5)
    with torch.no_grad():
        if source == "neuron_noise":
            target = simulate_noisy(net, video, dt, state1, float(d.get("neuron_noise_sd", 0.05)), seed)
        else:
            target = simulate(net, video, dt, state1)
    print(f"{source}: input {vid_np.shape}, fit reads {int(tw_np.sum())} frames; target activity sd {float(target.std()):.3f}",
          flush=True)
    stages = [st for st in stages if not [t for t in st if t not in index] or print(f"skipping {st}: not in this model")]
    cells_of = {tuple(st): np.concatenate([index[t] for t in st]) for st in stages}
    tasks = [(st, w) for st in stages for w in ("inversion", "control")]
    per_chunk = len(tasks) if batch <= 0 else batch
    videos, traces, steps_done = {}, {}, {}
    t_all = time.time()
    with GpuSampler() as gpu:
        for i in range(0, len(tasks), per_chunk):
            chunk = tasks[i:i + per_chunk]
            print(f"=== {source}: batch of {len(chunk)} tasks on {dev}", flush=True)
            tg = torch.cat([target if w == "inversion" else shuffled(target, cells_of[tuple(st)], rng) for st, w in chunk])
            state = net.steady_state(t_pre, dt, batch_size=len(chunk), value=0.5)
            twb = torch.as_tensor(tw_np)[None].repeat(len(chunk), 1)
            vid, tr, n = invert_batch(net, tg, [cells_of[tuple(st)] for st, _ in chunk], dt=dt, state=state, steps=steps,
                                      lr=lr, tv=tv, plateau_steps=plateau_steps, plateau_tol=plateau_tol,
                                      plateau_floor=plateau_floor, log_every=25, time_weight=twb)
            for b, (st, w) in enumerate(chunk):
                videos[(tuple(st), w)], traces[(tuple(st), w)], steps_done[(tuple(st), w)] = vid[b].numpy()[:shown], tr[:, b], n
            del tg, vid
    seconds = time.time() - t_all
    records = []
    inp = vid_np[:shown]
    read = tw_np[:shown] > 0
    # what a recovered frame is scored against: the input frame, except in the
    # dark after a clip, where the input is black and the question is how much
    # of the clip's last frame the state still carries
    ref = np.repeat(inp[frames - 1][None], int(read.sum()), 0) if source == "dark_after" else inp[read]
    for st in stages:
        k = tuple(st)
        label = "T4T5" if len(st) == 8 else "_".join(st)
        tag = f"{tag_prefix}dream_{source}_{label}"
        r, c = videos[(k, "inversion")], videos[(k, "control")]
        has_picture = source != "neuron_noise"
        s_rec = pixcorr_per_frame(r[read], ref) if has_picture else np.zeros(int(read.sum()))
        s_ctrl = pixcorr_per_frame(c[read], ref) if has_picture else np.zeros(int(read.sum()))
        record = {"tag": tag, "model": model, "source": source, "types": st, "cells": int(len(cells_of[k])),
                  "frames": shown, "margin": margin, "read_frames": [int(np.argmax(read)), int(len(read) - np.argmax(read[::-1]))],
                  "steps": steps, "steps_run": int(steps_done[(k, "inversion")]), "lr": lr, "tv": tv, "dt": dt,
                  "seed": seed, "settings": {kk: vv for kk, vv in d.items()}, "batch": per_chunk,
                  "gpu_utilisation": gpu.mean, "seconds_all_stages": round(seconds, 1), "device": str(dev),
                  "inversion": float(np.mean(s_rec)) if has_picture else None,
                  "control": float(np.mean(s_ctrl)) if has_picture else None,
                  "score_against": "last clip frame" if source == "dark_after" else "input frame",
                  "contrast_input": contrast(inp[read]), "contrast_inversion": contrast(r[read]),
                  "contrast_control": contrast(c[read]),
                  "fit_first": float(traces[(k, "inversion")][0]), "fit_final": float(traces[(k, "inversion")][-1]),
                  "control_fit_final": float(traces[(k, "control")][-1])}
        print(f"{tag:<40} r {record['inversion'] if has_picture else '  n/a'!s:>7} (control {record['control'] if has_picture else 'n/a'!s:>7})"
              f"  contrast in {record['contrast_input']:.3f} / out {record['contrast_inversion']:.3f} / control {record['contrast_control']:.3f}"
              f"  fit {record['fit_first']:.4f} -> {record['fit_final']:.5f}", flush=True)
        arrays = {"true": inp, "recovered": r, "control": c, "s_rec": s_rec, "s_ctrl": s_ctrl, "read": read, "ref": ref,
                  "trace": traces[(k, "inversion")], "trace_control": traces[(k, "control")]}
        if out_root is not None:
            dd = out_root / tag
            dd.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dd / "recovered.npz", **arrays)
            (dd / "meta.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        records.append({**record, "arrays": arrays})
    if gpu.mean is not None:
        print(f"GPU utilisation: {gpu.mean:.0f}% (batch {per_chunk}), {seconds:.0f} s for {len(tasks)} tasks", flush=True)
    return records


def main(argv=None) -> int:
    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--source", choices=SOURCES, required=True)
    p.add_argument("--stages", default="", help="comma-separated, types joined by +; default: the ladder")
    p.add_argument("--steps", type=int, default=g.get("steps", 150))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag-prefix", default=f"{time.strftime('%Y-%m-%d')}_malecns_")
    a = p.parse_args(argv)
    st = [s.split("+") for s in a.stages.split(",")] if a.stages else LADDER
    run_dreams(a.model, a.source, st, frames=g.get("frames", 40), margin=g.get("margin", 5), steps=a.steps,
               lr=g.get("lr", 0.05), tv=g.get("tv", 0.02), dt=g.get("dt", 0.02), t_pre=g.get("t_pre", 1.0),
               batch=g.get("batch", 0), plateau_steps=g.get("plateau_steps", 0), plateau_tol=g.get("plateau_tol", 0.0),
               plateau_floor=g.get("plateau_floor", 1e-3), seed=a.seed, out_root=ROOT / "data" / "generate",
               tag_prefix=a.tag_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
