"""Manipulated states (ROADMAP item 12): the video most compatible with a
state assembled from several sources, or edited by hand.

    python -m flydream.generate.mix --model malecns

The target state is no longer one clip's activity. Three scenarios, from the
human's note `docs/ideas/mixed_brain_state_video_generation.md`, all in one
batch (`config.toml [generate.mix]`):

- **C, gain edits.** Clip A's state read on every ladder type, with one
  type's activity scaled: T4a × 0.5, T4a × 2, T5 (a-d) × 0, Mi4 × 1.5, and
  × 1 as the control that must give the clip back.
- **A, interpolation.** One type's state (T4a) at α·A + (1−α)·B for α in
  {0, ¼, ½, ¾, 1}, read on that type alone, beside the state the averaged
  video ½(A+B) causes (is the mix of states the state of the mix?).
- **B, hybrid.** Early types (L1, L3) from clip A with the motion types
  (T4a-d, T5a-d) from clip B, and the reverse; beside the two pure states
  (all from A, all from B) read on the same types.

**Per-type normalisation.** A task's loss is the mean over its types of that
type's normalised squared error: cell weight 1 / (k · n_t · var_t), with
var_t the variance of the type's activity over (time, cells) under clip A.
Without it T4/T5 (sd ≈ 0.4) drown L3 (sd ≈ 0.04) inside one task.

Every task is scored by r to clip A and r to clip B per frame (and to the
averaged video for the interpolation), so the winner of a contest is a
number. A hybrid is a counterfactual set of constraints, not a perception:
the video most compatible with conditions no input produces together.
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

T4 = ["T4a", "T4b", "T4c", "T4d"]
T5 = ["T5a", "T5b", "T5c", "T5d"]
READOUT = ["L1", "L3", "Mi1", "Mi4", "Tm5a", "Tm9"] + T4 + T5      # the ladder's types, for the gain edits


def mix_settings() -> dict:
    return settings().get("mix", {})


def type_weights(index: dict, types: list[str], target_a: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Cells of `types` and their weights 1 / (k · n_t · var_t); the target of
    clip A gives var_t. Returns (cells, weights) aligned."""
    cells, weights = [], []
    for t in types:
        idx = index[t]
        var = float(target_a[0][:, idx].var(unbiased=False)) + 1e-6
        cells.append(idx)
        weights.append(np.full(len(idx), 1.0 / (len(types) * len(idx) * var), np.float32))
    return np.concatenate(cells), np.concatenate(weights)


def build_tasks(index: dict, target_a: torch.Tensor, target_b: torch.Tensor, target_avg: torch.Tensor, m: dict) -> list[dict]:
    """One dict per task: name, scenario, types read, the target (1, T, N), cells, weights."""
    tasks = []

    def add(name, scenario, types, target, note=""):
        cells, w = type_weights(index, types, target_a)
        tasks.append({"name": name, "scenario": scenario, "types": types, "target": target, "cells": cells,
                      "weights": w, "note": note})

    # C: gain edits on the full readout of clip A's state
    edits = [("x1", {}, "control: unedited state"), ("T4a_x0.5", {"T4a": 0.5}, ""), ("T4a_x2", {"T4a": 2.0}, ""),
             ("T5_x0", {t: 0.0 for t in T5}, ""), ("Mi4_x1.5", {"Mi4": 1.5}, "")]
    for name, gains, note in edits:
        tg = target_a.clone()
        for t, g in gains.items():
            idx = torch.as_tensor(index[t], dtype=torch.long, device=tg.device)
            tg[:, :, idx] = tg[:, :, idx] * g
        add(f"C_{name}", "C", READOUT, tg, note)
    # A: interpolation of one type between A and B, and the averaged video's state
    t_int = m.get("interp_type", "T4a")
    for alpha in m.get("alphas", [0.0, 0.25, 0.5, 0.75, 1.0]):
        add(f"A_{t_int}_a{alpha:g}", "A", [t_int], alpha * target_a + (1 - alpha) * target_b, f"alpha {alpha:g} of A")
    add(f"A_{t_int}_avgvideo", "A", [t_int], target_avg, "control: the state of the averaged video")
    # B: hybrids and the pure states on the same types
    early, motion = ["L1", "L3"], T4 + T5
    both = early + motion

    def hybrid(src_early, src_motion):
        tg = torch.zeros_like(target_a)
        for t in early:
            idx = torch.as_tensor(index[t], dtype=torch.long, device=tg.device); tg[:, :, idx] = src_early[:, :, idx]
        for t in motion:
            idx = torch.as_tensor(index[t], dtype=torch.long, device=tg.device); tg[:, :, idx] = src_motion[:, :, idx]
        return tg
    add("B_allA", "B", both, target_a, "control: all types from A")
    add("B_earlyA_motionB", "B", both, hybrid(target_a, target_b), "L1/L3 from A, T4/T5 from B")
    add("B_earlyB_motionA", "B", both, hybrid(target_b, target_a), "L1/L3 from B, T4/T5 from A")
    add("B_allB", "B", both, target_b, "control: all types from B")
    return tasks


def run_mix(model: str, *, sample_a: int, sample_b: int, frames: int, margin: int, steps: int, lr: float, tv: float,
            dt: float, t_pre: float, batch: int = 0, plateau_steps: int = 0, plateau_tol: float = 0.0,
            plateau_floor: float = 1e-3, out_root: Path | None = None, tag_prefix: str = "",
            mix: dict | None = None) -> list[dict]:
    m = mix if mix is not None else mix_settings()
    net = load_network(model)
    dev = device_of(net)
    _, index = P.type_index(net.connectome)
    a_np, b_np = clip_from_sintel(sample_a, frames, dt, margin), clip_from_sintel(sample_b, frames, dt, margin)
    avg_np = 0.5 * (a_np + b_np)
    state1 = net.steady_state(t_pre, dt, batch_size=1, value=0.5)
    with torch.no_grad():
        ta = simulate(net, torch.as_tensor(a_np[None], device=dev), dt, state1)
        tb = simulate(net, torch.as_tensor(b_np[None], device=dev), dt, state1)
        tavg = simulate(net, torch.as_tensor(avg_np[None], device=dev), dt, state1)
    tasks = build_tasks(index, ta, tb, tavg, m)
    per_chunk = len(tasks) if batch <= 0 else batch
    videos, traces, steps_done = {}, {}, {}
    t_all = time.time()
    with GpuSampler() as gpu:
        for i in range(0, len(tasks), per_chunk):
            chunk = tasks[i:i + per_chunk]
            print(f"=== mix: batch of {len(chunk)} tasks on {dev}: " + ", ".join(t["name"] for t in chunk), flush=True)
            tg = torch.cat([t["target"] for t in chunk])
            state = net.steady_state(t_pre, dt, batch_size=len(chunk), value=0.5)
            vid, tr, n = invert_batch(net, tg, [t["cells"] for t in chunk], dt=dt, state=state, steps=steps, lr=lr, tv=tv,
                                      plateau_steps=plateau_steps, plateau_tol=plateau_tol, plateau_floor=plateau_floor,
                                      log_every=25, cell_weights=[t["weights"] for t in chunk])
            for b, t in enumerate(chunk):
                videos[t["name"]], traces[t["name"]], steps_done[t["name"]] = vid[b].numpy()[:frames], tr[:, b], n
            del tg, vid
    seconds = time.time() - t_all
    a_s, b_s, avg_s = a_np[:frames], b_np[:frames], avg_np[:frames]
    records = []
    for t in tasks:
        r = videos[t["name"]]
        s_a, s_b, s_avg = pixcorr_per_frame(r, a_s), pixcorr_per_frame(r, b_s), pixcorr_per_frame(r, avg_s)
        tag = f"{tag_prefix}mix_{t['name']}"
        record = {"tag": tag, "model": model, "scenario": t["scenario"], "name": t["name"], "note": t["note"],
                  "types": t["types"], "cells": int(len(t["cells"])), "sample_a": sample_a, "sample_b": sample_b,
                  "frames": frames, "margin": margin, "steps": steps, "steps_run": int(steps_done[t["name"]]),
                  "lr": lr, "tv": tv, "dt": dt, "batch": per_chunk, "gpu_utilisation": gpu.mean,
                  "seconds_all_tasks": round(seconds, 1), "device": str(dev), "settings": dict(m),
                  "r_a": float(np.mean(s_a)), "r_b": float(np.mean(s_b)), "r_avg": float(np.mean(s_avg)),
                  "contrast": float(r.std(1).mean()),
                  "fit_first": float(traces[t["name"]][0]), "fit_final": float(traces[t["name"]][-1])}
        print(f"{tag:<44} r_A {record['r_a']:+.2f}  r_B {record['r_b']:+.2f}  r_avg {record['r_avg']:+.2f}  "
              f"fit {record['fit_first']:.4f} -> {record['fit_final']:.5f}  {t['note']}", flush=True)
        arrays = {"clip_a": a_s, "clip_b": b_s, "clip_avg": avg_s, "recovered": r, "s_a": s_a, "s_b": s_b, "s_avg": s_avg,
                  "trace": traces[t["name"]]}
        if out_root is not None:
            d = out_root / tag
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "recovered.npz", **arrays)
            (d / "meta.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        records.append({**record, "arrays": arrays})
    if gpu.mean is not None:
        print(f"GPU utilisation: {gpu.mean:.0f}% (batch {per_chunk}), {seconds:.0f} s for {len(tasks)} tasks", flush=True)
    return records


def main(argv=None) -> int:
    g, m = settings(), mix_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--sample-a", type=int, default=m.get("sample_a", 3))
    p.add_argument("--sample-b", type=int, default=m.get("sample_b", 10))
    p.add_argument("--steps", type=int, default=g.get("steps", 150))
    p.add_argument("--tag-prefix", default=f"{time.strftime('%Y-%m-%d')}_malecns_")
    a = p.parse_args(argv)
    run_mix(a.model, sample_a=a.sample_a, sample_b=a.sample_b, frames=g.get("frames", 40), margin=g.get("margin", 5),
            steps=a.steps, lr=g.get("lr", 0.05), tv=g.get("tv", 0.02), dt=g.get("dt", 0.02), t_pre=g.get("t_pre", 1.0),
            batch=g.get("batch", 0), plateau_steps=g.get("plateau_steps", 0), plateau_tol=g.get("plateau_tol", 0.0),
            plateau_floor=g.get("plateau_floor", 1e-3), out_root=ROOT / "data" / "generate", tag_prefix=a.tag_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
