"""Encoder inversion: the video that drives the model's cells to a given state.

    python -m flydream.generate.invert --run 2026-09-18_decode_sintel_flyvis --sample 3 \
        --types T4a T4b T4c T4d T5a T5b T5c T5d --tag t4t5

Given a recorded activity of a set of cell types over a clip, find the input
video (frames x 721 hexals of luminance in [0, 1]) whose simulation through
the network reproduces that activity (Bauer et al. 2026, gradient descent on
the pixels; the recipe checked against flyvis 1.2.0 in
`research_notes/decoder_stack/inversion.md`). The network's parameters are
constants; the video is the only leaf. The loss is the mean squared error on
the chosen cells plus a small total-variation prior over neighbouring hexals
and over time. Start: flat grey (0.5), or `--init ridge` from a ridge
decoder's guess when that run is given.

Protocol. The target is not read from `pairs.npz` but re-simulated from the
clip's true video with the same initial state the inversion uses (a 1 s grey
steady state, no fade-in), so the only thing the optimiser has to explain is
the video. Two numbers per run, both PixCorr per frame between the recovered
and the true video: the inversion, and the **wrong-target control** (the same
optimisation aimed at another clip's activity: what a video looks like when it
explains the wrong state; its correlation with this clip is the floor).

The window is `frames + margin` (config [generate]: 40 + 5 at dt 0.02): the
activity at a frame depends on the frames after it, so the last frames of a
window with no future were unconstrained and blurred (ROADMAP item 9); the
margin is fitted and dropped on save. Runs on the owner's CPU: one Adam step
over 20 frames at batch 1 is about 2-4 s here (a T4: under 0.1 s), and cost
is linear in the window. Output under
`data/generate/<tag>/`: `recovered.npz` (true, recovered, control videos and
their per-frame scores) and `reports/figures/<tag>_inversion.png` (rows:
frames; columns: true video, recovered, control) plus a GIF.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flydream.model import ROOT, configure_flyvis_root

configure_flyvis_root()
import flyvis  # noqa: E402
from flyvis.network.stimulus import Stimulus  # noqa: E402

from flydream.decode import pairs as P  # noqa: E402
from flydream.decode.hexraster import neighbour_index, to_raster  # noqa: E402


def load_network(model: str):
    """A flyvis NetworkView name, or `malecns[:member]` for model zero (the
    MaleCNS export with that FlyVis member's parameters transplanted, the same
    network `flydream.decode.map --model malecns` reads)."""
    if model.startswith("malecns"):
        import tomllib
        from flydream.data.export import filters_path
        from flydream.model.zero import build_network, content_addressed, transplant

        cfg = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
        member = int(model.split(":")[1]) if ":" in model else 0
        fv = flyvis.NetworkView(f"flow/0000/{member:03d}").init_network(checkpoint="best")
        net = build_network(content_addressed(filters_path(cfg)), extent=cfg["data"]["extent"])
        transplant(fv, net, rescale=True, rescale_cap=float(cfg.get("model", {}).get("rescale_cap", 3.0)))
    else:
        net = flyvis.NetworkView(model).init_network(checkpoint="best")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def device_of(net) -> torch.device:
    return next(net.parameters()).device


def settings() -> dict:
    """`config.toml [generate]`: frames, margin, steps, lr, tv, dt, t_pre."""
    import tomllib

    return tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8")).get("generate", {})


def _lum(item) -> np.ndarray:
    lum = item["lum"]
    lum = lum.detach().cpu().numpy() if torch.is_tensor(lum) else np.asarray(lum)   # on a GPU worker it is a cuda tensor
    return lum.astype(np.float32).reshape(lum.shape[0], -1)


def extend_clip(clip: np.ndarray, following: np.ndarray | None, margin: int) -> tuple[np.ndarray, str]:
    """`clip` plus `margin` frames of future: the start of `following` when
    given, else the last frame held. Returns the video and where the margin
    came from ("next", "held", "none")."""
    if margin <= 0:
        return clip, "none"
    if following is not None and len(following) >= margin:
        return np.concatenate([clip, following[:margin]]), "next"
    return np.concatenate([clip, np.repeat(clip[-1:], margin, axis=0)]), "held"


def clip_from_sintel(sample: int, frames: int, dt: float, margin: int = 0) -> np.ndarray:
    """The luminance video (frames + margin, 721) of one AugmentedSintel clip,
    the same dataset and index the decoder map uses, so no pairs file is
    needed. The margin (config [generate]) is the next temporal chunk of the
    same scene and vertical split, which in this set is the next sample
    (checked 2026-09-19: last frame of clip 3 to first of clip 4, r = 0.97);
    when the next sample is another scene the last frame is held."""
    from flydream.decode.map import stimulus_set

    ds, _, _, _ = stimulus_set("sintel", dt)
    sample = int(sample)
    clip = _lum(ds[sample])[:frames]
    if margin <= 0:
        return clip
    following = None
    df = getattr(ds, "arg_df", None)
    if df is not None and sample + 1 < len(df) and df.iloc[sample]["name"] == df.iloc[sample + 1]["name"]:
        following = _lum(ds[sample + 1])
    video, source = extend_clip(clip, following, margin)
    print(f"  clip {sample}: {frames} frames + margin {margin} ({source})", flush=True)
    return video


def simulate(net, video: torch.Tensor, dt: float, state) -> torch.Tensor:
    """(B, T, H) luminance -> (B, T, n_nodes) activity, differentiable in `video`."""
    stim = Stimulus(net.connectome, video.shape[0], video.shape[1], init_buffer=False)
    if hasattr(stim, "buffer"):
        del stim.buffer
    stim.add_input(video[:, :, None, :])
    return net(stim(), dt, state=state)


def tv_prior(video: torch.Tensor, nb: torch.Tensor) -> torch.Tensor:
    """Mean squared difference to lattice neighbours and to the previous frame."""
    valid = nb >= 0
    v = video[..., nb.clamp(min=0)]                       # (B, T, H, 6)
    space = ((v - video[..., None]) ** 2 * valid).sum(-1).mean() / valid.float().mean().clamp(min=1e-6)
    time_ = ((video[:, 1:] - video[:, :-1]) ** 2).mean() if video.shape[1] > 1 else video.new_zeros(())
    return space + time_


def invert(net, target: torch.Tensor, cells: np.ndarray, *, dt: float, state, steps: int, lr: float,
           tv: float, init: torch.Tensor | None, log_every: int = 10) -> tuple[torch.Tensor, list[float]]:
    """Return the recovered video (B, T, H) and the loss trace."""
    B, T = target.shape[:2]
    dev = device_of(net)
    H = net.stimulus.n_input_elements if hasattr(net.stimulus, "n_input_elements") else 721
    video = (init.clone().to(dev) if init is not None else torch.full((B, T, H), 0.5, device=dev)).requires_grad_(True)
    opt = torch.optim.Adam([video], lr=lr)
    nb = torch.as_tensor(neighbour_index(H), dtype=torch.long, device=dev)
    idx = torch.as_tensor(cells, dtype=torch.long, device=dev)
    target = target.to(dev)
    trace = []
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        act = simulate(net, video, dt, state)[:, :, idx]
        fit = ((act - target[:, :, idx]) ** 2).mean()
        loss = fit + tv * tv_prior(video, nb)
        loss.backward()
        opt.step()
        with torch.no_grad():
            video.clamp_(0.0, 1.0)
        trace.append(float(fit))
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:4d}  fit {float(fit):.5f}  loss {float(loss):.5f}  {time.time() - t0:.0f}s", flush=True)
    return video.detach().cpu(), trace


def task_weights(cells: list[np.ndarray], n_nodes: int, device) -> torch.Tensor:
    """(B, n_nodes) weights: task b's cells at 1/len(cells_b), so the batched
    fit is each task's own mean squared error and the tasks stay independent
    (Adam is elementwise; the summed loss gives every video the gradient it
    would get alone)."""
    w = torch.zeros(len(cells), n_nodes, device=device)
    for b, c in enumerate(cells):
        w[b, torch.as_tensor(c, dtype=torch.long, device=device)] = 1.0 / len(c)
    return w


def invert_batch(net, targets: torch.Tensor, cells: list[np.ndarray], *, dt: float, state, steps: int, lr: float,
                 tv: float, plateau_steps: int = 0, plateau_tol: float = 0.0, plateau_floor: float = 1e-3,
                 log_every: int = 10, time_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, np.ndarray, int]:
    """B independent inversions in one pass: `targets` (B, T, n_nodes), task b
    read on `cells[b]`. Returns the videos (B, T, H), the fit trace (steps, B)
    and the number of steps run. Stops early when every task's fit changed by
    less than `plateau_tol` (relative, or relative to `plateau_floor` times
    its first fit once the fit is that small: Mi4 at 3e-5 of its start still
    moved 10 % per 20 steps) over the last `plateau_steps` steps (ROADMAP
    item 10'; on the 2026-09-19 ladder this rule stops at step 127 of 150
    with every fit within 2.4 % of its step-150 value)."""
    B, T = targets.shape[:2]
    dev = device_of(net)
    H = net.stimulus.n_input_elements if hasattr(net.stimulus, "n_input_elements") else 721
    video = torch.full((B, T, H), 0.5, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([video], lr=lr)
    nb = torch.as_tensor(neighbour_index(H), dtype=torch.long, device=dev)
    valid = nb >= 0
    w = task_weights(cells, targets.shape[2], dev)
    targets = targets.to(dev)
    # which frames the fit reads (B, T), normalised per task; all of them by
    # default; the "dark after a clip" dream reads only the dark ones
    tw = torch.ones(B, T, device=dev) if time_weight is None else time_weight.to(dev).float()
    tw = tw / tw.sum(1, keepdim=True).clamp(min=1e-12)
    trace = []
    t0 = time.time()
    steps_run = 0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        act = simulate(net, video, dt, state)
        fit = ((((act - targets) ** 2) * w[:, None, :]).sum(-1) * tw).sum(1)   # (B,) each task's own MSE
        v = video[..., nb.clamp(min=0)]
        space = ((v - video[..., None]) ** 2 * valid).sum(-1).mean((1, 2)) / valid.float().mean().clamp(min=1e-6)
        time_ = ((video[:, 1:] - video[:, :-1]) ** 2).mean((1, 2)) if T > 1 else video.new_zeros(B)
        loss = (fit + tv * (space + time_)).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            video.clamp_(0.0, 1.0)
        f = fit.detach().cpu().numpy()
        trace.append(f)
        steps_run = step + 1
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:4d}  fit mean {f.mean():.5f} max {f.max():.5f}  {time.time() - t0:.0f}s", flush=True)
        if plateau_steps and step >= plateau_steps:
            prev, d = trace[-1 - plateau_steps], np.abs(trace[-1 - plateau_steps] - f)
            if np.all((d <= plateau_tol * prev) | (d <= plateau_tol * plateau_floor * trace[0])):
                print(f"  plateau at step {step}: every task within {plateau_tol:.0%} of {plateau_steps} steps ago", flush=True)
                break
    return video.detach().cpu(), np.stack(trace), steps_run


class GpuSampler:
    """Mean GPU utilisation over a run, from nvidia-smi every few seconds (a
    number the report states beside the batch, AGENTS "Human gates")."""

    def __init__(self, every: float = 2.0):
        import shutil as _sh
        import threading

        self.samples, self.every, self._stop = [], every, threading.Event()
        self._ok = torch.cuda.is_available() and _sh.which("nvidia-smi") is not None
        self._t = threading.Thread(target=self._run, daemon=True) if self._ok else None

    def _run(self):
        import subprocess

        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                self.samples.append(float(out))
            except Exception:
                pass
            self._stop.wait(self.every)

    def __enter__(self):
        if self._t:
            self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)

    @property
    def mean(self) -> float | None:
        return float(np.mean(self.samples)) if self.samples else None


def run_ladder(model: str, sample: int, stages: list[list[str]], *, frames: int, steps: int, lr: float, tv: float,
               dt: float, t_pre: float, margin: int = 0, control_sample: int | None = None,
               out_root: Path | None = None, tag_prefix: str = "", batch: int = 0, plateau_steps: int = 0,
               plateau_tol: float = 0.0, plateau_floor: float = 1e-3) -> list[dict]:
    """The inversion of one clip from each stage in turn, one network load.
    The optimiser fits `frames + margin` frames; the saved and scored videos
    are the first `frames` (config [generate], ROADMAP item 9). Every stage's
    inversion and its wrong-target control are one task; tasks run `batch`
    at a time in one simulation (0 = all at once; ROADMAP item 10'). Returns
    one record per stage with the videos and scores; writes the same under
    `out_root/<tag>/recovered.npz` + `meta.json` when `out_root` is given
    (the shape `flydream.generate.figures` reads)."""
    net = load_network(model)
    dev = device_of(net)
    types, index = P.type_index(net.connectome)
    true_np = clip_from_sintel(sample, frames, dt, margin)
    ctrl_i = control_sample if control_sample is not None else sample + 7
    other_np = clip_from_sintel(ctrl_i, frames, dt, margin)
    true = torch.as_tensor(true_np[None], device=dev)
    other = torch.as_tensor(other_np[None], device=dev)
    state1 = net.steady_state(t_pre, dt, batch_size=1, value=0.5)
    with torch.no_grad():
        target, target_other = simulate(net, true, dt, state1), simulate(net, other, dt, state1)
    stages = [st for st in stages if not [t for t in st if t not in index] or print(f"skipping {st}: not in this model")]
    # one task per (stage, which target): the inversion and its control side by side
    tasks = [(st, which) for st in stages for which in ("inversion", "control")]
    cells_of = {tuple(st): np.concatenate([index[t] for t in st]) for st in stages}
    per_chunk = len(tasks) if batch <= 0 else batch
    videos, traces, seconds, steps_done, util = {}, {}, {}, {}, []
    with GpuSampler() as gpu:
        for i in range(0, len(tasks), per_chunk):
            chunk = tasks[i:i + per_chunk]
            print(f"=== batch of {len(chunk)} tasks on {dev}: " + ", ".join(f"{'_'.join(st)}/{w[:3]}" for st, w in chunk), flush=True)
            t0 = time.time()
            tg = torch.cat([(target if w == "inversion" else target_other) for _, w in chunk])
            state = net.steady_state(t_pre, dt, batch_size=len(chunk), value=0.5)
            vid, tr, n = invert_batch(net, tg, [cells_of[tuple(st)] for st, _ in chunk], dt=dt, state=state, steps=steps,
                                      lr=lr, tv=tv, plateau_steps=plateau_steps, plateau_tol=plateau_tol,
                                      plateau_floor=plateau_floor, log_every=25)
            per_task = (time.time() - t0) / len(chunk)
            for b, (st, w) in enumerate(chunk):
                videos[(tuple(st), w)], traces[(tuple(st), w)] = vid[b].numpy()[:frames], tr[:, b]
                seconds[(tuple(st), w)], steps_done[(tuple(st), w)] = per_task, n
            del tg, vid
    records = []
    shown = true_np[:frames]
    for st in stages:
        k = tuple(st)
        cells = cells_of[k]
        label = "T4T5" if len(st) == 8 else "_".join(st)
        tag = f"{tag_prefix}invert_{label}_s{sample}"
        r, c = videos[(k, "inversion")], videos[(k, "control")]
        trace, trace_c = traces[(k, "inversion")], traces[(k, "control")]
        s_rec, s_ctrl = pixcorr_per_frame(r, shown), pixcorr_per_frame(c, shown)
        record = {"tag": tag, "model": model, "sample": sample, "control_sample": ctrl_i, "types": st,
                  "cells": int(len(cells)), "frames": frames, "margin": margin, "steps": steps,
                  "steps_run": int(steps_done[(k, "inversion")]), "lr": lr, "tv": tv, "dt": dt,
                  "batch": per_chunk, "gpu_utilisation": gpu.mean,
                  "inversion": float(np.mean(s_rec)), "control": float(np.mean(s_ctrl)),
                  "fit_first": float(trace[0]), "fit_final": float(trace[-1]), "control_fit_final": float(trace_c[-1]),
                  "seconds": round(seconds[(k, "inversion")] + seconds[(k, "control")], 1), "device": str(dev)}
        print(f"{tag:<44} r = {record['inversion']:+.3f} (control {record['control']:+.3f}), "
              f"fit {trace[0]:.4f} -> {trace[-1]:.4f} in {record['steps_run']} steps, {record['seconds']} s per stage", flush=True)
        arrays = {"true": shown, "recovered": r, "control": c, "s_rec": s_rec, "s_ctrl": s_ctrl,
                  "trace": np.asarray(trace), "trace_control": np.asarray(trace_c)}
        if out_root is not None:
            d = out_root / tag
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "recovered.npz", **arrays)
            (d / "meta.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        records.append({**record, "arrays": arrays})
    if gpu.mean is not None:
        print(f"GPU utilisation over the ladder: {gpu.mean:.0f}% (batch {per_chunk})", flush=True)
    return records


def pixcorr_per_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = []
    for x, y in zip(a, b):
        sx, sy = x.std(), y.std()
        out.append(float(np.corrcoef(x, y)[0, 1]) if sx > 1e-6 and sy > 1e-6 else 0.0)
    return np.asarray(out)


def figure(true: np.ndarray, rec: np.ndarray, ctrl: np.ndarray, scores: dict, frames: list[int], path: Path,
           title: str, pix_per_hex: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [("stimulus\n(what the eye saw)", true), ("inversion\n(most compatible video)", rec),
            ("wrong-target control", ctrl)]
    fig, axes = plt.subplots(len(frames), len(cols), figsize=(2.3 * len(cols) + 0.4, 2.1 * len(frames) + 0.9),
                             facecolor="#fcfcfb", squeeze=False)
    for i, f in enumerate(frames):
        for j, (name, vid) in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(to_raster(vid[f], vid.shape[-1], pix_per_hex, fill=np.nan), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                r = scores.get(("inversion" if j == 1 else "control") if j else None)
                ax.set_title(name + (f"\nr = {r:+.2f}" if r is not None else ""), fontsize=8.5, color="#0b0b0b")
            if j == 0:
                ax.set_ylabel(f"frame {f}", fontsize=8.5, color="#52514e")
    fig.suptitle(title, fontsize=9.5, color="#0b0b0b", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="#fcfcfb")
    plt.close(fig)


def animation(true: np.ndarray, rec: np.ndarray, ctrl: np.ndarray, path: Path, pix_per_hex: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    vids = [true, rec, ctrl]
    names = ["stimulus", "inversion", "wrong-target control"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), facecolor="#fcfcfb")
    ims = []
    for ax, v, n in zip(axes, vids, names):
        ims.append(ax.imshow(to_raster(v[0], v.shape[-1], pix_per_hex, fill=np.nan), cmap="gray", vmin=0, vmax=1,
                             interpolation="nearest"))
        ax.set_title(n, fontsize=9, color="#0b0b0b"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()

    def update(f):
        for im, v in zip(ims, vids):
            im.set_data(to_raster(v[f], v.shape[-1], pix_per_hex, fill=np.nan))
        return ims
    FuncAnimation(fig, update, frames=len(true), blit=True).save(path, writer=PillowWriter(fps=10))
    plt.close(fig)


def main(argv=None) -> int:
    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="a run under data/decode/ holding pairs.npz (the clips)")
    p.add_argument("--model", default="flow/0000/000")
    p.add_argument("--sample", type=int, default=3, help="clip index in pairs.npz")
    p.add_argument("--control-sample", type=int, default=None, help="the wrong-target clip; default sample+7")
    p.add_argument("--types", nargs="+", required=True)
    p.add_argument("--frames", type=int, default=g.get("frames", 40), help="frames shown and scored")
    p.add_argument("--margin", type=int, default=g.get("margin", 5),
                   help="extra frames fitted after the last shown one, dropped on save (config [generate])")
    p.add_argument("--steps", type=int, default=g.get("steps", 150))
    p.add_argument("--lr", type=float, default=g.get("lr", 0.05))
    p.add_argument("--tv", type=float, default=g.get("tv", 0.02))
    p.add_argument("--dt", type=float, default=g.get("dt", 0.02))
    p.add_argument("--t-pre", type=float, default=g.get("t_pre", 1.0))
    p.add_argument("--tag", default=None)
    p.add_argument("--show-frames", nargs="*", type=int, default=[4, 19, 35])
    p.add_argument("--from-dataset", action="store_true",
                   help="take the clip from AugmentedSintel instead of pairs.npz (no pairs file needed)")
    a = p.parse_args(argv)
    tag = a.tag or f"{time.strftime('%Y-%m-%d')}_invert_{'_'.join(a.types[:2])}{'_etc' if len(a.types) > 2 else ''}_s{a.sample}"
    out = ROOT / "data" / "generate" / tag
    out.mkdir(parents=True, exist_ok=True)

    net = load_network(a.model)
    dev = device_of(net)
    if a.from_dataset:
        ctrl_i = a.control_sample if a.control_sample is not None else a.sample + 7
        true = torch.as_tensor(clip_from_sintel(a.sample, a.frames, a.dt, a.margin)[None], device=dev)
        other = torch.as_tensor(clip_from_sintel(ctrl_i, a.frames, a.dt, a.margin)[None], device=dev)
    else:
        pairs = P.Pairs.load(ROOT / "data" / "decode" / a.run / "pairs.npz")
        ctrl_i = a.control_sample if a.control_sample is not None else (a.sample + 7) % pairs.n_samples
        # pairs.npz holds the clip alone: the margin is the last frame held
        true = torch.as_tensor(extend_clip(pairs.stimulus[a.sample, :a.frames].astype(np.float32), None, a.margin)[0][None], device=dev)
        other = torch.as_tensor(extend_clip(pairs.stimulus[ctrl_i, :a.frames].astype(np.float32), None, a.margin)[0][None], device=dev)
    print(f"clip {a.sample} (control clip {ctrl_i}), {a.frames} frames + margin {a.margin}, types {a.types}, device {dev}")

    types, index = P.type_index(net.connectome)
    missing = [t for t in a.types if t not in index]
    if missing:
        print(f"not cell types of this model: {missing}"); return 1
    cells = np.concatenate([index[t] for t in a.types])
    print(f"{len(cells)} cells of {net.n_nodes}")
    state = net.steady_state(a.t_pre, a.dt, batch_size=1, value=0.5)
    with torch.no_grad():
        target = simulate(net, true, a.dt, state)
        target_other = simulate(net, other, a.dt, state)
    print(f"targets simulated; activity of the chosen cells: mean {float(target[:, :, cells].mean()):.3f}, "
          f"sd {float(target[:, :, cells].std()):.3f}")

    print("inversion:")
    rec, trace = invert(net, target, cells, dt=a.dt, state=state, steps=a.steps, lr=a.lr, tv=a.tv, init=None)
    print("wrong-target control:")
    ctrl, trace_c = invert(net, target_other, cells, dt=a.dt, state=state, steps=a.steps, lr=a.lr, tv=a.tv, init=None)

    t, r, c = true[0, :a.frames].cpu().numpy(), rec[0, :a.frames].numpy(), ctrl[0, :a.frames].numpy()
    s_rec, s_ctrl = pixcorr_per_frame(r, t), pixcorr_per_frame(c, t)
    scores = {"inversion": float(np.mean(s_rec)), "control": float(np.mean(s_ctrl)),
              "inversion_per_frame": s_rec.tolist(), "control_per_frame": s_ctrl.tolist(),
              "fit_final": trace[-1], "fit_first": trace[0], "control_fit_final": trace_c[-1]}
    print(f"\nPixCorr recovered vs true: {scores['inversion']:+.3f} (per frame min {s_rec.min():+.2f}, max {s_rec.max():+.2f}); "
          f"wrong-target control: {scores['control']:+.3f}; fit {trace[0]:.4f} -> {trace[-1]:.4f}")
    np.savez_compressed(out / "recovered.npz", true=t, recovered=r, control=c, s_rec=s_rec, s_ctrl=s_ctrl,
                        trace=np.asarray(trace), trace_control=np.asarray(trace_c))
    (out / "meta.json").write_text(json.dumps({**vars(a), "tag": tag, "cells": int(len(cells)), "control_sample": ctrl_i,
                                               **{k: v for k, v in scores.items() if not k.endswith("per_frame")}},
                                              indent=1), encoding="utf-8")
    figdir = ROOT / "reports" / "figures"
    figure(t, r, c, scores, [f for f in a.show_frames if f < a.frames], figdir / f"{tag}_inversion.png",
           f"Encoder inversion from {', '.join(a.types)} ({len(cells)} cells), {a.model}, clip {a.sample}: "
           f"r = {scores['inversion']:+.2f} against control {scores['control']:+.2f}")
    animation(t, r, c, figdir / f"{tag}_inversion.gif")
    print(f"wrote {out}, reports/figures/{tag}_inversion.png and .gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
