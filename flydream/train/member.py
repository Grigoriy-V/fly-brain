"""Train one member: flyvis's `MultiTaskSolver` on a connectome export.

    python -m flydream.train.member --connectome <path.json> --run 0100/000 \
        --n-iters 250000 --results-root <dir> [--init <state.pt>] [--resume] [--out result.json]

The same path `flyvis train-single` takes, composed from flyvis's own Hydra
config, with three overrides: the connectome file, `n_syn_fill=0` (the MaleCNS
filters are complete) and the iteration count. Loss, schedule, augmentation,
checkpoints: Lappalainen et al.'s. `--init` loads a state written by
`flydream.model.init_state` (FlyVis member parameters transplanted with the
gain rescaled and capped) before training; without it the solver starts from
flyvis's initialisation, the "from scratch on this wiring" control.

Runs on Linux (Modal) and, with `flydream.model` imported first for the
datamate patch, on the owner's Windows machine for timing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path


def compose(connectome_path: str, run: str, n_iters: int, batch_size: int, dt: float,
            delete_if_exists: bool = False):
    from hydra import compose as hydra_compose, initialize_config_dir
    from flyvis_cli.training.train_single import CONFIG_PATH, prepare_config

    with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.1"):
        args = hydra_compose(config_name="solver.yaml", overrides=[
            f"ensemble_and_network_id={run}", "task_name=flow", "train=true", "resume=false",
            f"delete_if_exists={'true' if delete_if_exists else 'false'}", "description=flydream_malecns",
            f"task.n_iters={n_iters}", f"task.batch_size={batch_size}", f"task.dataset.dt={dt}",
            f"network.connectome.file={connectome_path}", "network.connectome.n_syn_fill=0",
        ])
    return prepare_config(args)


def _fit_diagnostic_loaders(task, batch_size: int) -> None:
    """flyvis's one-batch diagnostic loaders (`val_batch`, `train_batch`) take
    the first `batch_size` sequences but keep `batch_size` as the loader's
    batch, and the checkpoint's `test()` builds the steady state at that
    number. With 16 validation sequences a batch of 32 dies in the first
    checkpoint ("size of tensor a (32) must match b (16)", the batch smoke
    of 2026-09-18). Shrink the loader's batch to what it can hold; the
    training loader (`drop_last=True`) and the full validation pass
    (batch 1) are untouched, so the measured training is the reference's."""
    from torch.utils.data import DataLoader
    from flyvis.utils.dataset_utils import IndexSampler

    for name, index in (("val_batch", task.val_seq_index), ("train_batch", task.train_seq_index)):
        n = min(batch_size, len(index))
        if n < batch_size:
            setattr(task, name, DataLoader(task.dataset, batch_size=n, sampler=IndexSampler(list(index)[:n])))


def train_member(connectome_path: str, run: str, n_iters: int, results_root: str, *,
                 batch_size: int = 4, dt: float = 0.02, init_path: str | None = None,
                 resume: bool = False, delete_if_exists: bool = False,
                 variant: str = "baseline", seed: int | None = None,
                 evidence_dir: str | None = None, profile: bool = False,
                 profile_wait: int = 2, profile_warmup: int = 1, profile_active: int = 3) -> dict:
    """Build, optionally initialise, train; return what happened and how long it took."""
    import torch
    from datamate import set_root_context
    from flyvis.solver import MultiTaskSolver
    from flydream.train.optimizations import optimized_solver

    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if profile and evidence_dir is None:
        raise ValueError("Profiling requires an evidence directory")

    os.makedirs(results_root, exist_ok=True)
    t0 = time.time()
    config = compose(connectome_path, run, n_iters, batch_size, dt, delete_if_exists)
    with set_root_context(results_root):
        solver = MultiTaskSolver(config=config, delete_if_exists=delete_if_exists)
    built = time.time() - t0
    _fit_diagnostic_loaders(solver.task, batch_size)
    if len(solver.task.train_data) == 0:
        raise ValueError("Batch size leaves zero training batches with drop_last=True")
    loaded = None
    if resume:
        solver.recover(network=True, decoder=True, optimizer=True, penalty=True,
                       checkpoint=-1, strict=True, force=False)
        loaded = "resumed from the last checkpoint"
    elif init_path:
        state = torch.load(init_path, map_location="cpu")
        missing, unexpected = solver.network.load_state_dict(state, strict=False)
        loaded = {"init": os.path.basename(init_path), "missing": len(missing), "unexpected": len(unexpected)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evidence = Path(evidence_dir) if evidence_dir else None
    if evidence:
        evidence.mkdir(parents=True, exist_ok=False)

    def state():
        return {"network": {k: v.detach().cpu().clone() for k, v in solver.network.state_dict().items()},
                "decoder": {name: {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                            for name, head in solver.decoder.items()}}

    if evidence:
        torch.save(state(), evidence / "initial.pt")
    before = int(solver.iteration)
    profiler = None
    if profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        def trace(p):
            p.export_chrome_trace(str(evidence / "trace.json"))
            (evidence / "operators.txt").write_text(p.key_averages().table(
                sort_by="self_cuda_time_total" if device == "cuda" else "self_cpu_time_total",
                row_limit=50), encoding="utf-8")
        profiler = torch.profiler.profile(activities=activities,
            schedule=torch.profiler.schedule(wait=profile_wait, warmup=profile_warmup,
                                            active=profile_active, repeat=1),
            on_trace_ready=trace, record_shapes=True, profile_memory=True)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with optimized_solver(solver, variant, profiler) as source_hash:
        with profiler if profiler is not None else nullcontext():
            t1 = time.perf_counter()
            solver.train(False)
            if device == "cuda":
                torch.cuda.synchronize()
            trained = time.perf_counter() - t1
    after = int(solver.iteration)
    completed = after - before
    if completed <= 0:
        raise ValueError("No optimizer iterations completed; timing would be invalid")
    if evidence:
        torch.save(state(), evidence / "final.pt")
        losses = [float(x) for x in solver.dir.loss[:]]
        (evidence / "losses.json").write_text(json.dumps(losses), encoding="utf-8")
    out = {
        "run": run, "connectome": os.path.basename(connectome_path), "n_iters": n_iters,
        "batch_size": batch_size, "dt": dt, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
        "n_nodes": int(solver.network.n_nodes), "n_edges": int(solver.network.n_edges),
        "build_s": round(built, 1), "train_s": round(trained, 1),
        "iteration_start": before, "iteration_end": after, "completed_iters": completed,
        "s_per_iter": round(trained / completed, 6), "init": loaded,
        "samples_per_s": round(completed * batch_size / trained, 4),
        "variant": variant, "seed": seed, "profiled": profile,
        "flyvis_train_source_sha256": source_hash,
        "timing_scope": "solver.train including steady states, logging and checkpoints; excludes build",
        "dir": str(solver.dir.path), "pid": os.getpid(),
    }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--connectome", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--n-iters", type=int, required=True)
    p.add_argument("--results-root", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--init", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--delete-if-exists", action="store_true")
    p.add_argument("--out", default=None, help="write the result json here")
    p.add_argument("--variant", choices=("baseline", "stats", "relu", "stats_relu"), default="baseline")
    p.add_argument("--seed", type=int)
    p.add_argument("--evidence-dir")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-wait", type=int, default=2)
    p.add_argument("--profile-warmup", type=int, default=1)
    p.add_argument("--profile-active", type=int, default=3)
    a = p.parse_args(argv)
    if sys.platform == "win32":
        from flydream.model import patch_datamate_for_windows
        patch_datamate_for_windows()
    out = train_member(a.connectome, a.run, a.n_iters, a.results_root, batch_size=a.batch_size, dt=a.dt,
                       init_path=a.init, resume=a.resume, delete_if_exists=a.delete_if_exists,
                       variant=a.variant, seed=a.seed, evidence_dir=a.evidence_dir, profile=a.profile,
                       profile_wait=a.profile_wait, profile_warmup=a.profile_warmup,
                       profile_active=a.profile_active)
    print(json.dumps(out), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
