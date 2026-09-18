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


def train_member(connectome_path: str, run: str, n_iters: int, results_root: str, *,
                 batch_size: int = 4, dt: float = 0.02, init_path: str | None = None,
                 resume: bool = False, delete_if_exists: bool = False) -> dict:
    """Build, optionally initialise, train; return what happened and how long it took."""
    import torch
    from datamate import set_root_context
    from flyvis.solver import MultiTaskSolver

    os.makedirs(results_root, exist_ok=True)
    t0 = time.time()
    config = compose(connectome_path, run, n_iters, batch_size, dt, delete_if_exists)
    with set_root_context(results_root):
        solver = MultiTaskSolver(config=config, delete_if_exists=delete_if_exists)
    built = time.time() - t0
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
    t1 = time.time()
    solver.train(False)
    trained = time.time() - t1
    out = {
        "run": run, "connectome": os.path.basename(connectome_path), "n_iters": n_iters,
        "batch_size": batch_size, "dt": dt, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "n_nodes": int(solver.network.n_nodes), "n_edges": int(solver.network.n_edges),
        "build_s": round(built, 1), "train_s": round(trained, 1),
        "s_per_iter": round(trained / max(1, n_iters), 4), "init": loaded,
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
    a = p.parse_args(argv)
    if sys.platform == "win32":
        from flydream.model import patch_datamate_for_windows
        patch_datamate_for_windows()
    out = train_member(a.connectome, a.run, a.n_iters, a.results_root, batch_size=a.batch_size, dt=a.dt,
                       init_path=a.init, resume=a.resume, delete_if_exists=a.delete_if_exists)
    print(json.dumps(out), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
