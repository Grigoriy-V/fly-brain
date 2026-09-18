"""Training the MaleCNS model on Modal: FlyVis's optic-flow task, the same solver.

Every function here is a priced worker and starts only on the human's explicit
word, per action, with the price stated first (AGENTS.md, Human gates). The
Modal account is the project's second one (profile `grigoriy98smile`).

    modal volume create flydream-data                                            # once
    modal volume create flydream-runs                                            # once
    modal volume put flydream-data data/ol/filters_R_w5wk50.json /ol/filters_R_w5wk50.json
    modal volume put flydream-data data/flyvis/SintelDataSet /flyvis/SintelDataSet     # 5.8 GB, once
    modal volume put flydream-data data/runs/init_R_w5wk50_m000.pt /init/init_R_w5wk50_m000.pt
    modal run deploy/modal/train_app.py::smoke --connectome filters_R_w5wk50.json       # 50 iterations, the timing
    modal run --detach deploy/modal/train_app.py::train --connectome filters_R_w5wk50.json --run 0100/000 --init init_R_w5wk50_m000.pt
    modal run --detach deploy/modal/train_app.py::train_ensemble --connectome ... --ensemble 0100 --members 8

State on the Volumes:

    flydream-data   /ol/<export>.json      the connectome export (flydream.data.export)
                    /flyvis/SintelDataSet  the stimuli, as flyvis expects them
                    /init/<state>.pt       a transplanted network state (flydream.model.init_state)
    flydream-runs   /results/flow/<ensemble>/<member>/   flyvis's own NetworkDir: chkpts, loss, config
                    /results/ConnectomeFromAvgFilters_*  datamate's cache of the built connectome
                    /results/renderings/                 the rendered Sintel, built once per dt

The solver is flyvis's `MultiTaskSolver` composed from its own Hydra config with
three overrides (the connectome file, `n_syn_fill=0`, the iteration count), the
same path `flyvis train-single` takes, so the loss, schedule, augmentation and
checkpointing are Lappalainen et al.'s and not this project's. What this project
adds is the connectome and, optionally, the starting point: `--init` loads a
network state made by `flydream.model.init_state` (FlyVis member parameters
transplanted with the gain rescaled and capped, DECISIONS 2026-09-18) before
training starts; without it the solver uses flyvis's own initialisation
(`syn_strength = 0.01 / n_syn`), which is the "trained from scratch on MaleCNS
wiring" control.

Price. The reference trains 250,000 iterations at batch 4 x 19 frames; the
MaleCNS export is 0.55x FlyVis's edge count. The local CPU timing is in
`reports/` (step 3); the GPU number comes from `smoke`, which is the first
priced call and is small. Nothing here is estimated in this docstring: the
report states the price before `train` is asked for.
"""
from __future__ import annotations

import os
import time

import modal

APP_NAME = "flydream-train"
DATA_VOL, RUNS_VOL = "flydream-data", "flydream-runs"
DATA, RUNS = "/data", "/runs"
MINUTES = 60
GPU = os.environ.get("FLYDREAM_GPU", "A100-40GB")

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOL, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOL, create_if_missing=True)

# flyvis 1.2.0 with its pinned torch on Linux (CUDA wheels from PyPI); the
# flydream package rides along so the transplant and the settings are the
# repository's, not a copy.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("flyvis==1.2.0", "hydra-core>=1.3", "h5py", "pyarrow", "pandas", "scipy")
    .env({"FLYVIS_ROOT_DIR": f"{DATA}/flyvis", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("flydream")
)


def _compose(connectome_path: str, run: str, n_iters: int, batch_size: int, dt: float):
    """flyvis's solver config with this project's overrides, as train_single composes it."""
    from hydra import compose, initialize_config_dir
    from flyvis_cli.training.train_single import CONFIG_PATH, prepare_config

    with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.1"):
        args = compose(config_name="solver.yaml", overrides=[
            f"ensemble_and_network_id={run}", "task_name=flow", "train=true", "resume=false",
            "delete_if_exists=false", "description=flydream_malecns",
            f"task.n_iters={n_iters}", f"task.batch_size={batch_size}", f"task.dataset.dt={dt}",
            f"network.connectome.file={connectome_path}", "network.connectome.n_syn_fill=0",
        ])
    return prepare_config(args)


def _train(connectome: str, run: str, n_iters: int, batch_size: int, dt: float,
           init: str | None, resume: bool) -> dict:
    """Build the solver under the runs Volume, optionally load a transplanted
    state, train, and return what happened. Shared by smoke, train and the
    ensemble members."""
    import torch
    from datamate import set_root_context
    from flyvis.solver import MultiTaskSolver

    root = f"{RUNS}/results"
    os.makedirs(root, exist_ok=True)
    connectome_path = f"{DATA}/ol/{connectome}"
    t0 = time.time()
    config = _compose(connectome_path, run, n_iters, batch_size, dt)
    with set_root_context(root):
        solver = MultiTaskSolver(config=config, delete_if_exists=False)
    built = time.time() - t0
    loaded = None
    if resume:
        solver.recover(network=True, decoder=True, optimizer=True, penalty=True,
                       checkpoint=-1, strict=True, force=False)
        loaded = "resumed from the last checkpoint"
    elif init:
        state = torch.load(f"{DATA}/init/{init}", map_location=solver.network.device if hasattr(solver.network, "device") else "cpu")
        missing, unexpected = solver.network.load_state_dict(state, strict=False)
        loaded = {"init": init, "missing": list(missing), "unexpected": list(unexpected)}
    t1 = time.time()
    solver.train(False)
    trained = time.time() - t1
    runs_volume.commit()
    out = {
        "run": run, "connectome": connectome, "n_iters": n_iters, "batch_size": batch_size, "dt": dt,
        "gpu": GPU, "n_nodes": int(solver.network.n_nodes), "n_edges": int(solver.network.n_edges),
        "build_s": round(built, 1), "train_s": round(trained, 1),
        "s_per_iter": round(trained / max(1, n_iters), 4), "init": loaded,
        "dir": str(solver.dir.path),
    }
    print(out, flush=True)
    return out


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=4, memory=16384, timeout=30 * MINUTES)
def smoke(connectome: str = "filters_R_w5wk50.json", n_iters: int = 50, batch_size: int = 4,
          dt: float = 0.02, init: str | None = None) -> dict:
    """The timing run: a few iterations on the GPU so the price of `train` is a
    measurement, not an extrapolation. Also builds the connectome cache and the
    Sintel rendering on the Volume, which `train` then reuses."""
    return _train(connectome, "9999/000", n_iters, batch_size, dt, init, resume=False)


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=4, memory=16384, timeout=24 * 60 * MINUTES)
def train(connectome: str, run: str, n_iters: int = 250_000, batch_size: int = 4,
          dt: float = 0.02, init: str | None = None, resume: bool = False) -> dict:
    """One member, the reference's full schedule by default. `run` is
    `<ensemble>/<member>`, e.g. `0100/000`; the ensemble id is this project's
    (the reference's are 0000-0099 in the pretrained release)."""
    return _train(connectome, run, n_iters, batch_size, dt, init, resume)


@app.function(image=image, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, timeout=24 * 60 * MINUTES)
def train_ensemble(connectome: str, ensemble: str = "0100", members: int = 8, n_iters: int = 250_000,
                   batch_size: int = 4, dt: float = 0.02, init: str | None = None) -> list:
    """N members in parallel, one GPU each, from different random seeds (the
    solver seeds from the member id). The count is the human's word."""
    runs = [f"{ensemble}/{m:03d}" for m in range(members)]
    return list(train.starmap([(connectome, r, n_iters, batch_size, dt, init, False) for r in runs]))


@app.local_entrypoint()
def main(connectome: str = "filters_R_w5wk50.json", n_iters: int = 50):
    """`modal run deploy/modal/train_app.py` is the smoke by default."""
    print(smoke.remote(connectome=connectome, n_iters=n_iters))
