"""Training the MaleCNS model on Modal: FlyVis's optic-flow task, the same solver.

Every function here is a priced worker and starts only on the human's explicit
word, per action, with the price stated first (AGENTS.md, Human gates). The
Modal account is the project's second one (profile `grigoriy98smile`). The GPU
is a T4, L4 as the alternative, nothing above (the human, 2026-09-18).

    modal volume put flydream-data data/ol/filters_R_w5wk50m500oc.json /ol/filters_R_w5wk50m500oc.json
    modal volume put flydream-data data/flyvis/SintelDataSet /flyvis/SintelDataSet     # 5.8 GB, once
    modal volume put flydream-data data/runs/init_w5wk50m500oc_m000.pt /init/init_w5wk50m500oc_m000.pt
    modal run deploy/modal/train_app.py::smoke --n-iters 50                    # one member, the timing
    modal run deploy/modal/train_app.py::smoke_packed --members 4 --n-iters 50 # four members on one card
    modal run --detach deploy/modal/train_app.py::train_packed --ensemble 0100 --members 4 --init init_w5wk50m500oc_m000.pt
    modal run --detach deploy/modal/train_app.py::train --run 0100/000 --init init_w5wk50m500oc_m000.pt

State on the Volumes:

    flydream-data   /ol/<export>.json      the connectome export (flydream.data.export)
                    /flyvis/SintelDataSet  the stimuli, as flyvis expects them
                    /init/<state>.pt       a transplanted network state (flydream.model.init_state)
    flydream-runs   /results/flow/<ensemble>/<member>/   flyvis's own NetworkDir: chkpts, loss, config
                    /results/ConnectomeFromAvgFilters_*  datamate's cache of the built connectome
                    /results/renderings/                 the rendered Sintel, built once per dt
                    /results/packed/<ensemble>/<member>.json   the timing record of each packed member

Packing. The model is small (31,526 nodes, 1.4M edges, about a gigabyte of
graph per iteration) and its computation is sequential over 40 time steps, so
one member leaves a T4 mostly idle. `smoke_packed` and `train_packed` run N
members as N processes in one container sharing the card; the per-member
cost is what `smoke_packed` measures at N = 1, 2, 4, 8, and the report picks
N by dollars per 250,000 iterations per member, not by assumption. The first
member runs a warm-up alone so the connectome cache and the Sintel rendering
are built once, then the rest start.

The solver is flyvis's `MultiTaskSolver` composed from its own Hydra config
(`flydream.train.member`), the same path `flyvis train-single` takes; the
loss, schedule, augmentation and checkpointing are Lappalainen et al.'s.
`--init` starts from a transplanted state; without it the solver starts from
flyvis's own initialisation, the "from scratch on MaleCNS wiring" control.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import modal

APP_NAME = "flydream-train"
DATA_VOL, RUNS_VOL = "flydream-data", "flydream-runs"
DATA, RUNS = "/data", "/runs"
RESULTS = f"{RUNS}/results"
MINUTES = 60
GPU = os.environ.get("FLYDREAM_GPU", "T4")
CONNECTOME = "filters_R_w5wk50m500oc.json"

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOL, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOL, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("flyvis==1.2.0", "hydra-core>=1.3", "h5py", "pyarrow", "pandas", "scipy")
    .env({"FLYVIS_ROOT_DIR": f"{DATA}/flyvis", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("flydream")
)


def _member_cmd(connectome: str, run: str, n_iters: int, batch_size: int, dt: float,
                init: str | None, resume: bool, out: str | None) -> list[str]:
    cmd = [sys.executable, "-m", "flydream.train.member", "--connectome", f"{DATA}/ol/{connectome}",
           "--run", run, "--n-iters", str(n_iters), "--results-root", RESULTS,
           "--batch-size", str(batch_size), "--dt", str(dt)]
    if init:
        cmd += ["--init", f"{DATA}/init/{init}"]
    if resume:
        cmd += ["--resume"]
    if out:
        cmd += ["--out", out]
    return cmd


def _run_one(connectome: str, run: str, n_iters: int, batch_size: int, dt: float,
             init: str | None, resume: bool) -> dict:
    from flydream.train.member import train_member

    out = train_member(f"{DATA}/ol/{connectome}", run, n_iters, RESULTS, batch_size=batch_size, dt=dt,
                       init_path=f"{DATA}/init/{init}" if init else None, resume=resume)
    out["gpu_spec"] = GPU
    runs_volume.commit()
    print(json.dumps(out), flush=True)
    return out


def _run_packed(connectome: str, runs: list[str], n_iters: int, batch_size: int, dt: float,
                init: str | None, warmup_iters: int = 2) -> list[dict]:
    """N members as N processes on one card. A warm-up builds the shared caches first."""
    rec_dir = f"{RESULTS}/packed/{runs[0].split('/')[0]}"
    os.makedirs(rec_dir, exist_ok=True)
    t0 = time.time()
    # the warm-up dir is reused across calls, so it must be allowed to overwrite:
    # the second smoke_packed of 2026-09-18 died here because flyvis refuses an
    # existing NetworkDir unless told to delete it
    warm = subprocess.run(_member_cmd(connectome, "9998/000", warmup_iters, batch_size, dt, None, False, None)
                          + ["--delete-if-exists"], capture_output=True, text=True)
    if warm.returncode != 0:
        raise RuntimeError(f"warm-up failed:\n{warm.stdout[-3000:]}\n{warm.stderr[-3000:]}")
    warm_s = time.time() - t0
    procs, outs = [], []
    t1 = time.time()
    for r in runs:
        out = f"{rec_dir}/{r.split('/')[-1]}.json"
        outs.append(out)
        procs.append(subprocess.Popen(_member_cmd(connectome, r, n_iters, batch_size, dt, init, False, out),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    logs = [p.communicate()[0] for p in procs]
    wall = time.time() - t1
    results = []
    for r, out, p, log in zip(runs, outs, procs, logs):
        if p.returncode == 0 and os.path.exists(out):
            rec = json.load(open(out))
        else:
            rec = {"run": r, "error": f"exit {p.returncode}", "log_tail": log[-2000:]}
        rec.update({"packed": len(runs), "wall_s_all": round(wall, 1), "warmup_s": round(warm_s, 1), "gpu_spec": GPU})
        results.append(rec)
    runs_volume.commit()
    ok = [x for x in results if "s_per_iter" in x]
    if ok:
        per = sum(x["s_per_iter"] for x in ok) / len(ok)
        print(f"packed {len(runs)} on {GPU}: mean {per:.3f} s/iter per member, wall {wall:.0f}s for {n_iters} iters each "
              f"-> effective {wall / max(1, n_iters) / len(runs):.4f} card-seconds per member-iteration", flush=True)
    for x in results:
        print(json.dumps(x)[:600], flush=True)
    return results


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=4, memory=16384, timeout=30 * MINUTES)
def smoke(connectome: str = CONNECTOME, n_iters: int = 50, batch_size: int = 4, dt: float = 0.02,
          init: str | None = None) -> dict:
    """One member for a few iterations: builds the caches, measures s/iter on the card."""
    return _run_one(connectome, "9999/000", n_iters, batch_size, dt, init, resume=False)


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=8, memory=32768, timeout=60 * MINUTES)
def smoke_packed(connectome: str = CONNECTOME, members: int = 4, n_iters: int = 50, batch_size: int = 4,
                 dt: float = 0.02, init: str | None = None) -> list:
    """N members at once on one card, a few iterations each: the packing measurement."""
    runs = [f"9997/{m:03d}" for m in range(members)]
    return _run_packed(connectome, runs, n_iters, batch_size, dt, init)


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=4, memory=16384, timeout=60 * MINUTES)
def smoke_batch(connectome: str = CONNECTOME, batches: str = "4,8,16,32", n_iters: int = 30,
                dt: float = 0.02) -> list:
    """One member at several batch sizes, a few iterations each: does an
    iteration cost the same at batch 32 as at batch 4? If the per-step kernels
    are latency-bound (the hypothesis), samples per second scale almost
    linearly with the batch and the reference's sample budget (250k x 4) is
    reached in a fraction of the iterations. The optimisation is then a
    different one (larger batch, fewer steps) and must be validated, not
    assumed; this only prices it."""
    # One fresh process per batch size. Two solvers in one process share
    # something sized to the first batch (the run of 2026-09-18 died with
    # "size of tensor a (32) must match b (16) at dimension 0" from batch 32
    # on), and a subprocess also makes an OOM at a large batch a recorded
    # result instead of the end of the container.
    rec_dir = f"{RESULTS}/packed/9996"
    os.makedirs(rec_dir, exist_ok=True)
    out = []
    for i, b in enumerate(int(x) for x in batches.split(",")):
        rec = f"{rec_dir}/batch_{b}.json"
        cmd = _member_cmd(connectome, f"9996/{i:03d}", n_iters, b, dt, None, False, rec) + ["--delete-if-exists"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0 and os.path.exists(rec):
            r = json.load(open(rec))
            r["samples_per_s"] = round(b / max(r["s_per_iter"], 1e-9), 1)
        else:
            tail = (p.stdout + p.stderr)[-1500:]
            err = "OOM" if "out of memory" in tail.lower() else tail[-300:]
            r = {"batch_size": b, "error": err, "s_per_iter": None, "samples_per_s": None, "train_s": None}
        r["gpu_spec"] = GPU
        print(json.dumps({k: r.get(k) for k in ("batch_size", "s_per_iter", "samples_per_s", "train_s", "peak_mem_gb", "gpu", "error")}), flush=True)
        out.append(r)
    runs_volume.commit()
    return out


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=4, memory=16384, timeout=24 * 60 * MINUTES)
def train(connectome: str = CONNECTOME, run: str = "0100/000", n_iters: int = 250_000, batch_size: int = 4,
          dt: float = 0.02, init: str | None = None, resume: bool = False) -> dict:
    """One member, the reference's full schedule by default, alone on its card."""
    return _run_one(connectome, run, n_iters, batch_size, dt, init, resume)


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=8, memory=32768, timeout=24 * 60 * MINUTES)
def train_packed(connectome: str = CONNECTOME, ensemble: str = "0100", members: int = 4, first: int = 0,
                 n_iters: int = 250_000, batch_size: int = 4, dt: float = 0.02, init: str | None = None) -> list:
    """`members` members `first..first+members-1` of `ensemble`, packed on one card."""
    runs = [f"{ensemble}/{m:03d}" for m in range(first, first + members)]
    return _run_packed(connectome, runs, n_iters, batch_size, dt, init)


@app.local_entrypoint()
def main(connectome: str = CONNECTOME, n_iters: int = 50):
    """`modal run deploy/modal/train_app.py` is the single-member smoke."""
    print(smoke.remote(connectome=connectome, n_iters=n_iters))


@app.function(image=image.add_local_file("config.toml", "/opt/flydream-config.toml"),
              gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=4, memory=16384, timeout=30 * MINUTES)
def benchmark(connectome: str = CONNECTOME):
    """Four variants x two repeats + a separate profile. Priced: explicit gate."""
    from flydream.train.benchmark import execute
    try:
        return execute("/opt/flydream-config.toml", f"{DATA}/ol/{connectome}",
                       f"{RUNS}/benchmarks")
    finally:
        # Preserve diagnostics on failure as well as on success.
        runs_volume.commit()
