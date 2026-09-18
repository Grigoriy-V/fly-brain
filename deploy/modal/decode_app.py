"""The decodability map on Modal: ten members x five splits x the lag windows.

    modal run deploy/modal/decode_app.py --dry                      # the plan and the price, no worker
    modal run deploy/modal/decode_app.py --members 0,5 --seeds 0,1  # a subset
    modal run --detach deploy/modal/decode_app.py                   # everything in config.toml [decode]
    modal volume get flydream-runs /decode/data/decode/<run> data/decode/<run>   # read back one run

Two kinds of worker. `simulate` (T4) runs one FlyVis member over the Sintel
clips and caches the stimulus/activity pairs on the Volume (1.3 GB per
member, `flydream.decode.pairs`). `map_window` (CPU, no GPU) fits the ridge
decoders of every cell type with both controls for one (member, split,
window) from those pairs; it is SVD-bound, so it runs on cores, not on a
card. The local entrypoint fans both out with `.map` and prints where the
results are. Every `map.csv` lands under `/decode/data/decode/<run>/` on
`flydream-runs`, one run per (member, seed, window), named
`<date>_decode_sintel_m<member>_s<seed>_lag_<lags>`, so `flydream.decode.sweep`
and the ensemble aggregation read them exactly as local runs.

The human allowed long decode jobs on Modal without a per-action gate
(2026-09-18: "если на локале так долго, разрешаю запускать такое на модале");
the price is still stated by `--dry` and in the report. Measured locally
(2026-09-18, cProfile on three types at the 80 ms window): 22 s per type, of
which SVD 58% and SSIM 17%; a full 65-type window is 12-47 min on the
owner's CPU.
"""
from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import sys
import time

import modal

APP_NAME = "flydream-decode"
DATA_VOL, RUNS_VOL = "flydream-data", "flydream-runs"
DATA, RUNS = "/data", "/runs"
DECODE_ROOT = f"{RUNS}/decode"           # FLYDREAM_ROOT on the worker: config.toml + data/decode/
MINUTES = 60
GPU = os.environ.get("FLYDREAM_GPU", "T4")
CPU_CORES = 8

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOL, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOL, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("flyvis==1.2.0", "hydra-core>=1.3", "h5py", "pyarrow", "pandas", "scipy",
                    "scikit-image", "matplotlib")
    .env({"FLYVIS_ROOT_DIR": f"{DATA}/flyvis", "FLYDREAM_ROOT": DECODE_ROOT, "PYTHONUNBUFFERED": "1"})
    .add_local_file("config.toml", "/opt/flydream/config.toml")
    .add_local_python_source("flydream")
)


def _prepare_root() -> None:
    """The worker's project root on the Volume: config.toml beside data/decode/."""
    os.makedirs(f"{DECODE_ROOT}/data/decode", exist_ok=True)
    shutil.copyfile("/opt/flydream/config.toml", f"{DECODE_ROOT}/config.toml")


def run_name(member: int, seed: int, lags: list[int], date: str) -> str:
    return f"{date}_decode_sintel_m{member:03d}_s{seed}_lag_{'_'.join(str(x) for x in lags)}"


def pairs_run(member: int, date: str) -> str:
    return f"{date}_decode_sintel_pairs_m{member:03d}"


def _map_cmd(model: str, run: str, lags: list[int], seed: int, pairs_from: str | None, cache: bool) -> list[str]:
    cmd = [sys.executable, "-m", "flydream.decode.map", "--stimuli", "sintel", "--model", model,
           "--run", run, "--lags", *map(str, lags), "--seed", str(seed)]
    if pairs_from:
        cmd += ["--pairs-from", pairs_from]
    if cache:
        cmd += ["--cache"]
    return cmd


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=4, memory=24576, timeout=120 * MINUTES)
def simulate(member: int, date: str) -> dict:
    """One member over the Sintel clips: cache pairs.npz (every cell type) on the
    Volume and stop; the fits are the CPU workers' job. (The first pilot passed
    `--types R1` to keep the worker short and got a pairs file holding R1 only:
    `--types` also restricts what is cached. `--simulate-only` is the right knob.)"""
    _prepare_root()
    run = pairs_run(member, date)
    t0 = time.time()
    if os.path.exists(f"{DECODE_ROOT}/data/decode/{run}/pairs.npz"):
        return {"member": member, "run": run, "cached": True, "s": 0.0}
    cmd = _map_cmd(f"flow/0000/{member:03d}", run, [0], 0, None, cache=True) + ["--simulate-only"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    runs_volume.commit()
    data_volume.commit()   # flyvis's Sintel rendering cache lives under FLYVIS_ROOT_DIR, built once
    if p.returncode != 0:
        raise RuntimeError(f"simulate {member} failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return {"member": member, "run": run, "cached": False, "s": round(time.time() - t0, 1), "gpu": GPU}


@app.function(image=image, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=CPU_CORES, memory=16384, timeout=180 * MINUTES)
def map_window(job: dict) -> dict:
    """One (member, seed, window): all cell types, both controls, from cached pairs."""
    _prepare_root()
    runs_volume.reload()
    member, seed, lags, date = job["member"], job["seed"], job["lags"], job["date"]
    run = run_name(member, seed, lags, date)
    t0 = time.time()
    if os.path.exists(f"{DECODE_ROOT}/data/decode/{run}/map.csv"):
        return {**job, "run": run, "cached": True, "s": 0.0}
    cmd = _map_cmd(f"flow/0000/{member:03d}", run, lags, seed, pairs_run(member, date), cache=False)
    p = subprocess.run(cmd, capture_output=True, text=True)
    runs_volume.commit()
    if p.returncode != 0:
        raise RuntimeError(f"map {run} failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    n_types = sum(1 for line in p.stdout.splitlines() if line.startswith("["))
    return {**job, "run": run, "cached": False, "s": round(time.time() - t0, 1), "n_types": n_types}


def _settings() -> dict:
    import tomllib
    with open("config.toml", "rb") as f:
        return tomllib.load(f)["decode"]


@app.local_entrypoint()
def main(members: str = "", seeds: str = "", windows: str = "", date: str = "", dry: bool = False):
    """Fan out: simulate each member on a T4, then every (member, seed, window) on CPUs."""
    s = _settings()
    ms = [int(x) for x in members.split(",")] if members else list(s["members"])
    ss = [int(x) for x in seeds.split(",")] if seeds else list(s["splits"])
    ws = [[int(x) for x in w.split("_")] for w in windows.split(",")] if windows else list(s["lag_windows"])
    date = date or time.strftime("%Y-%m-%d")
    jobs = [{"member": m, "seed": sd, "lags": w, "date": date} for m, sd, w in itertools.product(ms, ss, ws)]
    # price: T4 $0.59/h + 4 cores; CPU $0.0472/core/h, RAM $0.008/GiB/h (modal.com/pricing, 2026-09-18);
    # a window is ~25-50 min of 65 types (local: 22 s/type at 80 ms), a simulation ~10 min on a T4
    sim_h = len(ms) * 10 / 60
    map_h = len(jobs) * 35 / 60
    price = sim_h * (0.59 + 4 * 0.0472 + 24 * 0.008) + map_h * (CPU_CORES * 0.0472 + 16 * 0.008)
    print(f"{len(ms)} members x {len(ss)} splits x {len(ws)} windows = {len(jobs)} map jobs; "
          f"~{sim_h:.1f} T4-hours + ~{map_h:.0f} CPU-container-hours, about ${price:.0f}")
    if dry:
        return
    t0 = time.time()
    for r in simulate.map(ms, kwargs={"date": date}):
        took = "cached" if r["cached"] else f"{r['s']:.0f} s"
        print(f"simulated member {r['member']:03d}: {took}", flush=True)
    for r in map_window.map(jobs, order_outputs=False):
        took = "cached" if r["cached"] else f"{r['s']:.0f} s, {r.get('n_types')} types"
        print(f"{r['run']}: {took}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.0f} min; read back with "
          f"modal volume get flydream-runs /decode/data/decode/{date}_decode_sintel_m<member>_s<seed>_lag_<lags> data/decode/")
