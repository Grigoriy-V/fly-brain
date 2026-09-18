"""The ensemble map on the owner's machine: N map processes at once.

    python tools/map_local.py --dry                                  # the plan
    python tools/map_local.py --members 0,5 --seeds 0,1 --windows 0,0_2_4 --jobs 6
    python tools/map_local.py                                        # config.toml [decode]: members, splits, the first two lag_windows

Per member, one simulation (`flydream.decode.map --simulate-only --cache`,
624 s on this CPU for 189 Sintel clips, 1.3 GB of pairs) into
`data/decode/<date>_decode_sintel_pairs_m<member>/`, then one map process
per (member, seed, window) reading those pairs (`--pairs-from`), the same run
names `deploy/modal/decode_app.py` uses, so `flydream.decode.ensemble` reads
either. `--jobs` processes run concurrently (each numpy SVD is itself
multithreaded; 6 on 32 cores was the setting of 2026-09-18). Existing
`map.csv` / `pairs.npz` are skipped, so a stopped sweep resumes. Logs per job
under `data/decode/<run>/log.txt`, unbuffered.
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_name(member: int, seed: int, lags: list[int], date: str) -> str:
    return f"{date}_decode_sintel_m{member:03d}_s{seed}_lag_{'_'.join(str(x) for x in lags)}"


def pairs_run(member: int, date: str) -> str:
    return f"{date}_decode_sintel_pairs_m{member:03d}"


def _run(cmd: list[str], run: str) -> tuple[str, int, float]:
    out = ROOT / "data" / "decode" / run
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out / "log.txt", "w", encoding="utf-8") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                           env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    return run, p.returncode, time.time() - t0


def main(argv=None) -> int:
    s = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))["decode"]
    p = argparse.ArgumentParser()
    p.add_argument("--members", default="")
    p.add_argument("--seeds", default="")
    p.add_argument("--windows", default="", help="e.g. 0,0_2_4; default the first two of [decode] lag_windows")
    p.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args(argv)
    ms = [int(x) for x in a.members.split(",")] if a.members else list(s["members"])
    ss = [int(x) for x in a.seeds.split(",")] if a.seeds else list(s["splits"])
    ws = ([[int(x) for x in w.split("_")] for w in a.windows.split(",")] if a.windows
          else [list(w) for w in s["lag_windows"][:2]])
    jobs = [(m, sd, w) for m, sd, w in itertools.product(ms, ss, ws)]
    todo_sim = [m for m in ms if not (ROOT / "data" / "decode" / pairs_run(m, a.date) / "pairs.npz").exists()]
    todo_map = [(m, sd, w) for m, sd, w in jobs
                if not (ROOT / "data" / "decode" / run_name(m, sd, w, a.date) / "map.csv").exists()]
    print(f"{len(ms)} members x {len(ss)} splits x {len(ws)} windows = {len(jobs)} map jobs; "
          f"to do: {len(todo_sim)} simulations (~10 min each, {a.jobs} at once) and {len(todo_map)} maps "
          f"(~25 min each, {a.jobs} at once) ~ {(len(todo_sim) * 10 + len(todo_map) * 25) / a.jobs / 60:.1f} h")
    if a.dry:
        return 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        sims = [ex.submit(_run, [PY, "-u", "-m", "flydream.decode.map", "--stimuli", "sintel",
                                 "--model", f"flow/0000/{m:03d}", "--run", pairs_run(m, a.date),
                                 "--cache", "--simulate-only"], pairs_run(m, a.date)) for m in todo_sim]
        for f in sims:
            run, rc, dt = f.result()
            print(f"{time.strftime('%H:%M')} {run}: {'ok' if rc == 0 else f'exit {rc}'} in {dt / 60:.1f} min", flush=True)
        if any(f.result()[1] for f in sims):
            print("a simulation failed; see its log.txt"); return 1
        maps = [ex.submit(_run, [PY, "-u", "-m", "flydream.decode.map", "--stimuli", "sintel",
                                 "--model", f"flow/0000/{m:03d}", "--run", run_name(m, sd, w, a.date),
                                 "--pairs-from", pairs_run(m, a.date), "--lags", *map(str, w), "--seed", str(sd)],
                          run_name(m, sd, w, a.date)) for m, sd, w in todo_map]
        for f in maps:
            run, rc, dt = f.result()
            print(f"{time.strftime('%H:%M')} {run}: {'ok' if rc == 0 else f'exit {rc}'} in {dt / 60:.1f} min", flush=True)
    print(f"done in {(time.time() - t0) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
