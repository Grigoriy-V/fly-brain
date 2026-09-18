"""Paired full-solver benchmark. Default CLI only prints the plan, never trains.

Execution is CUDA/Modal-only and requires explicit per-action permission.
References: reports/2026-09-18_training_optimization_bench.md.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib


def settings(path):
    s = tomllib.loads(Path(path).read_text(encoding="utf-8"))["benchmark"]
    if set(s["variants"]) != {"baseline", "stats", "relu", "stats_relu"} or len(s["variants"]) != 4:
        raise ValueError("The benchmark requires four distinct factorial variants")
    if min(s["n_iters"], s["batch_size"], s["repeats"], s["profile_active"]) < 1:
        raise ValueError("Benchmark counts must be positive")
    if s["profile_wait"] + s["profile_warmup"] + s["profile_active"] > s["n_iters"]:
        raise ValueError("Profile schedule exceeds requested iterations")
    return s


def plan(s):
    jobs = []
    for repeat in range(s["repeats"]):
        order = s["variants"] if repeat % 2 == 0 else list(reversed(s["variants"]))
        jobs.extend(dict(variant=v, repeat=repeat, profile=False) for v in order)
    jobs.append(dict(variant="baseline", repeat=0, profile=True))
    return jobs


def compare_states(reference, candidate, rtol, atol):
    import torch
    rows = []
    def visit(a, b, prefix):
        if isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError(f"State keys differ: {prefix}")
            for k in a:
                visit(a[k], b[k], f"{prefix}/{k}")
        else:
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"State shape/dtype differs: {prefix}")
            delta = (a.double() - b.double()).abs()
            rows.append({"tensor": prefix, "max_abs": float(delta.max()) if delta.numel() else 0.0,
                         "close": bool(torch.allclose(a, b, rtol=rtol, atol=atol))})
    visit(reference, candidate, "")
    return {"max_abs": max((r["max_abs"] for r in rows), default=0.0),
            "close": all(r["close"] for r in rows), "tensors": rows}


def execute(config, connectome, root):
    import modal
    import torch
    if modal.is_local() or not torch.cuda.is_available():
        raise RuntimeError("Benchmark execution is restricted to an authorized Modal CUDA worker")
    s = settings(config)
    source_paths = [Path(__file__), Path(__file__).with_name("member.py"),
                    Path(__file__).with_name("optimizations.py")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    identity = {"settings": s, "sources": hashes,
                "connectome_sha256": hashlib.sha256(Path(connectome).read_bytes()).hexdigest()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    run = datetime.now(timezone.utc).strftime("%Y-%m-%d_bench_%H%M%S_%f_") + digest
    out = Path(root) / run
    out.mkdir(parents=True, exist_ok=False)
    jobs = plan(s)
    manifest = {"run": run, **identity, "jobs": jobs, "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Isolate datamate/render caches from earlier measurements; all jobs share
    # this new root. An unscored warm-up builds it once before any timed job.
    results_root = out / "results"
    records = []
    for i, job in enumerate([dict(variant="baseline", repeat=-1, profile=False)] + jobs):
        tag = "warmup" if i == 0 else f"{i:02d}_{job['variant']}_r{job['repeat']}" + ("_profile" if job["profile"] else "")
        dest = out / tag
        dest.mkdir()
        cmd = [sys.executable, "-m", "flydream.train.member", "--connectome", connectome,
               "--run", f"0100/{i:03d}", "--n-iters", str(1 if i == 0 else s["n_iters"]),
               "--results-root", str(results_root), "--batch-size", str(s["batch_size"]),
               "--dt", str(s["dt"]), "--variant", job["variant"], "--seed", str(s["seed"]),
               "--out", str(dest / "result.json"), "--evidence-dir", str(dest / "evidence")]
        if job["profile"]:
            cmd += ["--profile", "--profile-wait", str(s["profile_wait"]),
                    "--profile-warmup", str(s["profile_warmup"]), "--profile-active", str(s["profile_active"])]
        (dest / "command.json").write_text(json.dumps(cmd), encoding="utf-8")
        start = time.perf_counter()
        with (dest / "stdout.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTHONHASHSEED": str(s["seed"])})
        if proc.returncode:
            raise RuntimeError(f"Benchmark stopped on failed {tag}; see {dest / 'stdout.log'}")
        rec = json.loads((dest / "result.json").read_text())
        rec.update(job, tag=tag, wall_s=time.perf_counter() - start)
        if i:
            records.append(rec)
        (out / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps({k: rec[k] for k in ("tag", "completed_iters", "s_per_iter", "wall_s")}), flush=True)
    comparisons = []
    for rec in records:
        if rec["profile"]:
            continue
        base = next(r for r in records if r["variant"] == "baseline" and not r["profile"] and r["repeat"] == rec["repeat"])
        if rec["completed_iters"] != base["completed_iters"]:
            raise RuntimeError("Unequal actual iteration counts invalidate the benchmark")
        checks = {}
        for phase in ("initial", "final"):
            a = torch.load(out / base["tag"] / "evidence" / f"{phase}.pt", weights_only=True)
            b = torch.load(out / rec["tag"] / "evidence" / f"{phase}.pt", weights_only=True)
            checks[phase] = compare_states(a, b, 0 if phase == "initial" else s["rtol"],
                                          0 if phase == "initial" else s["atol"])
        if not checks["initial"]["close"]:
            raise RuntimeError("Different initial states invalidate paired comparison")
        losses = []
        for record in (base, rec):
            loss_values = json.loads((out / record["tag"] / "evidence" / "losses.json").read_text())
            losses.append({"loss": torch.tensor(loss_values, dtype=torch.float64)})
        checks["losses"] = compare_states(losses[0], losses[1], s["rtol"], s["atol"])
        comparisons.append({"tag": rec["tag"], "control": base["tag"],
                            "speedup": base["train_s"] / rec["train_s"], **checks})
    (out / "comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    return {"run": run, "path": str(out), "records": records,
            "numerical_checks_pass": all(c["final"]["close"] and c["losses"]["close"] for c in comparisons),
            "note": "Short-run engineering check, not scientific validation or convergence evidence"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.toml")
    p.add_argument("--connectome")
    p.add_argument("--root")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.execute:
        if not a.connectome or not a.root:
            p.error("execution requires --connectome and --root")
        print(json.dumps(execute(a.config, a.connectome, a.root), indent=2))
    else:
        s = settings(a.config)
        print(json.dumps({"settings": s, "jobs": plan(s), "priced_worker_started": False}, indent=2))


if __name__ == "__main__":
    main()
