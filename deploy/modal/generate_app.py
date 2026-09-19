"""The inversion ladder on a T4: every stage of one clip, FlyVis or MaleCNS.

    modal run deploy/modal/generate_app.py --model flow/0000/000 --sample 3
    modal run deploy/modal/generate_app.py --model malecns --sample 3

One worker loads the model once, takes the clip from the Sintel set on the
Volume, simulates the target state, and inverts it from each stage in turn
(`flydream.generate.invert.run_ladder`); the recovered videos come back in
the return value (a few hundred KB) and are written locally under
`data/generate/<tag>/`, where `flydream.generate.figures` draws the ladder.
Nothing large moves. On the owner's CPU one Adam step over 20 frames took
2-5 s under load (a ladder of eight stages: an hour); on a T4 the same is
well under 0.1 s (minutes, ~$0.05). The GPU is used because the job is one
where it is more than 4× faster (DECISIONS 2026-09-18, night).

Window: `frames + margin` from `config.toml [generate]` (40 + 5 since ROADMAP
item 9, 2026-09-19); the 20-frame ladders of 2026-09-18 had no margin. Pass
--frames/--margin to override; -1 means the config value.
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import time
import tomllib
from pathlib import Path

import modal

APP_NAME = "flydream-generate"
DATA_VOL, RUNS_VOL = "flydream-data", "flydream-runs"
DATA, RUNS = "/data", "/runs"
GEN_ROOT = f"{RUNS}/generate"
MINUTES = 60
GPU = os.environ.get("FLYDREAM_GPU", "T4")
CONNECTOME = "filters_R_w5wk50m500oc.json"
STAGES = [["R1"], ["L1"], ["L3"], ["Mi1"], ["Mi4"], ["Tm5a"], ["Tm9"], ["T4a"], ["T5a"],
          ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]]

def _generate_settings_all() -> dict:
    """config.toml [generate] with its subsections: the repository copy locally, the image copy on the worker."""
    for c in (Path(__file__).resolve().parent.parent.parent / "config.toml", Path("/opt/flydream/config.toml")):
        if c.exists():
            return tomllib.loads(c.read_text(encoding="utf-8")).get("generate", {})
    return {}


def _generate_settings() -> dict:
    return _generate_settings_all()


GEN = _generate_settings()

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOL, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOL, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("flyvis==1.2.0", "hydra-core>=1.3", "h5py", "pyarrow", "pandas", "scipy", "matplotlib")
    .env({"FLYVIS_ROOT_DIR": f"{DATA}/flyvis", "FLYDREAM_ROOT": GEN_ROOT, "PYTHONUNBUFFERED": "1"})
    .add_local_file("config.toml", "/opt/flydream/config.toml")
    .add_local_python_source("flydream")
)


def _prepare_root() -> None:
    os.makedirs(f"{GEN_ROOT}/data/ol", exist_ok=True)
    os.makedirs(f"{GEN_ROOT}/data/generate", exist_ok=True)
    shutil.copyfile("/opt/flydream/config.toml", f"{GEN_ROOT}/config.toml")
    src, dst = f"{DATA}/ol/{CONNECTOME}", f"{GEN_ROOT}/data/ol/{CONNECTOME}"
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=1, memory=3072, timeout=60 * MINUTES)
def ladder(model: str = "flow/0000/000", sample: int = 3, frames: int = 40, margin: int = 5, steps: int = 150,
           lr: float = 0.05, tv: float = 0.02, dt: float = 0.02, t_pre: float = 1.0, stages: str = "",
           batch: int = 0, plateau_steps: int = 0, plateau_tol: float = 0.0, plateau_floor: float = 1e-3) -> list[dict]:
    import numpy as np

    _prepare_root()          # before the import: flydream.model reads ROOT/config.toml on import
    from flydream.generate.invert import run_ladder

    st = [s.split("+") for s in stages.split(",")] if stages else STAGES
    prefix = f"{time.strftime('%Y-%m-%d')}_{'malecns' if model.startswith('malecns') else 'flyvis'}_"
    t0 = time.time()
    records = run_ladder(model, sample, st, frames=frames, margin=margin, steps=steps, lr=lr, tv=tv, dt=dt, t_pre=t_pre,
                         out_root=Path(GEN_ROOT) / "data" / "generate", tag_prefix=prefix,
                         batch=batch, plateau_steps=plateau_steps, plateau_tol=plateau_tol, plateau_floor=plateau_floor)
    runs_volume.commit()
    out = []
    for r in records:
        buf = io.BytesIO()
        np.savez_compressed(buf, **r["arrays"])
        out.append({**{k: v for k, v in r.items() if k != "arrays"}, "npz_b64": base64.b64encode(buf.getvalue()).decode()})
    print(f"ladder of {len(out)} stages in {time.time() - t0:.0f} s on {GPU}", flush=True)
    return out


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=1, memory=3072, timeout=60 * MINUTES)
def dreams(model: str = "malecns", sources: str = "eye_noise,flash,dark_after,neuron_noise", frames: int = 40,
           margin: int = 5, steps: int = 150, lr: float = 0.05, tv: float = 0.02, dt: float = 0.02,
           t_pre: float = 1.0, stages: str = "", batch: int = 0, plateau_steps: int = 0, plateau_tol: float = 0.0,
           plateau_floor: float = 1e-3, seed: int = 0) -> list[dict]:
    """ROADMAP item 11: every source in `sources`, every stage, inversion beside
    its shuffled-state control, one batch per source."""
    import numpy as np

    _prepare_root()
    from flydream.generate.dreams import run_dreams

    st = [s.split("+") for s in stages.split(",")] if stages else STAGES
    prefix = f"{time.strftime('%Y-%m-%d')}_{'malecns' if model.startswith('malecns') else 'flyvis'}_"
    t0 = time.time()
    out = []
    for source in sources.split(","):
        records = run_dreams(model, source.strip(), st, frames=frames, margin=margin, steps=steps, lr=lr, tv=tv, dt=dt,
                             t_pre=t_pre, batch=batch, plateau_steps=plateau_steps, plateau_tol=plateau_tol,
                             plateau_floor=plateau_floor, seed=seed, out_root=Path(GEN_ROOT) / "data" / "generate",
                             tag_prefix=prefix)
        for r in records:
            buf = io.BytesIO()
            np.savez_compressed(buf, **r["arrays"])
            out.append({**{k: v for k, v in r.items() if k != "arrays"}, "npz_b64": base64.b64encode(buf.getvalue()).decode()})
    runs_volume.commit()
    print(f"dreams: {len(out)} records in {time.time() - t0:.0f} s on {GPU}", flush=True)
    return out


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=1, memory=3072, timeout=60 * MINUTES)
def mix(model: str = "malecns", sample_a: int = 3, sample_b: int = 10, frames: int = 40, margin: int = 5,
        steps: int = 150, lr: float = 0.05, tv: float = 0.02, dt: float = 0.02, t_pre: float = 1.0, stages: str = "",
        batch: int = 0, plateau_steps: int = 0, plateau_tol: float = 0.0, plateau_floor: float = 1e-3) -> list[dict]:
    """ROADMAP item 12: gain edits, interpolation and hybrids of two clips' states, one batch."""
    import numpy as np

    _prepare_root()
    from flydream.generate.mix import run_mix

    prefix = f"{time.strftime('%Y-%m-%d')}_{'malecns' if model.startswith('malecns') else 'flyvis'}_"
    t0 = time.time()
    records = run_mix(model, sample_a=sample_a, sample_b=sample_b, frames=frames, margin=margin, steps=steps, lr=lr,
                      tv=tv, dt=dt, t_pre=t_pre, batch=batch, plateau_steps=plateau_steps, plateau_tol=plateau_tol,
                      plateau_floor=plateau_floor, out_root=Path(GEN_ROOT) / "data" / "generate", tag_prefix=prefix)
    runs_volume.commit()
    out = []
    for r in records:
        buf = io.BytesIO()
        np.savez_compressed(buf, **r["arrays"])
        out.append({**{k: v for k, v in r.items() if k != "arrays"}, "npz_b64": base64.b64encode(buf.getvalue()).decode()})
    print(f"mix: {len(out)} tasks in {time.time() - t0:.0f} s on {GPU}", flush=True)
    return out


@app.function(image=image, volumes={DATA: data_volume}, cpu=2, memory=6144, timeout=120 * MINUTES)
def pairs13_videos(procedural: str = "pairs13/procedural_800_s0.npz", frames: int = 45, dt: float = 0.02,
                   out: str = "pairs13/videos.npz") -> dict:
    """ROADMAP 13A data, the CPU half: render the augmented Sintel clips (flyvis
    caches the render on the data volume), append the uploaded procedural
    clips, write every video with its meta to /data/<out>. No GPU: AGENTS
    "a GPU function does GPU work only"."""
    import numpy as np

    _prepare_root()
    from flydream.generate.pairs13 import sintel_videos

    t0 = time.time()
    z = np.load(f"{DATA}/{procedural}")
    proc_v, proc_p = z["videos"], [json.loads(x) for x in z["params"]]
    print(f"procedural: {proc_v.shape}", flush=True)
    sin_v, scenes, names = sintel_videos(frames, dt)
    print(f"sintel augmented: {sin_v.shape}, {len(set(scenes))} scenes, {time.time() - t0:.0f} s", flush=True)
    videos = np.concatenate([sin_v, proc_v])
    meta = [{"source": "sintel", "scene": sc, "name": nm} for sc, nm in zip(scenes, names)] + \
           [{"source": "procedural", "class": p["class"], "params": p} for p in proc_p]
    np.savez(f"{DATA}/{out}", videos=videos, meta=np.array([json.dumps(m) for m in meta]))
    data_volume.commit()
    print(f"videos: {videos.shape} in {time.time() - t0:.0f} s (CPU)", flush=True)
    return {"n": int(len(videos)), "n_sintel": int(len(sin_v)), "n_procedural": int(len(proc_v)), "seconds": round(time.time() - t0, 1)}


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=1, memory=4096, timeout=120 * MINUTES)
def pairs13(videos_file: str = "pairs13/videos.npz", model: str = "malecns", frames: int = 45,
            dt: float = 0.02, t_pre: float = 1.0, sim_batch: int = 32, shard: int = 512, seed: int = 0,
            held_scenes: str = "", held_classes: str = "", val_fraction: float = 0.1, out: str = "pairs13") -> dict:
    """ROADMAP 13A data, the GPU half: the assembled videos of `pairs13_videos`
    simulated through model zero; the 14 ladder types' activity written
    float16 in shards under /runs/<out>/ with a manifest (meta per clip, the
    split). Starts from finished inputs; the card's utilisation is sampled."""
    import numpy as np

    _prepare_root()
    from flydream.generate.invert import GpuSampler, load_network
    from flydream.generate.pairs13 import LADDER_TYPES, simulate_states, split_indices
    from flydream.decode import pairs as P

    t0 = time.time()
    z = np.load(f"{DATA}/{videos_file}")
    videos, meta = z["videos"], [json.loads(x) for x in z["meta"]]
    sin_v = [m for m in meta if m["source"] == "sintel"]; proc_v = [m for m in meta if m["source"] == "procedural"]
    print(f"videos: {videos.shape} ({len(sin_v)} sintel, {len(proc_v)} procedural)", flush=True)
    split = split_indices(meta, held_scenes.split(",") if held_scenes else [], held_classes.split(",") if held_classes else [],
                          val_fraction, seed)
    net = load_network(model)
    _, index = P.type_index(net.connectome)
    cells = np.concatenate([index[t] for t in LADDER_TYPES])
    type_of = np.concatenate([[t] * len(index[t]) for t in LADDER_TYPES])
    outdir = Path(RUNS) / out
    outdir.mkdir(parents=True, exist_ok=True)
    shards = []
    with GpuSampler() as gpu:
        for i in range(0, len(videos), shard):
            act = simulate_states(net, videos[i:i + shard], cells, dt, t_pre, sim_batch)
            f = outdir / f"shard_{i // shard:03d}.npz"
            np.savez(f, videos=videos[i:i + shard], states=act, index=np.arange(i, i + len(act)))
            shards.append(f.name)
            print(f"  {f.name}: {act.shape} ({time.time() - t0:.0f} s)", flush=True)
    manifest = {"model": model, "frames": frames, "dt": dt, "t_pre": t_pre, "types": LADDER_TYPES,
                "cells": cells.tolist(), "type_of_cell": type_of.tolist(), "n": int(len(videos)),
                "n_sintel": int(len(sin_v)), "n_procedural": int(len(proc_v)), "shards": shards, "shard_size": shard,
                "meta": meta, "split": split, "seconds": round(time.time() - t0, 1), "gpu": GPU,
                "gpu_utilisation": gpu.mean}
    print(f"GPU utilisation over the simulation: {gpu.mean}", flush=True)
    (outdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    runs_volume.commit()
    print(f"pairs13: {len(videos)} clips in {len(shards)} shards, {time.time() - t0:.0f} s on {GPU}", flush=True)
    return {k: v for k, v in manifest.items() if k not in ("meta", "cells", "type_of_cell")} | {
        "split_sizes": {k: len(v) for k, v in split.items()}}


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=2, memory=12288, timeout=120 * MINUTES)
def train13(run: str = "pairs13", conditions: str = "early,deep,all", epochs: int = 12, batch: int = 16,
            lr: float = 2e-3, width: int = 32, depth: int = 3, rings: int = 1, taps: int = 5, frames: int = 40,
            margin: int = 5, n_roundtrip: int = 8, inv_steps: int = 150, model: str = "malecns", seed: int = 0,
            out: str = "train13", epochs_cnn: int = -1, resume: str = "") -> dict:
    """ROADMAP 13A: per condition, the linear hex-temporal decoder and the
    hex+temporal CNN trained on the pairs13 shards; r on val and test (by
    source and class); the round trip through the frozen brain on
    `n_roundtrip` test clips for both models and for the Adam inversion of the
    same clips (per-type normalised loss, as item 12). Predictions of those
    clips are saved for the figures. `epochs_cnn` (default = epochs) lets the
    CNN, 40× slower per epoch, train for fewer; `resume` names an earlier
    `out` directory whose checkpoints initialise the models (fine-tuning)."""
    import numpy as np
    import torch

    _prepare_root()
    from flydream.generate import learned as L
    from flydream.generate.invert import GpuSampler, invert_batch, load_network, simulate
    from flydream.generate.mix import type_weights
    from flydream.decode import pairs as P

    torch.manual_seed(seed); np.random.seed(seed)
    gpu = GpuSampler(); gpu.__enter__()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(RUNS) / run
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads(Path(f"{DATA}/pairs13/columns.json").read_text(encoding="utf-8"))
    split, meta = manifest["split"], manifest["meta"]
    dt, t_pre = manifest["dt"], manifest["t_pre"]
    outdir = Path(RUNS) / out
    outdir.mkdir(parents=True, exist_ok=True)
    net = load_network(model)
    _, index = P.type_index(net.connectome)
    cells_all = np.asarray(manifest["cells"]); type_of = np.asarray(manifest["type_of_cell"])
    t0 = time.time()
    results = {}

    def by_group(r, ids):
        g = {}
        for v, i in zip(r, ids):
            m = meta[int(i)]
            g.setdefault(m["scene"] if m["source"] == "sintel" else m["class"], []).append(float(v))
        return {k: [float(np.mean(v)), len(v)] for k, v in g.items()}

    for cond in conditions.split(","):
        types = L.CONDITIONS[cond]
        print(f"=== condition {cond}: {types}", flush=True)
        train = L.ShardSet(root, manifest, columns, types, split["train"])
        val = L.ShardSet(root, manifest, columns, types, split["val"])
        test = L.ShardSet(root, manifest, columns, types, split["test"])
        print(f"  train {len(train.x)}, val {len(val.x)}, test {len(test.x)}, maps {train.x.shape[1:]}, {time.time() - t0:.0f} s", flush=True)
        res = {"types": types, "n": {"train": len(train.x), "val": len(val.x), "test": len(test.x)}, "models": {}}
        rt_ids = test.index[:: max(1, len(test.index) // n_roundtrip)][:n_roundtrip]
        rt_pos = np.array([int(np.where(test.index == i)[0][0]) for i in rt_ids])
        parts = []
        for name in manifest["shards"]:
            z = np.load(root / name)
            keep = np.isin(z["index"], rt_ids)
            if keep.any():
                parts.append((z["index"][keep], z["states"][keep]))
        order = np.concatenate([i for i, _ in parts]); st = np.concatenate([x for _, x in parts])
        lookup = {int(i): k for k, i in enumerate(order)}
        tg_states = st[[lookup[int(i)] for i in rt_ids]]
        for kind in ("linear", "cnn"):
            k = len(types)
            mdl = L.LinearHexTemporal(k, rings, taps) if kind == "linear" else L.HexTemporalCNN(k, width, depth, rings, taps)
            n_par = sum(q.numel() for q in mdl.parameters())
            ck = Path(RUNS) / resume / f"{cond}_{kind}.pt" if resume else None
            if ck is not None and ck.exists():
                mdl.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"])
                print(f"  {kind}: {n_par} parameters, resumed from {ck}", flush=True)
            else:
                print(f"  {kind}: {n_par} parameters", flush=True)
            fit = L.train_model(mdl, train, val, t_out=frames, epochs=(epochs if kind == "linear" or epochs_cnn < 0 else epochs_cnn),
                                batch=batch, lr=lr, device=dev, log=lambda s_: print(s_, flush=True))
            ev_val = L.evaluate(mdl, val, fit["mean"], fit["std"], frames, dev)
            ev_test = L.evaluate(mdl, test, fit["mean"], fit["std"], frames, dev)
            pred_rt = ev_test["pred"][rt_pos]
            rt = L.round_trip_error(net, pred_rt, tg_states, cells_all, type_of, types, dt, t_pre, margin)
            torch.save({"state_dict": mdl.state_dict(), "mean": fit["mean"], "std": fit["std"], "kind": kind, "types": types,
                        "rings": rings, "taps": taps, "width": width, "depth": depth}, outdir / f"{cond}_{kind}.pt")
            np.savez_compressed(outdir / f"{cond}_{kind}_roundtrip.npz", ids=rt_ids, pred=pred_rt,
                                true=test.y[rt_pos, :frames], r=ev_test["r"][rt_pos], rt_per_clip=rt["per_clip"])
            res["models"][kind] = {"parameters": int(n_par), "history": fit["history"],
                                   "val_r": float(ev_val["r"].mean()), "test_r": float(ev_test["r"].mean()),
                                   "test_r_by_group": by_group(ev_test["r"], test.index),
                                   "roundtrip_per_type": rt["per_type"], "roundtrip_per_clip": rt["per_clip"].tolist(),
                                   "roundtrip_ids": rt_ids.tolist()}
            print(f"  {kind}: val r {res['models'][kind]['val_r']:.3f}  test r {res['models'][kind]['test_r']:.3f}  "
                  f"round trip {rt['per_clip'].mean():.4f}  {time.time() - t0:.0f} s", flush=True)
            del mdl, ev_val, ev_test
            torch.cuda.empty_cache()
        vids_true = test.y[rt_pos]
        with torch.no_grad():
            state1 = net.steady_state(t_pre, dt, batch_size=1, value=0.5)
            targets = torch.cat([simulate(net, torch.as_tensor(v[None].astype(np.float32), device=dev), dt, state1) for v in vids_true])
        w_cells, w = type_weights(index, types, targets[:1])
        state = net.steady_state(t_pre, dt, batch_size=len(rt_ids), value=0.5)
        inv, tr, n_steps = invert_batch(net, targets, [w_cells] * len(rt_ids), dt=dt, state=state, steps=inv_steps, lr=0.05,
                                        tv=0.02, plateau_steps=20, plateau_tol=0.01, log_every=50, cell_weights=[w] * len(rt_ids))
        inv_np = inv.numpy()[:, :frames]
        rt_inv = L.round_trip_error(net, inv_np, tg_states, cells_all, type_of, types, dt, t_pre, margin)
        r_inv = L.pixcorr(inv_np, vids_true[:, :frames])
        np.savez_compressed(outdir / f"{cond}_inversion_roundtrip.npz", ids=rt_ids, pred=inv_np, true=vids_true[:, :frames],
                            r=r_inv, rt_per_clip=rt_inv["per_clip"])
        res["inversion"] = {"test_r": float(r_inv.mean()), "roundtrip_per_type": rt_inv["per_type"],
                            "roundtrip_per_clip": rt_inv["per_clip"].tolist(), "steps": int(n_steps)}
        print(f"  inversion: r {r_inv.mean():.3f}  round trip {rt_inv['per_clip'].mean():.4f}", flush=True)
        results[cond] = res
        del train, val, test, targets
        torch.cuda.empty_cache()
    gpu.__exit__(None, None, None)
    summary = {"run": run, "epochs": epochs, "batch": batch, "lr": lr, "width": width, "depth": depth, "rings": rings,
               "taps": taps, "frames": frames, "margin": margin, "seconds": round(time.time() - t0, 1), "gpu": GPU,
               "gpu_utilisation": gpu.mean, "conditions": results}
    print(f"GPU utilisation over train13: {gpu.mean}", flush=True)
    (outdir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    runs_volume.commit()
    print(f"train13 done in {time.time() - t0:.0f} s on {GPU}", flush=True)
    return summary


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, memory=4096,
              timeout=60 * MINUTES)
def roundtrip13(model: str = "malecns", ckpt: str = "train13", frames: int = 40, margin: int = 5, dt: float = 0.02,
                t_pre: float = 1.0, inv_steps: int = 150, seed: int = 0, out: str = "roundtrip13") -> dict:
    """ROADMAP 13A, last part: the train13 decoders on the states of items
    11-12 (rebuilt here from their recipes), beside the inversion and a
    shuffled-state control; `flydream.generate.roundtrip13`. Videos saved
    under /runs/<out>/<cond>.npz, the summary returned."""
    import numpy as np

    _prepare_root()
    from flydream.generate import roundtrip13 as R
    from flydream.generate.invert import GpuSampler

    manifest = json.loads((Path(RUNS) / "pairs13" / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads(Path(f"{DATA}/pairs13/columns.json").read_text(encoding="utf-8"))
    t0 = time.time()
    with GpuSampler() as gpu:
        r = R.run(model, Path(RUNS) / ckpt, manifest, columns, frames=frames, margin=margin, dt=dt, t_pre=t_pre,
                  inv_steps=inv_steps, seed=seed, log=lambda s_: print(s_, flush=True))
    outdir = Path(RUNS) / out
    outdir.mkdir(parents=True, exist_ok=True)
    arrays = r.pop("arrays")
    for cond, arr in arrays.items():
        flat = {}
        for name, a in arr.items():
            for k, v in a["videos"].items():
                flat[f"{name}__{k}"] = v
            for k, v in a["refs"].items():
                flat[f"{name}__ref_{k}"] = v
            if a["input"] is not None:
                flat[f"{name}__input"] = a["input"]
        np.savez_compressed(outdir / f"{cond}.npz", **flat)
    r.update({"gpu": GPU, "gpu_utilisation": gpu.mean, "cpu": 1, "memory_mb": 4096, "seconds_worker": round(time.time() - t0, 1)})
    (outdir / "summary.json").write_text(json.dumps(r), encoding="utf-8")
    runs_volume.commit()
    print(f"GPU utilisation: {gpu.mean}; roundtrip13 done in {time.time() - t0:.0f} s", flush=True)
    return r


# ----------------------------------------------------------------- 13B


def _deep_layout(manifest, columns):
    from flydream.generate import learned as L
    from flydream.generate.gen13b import DEEP
    return L.channel_layout(manifest, columns, DEEP)


@app.function(image=image, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=2, memory=12288, timeout=60 * MINUTES)
def maps13b(run: str = "pairs13", out: str = "gen13b/maps_deep.npz") -> dict:
    """13B data, CPU: the pairs13 shards -> (N, T, 8, 721) float16 maps of the
    T4/T5 types (z-scored per type over the training split), the videos and
    the split, one file on /runs/<out>. The GPU functions start from it."""
    import numpy as np
    from flydream.generate import learned as L
    from flydream.generate.gen13b import DEEP

    t0 = time.time()
    root = Path(RUNS) / run
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads(Path(f"{DATA}/pairs13/columns.json").read_text(encoding="utf-8"))
    layout = L.channel_layout(manifest, columns, DEEP)
    xs, ys, ids = [], [], []
    for name in manifest["shards"]:
        z = np.load(root / name)
        xs.append(L.to_maps(z["states"], layout, len(DEEP))); ys.append(z["videos"]); ids.append(z["index"])
        print(f"  {name} {time.time() - t0:.0f} s", flush=True)
    x, y, index = np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)
    split = manifest["split"]
    tr = np.isin(index, split["train"])
    k = x.shape[2]
    s1 = np.zeros(k); s2 = np.zeros(k); n = 0
    for i in range(0, tr.sum(), 256):
        xb = x[tr][i:i + 256].astype(np.float32)
        s1 += xb.sum((0, 1, 3)); s2 += (xb ** 2).sum((0, 1, 3)); n += xb.shape[0] * xb.shape[1] * xb.shape[3]
    mean = (s1 / n).astype(np.float32); std = (np.sqrt(np.maximum(s2 / n - mean ** 2, 0)) + 1e-6).astype(np.float32)
    for i in range(0, len(x), 256):
        x[i:i + 256] = ((x[i:i + 256].astype(np.float32) - mean[None, None, :, None]) / std[None, None, :, None]).astype(np.float16)
    outp = Path(RUNS) / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, maps=x, videos=y, index=index, mean=mean, std=std,
             train=np.array(split["train"]), val=np.array(split["val"]), test=np.array(split["test"]))
    runs_volume.commit()
    r = {"n": int(len(x)), "maps": list(x.shape), "gb": round(outp.stat().st_size / 1e9, 2), "seconds": round(time.time() - t0, 1), "cpu": 2, "memory_mb": 12288}
    print(json.dumps(r), flush=True)
    return r


def _load_maps(device, file: str = "gen13b/maps_deep.npz", subset: str = "train", limit: int = 0):
    """(videos, maps) float16 tensors of one split on `device`, plus the index."""
    import numpy as np
    import torch

    z = np.load(Path(RUNS) / file)
    keep = np.isin(z["index"], z[subset])
    idx = np.where(keep)[0]
    if limit:
        idx = idx[:limit]
    v = torch.as_tensor(z["videos"][idx][:, :40], device=device)
    m = torch.as_tensor(z["maps"][idx], device=device)
    return v, m, z["index"][idx], {"mean": z["mean"], "std": z["std"]}


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, memory=8192, timeout=30 * MINUTES)
def bench13b(steps: int = 200, batch: int = 32, kinds: str = "hexresnet,sit", width_hex: int = 64, depth_hex: int = 6,
             width_sit: int = 128, depth_sit: int = 4, compile_model: bool = False, limit: int = 2048) -> dict:
    """13B step (b): each backbone for `steps` steps on the training maps
    (first `limit` clips): seconds per step, peak memory, parameters, loss,
    GPU utilisation. Decides the backbone and the price of (c)."""
    import torch
    from flydream.generate import gen13b as G
    from flydream.generate.invert import GpuSampler

    dev = torch.device("cuda")
    t0 = time.time()
    v, m, _, _ = _load_maps(dev, limit=limit)
    print(f"data on GPU: {tuple(v.shape)} {tuple(m.shape)} in {time.time() - t0:.0f} s", flush=True)
    out = {}
    for kind in kinds.split(","):
        kw = dict(width=width_hex, depth=depth_hex) if kind == "hexresnet" else dict(width=width_sit, depth=depth_sit)
        for comp in ([False, True] if compile_model else [False]):
            with GpuSampler() as gpu:
                try:
                    r = G.bench(kind, v, m, steps=steps, batch=batch, compile_model=comp, log=lambda s_: print(s_, flush=True), **kw)
                except Exception as e:                                  # a compile failure is a result, not an abort
                    r = {"kind": kind, "compiled": comp, "error": str(e)[:300]}
            r["gpu_utilisation"] = gpu.mean
            out[f"{kind}{'_compiled' if comp else ''}"] = r
            print(json.dumps(r), flush=True)
            torch.cuda.empty_cache()
    out["_"] = {"gpu": GPU, "cpu": 1, "memory_mb": 8192, "seconds": round(time.time() - t0, 1), "torch": torch.__version__}
    return out


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, memory=8192, timeout=90 * MINUTES)
def train13b(kind: str = "hexresnet", steps: int = 6000, batch: int = 32, lr: float = 3e-4, width: int = 64, depth: int = 6,
             heads: int = 4, compile_model: bool = False, seed: int = 0, out: str = "gen13b") -> dict:
    """13B step (c): the generator trained with the optimised loop
    (`gen13b.train`); validation loss every 500 steps; checkpoint with EMA
    weights on /runs/<out>/<kind>.pt."""
    import torch
    from flydream.generate import gen13b as G
    from flydream.generate.invert import GpuSampler

    dev = torch.device("cuda")
    t0 = time.time()
    v, m, _, stats = _load_maps(dev)
    vv, mv, _, _ = _load_maps(dev, subset="val", limit=256)
    print(f"train {tuple(v.shape)}, val {tuple(vv.shape)} on GPU in {time.time() - t0:.0f} s", flush=True)
    model = G.build(kind, frames=40, width=width, depth=depth, heads=heads)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{kind}: {n_par} parameters", flush=True)
    with GpuSampler() as gpu:
        r = G.train(model, v, m, steps=steps, batch=batch, lr=lr, seed=seed, compile_model=compile_model, log_every=100,
                    log=lambda s_: print(s_, flush=True), val=(vv, mv), val_every=500)
    outdir = Path(RUNS) / out
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"kind": kind, "frames": 40, "width": width, "depth": depth, "heads": heads, "steps": steps, "batch": batch, "lr": lr,
            "parameters": int(n_par), "seed": seed, "mean": stats["mean"].tolist(), "std": stats["std"].tolist()}
    G.save(outdir / f"{kind}.pt", model, r["ema"], meta)
    summary = {**meta, "history": r["history"], "seconds": r["seconds"], "seconds_worker": round(time.time() - t0, 1),
               "gpu": GPU, "gpu_utilisation": gpu.mean, "cpu": 1, "memory_mb": 8192}
    (outdir / f"{kind}_train.json").write_text(json.dumps(summary), encoding="utf-8")
    runs_volume.commit()
    print(f"train13b done: {r['seconds']} s, GPU utilisation {gpu.mean}", flush=True)
    return summary


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, memory=6144, timeout=40 * MINUTES)
def multi_init13b(model: str = "malecns", n_clips: int = 4, n_starts: int = 8, inv_steps: int = 150, init_sd: float = 0.25,
                  seed: int = 0, out: str = "gen13b") -> dict:
    """13B step (a): the Adam inversion of one deep state from `n_starts`
    random starts (grey + N(0, init_sd), clipped) and the grey start, for
    `n_clips` held-out clips; every solution's round trip and the pairwise
    r between solutions. Decides whether sample spread is a goal."""
    import numpy as np
    import torch
    from flydream.generate import learned as L
    from flydream.generate.gen13b import DEEP
    from flydream.generate.invert import GpuSampler, invert_batch, load_network, simulate
    from flydream.generate.mix import type_weights
    from flydream.generate.roundtrip13 import round_trip
    from flydream.decode import pairs as P

    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    dev = torch.device("cuda")
    root = Path(RUNS) / "pairs13"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cells_all = np.asarray(manifest["cells"]); type_of = np.asarray(manifest["type_of_cell"])
    dt, t_pre, margin = manifest["dt"], manifest["t_pre"], 5
    ids = np.load(root.parent / "train13" / "deep_inversion_roundtrip.npz")["ids"][:n_clips]
    vids, sts = [], []
    for name in manifest["shards"]:
        z = np.load(root / name)
        for i in ids:
            w = np.where(z["index"] == i)[0]
            if len(w):
                vids.append((int(i), z["videos"][w[0]])); sts.append((int(i), z["states"][w[0]]))
    order = {int(i): k for k, i in enumerate(ids)}
    vids = np.stack([v for _, v in sorted(vids, key=lambda q: order[q[0]])]).astype(np.float32)
    tg_states = np.stack([v for _, v in sorted(sts, key=lambda q: order[q[0]])])
    net = load_network(model)
    _, index = P.type_index(net.connectome)
    with torch.no_grad():
        state1 = net.steady_state(t_pre, dt, batch_size=1, value=0.5)
        targets = torch.cat([simulate(net, torch.as_tensor(v[None], device=dev), dt, state1) for v in vids])
    w_cells, w = type_weights(index, DEEP, targets[:1])
    var_ref = {t: float(targets[0][:, index[t]].var(unbiased=False)) + 1e-6 for t in DEEP}
    T = vids.shape[1]
    B = n_clips * (n_starts + 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.full((B, T, 721), 0.5)
    for c in range(n_clips):
        for k in range(n_starts):
            init[c * (n_starts + 1) + 1 + k] = (0.5 + init_sd * torch.randn(T, 721, generator=g)).clamp(0, 1)
    tg = targets.repeat_interleave(n_starts + 1, 0)
    state = net.steady_state(t_pre, dt, batch_size=B, value=0.5)
    with GpuSampler() as gpu:
        inv, tr, n_steps = invert_batch(net, tg, [w_cells] * B, dt=dt, state=state, steps=inv_steps, lr=0.05, tv=0.02,
                                        plateau_steps=20, plateau_tol=0.01, log_every=50, cell_weights=[w] * B, init=init)
        inv_np = inv.numpy()[:, :40]
        rt = round_trip(net, inv_np, tg_states[:, :, cells_all].repeat(n_starts + 1, 0), cells_all, type_of, DEEP, dt, t_pre,
                        margin, (0, T), var_ref)
    res = {}
    for c in range(n_clips):
        sl = slice(c * (n_starts + 1), (c + 1) * (n_starts + 1))
        sols = inv_np[sl]
        rr = np.array([[float(L.pixcorr(sols[a][None], sols[b][None])[0]) for b in range(len(sols))] for a in range(len(sols))])
        off = rr[~np.eye(len(sols), dtype=bool)]
        res[int(ids[c])] = {"roundtrip": rt[sl].tolist(), "r_true": L.pixcorr(sols, np.repeat(vids[c][None, :40], len(sols), 0)).tolist(),
                            "r_pairwise_mean": float(off.mean()), "r_pairwise_min": float(off.min()),
                            "fit_final": tr[-1, sl].tolist()}
        print(f"clip {ids[c]}: round trip grey {rt[sl][0]:.4f}, random starts {rt[sl][1:].mean():.4f} "
              f"[{rt[sl][1:].min():.4f}-{rt[sl][1:].max():.4f}]; pairwise r {off.mean():.3f} (min {off.min():.3f})", flush=True)
    outdir = Path(RUNS) / out
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "multi_init.npz", ids=ids, videos=inv_np, true=vids[:, :40], roundtrip=rt, init=init.numpy()[:, :40])
    summary = {"clips": res, "n_starts": n_starts, "init_sd": init_sd, "steps": int(n_steps), "batch": B, "gpu": GPU,
               "gpu_utilisation": gpu.mean, "cpu": 1, "memory_mb": 6144, "seconds": round(time.time() - t0, 1)}
    (outdir / "multi_init.json").write_text(json.dumps(summary), encoding="utf-8")
    runs_volume.commit()
    return summary


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, memory=8192, timeout=40 * MINUTES)
def sample13b(model: str = "malecns", ckpt: str = "gen13b/hexresnet.pt", n_seeds: int = 4, sample_steps: int = 20,
              guidances: str = "1,2,4", seed: int = 0, out: str = "gen13b") -> dict:
    """13B step (d): samples of the generator and their round trips, all in
    batches. Sets: held-out test clips (the 8 of train13), item 11-12
    states (clips A/B, edits, the 0.5 mix, the averaged video's state, a
    hybrid, eye noise, neuron noise), single-type prompts on clip A's
    state, the conditioning-strength test (one z: true / shuffled / zero
    state), guidance on two states, the shuffled-state control."""
    import numpy as np
    import torch
    from flydream.generate import gen13b as G
    from flydream.generate import learned as L
    from flydream.generate.gen13b import DEEP
    from flydream.generate.invert import GpuSampler, load_network
    from flydream.generate.roundtrip13 import build_states, round_trip
    from flydream.decode import pairs as P

    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    dev = torch.device("cuda")
    root = Path(RUNS) / "pairs13"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads(Path(f"{DATA}/pairs13/columns.json").read_text(encoding="utf-8"))
    layout = _deep_layout(manifest, columns)
    cells_all = np.asarray(manifest["cells"]); type_of = np.asarray(manifest["type_of_cell"])
    dt, t_pre, margin, frames = manifest["dt"], manifest["t_pre"], 5, 40
    T = frames + margin
    gen, meta = G.load(Path(RUNS) / ckpt, dev)
    mean = torch.as_tensor(np.array(meta["mean"], np.float32), device=dev)[None, None, :, None]
    std = torch.as_tensor(np.array(meta["std"], np.float32), device=dev)[None, None, :, None]
    net = load_network(model)
    _, index = P.type_index(net.connectome)
    rng = np.random.default_rng(seed)

    # --- the states: (name, target (1,T,cells_all) numpy, refs) ---
    states, refs = {}, {}
    ids = np.load(root.parent / "train13" / "deep_inversion_roundtrip.npz")["ids"]
    for name in manifest["shards"]:
        z = np.load(root / name)
        for i in ids:
            w = np.where(z["index"] == i)[0]
            if len(w):
                states[f"test_{int(i)}"] = z["states"][w[0]][None]
                refs[f"test_{int(i)}"] = {"clip": z["videos"][w[0]][:frames].astype(np.float32)}
    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    keep = {"clip_A", "clip_B", "C_T4a_x0.5", "C_T4a_x2", "C_T5_x0", "A_a0.5", "A_avgvideo", "B_earlyA_motionB", "eye_noise", "neuron_noise"}
    for s in built:
        if s["name"] in keep:
            w0, w1 = s["window"]
            states[s["name"]] = s["target"][:, w0:w1, :][:, :, cells_all].cpu().numpy().astype(np.float16)
            refs[s["name"]] = {k: v for k, v in s["refs"].items()}
    ta = states["clip_A"]
    var_ref = {t: float(np.asarray(ta[0][:, np.where(type_of == t)[0]], np.float32).var()) + 1e-6 for t in DEEP}
    cells_deep = np.concatenate([index[t] for t in DEEP])

    def maps_of(st):                                                     # (1,T,cells) -> (1,T,8,721) z-scored torch
        x = torch.as_tensor(L.to_maps(st.astype(np.float16), layout, len(DEEP)).astype(np.float32), device=dev)
        return (x - mean) / std

    # --- the jobs: (key, maps, mask, guidance, seed) ---
    jobs = []
    gs = [float(g) for g in guidances.split(",")]

    def add(state, mask_name, g, seeds, maps=None, tag=None):
        mp = maps if maps is not None else maps_of(states[state])
        for k in seeds:
            jobs.append((f"{tag or state}__{mask_name}__s{g:g}__seed{k}", state, mp, G.named_mask(mask_name), g, k))
    seeds = list(range(n_seeds))
    for name in states:
        add(name, "full", 1.0, seeds)
    for mk in ("T4a", "t4", "t5", "dir_a"):
        add("clip_A", mk, 1.0, seeds)
    for g in gs[1:]:
        add("clip_A", "full", g, seeds); add(f"test_{int(ids[0])}", "full", g, seeds)
    # conditioning strength: one z, three states (seed 0)
    st_sh = ta.copy(); perm = rng.permutation(len(cells_deep))
    pos = np.array([int(np.where(cells_all == c)[0][0]) for c in cells_deep])
    st_sh[:, :, pos] = ta[:, :, pos[perm]]
    add("clip_A", "full", 1.0, [0], tag="strength_true")
    add("clip_A", "full", 1.0, [0], maps=maps_of(st_sh), tag="strength_shuffled")
    add("clip_A", "full", 1.0, [0], maps=torch.zeros(1, T, 8, 721, device=dev), tag="strength_zero")
    add("clip_A", "full", 1.0, seeds, maps=maps_of(st_sh), tag="control_shuffled")
    states["control_shuffled"] = st_sh; refs["control_shuffled"] = refs["clip_A"]
    states["strength_shuffled"] = st_sh; states["strength_zero"] = ta; states["strength_true"] = ta
    for k in ("strength_true", "strength_shuffled", "strength_zero"):
        refs[k] = refs["clip_A"]
    print(f"{len(jobs)} sampling jobs, {len(states)} states, {time.time() - t0:.0f} s", flush=True)

    # --- sampling in batches, grouped by guidance (one generator per seed keeps z shared across states) ---
    videos = {}
    with GpuSampler() as gpu:
        by_g = {}
        for j in jobs:
            by_g.setdefault(j[4], []).append(j)
        for g, js in by_g.items():
            for i in range(0, len(js), 32):
                chunk = js[i:i + 32]
                cond = torch.cat([c[2] for c in chunk]).float()
                mask = torch.as_tensor(np.stack([c[3] for c in chunk]), device=dev)
                # z per job from its seed: draw each job's noise with its own generator, then integrate together
                x0 = torch.stack([torch.randn(T, 721, device=dev, generator=torch.Generator(device=dev).manual_seed(1000 + c[5])) for c in chunk])
                vid = _sample_from(gen, x0, cond, mask, sample_steps, g)
                for c, v in zip(chunk, vid.cpu().numpy()):
                    videos[c[0]] = v[:frames]
        # round trips of every sample against its state, one batch
        keys = list(videos)
        vids_all = np.stack([videos[k] for k in keys])
        state_of = {j[0]: j[1] for j in jobs}
        tgs = np.concatenate([states[state_of[k]] for k in keys])
        rts = round_trip(net, vids_all, tgs, cells_all, type_of, DEEP, dt, t_pre, margin, (0, T), var_ref)
    rt_of = dict(zip(keys, rts.tolist()))
    # --- scores per (state, mask, guidance): median / best / spread / r ---
    groups = {}
    for k in keys:
        st, mk, g, sd = k.split("__")
        groups.setdefault((st, mk, g), []).append(k)
    scores = {}
    for (st, mk, g), ks in groups.items():
        v = np.stack([videos[k] for k in ks]); r = np.array([rt_of[k] for k in ks])
        sc = {"n": len(ks), "roundtrip_median": float(np.median(r)), "roundtrip_best": float(r.min()), "roundtrip_all": r.tolist()}
        if len(ks) > 1:
            pw = [float(L.pixcorr(v[a][None], v[b][None])[0]) for a in range(len(ks)) for b in range(a + 1, len(ks))]
            sc["r_between_samples"] = float(np.mean(pw))
        for rn, ref in refs.get(st, {}).items():
            if ref.shape[0] >= frames:
                sc[f"r_{rn}"] = float(L.pixcorr(v, np.repeat(ref[None, :frames], len(ks), 0)).mean())
        scores[f"{st}__{mk}__{g}"] = sc
        print(f"  {st:<18} {mk:<5} {g:<3} rt median {sc['roundtrip_median']:.3f} best {sc['roundtrip_best']:.3f}"
              + (f"  spread r {sc['r_between_samples']:.2f}" if "r_between_samples" in sc else "")
              + "".join(f"  {q} {sc[q]:+.2f}" for q in sc if q.startswith("r_") and q != "r_between_samples"), flush=True)
    # the strength test: r between the three videos of one z
    s_t, s_s, s_z = (videos[f"strength_{q}__full__s1__seed0"] for q in ("true", "shuffled", "zero"))
    strength = {"r_true_shuffled": float(L.pixcorr(s_t[None], s_s[None])[0]), "r_true_zero": float(L.pixcorr(s_t[None], s_z[None])[0]),
                "r_shuffled_zero": float(L.pixcorr(s_s[None], s_z[None])[0])}
    print(f"strength test (one z): r true~shuffled {strength['r_true_shuffled']:.2f}, true~zero {strength['r_true_zero']:.2f}", flush=True)
    outdir = Path(RUNS) / out
    outdir.mkdir(parents=True, exist_ok=True)
    flat = {k: v for k, v in videos.items()}
    for st, rf in refs.items():
        for rn, ref in rf.items():
            flat[f"ref__{st}__{rn}"] = ref
    np.savez_compressed(outdir / "samples.npz", **flat)
    summary = {"ckpt": ckpt, "n_seeds": n_seeds, "sample_steps": sample_steps, "guidances": gs, "scores": scores, "strength": strength,
               "roundtrip_per_sample": rt_of, "gpu": GPU, "gpu_utilisation": gpu.mean, "cpu": 1, "memory_mb": 8192,
               "seconds": round(time.time() - t0, 1), "frames": frames}
    (outdir / "samples.json").write_text(json.dumps(summary), encoding="utf-8")
    runs_volume.commit()
    print(f"sample13b done in {time.time() - t0:.0f} s, GPU utilisation {gpu.mean}", flush=True)
    return summary


def _sample_from(gen, x0, cond, mask, steps, guidance):
    """`gen13b.sample` from a given initial noise (shared z across states)."""
    import torch
    x = x0.clone()
    B = x.shape[0]
    zero = torch.zeros_like(mask)
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((B,), i / steps, device=x.device)
            v = gen(x, t, cond, mask)
            if guidance != 1.0:
                vu = gen(x, t, cond, zero)
                v = vu + guidance * (v - vu)
            x = x + v / steps
    return x.clamp(0, 1)


@app.local_entrypoint()
def main(model: str = "flow/0000/000", sample: int = 3, frames: int = -1, margin: int = -1, steps: int = -1,
         stages: str = "", dream_sources: str = "", seed: int = 0, mix_clips: str = "", pairs13_run: bool = False,
         train13_run: bool = False, epochs: int = 12, pairs13_videos_run: bool = False, epochs_cnn: int = -1,
         resume: str = "", roundtrip13_run: bool = False, maps13b_run: bool = False, bench13b_run: bool = False,
         train13b_run: bool = False, multi_init13b_run: bool = False, sample13b_run: bool = False, kind: str = "hexresnet",
         steps13b: int = 6000, batch13b: int = 32, width13b: int = 64, depth13b: int = 6, compile13b: bool = False):
    """`--dream-sources eye_noise,flash,dark_after,neuron_noise` runs item 11
    instead of the clip ladder; `--mix-clips 3,10` runs item 12."""
    root = Path(__file__).resolve().parents[2]
    frames = GEN.get("frames", 40) if frames < 0 else frames
    margin = GEN.get("margin", 5) if margin < 0 else margin
    steps = GEN.get("steps", 150) if steps < 0 else steps
    common = dict(model=model, frames=frames, margin=margin, steps=steps, lr=GEN.get("lr", 0.05), tv=GEN.get("tv", 0.02),
                  dt=GEN.get("dt", 0.02), t_pre=GEN.get("t_pre", 1.0), stages=stages, batch=GEN.get("batch", 0),
                  plateau_steps=GEN.get("plateau_steps", 0), plateau_tol=GEN.get("plateau_tol", 0.0),
                  plateau_floor=GEN.get("plateau_floor", 1e-3))
    t0 = time.time()
    if maps13b_run or bench13b_run or train13b_run or multi_init13b_run or sample13b_run:
        d = root / "data" / "gen13b"
        d.mkdir(parents=True, exist_ok=True)
        if maps13b_run:
            print("maps13b on CPU (2 cores, 12 GB): pairs13 shards -> deep maps"); r = maps13b.remote()
            (d / "maps.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        if bench13b_run:
            print(f"bench13b on {GPU}: hexresnet {width13b}x{depth13b} and sit, {steps13b} steps, batch {batch13b}")
            r = bench13b.remote(steps=steps13b, batch=batch13b, width_hex=width13b, depth_hex=depth13b, compile_model=compile13b)
            (d / "bench.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        if multi_init13b_run:
            print(f"multi_init13b on {GPU}: {model}"); r = multi_init13b.remote(model=model, seed=seed)
            (d / "multi_init.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        if train13b_run:
            print(f"train13b on {GPU}: {kind} {width13b}x{depth13b}, {steps13b} steps, batch {batch13b}")
            r = train13b.remote(kind=kind, steps=steps13b, batch=batch13b, width=width13b, depth=depth13b, compile_model=compile13b, seed=seed)
            (d / f"{kind}_train.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        if sample13b_run:
            print(f"sample13b on {GPU}: {model}, {kind}"); r = sample13b.remote(model=model, ckpt=f"gen13b/{kind}.pt", seed=seed)
            (d / "samples.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in r.items() if k not in ("history", "scores", "roundtrip_per_sample", "clips")}, indent=1)[:3000])
        print(f"done in {time.time() - t0:.0f} s")
        return
    if roundtrip13_run:
        print(f"roundtrip13 on {GPU}: {model}, decoders of /runs/train13, states of items 11-12")
        r = roundtrip13.remote(model=model, frames=frames, margin=margin, seed=seed)
        d = root / "data" / "roundtrip13"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        for cond, res in r["conditions"].items():
            for name, sc in res["states"].items():
                print(f"{cond:>6} {name:<18} rt linear {sc['linear']['round_trip']:.3f}  cnn {sc['cnn']['round_trip']:.3f}  "
                      f"inversion {sc['inversion']['round_trip']:.3f}  shuffled {sc['shuffled']['round_trip']:.3f}")
        print(f"done in {time.time() - t0:.0f} s; GPU utilisation {r['gpu_utilisation']}")
        return
    if train13_run:
        import numpy as np
        print(f"train13 on {GPU}: {model}, {epochs} epochs (cnn {epochs_cnn if epochs_cnn >= 0 else epochs}){', resume ' + resume if resume else ''}")
        r = train13.remote(model=model, epochs=epochs, frames=frames, margin=margin, seed=seed, epochs_cnn=epochs_cnn, resume=resume)
        d = root / "data" / "train13"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        for cond, res in r["conditions"].items():
            for kind, m in res["models"].items():
                print(f"{cond:>6} {kind:<9} params {m['parameters']:>7}  val r {m['val_r']:+.3f}  test r {m['test_r']:+.3f}  "
                      f"round trip {np.mean(m['roundtrip_per_clip']):.4f}")
            print(f"{cond:>6} {'inversion':<9} r {res['inversion']['test_r']:+.3f}  round trip {np.mean(res['inversion']['roundtrip_per_clip']):.4f}")
        print(f"done in {time.time() - t0:.0f} s")
        return
    if pairs13_videos_run:
        P13 = _generate_settings_all().get("pairs13", {})
        print(f"pairs13 videos on CPU: {frames + margin} frames, procedural {P13.get('n_per_class', 800)}/class")
        r = pairs13_videos.remote(procedural=f"pairs13/procedural_{P13.get('n_per_class', 800)}_s{P13.get('seed', 0)}.npz",
                                  frames=frames + margin, dt=GEN.get("dt", 0.02))
        print(json.dumps(r)); print(f"done in {time.time() - t0:.0f} s")
        return
    if pairs13_run:
        P13 = _generate_settings_all().get("pairs13", {})
        print(f"pairs13 on {GPU}: {model}, {frames + margin} frames, from /data/pairs13/videos.npz")
        r = pairs13.remote(videos_file="pairs13/videos.npz",
                           model=model, frames=frames + margin, dt=GEN.get("dt", 0.02), t_pre=GEN.get("t_pre", 1.0),
                           sim_batch=P13.get("sim_batch", 32), seed=P13.get("seed", 0),
                           held_scenes=",".join(P13.get("held_scenes", [])), held_classes=",".join(P13.get("held_classes", [])),
                           val_fraction=P13.get("val_fraction", 0.1))
        d = root / "data" / "pairs13"
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest_summary.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps(r, indent=1)); print(f"done in {time.time() - t0:.0f} s")
        return
    if mix_clips:
        sa, sb = (int(x) for x in mix_clips.split(","))
        print(f"mix on {GPU}: {model}, clips {sa} and {sb}, {frames} frames + margin {margin}, {steps} steps")
        records = mix.remote(sample_a=sa, sample_b=sb, **common)
        for r in records:
            d = root / "data" / "generate" / r["tag"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "recovered.npz").write_bytes(base64.b64decode(r.pop("npz_b64")))
            (d / "meta.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
            print(f"{r['tag']:<44} r_A {r['r_a']:+.2f}  r_B {r['r_b']:+.2f}  {r['note']}")
        print(f"done in {time.time() - t0:.0f} s; draw with: python tools/fig_mix.py --prefix {records[0]['tag'].split('_mix_')[0]}")
        return
    if dream_sources:
        print(f"dreams on {GPU}: {model}, sources {dream_sources}, {frames} frames + margin {margin}, {steps} steps")
        records = dreams.remote(sources=dream_sources, seed=seed, **common)
        for r in records:
            d = root / "data" / "generate" / r["tag"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "recovered.npz").write_bytes(base64.b64decode(r.pop("npz_b64")))
            (d / "meta.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
            rr = f"r {r['inversion']:+.3f} (control {r['control']:+.3f})" if r["inversion"] is not None else "r n/a"
            print(f"{r['tag']:<44} {rr}  contrast out {r['contrast_inversion']:.3f} / control {r['contrast_control']:.3f}")
        print(f"done in {time.time() - t0:.0f} s; draw with: python tools/fig_dreams.py --prefix {records[0]['tag'].split('_dream_')[0]}")
        return
    print(f"ladder on {GPU}: {model}, clip {sample}, {frames} frames + margin {margin}, {steps} steps")
    records = ladder.remote(sample=sample, **common)
    tags = []
    for r in records:
        d = root / "data" / "generate" / r["tag"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "recovered.npz").write_bytes(base64.b64decode(r.pop("npz_b64")))
        (d / "meta.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        tags.append(r["tag"])
        print(f"{r['tag']:<40} {r['cells']:>6} cells  r {r['inversion']:+.3f}  control {r['control']:+.3f}  "
              f"{r.get('steps_run', r['steps'])} steps  {r['seconds']} s")
    if records and records[0].get("gpu_utilisation") is not None:
        print(f"batch {records[0]['batch']}, GPU utilisation {records[0]['gpu_utilisation']:.0f}%")
    print(f"done in {time.time() - t0:.0f} s; draw with:\n  python -m flydream.generate.figures --tags {' '.join(tags)} "
          f"--out {tags[0].rsplit('_invert_', 1)[0]}_inversion_ladder --model {model}")
