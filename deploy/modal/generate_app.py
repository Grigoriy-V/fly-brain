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
def maps13b(run: str = "pairs13", out: str = "gen13b/maps_deep.npz", stats_from: str = "", dct_k: int = 0,
            dct_out: str = "", dct_frames: int = 40) -> dict:
    """13B data, CPU: the pairs13 shards -> (N, T, 8, 721) float16 maps of the
    T4/T5 types (z-scored per type over the training split), the videos and
    the split, one file on /runs/<out>. The GPU functions start from it.

    Item 18 adds two arguments. `stats_from` takes the per-type mean and sd of
    an existing maps file instead of computing new ones — the corpus states
    must be on **13B's** scale, or 13B is handed conditioning it was never
    trained to read and the check of 18.2(b) fails for a trivial reason.
    `dct_k` also writes the compact file the prior trains on: the first `k`
    temporal DCT coefficients of the first `dct_frames` frames, z-scored per
    coefficient over the training split (17.1b's representation, precomputed
    here because a corpus-sized maps file no longer fits a T4 as float32)."""
    import numpy as np
    from flydream.generate import learned as L
    from flydream.generate import prior17 as R
    from flydream.generate.gen13b import DEEP

    t0 = time.time()
    root = Path(RUNS) / run
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads(Path(f"{DATA}/pairs13/columns.json").read_text(encoding="utf-8"))
    layout = L.channel_layout(manifest, columns, DEEP)
    n_total = int(manifest["n"])
    x = y = None
    index = np.empty(n_total, np.int64)
    at = 0
    for name in manifest["shards"]:                                  # filled in place: a corpus-sized
        z = np.load(root / name)                                     # set cannot afford two copies
        xm = L.to_maps(z["states"], layout, len(DEEP))
        if x is None:
            x = np.empty((n_total, *xm.shape[1:]), np.float16)
            y = np.empty((n_total, *z["videos"].shape[1:]), np.float16)
        x[at:at + len(xm)] = xm
        y[at:at + len(xm)] = z["videos"]
        index[at:at + len(xm)] = z["index"]
        at += len(xm)
        del xm, z
        print(f"  {name} {at}/{n_total} {time.time() - t0:.0f} s", flush=True)
    if at != n_total:
        raise SystemExit(f"shards hold {at} clips, the manifest says {n_total}")
    split = manifest["split"]
    tr = np.isin(index, split["train"])
    k = x.shape[2]
    if stats_from:
        zs = np.load(Path(RUNS) / stats_from)
        mean, std = zs["mean"].astype(np.float32), zs["std"].astype(np.float32)
        print(f"  per-type scale taken from {stats_from}: mean {np.round(mean, 3).tolist()}", flush=True)
    else:
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
    per_type = {t: {"mean": float(mean[i]), "sd": float(std[i])} for i, t in enumerate(DEEP)}
    var = np.empty(len(x), np.float32)                               # is every clip a live state?
    energy = np.zeros(k, np.float64)
    for i in range(0, len(x), 256):
        xb = x[i:i + 256].astype(np.float32)
        var[i:i + 256] = xb.reshape(len(xb), -1).var(1)
        energy += (xb ** 2).mean((0, 1, 3)) * len(xb)
    energy = (energy / len(x)).astype(float)
    r = {"n": int(len(x)), "maps": list(x.shape), "gb": round(outp.stat().st_size / 1e9, 2),
         "stats_from": stats_from or None, "cpu": 2,
         "state_per_type": per_type,
         "state_energy_per_type": {t: round(float(energy[i]), 3) for i, t in enumerate(DEEP)},
         "clip_variance": {"median": float(np.median(var)), "p01": float(np.percentile(var, 1)),
                           "min": float(var.min()), "dead_below_0.01": int((var < 0.01).sum())}}
    print(f"  states: per-type sd {np.round(std, 3).tolist()}, clip variance median {np.median(var):.3f} "
          f"(min {var.min():.4f}, {int((var < 0.01).sum())} clips below 0.01)", flush=True)
    if dct_k:                                                        # the compact file the prior trains on
        tf = min(int(dct_frames), x.shape[1])
        dmat = R.dct_matrix(tf, int(dct_k))
        c = np.empty((len(x), int(dct_k), k, x.shape[3]), np.float16)
        num = den = 0.0
        for i in range(0, len(x), 256):
            xb = x[i:i + 256, :tf].astype(np.float32)
            cb = R.to_dct(xb, dmat)                                  # orthonormal: energy is comparable
            num += float((cb ** 2).sum()); den += float((xb ** 2).sum())
            c[i:i + 256] = cb.astype(np.float16)
        s1 = np.zeros((dct_k, k)); s2 = np.zeros((dct_k, k)); n = 0
        idx_tr = np.where(tr)[0]
        for i in range(0, len(idx_tr), 256):
            cb = c[idx_tr[i:i + 256]].astype(np.float32)
            s1 += cb.sum((0, 3)); s2 += (cb ** 2).sum((0, 3)); n += cb.shape[0] * cb.shape[3]
        cm = (s1 / n).astype(np.float32); cs = (np.sqrt(np.maximum(s2 / n - cm ** 2, 0)) + 1e-6).astype(np.float32)
        for i in range(0, len(c), 256):
            c[i:i + 256] = ((c[i:i + 256].astype(np.float32) - cm[None, :, :, None]) / cs[None, :, :, None]).astype(np.float16)
        dp = Path(RUNS) / (dct_out or f"{out.rsplit('/', 1)[0]}/maps_dct{dct_k}.npz")
        dp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dp, maps=c, index=index, mean=mean, std=std, coef_mean=cm, coef_std=cs,
                 time_frames=np.array(tf), dct_k=np.array(int(dct_k)),
                 train=np.array(split["train"]), val=np.array(split["val"]), test=np.array(split["test"]))
        r["dct_file"] = str(dp.relative_to(RUNS)); r["dct_shape"] = list(c.shape)
        r["dct_gb"] = round(dp.stat().st_size / 1e9, 2)
        r["dct_energy_kept"] = round(num / max(den, 1e-9), 5)        # 17.1b measured 0.9955 on the Sintel states
        r["dct_coefficient_sd"] = np.round(cs.mean(1), 3).tolist()
        print(f"  DCT-{dct_k}: keeps {100 * num / max(den, 1e-9):.2f} % of the state energy, "
              f"coefficient sd by index {np.round(cs.mean(1), 3).tolist()}", flush=True)
        print(f"  DCT-{dct_k} over {tf} frames: {c.shape}, {r['dct_gb']} GB", flush=True)
    runs_volume.commit()
    r["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(r), flush=True)
    return r


def _load_maps(device, file: str = "gen13b/maps_deep.npz", subset: str = "train", limit: int = 0,
               with_videos: bool = True, sources: str = "all", run: str = "pairs13"):
    """(videos, maps) float16 tensors of one split on `device`, plus the index.
    `with_videos=False` leaves the videos on the volume (17.1 trains on the
    states alone and the card need not hold them). `sources="sintel"` keeps
    only the scene clips, `"procedural"` only the stimuli (17.1b: the human,
    2026-09-20 — the set is 76 % procedural and the task is a video
    generator)."""
    import numpy as np
    import torch

    z = np.load(Path(RUNS) / file)
    keep = np.isin(z["index"], z[subset])
    if sources != "all":
        n_s = int(json.loads((Path(RUNS) / run / "manifest.json").read_text(encoding="utf-8"))["n_sintel"])
        keep &= (z["index"] < n_s) if sources == "sintel" else (z["index"] >= n_s)
    idx = np.where(keep)[0]
    if limit:
        idx = idx[:limit]
    dct = "coef_mean" in z.files                                     # item 18: the maps file is already compact
    v = torch.as_tensor(z["videos"][idx][:, :40], device=device) if with_videos and not dct else None
    m = torch.as_tensor(z["maps"][idx] if dct else z["maps"][idx][:, :40], device=device)
    stats = {"mean": z["mean"], "std": z["std"]}
    if dct:
        stats |= {"coef_mean": z["coef_mean"], "coef_std": z["coef_std"],
                  "time_frames": int(z["time_frames"]), "dct_k": int(z["dct_k"])}
    return v, m, z["index"][idx], stats


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, memory=12288, timeout=30 * MINUTES)
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
    out["_"] = {"gpu": GPU, "cpu": 1, "memory_mb": 12288, "seconds": round(time.time() - t0, 1), "torch": torch.__version__}
    return out


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, memory=12288, timeout=90 * MINUTES)
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
               "gpu": GPU, "gpu_utilisation": gpu.mean, "cpu": 1, "memory_mb": 12288}
    (outdir / f"{kind}_train.json").write_text(json.dumps(summary), encoding="utf-8")
    runs_volume.commit()
    print(f"train13b done: {r['seconds']} s, GPU utilisation {gpu.mean}", flush=True)
    return summary


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, memory=12288, timeout=90 * MINUTES)
def train_vae18(steps: int = 20000, batch: int = 32, lr: float = 1e-3, width: int = 192, depth: int = 4,
                heads: int = 4, z_col: int = 3, beta: float = 1e-4, free_bits: float = 0.0,
                warmup_frac: float = 0.3, seed: int = 0, out: str = "prior18", name: str = "vae_z3",
                maps_file: str = "gen18/maps_dct16.npz", run: str = "pairs18", sources: str = "all") -> dict:
    """18.24: the encoder and decoder whose latent is drawable (DECISIONS 2026-09-20).

    Trains on the same compact maps every arm of item 18 used, so the states
    are identical and only the model differs. The number that decides the run
    is not the loss: it is what the held-out latents look like. A Gaussian in D
    dimensions sits at radius sqrt(D) with spread 0.71, so the summary reports
    the held-out z's per-axis sd, its radius and the spread of that radius
    beside sqrt(D) — the acceptance condition the human set is sd 0.99-1.01 and
    radius within about 1.4 of sqrt(D), together with a reconstruction good
    enough to render.
    """
    import numpy as np
    import torch
    from flydream.generate import vae18 as V
    from flydream.generate.invert import GpuSampler

    dev = torch.device("cuda")
    t0 = time.time()
    _, m, _, stats = _load_maps(dev, file=maps_file, with_videos=False, sources=sources, run=run)
    _, mv, _, _ = _load_maps(dev, file=maps_file, subset="val", limit=256, with_videos=False,
                             sources=sources, run=run)
    frames, k = int(m.shape[1]), int(m.shape[2])
    print(f"train {tuple(m.shape)}, val {tuple(mv.shape)} on GPU in {time.time() - t0:.0f} s", flush=True)
    model = V.build(frames=frames, k=k, n=int(m.shape[3]), z_col=z_col, width=width, depth=depth, heads=heads)
    n_par = sum(p.numel() for p in model.parameters())
    D = int(m.shape[3]) * z_col
    print(f"state VAE: {n_par} parameters, latent {int(m.shape[3])} x {z_col} = {D} "
          f"(sqrt(D) = {np.sqrt(D):.1f}), compression {int(np.prod(m.shape[1:])) / D:.1f}x", flush=True)
    with GpuSampler() as gpu:
        r = V.train(model, m, steps=steps, batch=batch, lr=lr, beta=beta, free_bits=free_bits,
                    warmup_frac=warmup_frac, seed=seed, log_every=200,
                    log=lambda s_: print(s_, flush=True), val=mv, val_every=1000)
    outdir = Path(RUNS) / out
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"kind": "state_vae", "frames": frames, "k": k, "n": int(m.shape[3]), "z_col": z_col,
            "width": width, "depth": depth, "heads": heads, "latent_dims": D, "steps": steps,
            "batch": batch, "lr": lr, "beta": beta, "free_bits": free_bits, "warmup_frac": warmup_frac,
            "parameters": int(n_par), "seed": seed, "sources": sources, "maps_file": maps_file, "run": run,
            "mean": stats["mean"].tolist(), "std": stats["std"].tolist(),
            "dct_k": int(stats.get("dct_k", 0) or 0), "time_frames": int(stats.get("time_frames", 0) or 0),
            "coef_mean": None if "coef_mean" not in stats else stats["coef_mean"].tolist(),
            "coef_std": None if "coef_std" not in stats else stats["coef_std"].tolist(),
            "n_train": int(m.shape[0])}
    V.save(outdir / f"{name}.pt", model, r["ema"], meta)
    last = r["history"][-1]
    summary = {**meta, "history": r["history"], "seconds": r["seconds"],
               "seconds_worker": round(time.time() - t0, 1), "gpu": GPU, "gpu_utilisation": gpu.mean,
               "cpu": 1, "memory_mb": 12288,
               "acceptance": {kx: last.get(kx) for kx in
                              ("val_rec", "val_z_sd", "val_radius_mean", "val_radius_sd",
                               "val_typical_radius", "val_dims")}}
    (outdir / f"{name}_train.json").write_text(json.dumps(summary), encoding="utf-8")
    runs_volume.commit()
    a = summary["acceptance"]
    print(f"train_vae18 done: {r['seconds']} s, GPU {gpu.mean}", flush=True)
    print(f"  held-out z: sd {a['val_z_sd']:.3f} (target 1.00), radius {a['val_radius_mean']:.1f} "
          f"+- {a['val_radius_sd']:.2f} against sqrt(D) = {a['val_typical_radius']:.1f} +- 0.71; "
          f"rec {a['val_rec']:.4f}", flush=True)
    return summary


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, memory=16384, timeout=60 * MINUTES)
def invert17(prior: str = "prior18/corpus_dct16_w192_lr1e3_c.pt", out: str = "prior18/eps_w192.npz",
             maps_file: str = "gen18/maps_dct16.npz", run: str = "pairs18", sources: str = "all",
             steps: int = 20, fixed_point: int = 3, batch: int = 64, normalise: str = "sd") -> dict:
    """18.23: every training state's own preimage, put where a draw lands.

    The human, 2026-09-20: "пары работают, они просто лежат не там". `to_noise`
    already gives each real state the noise this prior would have drawn it
    from, and the round trip through it reproduces the clip at r = 0.957. The
    only defect is where those pairs sit: radius 257 with per-axis sd 0.848
    against the shell at 303.8, which no draw ever reaches.

    This inverts the whole training set once and rescales the result to a
    standard normal, so `train17(couple="file")` can be retrained on pairs that
    lie where the sampler actually draws. Unlike 18.20's arbitrary assignment,
    the map from noise to state here is the smooth inverse of a neural ODE, so
    nearby states keep nearby preimages and the pairing is learnable.

    `normalise="sd"` divides by the measured per-axis sd (keeps the relative
    spread of radii); `"shell"` puts every sample on sqrt(D) exactly; `"none"`
    saves the raw preimages.
    """
    import numpy as np
    import torch
    from flydream.generate import prior17 as R
    from flydream.generate.invert import GpuSampler

    dev = torch.device("cuda")
    t0 = time.time()
    _, m, idx, _ = _load_maps(dev, file=maps_file, with_videos=False, sources=sources, run=run)
    model, meta = R.load(Path(RUNS) / prior, dev)
    D = int(np.prod(m.shape[1:]))
    print(f"inverting {tuple(m.shape)} with {prior} (D = {D}, sqrt(D) = {np.sqrt(D):.1f})", flush=True)

    eps = np.empty((len(m),) + tuple(m.shape[1:]), np.float16)
    with GpuSampler() as gpu:
        for i in range(0, len(m), batch):
            x = m[i:i + batch].float()
            e = R.invert(model, x, steps=steps, fixed_point=fixed_point)
            eps[i:i + batch] = e.cpu().numpy().astype(np.float16)
            if (i // batch) % 20 == 0:
                print(f"  {i + len(x)}/{len(m)}, {time.time() - t0:.0f} s", flush=True)
    raw = eps.reshape(len(eps), -1).astype(np.float32)
    sd0, r0 = float(raw.std()), float(np.linalg.norm(raw, axis=1).mean())
    if normalise == "sd":
        eps = (raw / sd0).reshape(eps.shape).astype(np.float16)
    elif normalise == "shell":
        eps = (raw * (np.sqrt(D) / np.linalg.norm(raw, axis=1, keepdims=True))).reshape(eps.shape).astype(np.float16)
    f = eps.reshape(len(eps), -1).astype(np.float32)
    rad = np.linalg.norm(f, axis=1)
    stats = {"prior": prior, "maps_file": maps_file, "run": run, "sources": sources, "n": int(len(eps)),
             "dims": D, "typical_radius": float(np.sqrt(D)), "steps": steps, "fixed_point": fixed_point,
             "normalise": normalise, "sd_before": sd0, "radius_before": r0,
             "sd_after": float(f.std()), "radius_after_mean": float(rad.mean()),
             "radius_after_sd": float(rad.std()), "radius_sd_of_gaussian": float(np.sqrt(0.5)),
             "gpu": GPU, "gpu_utilisation": gpu.mean, "cpu": 1, "memory_mb": 16384,
             "seconds": round(time.time() - t0, 1)}
    outp = Path(RUNS) / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, eps=eps, index=idx, stats=json.dumps(stats))
    runs_volume.commit()
    print(f"invert17 done in {stats['seconds']} s: sd {sd0:.3f} -> {stats['sd_after']:.3f}, "
          f"radius {r0:.1f} -> {stats['radius_after_mean']:.1f} +- {stats['radius_after_sd']:.2f} "
          f"(a Gaussian sits at {np.sqrt(D):.1f} +- 0.71), GPU {gpu.mean}", flush=True)
    return stats


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, memory=12288, timeout=90 * MINUTES)
def train17(steps: int = 20000, batch: int = 32, lr: float = 3e-4, width: int = 128, depth: int = 4, heads: int = 4,
            seed: int = 0, out: str = "prior17", sources: str = "all", dct_k: int = 0, name: str = "state_flow",
            maps_file: str = "gen13b/maps_deep.npz", run: str = "pairs13", compile_mode: str = "",
            classes: bool = False, loss_weight_p: float = 0.0, label_drop: float = 0.0,
            couple: str = "random", couple_file: str = "", couple_norm: str = "shell") -> dict:
    """17.1: the prior over T4/T5 states — flow matching on the same maps 13B
    was conditioned on (`prior17.train`), no condition of its own; validation
    loss every 500 steps; checkpoint with EMA weights on /runs/<out>/<name>.pt.
    One T4, the states alone on the card.

    17.1b (`sources="sintel"`, `dct_k=16`): scene states only, and the time
    axis compressed to its first DCT coefficients, each z-scored over the
    training subset. K = 16 keeps 99.55 % of the energy and costs a round trip
    of 0.045 on real states; K = 8 costs 0.58 (measured before the run)."""
    import numpy as np
    import torch
    from flydream.generate import prior17 as R
    from flydream.generate.invert import GpuSampler

    dev = torch.device("cuda")
    t0 = time.time()
    _, m, idx_tr, stats = _load_maps(dev, file=maps_file, with_videos=False, sources=sources, run=run)
    _, mv, idx_val, _ = _load_maps(dev, file=maps_file, subset="val", limit=256, with_videos=False, sources=sources, run=run)
    print(f"train {tuple(m.shape)}, val {tuple(mv.shape)} on GPU in {time.time() - t0:.0f} s "
          f"({m.element_size() * m.nelement() / 1e9:.1f} GB)", flush=True)
    time_frames, coef_mean, coef_std = int(m.shape[1]), None, None
    if "coef_mean" in stats:                                         # item 18: already compact on the volume
        dct_k, time_frames = int(stats["dct_k"]), int(stats["time_frames"])
        coef_mean, coef_std = stats["coef_mean"], stats["coef_std"]
        print(f"maps file is DCT-{dct_k} over {time_frames} frames, coefficients z-scored on the volume", flush=True)
    elif dct_k:
        dmat = R.dct_matrix(time_frames, dct_k)
        m = R.to_dct(m.float(), dmat)
        mv = R.to_dct(mv.float(), dmat)
        coef_mean = m.mean((0, 3), keepdim=True)
        coef_std = m.std((0, 3), keepdim=True) + 1e-6
        m = ((m - coef_mean) / coef_std).half()
        mv = ((mv - coef_mean) / coef_std).half()
        coef_mean = coef_mean[0, :, :, 0].cpu().numpy(); coef_std = coef_std[0, :, :, 0].cpu().numpy()
        print(f"DCT-{dct_k}: train {tuple(m.shape)}, coefficient sd per index "
              f"{np.round(coef_std.mean(1), 3).tolist()}", flush=True)
    names, labels, val_labels, trained = [], None, None, None
    if classes:                                                      # 18.4e: the label the clip came with
        meta = json.loads((Path(RUNS) / run / "manifest.json").read_text(encoding="utf-8"))["meta"]
        of = [_clip_label(r) for r in meta]
        names = sorted(set(of))
        ids = {n: i for i, n in enumerate(names)}
        by_clip = np.array([ids[n] for n in of], np.int64)
        labels = torch.as_tensor(by_clip[idx_tr], device=dev)
        val_labels = torch.as_tensor(by_clip[idx_val], device=dev)
        cnt = np.bincount(by_clip[idx_tr], minlength=len(names))
        trained = np.where(cnt > 0)[0].tolist()                      # ISS-0007: the split holds out whole
        print(f"classes: {len(names)} labels, {cnt.min()}-{cnt.max()} training clips each; "
              f"{len(names) - len(trained)} have none and must not be sampled from "
              f"({', '.join(names[i] for i in range(len(names)) if cnt[i] == 0)})", flush=True)
    cw = None
    if loss_weight_p:                                                # 18.6: what a coefficient is actually worth
        if coef_std is None:
            raise SystemExit("loss_weight_p needs a DCT maps file; this one has no coefficient scale")
        sd = np.asarray(coef_std, np.float32).mean(1)                # (k,) averaged over the eight types
        wv = sd ** float(loss_weight_p)
        wv = (wv / wv.mean()).astype(np.float32)
        cw = torch.as_tensor(wv, device=dev)
        sh = wv / wv.sum()
        print(f"loss weight sd^{loss_weight_p}: coefficient 0 takes {100 * sh[0]:.1f} % of the loss, "
              f"the top half {100 * sh[len(sh) // 2:].sum():.2f} %, effective K {1 / (sh ** 2).sum():.2f} "
              f"of {len(sh)}", flush=True)
    if label_drop and not names:
        raise SystemExit("label_drop needs classes=True; there is nothing to drop")
    model = R.build(frames=m.shape[1], k=m.shape[2], width=width, depth=depth, heads=heads, n_classes=len(names),
                    null_class=bool(label_drop))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"state flow: {n_par} parameters", flush=True)
    couple_eps = None
    if couple == "file":                                             # 18.23: pairs from invert17, already standardised
        z = np.load(Path(RUNS) / couple_file)
        if not np.array_equal(np.asarray(z["index"]), np.asarray(idx_tr)):
            raise ValueError("the coupling file was inverted from a different subset of the maps")
        couple_eps = torch.as_tensor(z["eps"], device=dev)
        # 18.23: scaling by the per-axis sd fixes the marginal and leaves the
        # cloud elongated - measured, radius 302.7 +- 25.4 where a Gaussian in
        # D dimensions sits at sqrt(D) +- 0.71, i.e. 36x too wide. Putting
        # every pair on the shell is the far better stand-in for a draw, and
        # it is what the sampler will actually meet. Both the rescale and the
        # report run in chunks: the whole set in float32 is 5 GB and does not
        # fit beside the maps on a T4.
        Dc = int(np.prod(couple_eps.shape[1:]))
        tgt, chunk, rs, ss = float(Dc) ** 0.5, 512, [], 0.0
        for i in range(0, len(couple_eps), chunk):
            c = couple_eps[i:i + chunk].float().reshape(-1, Dc)
            if couple_norm == "shell":
                c *= tgt / c.norm(dim=1, keepdim=True)
                couple_eps[i:i + chunk] = c.reshape(couple_eps[i:i + chunk].shape).to(couple_eps.dtype)
            rs.append(c.norm(dim=1)); ss += float((c ** 2).sum())
            del c
        rr = torch.cat(rs)
        print(f"coupling from {couple_file} ({couple_norm}): {tuple(couple_eps.shape)}, "
              f"sd {(ss / (len(couple_eps) * Dc)) ** 0.5:.4f}, radius {float(rr.mean()):.1f} "
              f"+- {float(rr.std()):.2f} (a Gaussian sits at {tgt:.1f} +- 0.71)", flush=True)
    with GpuSampler() as gpu:
        r = R.train(model, m, steps=steps, batch=batch, lr=lr, seed=seed, compile_mode=compile_mode,
                    log_every=100, log=lambda s_: print(s_, flush=True), val=mv, val_every=500,
                    labels=labels, val_labels=val_labels, coef_weight=cw, label_drop=label_drop,
                    couple=couple, couple_eps=couple_eps)
    outdir = Path(RUNS) / out
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"kind": "sit_states", "frames": int(m.shape[1]), "k": int(m.shape[2]), "width": width, "depth": depth,
            "heads": heads, "steps": steps, "batch": batch, "lr": lr, "parameters": int(n_par), "seed": seed,
            "compile_mode": compile_mode, "n_classes": len(names), "class_names": names,
            "trained_classes": trained,
            "loss_weight_p": float(loss_weight_p), "label_drop": float(label_drop), "couple": couple,
            "couple_file": couple_file, "couple_norm": couple_norm,
            "mean": stats["mean"].tolist(), "std": stats["std"].tolist(), "sources": sources, "dct_k": int(dct_k),
            "time_frames": time_frames, "n_train": int(m.shape[0]),
            "coef_mean": None if coef_mean is None else coef_mean.tolist(),
            "coef_std": None if coef_std is None else coef_std.tolist()}
    R.save(outdir / f"{name}.pt", model, r["ema"], meta)
    summary = {**meta, "history": r["history"], "seconds": r["seconds"], "seconds_worker": round(time.time() - t0, 1),
               "gpu": GPU, "gpu_utilisation": gpu.mean, "cpu": 1, "memory_mb": 12288}
    (outdir / f"{name}_train.json").write_text(json.dumps(summary), encoding="utf-8")
    runs_volume.commit()
    print(f"train17 done: {r['seconds']} s, GPU utilisation {gpu.mean}", flush=True)
    return summary


def _bench_step_ms(R, model, states, *, steps: int, warmup: int, batch: int, seed: int,
                   host_sync: bool, foreach_ema: bool, net=None) -> float:
    """One timed training loop, milliseconds per step. The flags isolate the
    two places `prior17.train` pays for being written step-at-a-time:
    `host_sync` keeps its `loss.item()` (a device synchronisation every step),
    `foreach_ema` replaces the per-parameter EMA loop with two fused kernels.
    Everything else — data on the device, AMP fp16 with a GradScaler, fused
    AdamW, clipping — is the real loop (`research_notes/...
    /single_gpu_throughput.md`: 56.4 ms/step against a 15-23 ms floor)."""
    import time

    import numpy as np
    import torch

    dev = states.device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = net if net is not None else model
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.99), weight_decay=0.01, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    shadow = [p.detach().clone() for p in params]
    acc, t0 = torch.zeros((), device=dev), None
    for step in range(warmup + steps):
        if step == warmup:                                           # compile and cuDNN warmup are not timed
            torch.cuda.synchronize(); t0 = time.time()
        idx = torch.as_tensor(rng.integers(0, len(states), batch), device=dev)
        x1 = states[idx].float()
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            loss = R.loss_fn(net, x1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        with torch.no_grad():
            if foreach_ema:
                torch._foreach_mul_(shadow, 0.999)
                torch._foreach_add_(shadow, params, alpha=0.001)
            else:
                for s, p in zip(shadow, params):
                    s.mul_(0.999).add_(p.detach(), alpha=0.001)
        acc = acc + (loss.item() if host_sync else loss.detach())
    torch.cuda.synchronize()
    return round((time.time() - t0) / steps * 1000.0, 2)


@app.function(image=image, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=2, memory=12288,
              timeout=30 * MINUTES)
def class_probe18(maps_file: str = "gen18/maps_dct16.npz", run: str = "pairs18", shuffles: int = 3,
                  seed: int = 0) -> dict:
    """18.15: is there anything in the state to condition on, model aside?

    ISS-0008 voided both conditional arms — their label tables never left
    their initialisation — so "the label carries nothing" was never actually
    tested. This tests the data alone, with no generator in the way: how much
    of the state's variance the class explains, and whether the class can be
    read back from a held-out state.

    Neither number means anything by itself, so both carry controls. The
    **shuffle null** is the same statistic with the labels permuted — what a
    label carrying nothing must score. The **positive control** is the nine
    procedural generator classes (bar, dots, flash, grating, …): drastically
    different stimuli that *must* separate, so if they do not, the
    measurement is broken rather than the labels.

    The classifier is the nearest class mean in the state's own space, which
    is a linear classifier and needs no reduction. CPU only — a covariance
    and two matrix products, no GPU work.
    """
    import numpy as np

    t0 = time.time()
    z = np.load(Path(RUNS) / maps_file)
    meta = json.loads((Path(RUNS) / run / "manifest.json").read_text(encoding="utf-8"))["meta"]
    of = np.array([_clip_label(r) for r in meta])
    index, maps = z["index"], z["maps"]
    print(f"maps {tuple(maps.shape)} {maps.dtype}, {len(meta)} clips in the manifest", flush=True)
    part = {k: np.isin(index, z[k]) for k in ("train", "val")}
    has_colon = np.char.count(of[index], ":") > 0
    rng = np.random.default_rng(seed)

    def flat(rows) -> np.ndarray:
        x = np.asarray(maps[rows], np.float32)
        return x.reshape(len(x), -1)

    def between(Xc: np.ndarray, y: np.ndarray, n_cls: int, tot: float) -> float:
        """Share of the (already centred) total variance held by the class means."""
        b = 0.0
        for i in range(n_cls):
            m = y == i
            if m.any():
                b += float(m.sum()) * float(np.einsum("j,j->", Xc[m].mean(0), Xc[m].mean(0)))
        return b / (tot + 1e-12)

    out = {"maps_file": maps_file, "run": run, "shuffles": shuffles, "seed": seed, "groups": {}}
    for gname, want in (("actions_ucf101", False), ("procedural_control", True)):
        tr = np.where(part["train"] & (has_colon == want))[0]
        va = np.where(part["val"] & (has_colon == want))[0]
        names = sorted(set(of[index[tr]]))
        ids = {n: i for i, n in enumerate(names)}
        ytr = np.array([ids[n] for n in of[index[tr]]])
        keepv = np.array([n in ids for n in of[index[va]]], bool)
        va = va[keepv]
        yva = np.array([ids[n] for n in of[index[va]]])
        print(f"{gname}: {len(tr)} train, {len(va)} val, {len(names)} classes", flush=True)

        X = flat(tr)                                                 # (n, 16*8*721) float32
        mu = X.mean(0, keepdims=True)
        X -= mu                                                      # in place: the copy would be gigabytes
        tot = float(np.einsum("ij,ij->", X, X))
        e_real = between(X, ytr, len(names), tot)
        e_null = float(np.mean([between(X, rng.permutation(ytr), len(names), tot) for _ in range(shuffles)]))

        def score(y: np.ndarray) -> tuple[float, float]:
            cm = np.stack([X[y == i].mean(0) if (y == i).any() else np.zeros(X.shape[1], np.float32)
                           for i in range(len(names))])
            d = (cm * cm).sum(1)[None] - 2.0 * ((flat(va) - mu) @ cm.T)     # ‖x‖² is common to a row
            o = np.argsort(d, 1)
            return float((o[:, 0] == yva).mean()), float(np.mean([yva[i] in o[i, :5] for i in range(len(yva))]))

        t1, t5 = score(ytr)
        n1, n5 = score(rng.permutation(ytr))
        out["groups"][gname] = {
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_classes": len(names),
            "eta2": e_real, "eta2_shuffled": e_null, "eta2_ratio": e_real / (e_null + 1e-12),
            "top1": t1, "top5": t5, "top1_shuffled": n1, "top5_shuffled": n5,
            "chance": 1.0 / len(names), "top1_over_shuffled": t1 / (n1 + 1e-12)}
        print(f"  eta^2 {e_real:.4f} vs shuffled {e_null:.4f} ({e_real / (e_null + 1e-12):.2f}x) | "
              f"top-1 {100 * t1:.1f} % vs shuffled {100 * n1:.1f} % (chance {100 / len(names):.1f} %), "
              f"top-5 {100 * t5:.1f} % vs {100 * n5:.1f} %", flush=True)
        del X

    out["seconds"] = round(time.time() - t0, 1)
    out["cpu"], out["memory_mb"] = 2, 12288
    (Path(RUNS) / "prior18").mkdir(parents=True, exist_ok=True)
    (Path(RUNS) / "prior18" / "class_probe18.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                               encoding="utf-8")
    runs_volume.commit()
    print(f"class_probe18 done in {out['seconds']} s", flush=True)
    return out


def _clip_label(r: dict) -> str:
    """The class a clip carries. Ordinary video brings its UCF101 action; the
    procedural minority has no action, so its own generator class stands in —
    the alternative (one bucket for 3,103 stimuli of six kinds) would put the
    most different clips in the set under a single label."""
    if r.get("label"):
        return str(r["label"])
    kind = (r.get("params") or {}).get("class") or r.get("class") or r.get("scene") or "other"
    return f"{r.get('source', 'other')}:{kind}"


@app.function(image=image, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=2, memory=24576, timeout=45 * MINUTES)
def dct_maps(src: str = "gen18/maps_deep.npz", dct_out: str = "", dct_k: int = 32, dct_frames: int = 40) -> dict:
    """A second compact file at a different K from the maps that already exist
    — CPU only, and it **never touches `src`**. 18.3 left a floor of 0.021 that
    belongs to DCT-16 itself (a real state band-limited to it scores exactly
    that), so K is the one knob with an unambiguous direction; re-running
    `maps13b` would rebuild and overwrite a 9 GB file to get it."""
    import numpy as np
    from flydream.generate import prior17 as R

    t0 = time.time()
    z = np.load(Path(RUNS) / src)
    x, index = z["maps"], z["index"]
    mean, std, split = z["mean"], z["std"], {k: z[k] for k in ("train", "val", "test")}
    tf, k = min(int(dct_frames), x.shape[1]), x.shape[2]
    print(f"{src}: {tuple(x.shape)}, DCT-{dct_k} over {tf} frames in {time.time() - t0:.0f} s", flush=True)
    dmat = R.dct_matrix(tf, int(dct_k))
    c = np.empty((len(x), int(dct_k), k, x.shape[3]), np.float16)
    num = den = 0.0
    for i in range(0, len(x), 256):
        xb = x[i:i + 256, :tf].astype(np.float32)
        cb = R.to_dct(xb, dmat)                                      # orthonormal: energy is comparable
        num += float((cb ** 2).sum()); den += float((xb ** 2).sum())
        c[i:i + 256] = cb.astype(np.float16)
    del x, z
    s1 = np.zeros((dct_k, k)); s2 = np.zeros((dct_k, k)); n = 0      # z-scored on the training split only
    idx_tr = np.where(np.isin(index, split["train"]))[0]
    for i in range(0, len(idx_tr), 256):
        cb = c[idx_tr[i:i + 256]].astype(np.float32)
        s1 += cb.sum((0, 3)); s2 += (cb ** 2).sum((0, 3)); n += cb.shape[0] * cb.shape[3]
    cm = (s1 / n).astype(np.float32); cs = (np.sqrt(np.maximum(s2 / n - cm ** 2, 0)) + 1e-6).astype(np.float32)
    for i in range(0, len(c), 256):
        c[i:i + 256] = ((c[i:i + 256].astype(np.float32) - cm[None, :, :, None]) / cs[None, :, :, None]).astype(np.float16)
    dp = Path(RUNS) / (dct_out or f"{src.rsplit('/', 1)[0]}/maps_dct{dct_k}.npz")
    if dp.exists():                                                  # AGENTS: a dataset is never overwritten
        raise SystemExit(f"{dp} already exists; pass a different dct_out")
    dp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dp, maps=c, index=index, mean=mean, std=std, coef_mean=cm, coef_std=cs,
             time_frames=np.array(tf), dct_k=np.array(int(dct_k)),
             train=split["train"], val=split["val"], test=split["test"])
    r = {"src": src, "dct_file": str(dp.relative_to(RUNS)), "dct_shape": list(c.shape), "dct_k": int(dct_k),
         "time_frames": tf, "dct_gb": round(dp.stat().st_size / 1e9, 2),
         "dct_energy_kept": round(num / max(den, 1e-9), 5),          # DCT-16 kept 0.9932 on this corpus
         "dct_coefficient_sd": np.round(cs.mean(1), 3).tolist(), "cpu": 2, "memory_mb": 24576,
         "seconds": round(time.time() - t0, 1)}
    runs_volume.commit()
    print(f"DCT-{dct_k}: keeps {100 * num / max(den, 1e-9):.2f} % of the state energy, "
          f"{c.shape} -> {r['dct_gb']} GB in {r['seconds']} s", flush=True)
    return r


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, memory=12288, timeout=30 * MINUTES)
def bench17(steps: int = 60, warmup: int = 15, batch: int = 32, batches: str = "64,128,256", width: int = 128,
            depth: int = 4, heads: int = 4, seed: int = 0, sources: str = "all",
            maps_file: str = "gen18/maps_dct16.npz", run: str = "pairs18",
            compile_mode: str = "reduce-overhead") -> dict:
    """Where 18.3's 56.4 ms/step goes. Three measurements in one container:
    (a) the loop's own two step-at-a-time costs, isolated one at a time;
    (b) the same loop across batches, so the affine fit separates the fixed
    per-step cost from the per-sample one; (c) `torch.compile`, which is what
    the note names for the launch-bound symptom. No checkpoint is written and
    nothing on the volume is touched — this only measures."""
    import numpy as np
    import torch
    from flydream.generate import prior17 as R
    from flydream.generate.invert import GpuSampler

    dev = torch.device("cuda")
    t0 = time.time()
    _, m, _, stats = _load_maps(dev, file=maps_file, with_videos=False, sources=sources, run=run)
    print(f"train {tuple(m.shape)} on GPU in {time.time() - t0:.0f} s", flush=True)

    def fresh():                                                     # every variant starts from the same weights
        torch.manual_seed(seed)
        return R.build(frames=m.shape[1], k=m.shape[2], width=width, depth=depth, heads=heads).to(dev).train()

    rows = []

    def run_one(label, *, batch_, host_sync, foreach_ema, compiled=""):
        model = fresh()
        net = model
        if compiled:
            net = torch.compile(model, mode=compiled)
        ms = _bench_step_ms(R, model, m, steps=steps, warmup=warmup, batch=batch_, seed=seed,
                            host_sync=host_sync, foreach_ema=foreach_ema, net=net)
        rows.append({"variant": label, "batch": batch_, "host_sync": host_sync, "foreach_ema": foreach_ema,
                     "compile": compiled, "ms_per_step": ms, "samples_per_s": round(batch_ / ms * 1000.0, 1)})
        print(f"  {label:28s} batch {batch_:4d}  {ms:7.2f} ms/step  {rows[-1]['samples_per_s']:8.1f} samples/s",
              flush=True)
        del model, net
        torch.cuda.empty_cache()
        return ms

    with GpuSampler() as gpu:
        print(f"a) the loop's own costs at batch {batch}", flush=True)
        base = run_one("base (18.3's loop)", batch_=batch, host_sync=True, foreach_ema=False)
        run_one("no loss.item()", batch_=batch, host_sync=False, foreach_ema=False)
        run_one("foreach EMA", batch_=batch, host_sync=True, foreach_ema=True)
        best = run_one("both", batch_=batch, host_sync=False, foreach_ema=True)
        lean = best <= base
        print(f"b) batches {batches} on the {'leaner' if lean else 'original'} loop", flush=True)
        for b in [int(x) for x in batches.split(",") if x.strip()]:
            try:
                run_one(f"both, batch {b}", batch_=b, host_sync=not lean, foreach_ema=lean)
            except torch.cuda.OutOfMemoryError:                       # the card's limit is a measurement too
                rows.append({"variant": f"both, batch {b}", "batch": b, "ms_per_step": None, "note": "CUDA OOM"})
                print(f"  batch {b:4d}: CUDA OOM", flush=True)
                torch.cuda.empty_cache()
                break
        if compile_mode:
            print(f"c) torch.compile(mode={compile_mode!r}) at batch {batch}", flush=True)
            t_c = time.time()
            try:
                run_one(f"compile {compile_mode}", batch_=batch, host_sync=not lean, foreach_ema=lean,
                        compiled=compile_mode)
                rows[-1]["compile_warmup_s"] = round(time.time() - t_c, 1)
            except Exception as e:                                    # a compile failure must not lose (a) and (b)
                rows.append({"variant": f"compile {compile_mode}", "batch": batch, "ms_per_step": None,
                             "note": f"{type(e).__name__}: {e}"[:300]})
                print(f"  compile failed: {type(e).__name__}: {e}"[:300], flush=True)

    fit = {}
    pts = [(r["batch"], r["ms_per_step"]) for r in rows
           if r.get("ms_per_step") and r["variant"].startswith("both") and not r.get("compile")]
    if len(pts) >= 2:                                                # ms = fixed + per_sample * batch
        b_, y_ = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
        slope, inter = np.polyfit(b_, y_, 1)
        fit = {"fixed_ms": round(float(inter), 2), "per_sample_ms": round(float(slope), 4),
               "fixed_share_at_32": round(float(inter / (inter + slope * 32)), 3), "points": len(pts)}
        print(f"affine fit: {inter:.2f} ms fixed + {slope:.4f} ms per sample "
              f"({fit['fixed_share_at_32']:.0%} of a batch-32 step is fixed cost)", flush=True)
    out = {"rows": rows, "fit": fit, "baseline_18_3_ms": 56.4, "steps": steps, "warmup": warmup,
           "n_train": int(m.shape[0]), "shape": list(m.shape[1:]), "gpu": GPU, "gpu_utilisation": gpu.mean,
           "cpu": 1, "memory_mb": 12288, "seconds_worker": round(time.time() - t0, 1)}
    print(f"bench17 done in {out['seconds_worker']} s, GPU utilisation {gpu.mean}", flush=True)
    return out


def _nn_stream(arr, keep, qf, chunk: int = 512):
    """Nearest row of `arr[keep]` to each centred query row of `qf` (N, D):
    (bank index, correlation, normalised squared distance). Streamed in
    chunks so the bank is never copied beside itself."""
    import numpy as np

    qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    best_r = np.full(len(qf), -np.inf, np.float32); best_i = np.zeros(len(qf), np.int64); best_d = np.zeros(len(qf), np.float32)
    for s in range(0, len(keep), chunk):
        ids = keep[s:s + chunk]
        x = np.asarray(arr[ids], np.float32).reshape(len(ids), -1)
        x -= x.mean(1, keepdims=True)
        xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        r = (qn @ xn.T).astype(np.float32)
        j = r.argmax(1); v = r[np.arange(len(qf)), j]
        upd = np.where(v > best_r)[0]
        if len(upd):
            best_r[upd] = v[upd]; best_i[upd] = ids[j[upd]]
            best_d[upd] = ((qf[upd] - x[j[upd]]) ** 2).sum(1) / ((qf[upd] ** 2).sum(1) + 1e-12)
    return best_i, best_r, best_d


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume}, cpu=1, memory=8192, timeout=40 * MINUTES)
def sample17(model: str = "malecns", ckpt: str = "prior17/state_flow.pt", gen_ckpt: str = "gen13b/sit.pt",
             n_samples: int = 16, n_clips: int = 8, sample_steps: int = 20, seed: int = 0, out: str = "prior17") -> dict:
    """17.2: `prior -> state -> 13B -> video -> frozen brain -> state'`, and the
    gates. Round trip against the sampled state, novelty of the state and of
    the video against the training split, diversity of both, and the direction
    the brain reads. Every control is rebuilt here, in this path: a held-out
    clip's own state, white and structured noise in the types, a shuffled clip
    state — so all the numbers of the table share one scale."""
    import numpy as np
    import torch
    from flydream.decode import pairs as P
    from flydream.generate import gen13b as G
    from flydream.generate import learned as L
    from flydream.generate import prior17 as R
    from flydream.generate.gen13b import DEEP
    from flydream.generate.invert import GpuSampler, load_network
    from flydream.generate.pairs13 import simulate_states
    from flydream.generate.prompts14 import Deep, direction_energy
    from flydream.generate.roundtrip13 import build_states, round_trip

    t0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(seed); np.random.seed(seed)
    root = Path(RUNS) / "pairs13"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads(Path(f"{DATA}/pairs13/columns.json").read_text(encoding="utf-8"))
    d = Deep(manifest, columns)
    net = load_network(model)
    _, index = P.type_index(net.connectome)
    dt, t_pre, margin, frames = manifest["dt"], manifest["t_pre"], 5, 40
    T = frames + margin
    prior, pmeta = R.load(Path(RUNS) / ckpt, dev)
    gen, gmeta = G.load(Path(RUNS) / gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)
    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]
    ta = sa["target"][:, w0:w1, :][:, :, d.cells_all].cpu().numpy().astype(np.float32)
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}      # 13B's and item 14's normalisation
    print(f"model, prior ({pmeta['parameters']} par) and 13B loaded in {time.time() - t0:.0f} s", flush=True)

    # --- the states to render: the prior's samples and the controls, all in 13B's conditioning units ---
    z = np.load(Path(RUNS) / "gen13b/maps_deep.npz")
    idx_all = z["index"]
    tr = np.where(np.isin(idx_all, z["train"]))[0]
    te = np.where(np.isin(idx_all, z["test"]))[0][:n_clips]
    maps_all = z["maps"]                                                    # one 4.9 GB copy, reused for the bank below
    clip_maps = maps_all[te][:, :frames].astype(np.float32)
    g = torch.Generator(device=dev).manual_seed(2000 + seed)
    with torch.no_grad():
        prior_states = R.sample(prior, n_samples, frames=frames, k=len(DEEP), steps=sample_steps, device=dev,
                                generator=g).cpu().numpy().astype(np.float32)
    rng = np.random.default_rng(seed)
    ring = L.ring_index(1)
    white = rng.standard_normal((frames, len(DEEP), 721)).astype(np.float32)
    sm = white.copy()
    for _ in range(5):                                                      # ~100 ms in time, one hex ring in space
        sm = 0.5 * sm + 0.5 * np.concatenate([sm[:1], sm[:-1]], 0)
        sm = np.nanmean(np.where(ring[None, None] >= 0, sm[:, :, np.clip(ring, 0, None)], np.nan), -1)
    sm = (sm - sm.mean((0, 2), keepdims=True)) / (sm.std((0, 2), keepdims=True) + 1e-6)
    jobs = {f"prior_{k}": prior_states[k] for k in range(n_samples)}
    jobs.update({f"clip_{int(idx_all[i])}": clip_maps[j] for j, i in enumerate(te)})
    jobs["noise_white"] = white
    jobs["noise_structured"] = sm.astype(np.float32)
    jobs["shuffled_clip"] = clip_maps[0][:, :, rng.permutation(721)]
    names = list(jobs)
    kinds = {n: ("prior" if n.startswith("prior") else "clip" if n.startswith("clip") else "control") for n in names}

    # --- 13B renders them all, z shared across the states of a chunk ---
    cond = torch.as_tensor(np.stack([jobs[n] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    with GpuSampler() as gpu:
        vids = []
        with torch.no_grad():
            for i in range(0, len(names), 8):
                gg = torch.Generator(device=dev).manual_seed(1000 + seed)
                vids.append(G.sample(gen, cond[i:i + 8], mask[i:i + 8], steps=sample_steps, generator=gg).cpu().numpy())
        videos = np.concatenate(vids).astype(np.float32)
        print(f"{len(videos)} videos rendered in {time.time() - t0:.0f} s", flush=True)

        # --- the frozen brain: the round trip against the state that was asked for ---
        target_raw = np.stack([R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0] for n in names])
        rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
        vids_m = np.concatenate([videos, np.repeat(videos[:, -1:], margin, 1)], 1).astype(np.float16)
        st_back = simulate_states(net, vids_m, d.cells_all, dt, t_pre, 16).astype(np.float32)
        st_grey = simulate_states(net, np.full((1, T, 721), 0.5, np.float16), d.cells_all, dt, t_pre, 1).astype(np.float32)[0]
        print(f"round trips in {time.time() - t0:.0f} s", flush=True)

    # --- novelty: the nearest training state and the nearest training video ---
    def centred(x):
        f = np.asarray(x, np.float32).reshape(len(x), -1)
        return f - f.mean(1, keepdims=True)

    q_states = centred(np.stack([jobs[n] for n in names]))
    i_s, r_s, d_s = _nn_stream(maps_all, tr, q_states)
    del q_states, maps_all, clip_maps
    vids_all = z["videos"]
    q_vids = centred(videos)
    i_v, r_v, d_v = _nn_stream(vids_all, tr, q_vids)
    print(f"nearest neighbours over {len(tr)} training clips in {time.time() - t0:.0f} s", flush=True)

    def pw(x):
        n = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        c = n @ n.T
        iu = np.triu_indices(len(c), 1)
        return {"mean": float(c[iu].mean()), "max": float(c[iu].max())}

    sel = {k: [i for i, n in enumerate(names) if kinds[n] == k] for k in ("prior", "clip")}
    diversity = {f"videos_{k}": pw(q_vids[v]) for k, v in sel.items() if len(v) > 1}
    diversity.update({f"states_{k}": pw(centred(np.stack([jobs[names[i]] for i in v]))) for k, v in sel.items() if len(v) > 1})

    scores = {}
    for i, n in enumerate(names):
        de_v = direction_energy(st_back[i], st_grey, d)
        de_s = direction_energy(R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0], st_grey, d)
        scores[n] = {"kind": kinds[n], "round_trip": float(rts[i]),
                     "state_nn_r": float(r_s[i]), "state_nn_distance": float(d_s[i]), "state_nn_index": int(idx_all[i_s[i]]),
                     "video_nn_r": float(r_v[i]), "video_nn_distance": float(d_v[i]), "video_nn_index": int(idx_all[i_v[i]]),
                     "direction_state": de_s["T4_argmax"], "direction_video": de_v["T4_argmax"],
                     "video_mean": float(videos[i].mean()), "video_sd": float(videos[i].std()),
                     "frame_to_frame": float(np.abs(np.diff(videos[i], axis=0)).mean())}
        print(f"  {n:<16} {kinds[n]:<7} rt {scores[n]['round_trip']:.3f}  state nn r {scores[n]['state_nn_r']:+.2f}"
              f"  video nn r {scores[n]['video_nn_r']:+.2f}  dir {de_s['T4_argmax']}->{de_v['T4_argmax']}", flush=True)

    def group(k, q):
        v = [scores[n][q] for n in names if kinds[n] == k]
        return {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}

    summary = {"ckpt": ckpt, "gen_ckpt": gen_ckpt, "n_samples": n_samples, "n_clips": n_clips, "frames": frames,
               "sample_steps": sample_steps, "seed": seed, "prior": {k: pmeta[k] for k in ("width", "depth", "steps", "parameters")},
               "bank": {"train": int(len(tr))},
               "gates": {k: {q: group(k, q) for q in ("round_trip", "state_nn_r", "video_nn_r")} for k in ("prior", "clip")},
               "diversity": diversity, "scores": scores, "gpu": GPU, "gpu_utilisation": gpu.mean, "cpu": 1, "memory_mb": 8192,
               "seconds": round(time.time() - t0, 1)}
    outdir = Path(RUNS) / out
    outdir.mkdir(parents=True, exist_ok=True)
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    arrays.update({f"T4a__{n}": R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0][:, d.pos["T4a"]]
                   for n in names})
    arrays.update({f"nnvideo__{n}": np.asarray(vids_all[i_v[i]][:frames], np.float32) for i, n in enumerate(names)})
    np.savez_compressed(outdir / "samples17.npz", **arrays)
    (outdir / "samples17.json").write_text(json.dumps(summary), encoding="utf-8")
    runs_volume.commit()
    print(f"sample17 done in {time.time() - t0:.0f} s, GPU utilisation {gpu.mean}", flush=True)
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
    rng = np.random.default_rng(seed)
    init_np = np.full((B, T, 721), 0.5, np.float32)
    for c in range(n_clips):
        for k in range(n_starts):
            init_np[c * (n_starts + 1) + 1 + k] = np.clip(0.5 + init_sd * rng.standard_normal((T, 721)), 0, 1)
    init = torch.as_tensor(init_np, device=dev)
    tg = targets.repeat_interleave(n_starts + 1, 0)
    state = net.steady_state(t_pre, dt, batch_size=B, value=0.5)
    with GpuSampler() as gpu:
        inv, tr, n_steps = invert_batch(net, tg, [w_cells] * B, dt=dt, state=state, steps=inv_steps, lr=0.05, tv=0.02,
                                        plateau_steps=20, plateau_tol=0.01, log_every=50, cell_weights=[w] * B, init=init)
        inv_np = inv.numpy()[:, :40]
        rt = round_trip(net, inv_np, tg_states.repeat(n_starts + 1, 0), cells_all, type_of, DEEP, dt, t_pre,
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
    np.savez_compressed(outdir / "multi_init.npz", ids=ids, videos=inv_np, true=vids[:, :40], roundtrip=rt, init=init_np[:, :40])
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
        x = torch.as_tensor(L.to_maps(st.astype(np.float16), layout, len(DEEP)).astype(np.float32), device=dev)[:, :frames]
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
    add("clip_A", "full", 1.0, [0], maps=torch.zeros(1, frames, 8, 721, device=dev), tag="strength_zero")
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
                x0 = torch.stack([torch.randn(frames, 721, device=dev, generator=torch.Generator(device=dev).manual_seed(1000 + c[5])) for c in chunk])
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
         steps13b: int = 6000, batch13b: int = 32, width13b: int = 64, depth13b: int = 6, compile13b: bool = False,
         train17_run: bool = False, steps17: int = 20000, batch17: int = 32, width17: int = 128, depth17: int = 4,
         sample17_run: bool = False, samples17: int = 16, sources17: str = "all", dct17: int = 0,
         name17: str = "state_flow", maps_file17: str = "gen13b/maps_deep.npz", run17: str = "pairs13",
         pairs18_run: bool = False, maps18_run: bool = False, videos18: str = "corpus18/videos.npz",
         out18: str = "pairs18", gen18: str = "gen18", held18: str = "", mem18: int = 16384,
         sim_batch18: int = 32, model18: str = "malecns"):
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
    if pairs18_run:                                                  # ROADMAP 18.1: the corpus through the frozen brain
        print(f"pairs18 on {GPU}: {videos18} -> states on /runs/{out18} with {model18}, batch {sim_batch18}, "
              f"held-out labels: {held18 or '(none)'}")
        r = pairs13.remote(videos_file=videos18, model=model18, frames=45, held_scenes=held18, val_fraction=0.05,
                           sim_batch=sim_batch18, out=out18)
        d = root / "data" / out18
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest_summary.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps(r, indent=1)[:1200])
        print(f"done in {time.time() - t0:.0f} s")
        return
    if maps18_run:                                                   # the maps on 13B's scale, plus the compact file
        k = dct17 or 16
        print(f"maps18 on CPU (2 cores, {mem18 // 1024} GB): /runs/{out18} shards -> {gen18}/maps_deep.npz and "
              f"maps_dct{k}.npz, per-type scale from gen13b/maps_deep.npz")
        r = maps13b.with_options(memory=mem18).remote(                # the corpus needs more than 13A's 12 GB
            run=out18, out=f"{gen18}/maps_deep.npz", stats_from="gen13b/maps_deep.npz",
            dct_k=k, dct_out=f"{gen18}/maps_dct{k}.npz")
        d = root / "data" / gen18
        d.mkdir(parents=True, exist_ok=True)
        (d / "maps.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps(r, indent=1))
        print(f"done in {time.time() - t0:.0f} s")
        return
    if train17_run:
        print(f"train17 on {GPU}: state flow {width17}x{depth17}, {steps17} steps, batch {batch17}, "
              f"sources {sources17}, DCT {dct17 or 'off'}, maps {maps_file17} -> {name17}.pt")
        r = train17.remote(steps=steps17, batch=batch17, width=width17, depth=depth17, seed=seed,
                           sources=sources17, dct_k=dct17, name=name17, maps_file=maps_file17, run=run17)
        d = root / "data" / "prior17"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name17}_train.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        h = r["history"]
        print(json.dumps({k: v for k, v in r.items() if k not in ("history", "mean", "std")}, indent=1)[:1500])
        print(f"loss {h[0]['loss']:.4f} -> {h[-1]['loss']:.4f}, val {h[-1].get('val_loss')}")
        print(f"done in {time.time() - t0:.0f} s; GPU utilisation {r['gpu_utilisation']}")
        return
    if sample17_run:
        print(f"sample17 on {GPU}: {samples17} states from the prior -> 13B -> brain, with the controls")
        r = sample17.remote(model=model, n_samples=samples17, seed=seed)
        d = root / "data" / "prior17"
        d.mkdir(parents=True, exist_ok=True)
        (d / "samples17.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        for k, v in r["gates"].items():
            print(f"{k:>6}: round trip {v['round_trip']['median']:.3f} ({v['round_trip']['min']:.3f}-{v['round_trip']['max']:.3f})"
                  f"  state nn r {v['state_nn_r']['median']:+.2f}  video nn r {v['video_nn_r']['median']:+.2f}")
        for n in ("noise_white", "noise_structured", "shuffled_clip"):
            c = r["scores"][n]
            print(f"{n:>16}: round trip {c['round_trip']:.3f}  state nn r {c['state_nn_r']:+.2f}")
        print(json.dumps(r["diversity"], indent=1))
        print(f"done in {time.time() - t0:.0f} s; GPU utilisation {r['gpu_utilisation']}")
        return
    if maps13b_run or bench13b_run or train13b_run or multi_init13b_run or sample13b_run:
        d = root / "data" / "gen13b"
        d.mkdir(parents=True, exist_ok=True)
        if maps13b_run:
            print("maps13b on CPU (2 cores, 12 GB): pairs13 shards -> deep maps"); r = maps13b.remote()
            (d / "maps.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        if bench13b_run:
            print(f"bench13b on {GPU}: hexresnet {width13b}x{depth13b} and sit, 200 steps, batch {batch13b}")
            r = bench13b.remote(steps=200, batch=batch13b, width_hex=width13b, depth_hex=depth13b, compile_model=compile13b)
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
