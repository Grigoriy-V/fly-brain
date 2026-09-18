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
            out: str = "train13") -> dict:
    """ROADMAP 13A: per condition, the linear hex-temporal decoder and the
    hex+temporal CNN trained on the pairs13 shards; r on val and test (by
    source and class); the round trip through the frozen brain on
    `n_roundtrip` test clips for both models and for the Adam inversion of the
    same clips (per-type normalised loss, as item 12). Predictions of those
    clips are saved for the figures."""
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
            print(f"  {kind}: {n_par} parameters", flush=True)
            fit = L.train_model(mdl, train, val, t_out=frames, epochs=epochs, batch=batch, lr=lr, device=dev,
                                log=lambda s_: print(s_, flush=True))
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


@app.local_entrypoint()
def main(model: str = "flow/0000/000", sample: int = 3, frames: int = -1, margin: int = -1, steps: int = -1,
         stages: str = "", dream_sources: str = "", seed: int = 0, mix_clips: str = "", pairs13_run: bool = False,
         train13_run: bool = False, epochs: int = 12, pairs13_videos_run: bool = False):
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
    if train13_run:
        import numpy as np
        print(f"train13 on {GPU}: {model}, {epochs} epochs")
        r = train13.remote(model=model, epochs=epochs, frames=frames, margin=margin, seed=seed)
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
