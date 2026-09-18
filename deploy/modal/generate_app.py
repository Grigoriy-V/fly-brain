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
              cpu=2, memory=12288, timeout=60 * MINUTES)
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
              cpu=2, memory=12288, timeout=60 * MINUTES)
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
              cpu=2, memory=12288, timeout=60 * MINUTES)
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


@app.function(image=image, gpu=GPU, volumes={DATA: data_volume, RUNS: runs_volume},
              cpu=4, memory=24576, timeout=120 * MINUTES)
def pairs13(procedural: str = "pairs13/procedural_800_s0.npz", model: str = "malecns", frames: int = 45,
            dt: float = 0.02, t_pre: float = 1.0, sim_batch: int = 32, shard: int = 512, seed: int = 0,
            held_scenes: str = "", held_classes: str = "", val_fraction: float = 0.1, out: str = "pairs13") -> dict:
    """ROADMAP 13A data: the augmented Sintel clips (rendered here, cached on the
    data volume) plus the uploaded procedural clips, simulated through model
    zero; the 14 ladder types' activity written float16 in shards under
    /runs/<out>/ with a manifest (meta per clip, the split)."""
    import numpy as np

    _prepare_root()
    from flydream.generate.invert import load_network
    from flydream.generate.pairs13 import LADDER_TYPES, simulate_states, sintel_videos, split_indices
    from flydream.decode import pairs as P

    t0 = time.time()
    z = np.load(f"{DATA}/{procedural}")
    proc_v, proc_p = z["videos"], [json.loads(x) for x in z["params"]]
    print(f"procedural: {proc_v.shape}", flush=True)
    sin_v, scenes, names = sintel_videos(frames, dt)
    print(f"sintel augmented: {sin_v.shape}, {len(set(scenes))} scenes, {time.time() - t0:.0f} s", flush=True)
    data_volume.commit()          # the rendered sintel cache
    videos = np.concatenate([sin_v, proc_v])
    meta = [{"source": "sintel", "scene": sc, "name": nm} for sc, nm in zip(scenes, names)] +            [{"source": "procedural", "class": p["class"], "params": p} for p in proc_p]
    split = split_indices(meta, held_scenes.split(",") if held_scenes else [], held_classes.split(",") if held_classes else [],
                          val_fraction, seed)
    net = load_network(model)
    _, index = P.type_index(net.connectome)
    cells = np.concatenate([index[t] for t in LADDER_TYPES])
    type_of = np.concatenate([[t] * len(index[t]) for t in LADDER_TYPES])
    outdir = Path(RUNS) / out
    outdir.mkdir(parents=True, exist_ok=True)
    shards = []
    for i in range(0, len(videos), shard):
        act = simulate_states(net, videos[i:i + shard], cells, dt, t_pre, sim_batch)
        f = outdir / f"shard_{i // shard:03d}.npz"
        np.savez(f, videos=videos[i:i + shard], states=act, index=np.arange(i, i + len(act)))
        shards.append(f.name)
        print(f"  {f.name}: {act.shape} ({time.time() - t0:.0f} s)", flush=True)
    manifest = {"model": model, "frames": frames, "dt": dt, "t_pre": t_pre, "types": LADDER_TYPES,
                "cells": cells.tolist(), "type_of_cell": type_of.tolist(), "n": int(len(videos)),
                "n_sintel": int(len(sin_v)), "n_procedural": int(len(proc_v)), "shards": shards, "shard_size": shard,
                "meta": meta, "split": split, "seconds": round(time.time() - t0, 1), "gpu": GPU}
    (outdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    runs_volume.commit()
    print(f"pairs13: {len(videos)} clips in {len(shards)} shards, {time.time() - t0:.0f} s on {GPU}", flush=True)
    return {k: v for k, v in manifest.items() if k not in ("meta", "cells", "type_of_cell")} | {
        "split_sizes": {k: len(v) for k, v in split.items()}}


@app.local_entrypoint()
def main(model: str = "flow/0000/000", sample: int = 3, frames: int = -1, margin: int = -1, steps: int = -1,
         stages: str = "", dream_sources: str = "", seed: int = 0, mix_clips: str = "", pairs13_run: bool = False):
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
    if pairs13_run:
        P13 = _generate_settings_all().get("pairs13", {})
        print(f"pairs13 on {GPU}: {model}, {frames + margin} frames, procedural {P13.get('n_per_class', 800)}/class")
        r = pairs13.remote(procedural=f"pairs13/procedural_{P13.get('n_per_class', 800)}_s{P13.get('seed', 0)}.npz",
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
