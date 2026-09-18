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
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import time
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
def ladder(model: str = "flow/0000/000", sample: int = 3, frames: int = 20, steps: int = 150,
           lr: float = 0.05, tv: float = 0.02, dt: float = 0.02, stages: str = "") -> list[dict]:
    import numpy as np
    from flydream.generate.invert import run_ladder

    _prepare_root()
    st = [s.split("+") for s in stages.split(",")] if stages else STAGES
    prefix = f"{time.strftime('%Y-%m-%d')}_{'malecns' if model.startswith('malecns') else 'flyvis'}_"
    t0 = time.time()
    records = run_ladder(model, sample, st, frames=frames, steps=steps, lr=lr, tv=tv, dt=dt, t_pre=1.0,
                         out_root=Path(GEN_ROOT) / "data" / "generate", tag_prefix=prefix)
    runs_volume.commit()
    out = []
    for r in records:
        buf = io.BytesIO()
        np.savez_compressed(buf, **r["arrays"])
        out.append({**{k: v for k, v in r.items() if k != "arrays"}, "npz_b64": base64.b64encode(buf.getvalue()).decode()})
    print(f"ladder of {len(out)} stages in {time.time() - t0:.0f} s on {GPU}", flush=True)
    return out


@app.local_entrypoint()
def main(model: str = "flow/0000/000", sample: int = 3, frames: int = 20, steps: int = 150, stages: str = ""):
    root = Path(__file__).resolve().parents[2]
    t0 = time.time()
    records = ladder.remote(model=model, sample=sample, frames=frames, steps=steps, stages=stages)
    tags = []
    for r in records:
        d = root / "data" / "generate" / r["tag"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "recovered.npz").write_bytes(base64.b64decode(r.pop("npz_b64")))
        (d / "meta.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        tags.append(r["tag"])
        print(f"{r['tag']:<40} {r['cells']:>6} cells  r {r['inversion']:+.3f}  control {r['control']:+.3f}  {r['seconds']} s")
    print(f"done in {time.time() - t0:.0f} s; draw with:\n  python -m flydream.generate.figures --tags {' '.join(tags)} "
          f"--out {tags[0].rsplit('_invert_', 1)[0]}_inversion_ladder --model {model}")
