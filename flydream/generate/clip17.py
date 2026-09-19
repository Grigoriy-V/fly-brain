"""17.3 (first half): the 17.1b prior applied to one known clip's state.

    python -m flydream.generate.clip17 --sample 3        # local CPU, $0

The prior samples states from noise; to see what it does *with* a clip, the
clip's own state is taken to a noise level t and finished by the prior
(`prior17.refine`, the state-space analogue of image-to-image). t = 1 is the
state itself, t = 0 an unconditional sample. Every state — the real one, the
refined ones and a pure sample — goes through 13B and the frozen brain, so
each cell carries its round trip (is this a state the brain accepts?) and its
correlation to the clip the face came from (did the scene survive?).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flydream.decode import pairs as P
from flydream.generate import gen13b as G
from flydream.generate import learned as L
from flydream.generate import prior17 as R
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import clip_from_sintel, device_of, load_network, pixcorr_per_frame
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep, direction_energy
from flydream.generate.roundtrip13 import build_states, round_trip


def run(model: str, prior_ckpt, gen_ckpt, manifest: dict, columns: dict, *, sample: int = 3, levels=(0.8, 0.6, 0.3),
        n_prior: int = 1, frames: int = 40, margin: int = 5, dt: float = 0.02, t_pre: float = 1.0,
        sample_steps: int = 20, seed: int = 0, log=print) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    net = load_network(model); dev = device_of(net)
    _, index = P.type_index(net.connectome)
    d = Deep(manifest, columns)
    prior, pmeta = R.load(prior_ckpt, dev)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)
    T = frames + margin
    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]
    ta = sa["target"][:, w0:w1, :][:, :, d.cells_all].cpu().numpy().astype(np.float32)
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}
    log(f"model, prior ({pmeta.get('sources')}, DCT {pmeta.get('dct_k')}) and 13B loaded, {time.time() - t0:.0f} s")

    clip = clip_from_sintel(sample, frames, dt, margin)
    st = simulate_states(net, clip[None].astype(np.float16), d.cells_all, dt, t_pre, 1).astype(np.float32)
    real = (L.to_maps(st.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
            - mean[None, None, :, None]) / std[None, None, :, None]
    jobs = {"real": real[0]}
    for t in levels:
        g = torch.Generator(device=dev).manual_seed(3000 + seed)
        jobs[f"refined_t{t:g}"] = R.refine(prior, pmeta, real, t0=t, steps=sample_steps, device=dev, generator=g)[0]
    g = torch.Generator(device=dev).manual_seed(2000 + seed)
    pure = R.sample_states(prior, pmeta, n_prior, steps=sample_steps, device=dev, generator=g)
    for k in range(n_prior):
        jobs[f"prior_{k}"] = pure[k]
    names = list(jobs)

    cond = torch.as_tensor(np.stack([jobs[n] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    with torch.no_grad():
        gg = torch.Generator(device=dev).manual_seed(1000 + seed)
        videos = G.sample(gen, cond, mask, steps=sample_steps, generator=gg).cpu().numpy().astype(np.float32)
    target_raw = np.stack([R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0] for n in names])
    rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
    vids_m = np.concatenate([videos, np.repeat(videos[:, -1:], margin, 1)], 1).astype(np.float16)
    st_back = simulate_states(net, vids_m, d.cells_all, dt, t_pre, 8).astype(np.float32)
    st_grey = simulate_states(net, np.full((1, T, 721), 0.5, np.float16), d.cells_all, dt, t_pre, 1).astype(np.float32)[0]

    flat_real = real[0].reshape(-1); flat_real = flat_real - flat_real.mean()
    scores = {}
    for i, n in enumerate(names):
        q = jobs[n].reshape(-1); q = q - q.mean()
        scores[n] = {"round_trip": float(rts[i]),
                     "r_state_to_clip": float(q @ flat_real / (np.linalg.norm(q) * np.linalg.norm(flat_real) + 1e-9)),
                     "r_video_to_clip": float(pixcorr_per_frame(videos[i], clip[:frames]).mean()),
                     "direction_state": direction_energy(target_raw[i], st_grey, d)["T4_argmax"],
                     "direction_video": direction_energy(st_back[i], st_grey, d)["T4_argmax"],
                     "frame_to_frame": float(np.abs(np.diff(videos[i], axis=0)).mean())}
        log(f"  {n:<14} rt {scores[n]['round_trip']:6.3f}  r состояния к клипу {scores[n]['r_state_to_clip']:+.2f}"
            f"  r видео к клипу {scores[n]['r_video_to_clip']:+.2f}  dir {scores[n]['direction_state']}->{scores[n]['direction_video']}")
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    arrays.update({f"T4a__{n}": target_raw[i][:, d.pos["T4a"]] for i, n in enumerate(names)})
    arrays["clip"] = clip[:frames]
    summary = {"model": model, "prior_ckpt": str(prior_ckpt), "sample": sample, "levels": list(levels), "frames": frames,
               "sample_steps": sample_steps, "seed": seed, "scores": scores,
               "prior_meta": {k: pmeta.get(k) for k in ("sources", "dct_k", "n_train", "steps")},
               "seconds": round(time.time() - t0, 1)}
    return {"summary": summary, "arrays": arrays}


def main(argv=None) -> int:
    from flydream.generate.invert import settings
    from flydream.model import ROOT

    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--prior", default=str(ROOT / "data" / "prior17" / "scene_dct16.pt"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior17"))
    p.add_argument("--tag", default="clip17_local")
    p.add_argument("--sample", type=int, default=3)
    p.add_argument("--levels", default="0.8,0.6,0.3")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.prior), Path(a.gen), manifest, columns, sample=a.sample,
            levels=tuple(float(x) for x in a.levels.split(",")), frames=g.get("frames", 40), margin=g.get("margin", 5),
            dt=g.get("dt", 0.02), t_pre=g.get("t_pre", 1.0), seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    print(f"wrote {out / a.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
