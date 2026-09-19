"""17.3b: the prior run backwards — every state's own noise, and the space between two clips.

    python -m flydream.generate.noise17            # local CPU, no Modal, $0

The prior is a deterministic map: noise at t = 0 is carried to a state at
t = 1 by Euler steps. Run those steps backwards and a **real** state gets the
noise it would have been drawn from (`prior17.to_noise`), which is the
"состояние → шум и обратно" the human asked for. Three things are measured,
all in the code path the earlier gates used:

- **the inversion itself**: noise → state → noise recovers the noise it
  started from (the check the rest depends on), and a real clip's state comes
  back through its own noise — against the DCT-16 ceiling, which is the best
  this representation allows;
- **whether the prior covers a real state**: a Gaussian of dimension D
  concentrates on the radius ‖ε‖²/D = 1 ± 3·√(2/D). Where a real state's own
  noise lands on that scale says whether the prior assigns it any density —
  a number that the round trip alone cannot give;
- **the space between two clips**: the noises of clip A and clip B mixed on
  the sphere (`prior17.slerp`), each mixture carried forward to a state, then
  through 13B and the frozen brain. These states have no source clip, so they
  carry the round trip, the correlation to *both* clips and the nearest
  training video, beside the known-unreachable control (clip A's state with
  its columns permuted).
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
from flydream.generate.baseline17 import flat, nearest, video_bank
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import clip_from_sintel, device_of, load_network, pixcorr_per_frame
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep, direction_energy
from flydream.generate.roundtrip13 import build_states, round_trip


def radius(e: np.ndarray) -> np.ndarray:
    """‖ε‖²/D per sample — 1.0 for noise the prior was trained to start from."""
    f = np.asarray(e, np.float32).reshape(len(e), -1)
    return (f ** 2).sum(1) / f.shape[1]


def scene_name(sample: int, dt: float) -> str:
    """The AugmentedSintel row behind `clip_from_sintel`'s index — the manifest
    is indexed differently (2,268 augmented clips against the dataset's 189),
    so the label has to come from the dataset itself."""
    from flydream.decode.map import stimulus_set

    ds, _, _, _ = stimulus_set("sintel", dt)
    df = getattr(ds, "arg_df", None)
    if df is None:
        return f"sample {sample}"
    return str(df.iloc[int(sample)]["name"]).replace("sequence_", "").replace("_split_", " split ")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, np.float32).reshape(-1), np.asarray(b, np.float32).reshape(-1)
    x = x - x.mean(); y = y - y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def run(model: str, prior_ckpt, gen_ckpt, manifest: dict, columns: dict, procedural_file: Path, *,
        clip_a: int = 3, clip_b: int = 126, corpus: Path | None = None, alphas=(0.25, 0.5, 0.75),
        fixed_point: int = 4, n_check: int = 4,
        frames: int = 40, margin: int = 5, dt: float = 0.02, t_pre: float = 1.0, sample_steps: int = 20,
        seed: int = 0, log=print) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
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
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}      # 13B's and item 14's normalisation
    log(f"model, prior ({pmeta.get('sources')}, DCT {pmeta.get('dct_k')}) and 13B loaded, {time.time() - t0:.0f} s")

    def to_z(states: np.ndarray) -> np.ndarray:                              # (N,T,cells) -> z-scored maps
        m = L.to_maps(states.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
        return (m - mean[None, None, :, None]) / std[None, None, :, None]

    # --- 1. the inversion's own check: noise -> state -> noise ---
    g = torch.Generator(device=dev).manual_seed(4000 + seed)
    x0 = torch.randn(n_check, pmeta["frames"], pmeta.get("k", 8), 721, device=dev, generator=g)
    xs = R.integrate(prior, x0.clone(), steps=sample_steps)
    check = {}
    for fp in sorted({1, fixed_point}):
        back = R.invert(prior, xs.clone(), steps=sample_steps, fixed_point=fp)
        e0, e1 = x0.cpu().numpy(), back.cpu().numpy()
        check[f"fixed_point_{fp}"] = {
            "r_noise_recovered": float(np.mean([corr(e0[i], e1[i]) for i in range(n_check)])),
            "relative_error": float(np.mean(np.linalg.norm((e1 - e0).reshape(n_check, -1), axis=1)
                                            / (np.linalg.norm(e0.reshape(n_check, -1), axis=1) + 1e-12))),
            "model_calls_per_step": fp}
        log(f"  inversion check (fixed point {fp}): r {check[f'fixed_point_{fp}']['r_noise_recovered']:.5f}, "
            f"relative error {check[f'fixed_point_{fp}']['relative_error']:.4f}")

    # --- 2. two real clips -> their own noise -> back ---
    cz = cmeta = None
    if corpus is not None:                                                   # item 18: clips of the prior's own set
        cz = np.load(Path(corpus) / "videos.npz")
        cmeta = [json.loads(str(x)) for x in cz["meta"]]
    clips = {}
    for name, idx in (("A", clip_a), ("B", clip_b)):
        if cz is not None:
            v = np.asarray(cz["videos"][int(idx)], np.float32)
            label = cmeta[int(idx)].get("label") or cmeta[int(idx)].get("class", "?")
        else:
            v = clip_from_sintel(idx, frames, dt, margin)
            label = scene_name(idx, dt)
        st = simulate_states(net, v[None][:, :frames + margin].astype(np.float16), d.cells_all, dt, t_pre, 1).astype(np.float32)
        clips[name] = {"index": int(idx), "scene": label, "video": v[:frames], "state": to_z(st)[0]}
        log(f"  clip {name}: sample {idx}, {label}")
    shuffled = clips["A"]["state"][:, :, rng.permutation(721)]

    states_in = np.stack([clips["A"]["state"], clips["B"]["state"], shuffled])
    eps = R.to_noise(prior, pmeta, states_in, steps=sample_steps, fixed_point=fixed_point, device=dev)
    eps_a, eps_b, eps_s = eps[0:1], eps[1:2], eps[2:3]
    rec = R.from_noise(prior, pmeta, np.concatenate([eps_a, eps_b]), steps=sample_steps, device=dev)
    dmat = R.dct_matrix(frames, int(pmeta["dct_k"])) if pmeta.get("dct_k") else None
    ceiling = (R.from_dct(R.to_dct(states_in[:2], dmat), dmat).astype(np.float32) if dmat is not None
               else states_in[:2].copy())                                    # band-limited real state: the best possible
    expected = 1.0 + 3.0 * np.sqrt(2.0 / eps[0].size)                        # 3σ of ‖ε‖²/D for true N(0, I)
    rad = radius(eps)
    log(f"  noise radius ‖e‖²/D: clip A {rad[0]:.3f}, clip B {rad[1]:.3f}, shuffled A {rad[2]:.3f}"
        f"  (Gaussian 1.000 ± {expected - 1:.3f})")

    # --- 3. the space between the two clips ---
    jobs = {"clip_A": clips["A"]["state"], "A_through_noise": rec[0]}
    for al in alphas:
        jobs[f"mix_{al:g}"] = R.from_noise(prior, pmeta, R.slerp(eps_a, eps_b, float(al)), steps=sample_steps, device=dev)[0]
    jobs["B_through_noise"] = rec[1]
    jobs["clip_B"] = clips["B"]["state"]
    jobs["ceiling_A"] = ceiling[0]
    jobs["shuffled_A"] = shuffled
    names = list(jobs)

    # --- 4. 13B renders them, the frozen brain scores them ---
    cond = torch.as_tensor(np.stack([jobs[n] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), 8):
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)
            vids.append(G.sample(gen, cond[i:i + 8], mask[i:i + 8], steps=sample_steps, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    log(f"{len(videos)} videos rendered in {time.time() - t0:.0f} s")

    target_raw = np.stack([R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0] for n in names])
    rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
    vids_m = np.concatenate([videos, np.repeat(videos[:, -1:], margin, 1)], 1).astype(np.float16)
    st_back = simulate_states(net, vids_m, d.cells_all, dt, t_pre, 8).astype(np.float32)
    st_grey = simulate_states(net, np.full((1, T, 721), 0.5, np.float16), d.cells_all, dt, t_pre, 1).astype(np.float32)[0]
    log(f"round trips and states back in {time.time() - t0:.0f} s")

    # --- 5. novelty of the video against 13B's training split ---
    bank, labels = video_bank(manifest, procedural_file, T, dt, log=log)
    train_idx = np.asarray(manifest["split"]["train"])
    i_v, r_v, d_v = nearest(flat(videos, frames), flat(bank[train_idx], frames))

    scores = {}
    for i, n in enumerate(names):
        scores[n] = {
            "round_trip": float(rts[i]),
            "r_state_to_A": corr(jobs[n], clips["A"]["state"]), "r_state_to_B": corr(jobs[n], clips["B"]["state"]),
            "r_video_to_A": float(pixcorr_per_frame(videos[i], clips["A"]["video"]).mean()),
            "r_video_to_B": float(pixcorr_per_frame(videos[i], clips["B"]["video"]).mean()),
            "video_nn_r": float(r_v[i]), "video_nn_distance": float(d_v[i]),
            "video_nn_label": labels[int(train_idx[i_v[i]])], "video_nn_index": int(train_idx[i_v[i]]),
            "direction_state": direction_energy(target_raw[i], st_grey, d)["T4_argmax"],
            "direction_video": direction_energy(st_back[i], st_grey, d)["T4_argmax"],
            "frame_to_frame": float(np.abs(np.diff(videos[i], axis=0)).mean())}
        log(f"  {n:<16} rt {scores[n]['round_trip']:6.3f}  r состояния A {scores[n]['r_state_to_A']:+.2f} "
            f"B {scores[n]['r_state_to_B']:+.2f}  видео к A {scores[n]['r_video_to_A']:+.2f} B {scores[n]['r_video_to_B']:+.2f}"
            f"  nn {scores[n]['video_nn_r']:+.2f} ({scores[n]['video_nn_label']})")

    noise = {"dimension": int(eps[0].size), "gaussian_3sigma": float(expected),
             "radius": {"clip_A": float(rad[0]), "clip_B": float(rad[1]), "shuffled_A": float(rad[2])},
             "r_between_clip_noises": corr(eps_a, eps_b),
             "sd": {"clip_A": float(eps[0].std()), "clip_B": float(eps[1].std()), "shuffled_A": float(eps[2].std())},
             "mean": {"clip_A": float(eps[0].mean()), "clip_B": float(eps[1].mean()), "shuffled_A": float(eps[2].mean())}}
    reconstruction = {f"clip_{k}": {"r_state": corr(rec[j], states_in[j]),
                                    "r_state_ceiling": corr(ceiling[j], states_in[j]),
                                    "round_trip": float(rts[names.index(f"{k}_through_noise")]),
                                    "round_trip_real": float(rts[names.index(f"clip_{k}")])}
                      for j, k in enumerate(("A", "B"))}
    summary = {"model": model, "prior_ckpt": str(prior_ckpt), "gen_ckpt": str(gen_ckpt), "frames": frames,
               "sample_steps": sample_steps, "fixed_point": fixed_point, "seed": seed,
               "clips": {k: {"index": v["index"], "scene": v["scene"]} for k, v in clips.items()},
               "alphas": [float(a) for a in alphas],
               "prior_meta": {k: pmeta.get(k) for k in ("sources", "dct_k", "n_train", "steps", "width", "depth")},
               "inversion_check": check, "noise": noise, "reconstruction": reconstruction,
               "scores": scores, "seconds": round(time.time() - t0, 1)}
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    arrays.update({f"T4a__{n}": target_raw[i][:, d.pos["T4a"]] for i, n in enumerate(names)})
    arrays.update({f"nnvideo__{n}": bank[train_idx[i_v[i]]][:frames].astype(np.float32) for i, n in enumerate(names)})
    arrays["clip_A"] = clips["A"]["video"]; arrays["clip_B"] = clips["B"]["video"]
    arrays["noise_A"] = eps_a[0]; arrays["noise_B"] = eps_b[0]
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
    p.add_argument("--tag", default="noise17_local")
    p.add_argument("--clip-a", type=int, default=3)
    p.add_argument("--clip-b", type=int, default=126)
    p.add_argument("--alphas", default="0.25,0.5,0.75")
    p.add_argument("--fixed-point", type=int, default=4)
    p.add_argument("--corpus", default="")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    proc = sorted(pdir.glob("procedural_*.npz"))[0]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.prior), Path(a.gen), manifest, columns, proc, clip_a=a.clip_a, clip_b=a.clip_b,
            corpus=Path(a.corpus) if a.corpus else None,
            alphas=tuple(float(x) for x in a.alphas.split(",")), fixed_point=a.fixed_point,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    print(f"wrote {out / a.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
