"""Item 14: the trained 13B generator as a controllable generator — four
cheap tests on states no video caused.

    python -m flydream.generate.prompts14 --model malecns          # local CPU

14.0 random states (per-cell white noise at the clip's mean/sd; a
structured variant smoothed in time and over the lattice); 14.1
counterfactual edits of a clip's deep state (direction swaps and a 90°
rotation of the T4/T5 channels, left/right halves from two clips, one
direction amplified); 14.2 prompts without a video, assembled from the
brain's responses to procedural stimuli by regions of the eye, beside one
state written by hand; 14.3 the closed loop state → 13B → video → brain →
state′ → … Every video's score is the round trip (13A's metric on T4/T5)
and the direction the brain reads in it (`direction_energy`: T4/T5 activity
per direction over the grey baseline). Control: the shuffled state.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from flydream.decode import pairs as P
from flydream.generate import gen13b as G
from flydream.generate import learned as L
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import clip_from_sintel, device_of, load_network, pixcorr_per_frame, simulate
from flydream.generate.pairs13 import simulate_states
from flydream.generate.roundtrip13 import round_trip
from flydream.generate.stimuli import hex_xy

DIRS = "abcd"                                    # T4a front-to-back, b back-to-front, c up, d down (FlyVis convention)
ROT90 = {"a": "c", "c": "b", "b": "d", "d": "a"}  # +90°: right → up → left → down → right


# ----------------------------------------------------------------- state helpers


class Deep:
    """Deep-state bookkeeping: positions of each T4/T5 type inside the stored
    (T, cells) state, the lattice coordinates of their columns."""

    def __init__(self, manifest: dict, columns: dict):
        self.cells_all = np.asarray(manifest["cells"]); self.type_of = np.asarray(manifest["type_of_cell"])
        self.pos = {t: np.where(self.type_of == t)[0] for t in DEEP}
        self.col = {t: np.asarray(columns["column_of_cell"][t]) for t in DEEP}
        self.layout = L.channel_layout(manifest, columns, DEEP)
        self.xy = hex_xy()

    def get(self, st: np.ndarray, t: str) -> np.ndarray:            # (T, 721) in column order
        out = np.zeros((st.shape[0], 721), np.float32); out[:, self.col[t]] = st[:, self.pos[t]]; return out

    def put(self, st: np.ndarray, t: str, m: np.ndarray) -> None:
        st[:, self.pos[t]] = m[:, self.col[t]]

    def maps(self, st: np.ndarray, mean, std, device) -> torch.Tensor:  # (1, T, 8, 721) z-scored
        x = torch.as_tensor(L.to_maps(st[None].astype(np.float16), self.layout, len(DEEP)).astype(np.float32), device=device)
        return (x - mean) / std


def swap_types(d: Deep, st: np.ndarray, a: str, b: str) -> np.ndarray:
    out = st.copy(); ma, mb = d.get(st, a), d.get(st, b); d.put(out, a, mb); d.put(out, b, ma); return out


def rotate_90(d: Deep, st: np.ndarray) -> np.ndarray:
    """Every direction channel moved one step: a→c, c→b, b→d, d→a, for T4 and T5."""
    out = st.copy()
    for fam in ("T4", "T5"):
        maps = {x: d.get(st, f"{fam}{x}") for x in DIRS}
        for x in DIRS:
            d.put(out, f"{fam}{ROT90[x]}", maps[x])
    return out


def half_and_half(d: Deep, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = left.copy()
    for t in DEEP:
        m = d.get(left, t); mr = d.get(right, t); m[:, d.xy[:, 0] > 0] = mr[:, d.xy[:, 0] > 0]; d.put(out, t, m)
    return out


def window_only(d: Deep, inside: np.ndarray, outside: np.ndarray, radius: float) -> np.ndarray:
    """`inside`'s state within `radius` columns of the centre, `outside`'s elsewhere."""
    out = outside.copy(); sel = (d.xy ** 2).sum(1) < radius ** 2
    for t in DEEP:
        m = d.get(outside, t); mi = d.get(inside, t); m[:, sel] = mi[:, sel]; d.put(out, t, m)
    return out


def amplify(d: Deep, st: np.ndarray, base: np.ndarray, types: list[str], gain: float, others: float) -> np.ndarray:
    out = st.copy()
    for t in DEEP:
        m = d.get(st, t); b = d.get(base, t); g = gain if t in types else others
        d.put(out, t, b + g * (m - b))
    return out


def random_state(d: Deep, ref: np.ndarray, rng: np.random.Generator, structured: bool, ring: np.ndarray | None = None) -> np.ndarray:
    """Per type: noise with the type's mean and sd from `ref`; structured =
    AR(1) in time (τ ≈ 100 ms at dt 20 ms) and two ring-1 averages over the lattice."""
    out = ref.copy()
    for t in DEEP:
        v = ref[:, d.pos[t]]; mu, sd = float(v.mean()), float(v.std()) + 1e-6
        n = rng.standard_normal(v.shape).astype(np.float32)
        if structured:
            keep = np.exp(-1 / 5)
            for i in range(1, len(n)):
                n[i] = keep * n[i - 1] + np.sqrt(1 - keep ** 2) * n[i]
            m = np.zeros((len(n), 721), np.float32); m[:, d.col[t]] = n
            for _ in range(2):
                g = m[:, ring.clamp(min=0).numpy()] * (ring >= 0).numpy(); m = g.sum(-1) / (ring >= 0).sum(-1).numpy()
            n = m[:, d.col[t]]; n = n / (n.std() + 1e-6)
        out[:, d.pos[t]] = mu + sd * n
    return out


def hand_stripe(d: Deep, base: np.ndarray, t: str = "T4a", level_sd: float = 3.0) -> np.ndarray:
    """A state written by hand: `t` high in a horizontal stripe through the centre, all else the grey baseline."""
    out = base.copy(); m = d.get(base, t)
    sel = np.abs(d.xy[:, 1]) < 3
    m[:, sel] = m[:, sel] + level_sd * (base[:, d.pos[t]].std() + 1e-6); d.put(out, t, m)
    return out


# ----------------------------------------------------------------- scores


def direction_energy(st: np.ndarray, base: np.ndarray, d: Deep) -> dict:
    """Mean activity over cells and time of each T4/T5 type above the grey baseline; the argmax direction."""
    e = {t: float(st[:, d.pos[t]].mean() - base[:, d.pos[t]].mean()) for t in DEEP}
    t4 = np.array([e[f"T4{x}"] for x in DIRS]); t5 = np.array([e[f"T5{x}"] for x in DIRS])
    return {"per_type": e, "T4_argmax": DIRS[int(t4.argmax())], "T5_argmax": DIRS[int(t5.argmax())]}


# ----------------------------------------------------------------- the run


def run(model: str, ckpt, manifest: dict, columns: dict, *, frames: int = 40, margin: int = 5, dt: float = 0.02,
        t_pre: float = 1.0, sample_steps: int = 20, n_seeds: int = 2, loop_iters: int = 12, seed: int = 0, log=print) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    net = load_network(model); dev = device_of(net)
    _, index = P.type_index(net.connectome)
    d = Deep(manifest, columns)
    gen, meta = G.load(ckpt, dev)
    mean = torch.as_tensor(np.array(meta["mean"], np.float32), device=dev)[None, None, :, None]
    std = torch.as_tensor(np.array(meta["std"], np.float32), device=dev)[None, None, :, None]
    ring = torch.as_tensor(L.ring_index(1))
    T = frames + margin
    state1 = net.steady_state(t_pre, dt, batch_size=1, value=0.5)

    def brain(v: np.ndarray) -> np.ndarray:                      # (T,721) video -> (T, cells) deep-ladder state
        with torch.no_grad():
            return simulate(net, torch.as_tensor(v[None].astype(np.float32), device=dev), dt, state1)[0][:, d.cells_all].cpu().numpy()

    # --- source states ---
    a_np = clip_from_sintel(3, frames, dt, margin); b_np = clip_from_sintel(10, frames, dt, margin)
    grey = np.full((T, 721), 0.5, np.float32)
    sA, sB, sG = brain(a_np), brain(b_np), brain(grey)
    var_ref = {t: float(sA[:, d.pos[t]].var()) + 1e-6 for t in DEEP}
    from flydream.generate.stimuli import clip as stim_clip
    xy = d.xy

    def grating(theta, tf=4.0):
        t_ = np.arange(T) * dt
        ph = 2 * np.pi * (0.08 * (xy @ np.array([np.cos(theta), np.sin(theta)]))[None, :] - tf * t_[:, None])
        return np.clip(0.5 + 0.5 * np.sin(ph), 0, 1).astype(np.float32)
    stim = {"right": grating(0.0), "left": grating(np.pi), "up": grating(np.pi / 2)}
    for kind, sd_ in (("expand", 1), ("rotate", 2)):
        r_ = np.random.default_rng(sd_)
        while True:
            v, p = stim_clip("flow", T, r_, xy, dt)
            if p["kind"] == kind:
                stim[kind] = v; break
    S = {k: brain(v) for k, v in stim.items()}
    log(f"source states ready {time.time() - t0:.0f} s")

    states, refs, note = {}, {}, {}

    def add(name, st, ref=None, n=""):
        states[name] = st.astype(np.float32); note[name] = n
        if ref is not None:
            refs[name] = ref[:frames]
    # 14.0 random
    add("r0_white", random_state(d, sA, rng, False), n="per-cell white noise at clip A's mean/sd per type")
    add("r0_structured", random_state(d, sA, rng, True, ring), n="noise smoothed in time (τ 100 ms) and over the lattice")
    # 14.1 edits on clip A (and on the right-moving grating, where direction is unambiguous)
    add("e_A", sA, a_np, "clip A unedited")
    add("e_A_swap_ac", swap_types(d, swap_types(d, sA, "T4a", "T4c"), "T5a", "T5c"), a_np, "T4a↔T4c, T5a↔T5c")
    add("e_A_rot90", rotate_90(d, sA), a_np, "all direction channels +90°")
    add("e_A_left_B_right", half_and_half(d, sA, sB), a_np, "left half A, right half B")
    add("e_A_amp_a", amplify(d, sA, sG, ["T4a", "T5a"], 2.0, 0.5), a_np, "direction a × 2, others × 0.5")
    add("e_right", S["right"], stim["right"], "grating moving right")
    add("e_right_swap_ab", swap_types(d, swap_types(d, S["right"], "T4a", "T4b"), "T5a", "T5b"), stim["right"], "right grating, a↔b (reverse)")
    add("e_right_rot90", rotate_90(d, S["right"]), stim["right"], "right grating, +90°")
    add("e_right_timerev", S["right"][::-1].copy(), stim["right"][::-1].copy(), "right grating, state reversed in time (motion should reverse)")
    add("e_A_timerev", sA[::-1].copy(), a_np[::-1].copy(), "clip A, state reversed in time")
    # 14.2 prompts
    add("p_left_right_conflict", half_and_half(d, S["right"], S["left"]), n="left half: motion right; right half: motion left")
    add("p_right_expand", half_and_half(d, S["right"], S["expand"]), n="left half: motion right; right half: expansion")
    add("p_window_up", window_only(d, S["up"], sG, 6.0), n="motion up inside radius 6, grey elsewhere")
    add("p_rotate", S["rotate"], stim["rotate"], "rotation (brain's state of the stimulus)")
    add("p_hand_T4a_stripe", hand_stripe(d, sG), n="written by hand: T4a high in a horizontal stripe")
    # control
    sh = sA.copy(); cells_deep = np.concatenate([d.pos[t] for t in DEEP]); perm = rng.permutation(len(cells_deep))
    sh[:, cells_deep] = sA[:, cells_deep[perm]]
    add("c_shuffled", sh, a_np, "control: clip A's cells permuted")

    # --- sampling ---
    full = torch.ones(1, 8, device=dev)
    videos = {}
    with torch.no_grad():
        for name, st in states.items():
            mp = d.maps(st[:frames], mean, std, dev)
            for k in range(n_seeds):
                g = torch.Generator(device=dev).manual_seed(1000 + k)
                videos[f"{name}__seed{k}"] = G.sample(gen, mp, full, steps=sample_steps, generator=g)[0].cpu().numpy()
    log(f"{len(videos)} samples in {time.time() - t0:.0f} s")
    # --- round trips + direction, one batch ---
    keys = list(videos)
    vids = np.stack([videos[k] for k in keys]); tgs = np.stack([states[k.split("__")[0]] for k in keys])
    rts = round_trip(net, vids, tgs, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, T), var_ref)
    vids_m = np.concatenate([vids, np.repeat(vids[:, -1:], margin, 1)], 1)
    st_back = simulate_states(net, vids_m.astype(np.float16), d.cells_all, dt, t_pre, 16).astype(np.float32)
    scores = {}
    for i, k in enumerate(keys):
        name = k.split("__")[0]
        sc = {"round_trip": float(rts[i]), "direction_video": direction_energy(st_back[i], sG, d),
              "direction_state": direction_energy(states[name], sG, d)}
        if name in refs:
            sc["r_ref"] = float(pixcorr_per_frame(videos[k], refs[name]).mean())
        scores[k] = sc
        log(f"  {k:<32} rt {sc['round_trip']:.3f}  dir state T4 {sc['direction_state']['T4_argmax']} -> video T4 {sc['direction_video']['T4_argmax']}"
            + (f"  r_ref {sc['r_ref']:+.2f}" if "r_ref" in sc else ""))
    # --- 14.3 closed loop ---
    loops = {}
    starts = {"loop_clipA": sA, "loop_clipB": sB, "loop_grating": S["right"], "loop_white": states["r0_white"],
              "loop_hand": states["p_hand_T4a_stripe"], "loop_swap": states["e_right_swap_ab"], "loop_conflict": states["p_left_right_conflict"]}
    loop_refs = {"loop_clipA": a_np[:frames], "loop_clipB": b_np[:frames], "loop_grating": stim["right"][:frames]}
    for name, st in starts.items():
        cur, hist, vprev = st, [], None
        for it in range(loop_iters):
            with torch.no_grad():
                g = torch.Generator(device=dev).manual_seed(1000)
                v = G.sample(gen, d.maps(cur[:frames], mean, std, dev), full, steps=sample_steps, generator=g)[0].cpu().numpy()
            vm = np.concatenate([v, np.repeat(v[-1:], margin, 0)], 0)
            nxt = brain(vm)
            rt = round_trip(net, v[None], cur[None], d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, T), var_ref)[0]
            rec = {"iter": it, "round_trip": float(rt), "r_video_prev": None if vprev is None else float(pixcorr_per_frame(v, vprev).mean()),
                   "r_video_start": None if name not in loop_refs else float(pixcorr_per_frame(v, loop_refs[name]).mean()),
                   "state_change": float(np.abs(nxt[:, cells_deep] - cur[:, cells_deep]).mean() / (sA[:, cells_deep].std() + 1e-6)),
                   "direction_T4": direction_energy(nxt, sG, d)["T4_argmax"]}
            hist.append(rec); videos[f"{name}__it{it}"] = v
            cur, vprev = nxt, v
        loops[name] = hist
        log(f"  {name}: rt " + " ".join(f"{h['round_trip']:.3f}" for h in hist) + " | change " + " ".join(f"{h['state_change']:.3f}" for h in hist)
            + ("" if name not in loop_refs else " | r to start clip " + " ".join(f"{h['r_video_start']:.2f}" for h in hist)))
    return {"scores": scores, "loops": loops, "notes": note, "frames": frames, "n_seeds": n_seeds, "loop_iters": loop_iters,
            "seconds": round(time.time() - t0, 1), "arrays": {"videos": videos, "refs": refs, "stim": {k: v[:frames] for k, v in stim.items()},
                                                            "states_T4a": {k: d.get(v, "T4a")[:frames] for k, v in states.items()}}}


def main(argv=None) -> int:
    import argparse, json
    from pathlib import Path
    from flydream.model import ROOT

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--ckpt", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--out", default=str(ROOT / "data" / "prompts14"))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--loop-iters", type=int, default=12)
    a = p.parse_args(argv)
    manifest = json.loads((ROOT / "data" / "pairs13" / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((ROOT / "data" / "pairs13" / "columns.json").read_text(encoding="utf-8"))
    r = run(a.model, a.ckpt, manifest, columns, n_seeds=a.seeds, loop_iters=a.loop_iters)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arr = r.pop("arrays")
    flat = {f"video__{k}": v for k, v in arr["videos"].items()}
    flat.update({f"ref__{k}": v for k, v in arr["refs"].items()}); flat.update({f"stim__{k}": v for k, v in arr["stim"].items()})
    flat.update({f"T4a__{k}": v for k, v in arr["states_T4a"].items()})
    np.savez_compressed(out / "prompts14.npz", **flat)
    (out / "summary.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"wrote {out}; {r['seconds']} s")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
