"""13A, the last part: the decoders of train13 applied to the states of items
11-12 — states no held-out clip caused — beside the Adam inversion of the same
state on the same types and a shuffled-state control.

    modal run deploy/modal/generate_app.py --model malecns --roundtrip13-run

Every state is rebuilt on the worker from its recipe (the dream inputs of
`dreams.py`, the edits and mixes of `mix.py` on clips A and B), so nothing is
uploaded. Per condition (early / deep / all) and state: the linear and the
CNN readout of the state's maps, the inversion (item-12 loss, clip A's
variances), the linear readout of the state with its cells permuted. Every
video's round trip = simulated through the frozen brain, per-type normalised
distance to the target on the condition's types (`round_trip`); where a
reference video exists (the eye's input, clip A/B, the averaged video) the
PixCorr to it is also given.

Differences from item 12, chosen so one state feeds all three conditions:
the interpolation mixes the whole state (not T4a alone), and a hybrid takes
every pre-motion type (L1 … Tm9) from one clip and T4/T5 from the other.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from flydream.decode import pairs as P
from flydream.generate import learned as L
from flydream.generate.dreams import dream_settings, input_video, shuffled, simulate_noisy, temporal_corr
from flydream.generate.invert import clip_from_sintel, device_of, invert_batch, load_network, pixcorr_per_frame, simulate
from flydream.generate.mix import T4, T5, mix_settings, type_weights
from flydream.generate.pairs13 import LADDER_TYPES, simulate_states

PRE_MOTION = ["L1", "L3", "Mi1", "Mi4", "Tm5a", "Tm9"]


def build_states(net, index: dict, *, frames: int, margin: int, dt: float, t_pre: float, seed: int = 0,
                 dreams: dict | None = None, mix: dict | None = None) -> list[dict]:
    """The target states of items 11-12 as (1, T, n_nodes) tensors with their
    reference videos. A state's window is `frames + margin` steps; dark_after
    keeps the whole 85-step simulation and names the dark window."""
    d, m = dreams or dream_settings(), mix or mix_settings()
    dev = device_of(net)
    T = frames + margin
    state1 = net.steady_state(t_pre, dt, batch_size=1, value=0.5)
    out = []

    def sim(v):
        with torch.no_grad():
            return simulate(net, torch.as_tensor(v[None], device=dev), dt, state1)

    # item 11: the dreams
    for src in ("eye_noise", "flash", "dark_after", "neuron_noise"):
        rng = np.random.default_rng(seed)
        v, _ = input_video(src, frames, margin, dt, d, rng)
        if src == "neuron_noise":
            with torch.no_grad():
                tg = simulate_noisy(net, torch.as_tensor(v[None], device=dev), dt, state1,
                                    float(d.get("neuron_noise_sd", 0.05)), seed)
        else:
            tg = sim(v)
        s = {"name": src, "group": "dream", "target": tg, "input": v, "window": (0, T), "refs": {}}
        if src == "eye_noise":
            s["refs"]["input"] = v[:frames]
        elif src == "dark_after":
            s["window"] = (frames, frames + T)
            s["refs"]["last_frame"] = np.repeat(v[frames - 1:frames], frames, 0)
        elif src == "flash":
            s["refs"]["input"] = v[:frames]; s["temporal"] = True
        out.append(s)
    # item 12: clips A and B, gain edits, whole-state mixes, hybrids
    a_np = clip_from_sintel(int(m.get("sample_a", 3)), frames, dt, margin)
    b_np = clip_from_sintel(int(m.get("sample_b", 10)), frames, dt, margin)
    avg_np = 0.5 * (a_np + b_np)
    ta, tb, tavg = sim(a_np), sim(b_np), sim(avg_np)
    refs = {"clip_a": a_np[:frames], "clip_b": b_np[:frames], "clip_avg": avg_np[:frames]}

    def add(name, tg, note=""):
        out.append({"name": name, "group": "mix", "target": tg, "input": None, "window": (0, T), "refs": refs, "note": note})

    add("clip_A", ta, "control: clip A's own state")
    add("clip_B", tb, "control: clip B's own state")
    for name, gains in [("C_T4a_x0.5", {"T4a": 0.5}), ("C_T4a_x2", {"T4a": 2.0}),
                        ("C_T5_x0", {t: 0.0 for t in T5}), ("C_Mi4_x1.5", {"Mi4": 1.5})]:
        tg = ta.clone()
        for t, g in gains.items():
            idx = torch.as_tensor(index[t], dtype=torch.long, device=dev)
            tg[:, :, idx] = tg[:, :, idx] * g
        add(name, tg, "gain edit of clip A's state")
    for alpha in (0.25, 0.5, 0.75):
        add(f"A_a{alpha:g}", alpha * ta + (1 - alpha) * tb, f"whole state {alpha:g}·A + {1 - alpha:g}·B")
    add("A_avgvideo", tavg, "control: the state of the averaged video")

    def hybrid(src_early, src_motion):
        tg = torch.zeros_like(ta)
        for t in PRE_MOTION:
            idx = torch.as_tensor(index[t], dtype=torch.long, device=dev); tg[:, :, idx] = src_early[:, :, idx]
        for t in T4 + T5:
            idx = torch.as_tensor(index[t], dtype=torch.long, device=dev); tg[:, :, idx] = src_motion[:, :, idx]
        return tg
    add("B_earlyA_motionB", hybrid(ta, tb), "L1…Tm9 from A, T4/T5 from B")
    add("B_earlyB_motionA", hybrid(tb, ta), "L1…Tm9 from B, T4/T5 from A")
    return out


def load_decoder(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    k = len(ck["types"])
    mdl = (L.LinearHexTemporal(k, ck["rings"], ck["taps"]) if ck["kind"] == "linear"
           else L.HexTemporalCNN(k, ck["width"], ck["depth"], ck["rings"], ck["taps"]))
    mdl.load_state_dict(ck["state_dict"])
    return mdl.to(device).eval(), ck


def round_trip(net, videos: np.ndarray, target: np.ndarray, cells: np.ndarray, type_of: np.ndarray, types: list[str],
               dt: float, t_pre: float, margin: int, window: tuple[int, int], var_ref: dict) -> np.ndarray:
    """(N, T_v, 721) videos simulated from grey through the frozen brain (last
    frame held for `margin`), the state of `types` in `window` against
    `target` (N, T_w, cells); per-type squared error over clip A's variance
    (`var_ref`), mean over types, per clip."""
    vids = np.concatenate([videos, np.repeat(videos[:, -1:], margin, 1)], 1) if margin else videos
    st = simulate_states(net, vids.astype(np.float16), cells, dt, t_pre, 16).astype(np.float32)
    w0, w1 = window
    st = st[:, w0:w1] if st.shape[1] >= w1 else st
    tg = target.astype(np.float32)
    errs = []
    for t in types:
        sel = np.where(type_of == t)[0]
        errs.append(((st[:, :tg.shape[1], sel] - tg[:, :st.shape[1], sel]) ** 2).mean((1, 2)) / var_ref[t])
    return np.mean(errs, 0)


def run(model: str, ckpt_dir, manifest: dict, columns: dict, *, conditions=("early", "deep", "all"), frames: int = 40,
        margin: int = 5, dt: float = 0.02, t_pre: float = 1.0, inv_steps: int = 150, seed: int = 0, log=print) -> dict:
    """Everything of the module for one model; returns the summary and, under
    "arrays", per condition and state the videos (for the figures)."""
    from pathlib import Path

    torch.manual_seed(seed); np.random.seed(seed)
    net = load_network(model)
    dev = device_of(net)
    _, index = P.type_index(net.connectome)
    cells_all = np.asarray(manifest["cells"]); type_of = np.asarray(manifest["type_of_cell"])
    states = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    ta = next(s for s in states if s["name"] == "clip_A")["target"]
    var_ref = {t: float(ta[0][:, index[t]].var(unbiased=False)) + 1e-6 for t in LADDER_TYPES}
    T = frames + margin
    log(f"{len(states)} states built")
    rng = np.random.default_rng(seed)
    t0 = time.time()
    summary, arrays = {}, {}

    def window_of(s):
        w0, w1 = s["window"]
        return s["target"][:, w0:w1, :]

    for cond in conditions:
        types = L.CONDITIONS[cond]
        layout = L.channel_layout(manifest, columns, types)
        cells_cond = np.concatenate([index[t] for t in types])
        decoders = {k: load_decoder(Path(ckpt_dir) / f"{cond}_{k}.pt", dev) for k in ("linear", "cnn")}
        res, arr = {}, {}
        # 1. the decoders on every state, and the linear one on the shuffled state
        for s in states:
            tg = window_of(s)
            x = L.to_maps(tg[:, :, cells_all].cpu().numpy().astype(np.float16), layout, len(types))
            xs = L.to_maps(shuffled(tg, cells_cond, rng)[:, :, cells_all].cpu().numpy().astype(np.float16), layout, len(types))
            vids = {k: L.predict(m, x, ck["mean"], ck["std"], frames, dev)[0] for k, (m, ck) in decoders.items()}
            vids["shuffled"] = L.predict(decoders["linear"][0], xs, decoders["linear"][1]["mean"], decoders["linear"][1]["std"], frames, dev)[0]
            arr[s["name"]] = {"videos": vids, "refs": s["refs"], "input": s["input"]}
        # 2. the inversion of every state on the condition's types, one batch (dark_after: its own, 85 steps, dark window read)
        w_cells, w = type_weights(index, types, ta)
        plain = [s for s in states if s["name"] != "dark_after"]
        targets = torch.cat([s["target"] for s in plain])
        state = net.steady_state(t_pre, dt, batch_size=len(plain), value=0.5)
        inv, _, n_steps = invert_batch(net, targets, [w_cells] * len(plain), dt=dt, state=state, steps=inv_steps, lr=0.05, tv=0.02,
                                       plateau_steps=20, plateau_tol=0.01, log_every=50, cell_weights=[w] * len(plain))
        for s, v in zip(plain, inv.numpy()):
            arr[s["name"]]["videos"]["inversion"] = v[:frames]
        dark = next(s for s in states if s["name"] == "dark_after")
        tw = torch.zeros(dark["target"].shape[1]); tw[dark["window"][0]:dark["window"][1]] = 1.0
        state = net.steady_state(t_pre, dt, batch_size=1, value=0.5)
        inv_d, _, n_d = invert_batch(net, dark["target"], [w_cells], dt=dt, state=state, steps=inv_steps, lr=0.05, tv=0.02,
                                     plateau_steps=20, plateau_tol=0.01, log_every=50, time_weight=tw[None], cell_weights=[w])
        arr["dark_after"]["videos"]["inversion"] = inv_d.numpy()[0]        # 85 frames; the dark window is shown
        log(f"  {cond}: inversion {n_steps} + {n_d} steps, {time.time() - t0:.0f} s")
        # 3. the round trips, all 45-frame videos of the condition in one batch; dark_after's inversion (85 frames) alone
        tg_d = window_of(dark)[:, :, cells_all].cpu().numpy()
        rt_dark = round_trip(net, arr["dark_after"]["videos"]["inversion"][None], tg_d, cells_all, type_of, types, dt, t_pre, 0,
                             dark["window"], var_ref)[0]
        arr["dark_after"]["videos"]["inversion"] = arr["dark_after"]["videos"]["inversion"][dark["window"][0]:dark["window"][0] + frames]
        keys, vids_all, tgs = [], [], []
        for s in states:
            tg = window_of(s)[:, :, cells_all].cpu().numpy()[0]
            for k, v in arr[s["name"]]["videos"].items():
                if not (k == "inversion" and s["name"] == "dark_after"):
                    keys.append((s["name"], k)); vids_all.append(v); tgs.append(tg)
        rts = round_trip(net, np.stack(vids_all), np.stack(tgs), cells_all, type_of, types, dt, t_pre, margin, (0, T), var_ref)
        rt_of = dict(zip(keys, rts.tolist())); rt_of[("dark_after", "inversion")] = float(rt_dark)
        # 4. scores
        for s in states:
            sc = {}
            for k, v in arr[s["name"]]["videos"].items():
                sc[k] = {"round_trip": rt_of[(s["name"], k)]}
                for rname, ref in s["refs"].items():
                    sc[k][f"r_{rname}"] = (temporal_corr(v, ref) if s.get("temporal") else float(pixcorr_per_frame(v, ref).mean()))
            vids = arr[s["name"]]["videos"]
            sc["agreement"] = {"linear_cnn": float(pixcorr_per_frame(vids["linear"], vids["cnn"]).mean()),
                               "linear_inversion": float(pixcorr_per_frame(vids["linear"], vids["inversion"]).mean()),
                               "cnn_inversion": float(pixcorr_per_frame(vids["cnn"], vids["inversion"]).mean())}
            res[s["name"]] = sc
            log(f"  {cond:>5} {s['name']:<18} rt lin {sc['linear']['round_trip']:.3f} cnn {sc['cnn']['round_trip']:.3f} "
                f"inv {sc['inversion']['round_trip']:.3f} shuf {sc['shuffled']['round_trip']:.3f}  lin~cnn {sc['agreement']['linear_cnn']:.2f}")
        summary[cond] = {"types": types, "states": res, "inversion_steps": [int(n_steps), int(n_d)]}
        arrays[cond] = arr
        del decoders; torch.cuda.empty_cache()
    return {"conditions": summary, "frames": frames, "margin": margin, "seconds": round(time.time() - t0, 1),
            "state_names": [s["name"] for s in states], "notes": {s["name"]: s.get("note", "") for s in states}, "arrays": arrays}
