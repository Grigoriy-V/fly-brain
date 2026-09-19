"""18.4e, the diagnostic: does the label do anything at all?

    python -m flydream.generate.classcmp18            # local CPU, no Modal, $0

18.4e measured the class-conditional prior against the unconditional one and
found no difference at the gate (0.098 against 0.090, inside the spread). That
is an average over sixteen samples drawn from sixteen different noises, and it
cannot separate *"the label is ignored"* from *"the label changes the sample
but not its compatibility"*. This does:

**one noise vector, four ways out of it** — through the unconditional prior,
and through the conditional prior with three different labels. Everything
downstream is held fixed, 13B included (the same z for every render), so any
difference between the four videos is the prior's response to the label and
nothing else.

The number that answers the question is the correlation between the cells: if
switching the label leaves the video where it was, the embedding is not being
read.
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
from flydream.generate.invert import device_of, load_network
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep
from flydream.generate.roundtrip13 import build_states, round_trip


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two arrays of any matching shape, flattened."""
    x, y = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    x -= x.mean(); y -= y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def run(model: str, ctrl_ckpt, cls_ckpt, gen_ckpt, manifest: dict, columns: dict, *,
        labels: list[str], frames: int = 40, margin: int = 5, dt: float = 0.02, t_pre: float = 1.0,
        sample_steps: int = 20, seed: int = 0, log=print) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    net = load_network(model); dev = device_of(net)
    _, index = P.type_index(net.connectome)
    d = Deep(manifest, columns)
    ctrl, cmeta = R.load(ctrl_ckpt, dev)
    cond, ymeta = R.load(cls_ckpt, dev)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)
    names_all = list(ymeta.get("class_names") or [])
    if not names_all:
        raise SystemExit(f"{cls_ckpt} is not a class-conditional prior")
    if cmeta["frames"] != ymeta["frames"] or cmeta.get("k", 8) != ymeta.get("k", 8):
        raise SystemExit("the two priors do not share a model space; one noise cannot feed both")
    T = frames + margin
    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]
    ta = sa["target"][:, w0:w1, :][:, :, d.cells_all].cpu().numpy().astype(np.float32)
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}      # the same normalisation as every gate
    log(f"priors loaded ({cmeta['parameters']} par unconditional, {ymeta['parameters']} par over "
        f"{len(names_all)} classes), {time.time() - t0:.0f} s")

    missing = [x for x in labels if x not in names_all]
    if missing:
        raise SystemExit(f"unknown labels {missing}; the prior knows {len(names_all)}, e.g. {names_all[:6]}")

    # --- one noise, four ways out of it ---
    g = torch.Generator(device=dev).manual_seed(3000 + seed)
    eps = torch.randn(1, cmeta["frames"], cmeta.get("k", 8), 721, device=dev, generator=g)
    cells: list[tuple[str, str, np.ndarray]] = []
    with torch.no_grad():
        x = R.integrate(ctrl, eps.clone(), steps=sample_steps)
        cells.append(("control", "безусловный прайор", R.from_model_space(cmeta, x)[0]))
        for lab in labels:
            y = torch.full((1,), names_all.index(lab), dtype=torch.long, device=dev)
            x = R.integrate(R.conditioned(cond, y), eps.clone(), steps=sample_steps)
            cells.append((f"class_{lab}", f"класс «{lab}»", R.from_model_space(ymeta, x)[0]))
    log(f"{len(cells)} states from one noise in {time.time() - t0:.0f} s")

    # --- 13B renders them all with the same z, so only the state differs ---
    states = np.stack([c[2] for c in cells]).astype(np.float32)
    cnd = torch.as_tensor(states, device=dev)
    mask = torch.ones(len(cells), len(DEEP), device=dev)
    with torch.no_grad():
        gg = torch.Generator(device=dev).manual_seed(1000 + seed)
        videos = G.sample(gen, cnd, mask, steps=sample_steps, generator=gg).cpu().numpy().astype(np.float32)
    log(f"{len(videos)} videos rendered in {time.time() - t0:.0f} s")

    # --- the frozen brain reads them back ---
    target_raw = np.stack([R.from_maps(R.unscale(s[None], mean, std), d.layout, len(d.cells_all))[0] for s in states])
    rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
    log(f"round trips in {time.time() - t0:.0f} s")

    keys = [c[0] for c in cells]
    r_state = {f"{a}|{b}": corr(states[i], states[j]) for i, a in enumerate(keys) for j, b in enumerate(keys) if i < j}
    r_video = {f"{a}|{b}": corr(videos[i], videos[j]) for i, a in enumerate(keys) for j, b in enumerate(keys) if i < j}
    among = [v for k, v in r_video.items() if not k.startswith("control")]
    summary = {
        "model": model, "control_ckpt": str(ctrl_ckpt), "class_ckpt": str(cls_ckpt), "seed": seed,
        "frames": frames, "sample_steps": sample_steps, "labels": labels, "n_classes": len(names_all),
        "cells": [{"key": k, "title": t, "round_trip": float(rts[i])} for i, (k, t, _) in enumerate(cells)],
        "corr_state": r_state, "corr_video": r_video,
        "label_effect": {"video_r_between_labels_min": min(among) if among else None,
                         "video_r_between_labels_mean": float(np.mean(among)) if among else None,
                         "video_r_control_to_first_label": r_video[f"control|{keys[1]}"]},
        "seconds": round(time.time() - t0, 1),
    }
    arrays = {f"video__{k}": videos[i] for i, k in enumerate(keys)}
    arrays |= {f"T4a__{k}": states[i][:, DEEP.index("T4a")] for i, k in enumerate(keys)}
    return {"summary": summary, "arrays": arrays}


def main(argv=None) -> int:
    from flydream.generate.invert import settings
    from flydream.model import ROOT

    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--control", default=str(ROOT / "data" / "prior18" / "corpus_dct16_c.pt"))
    p.add_argument("--classes", default=str(ROOT / "data" / "prior18" / "corpus_dct16_cls_c.pt"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--tag", default="classcmp18_local")
    p.add_argument("--labels", default="Surfing,Archery,procedural:mixture")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.control), Path(a.classes), Path(a.gen), manifest, columns,
            labels=[x for x in a.labels.split(",") if x.strip()],
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    s = r["summary"]
    print("\nround trip per cell:")
    for c in s["cells"]:
        print(f"  {c['title']:32s} {c['round_trip']:.3f}")
    print("\ncorrelation between the videos:")
    for k, v in s["corr_video"].items():
        print(f"  {k:46s} {v:+.4f}")
    e = s["label_effect"]
    print(f"\nbetween labels, video r: mean {e['video_r_between_labels_mean']:+.4f}, "
          f"min {e['video_r_between_labels_min']:+.4f}; control to the first label {e['video_r_control_to_first_label']:+.4f}")
    print(f"wrote {out / a.tag}.json/.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
