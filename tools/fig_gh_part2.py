"""GitHub/article figures for part 2: from rendering a state to drawing one, in the house style.

    python tools/fig_gh_part2.py                    # all into docs/figures/
    python tools/fig_gh_part2.py --only priors shell

Nothing is recomputed on a worker; every figure reads saved runs:

- `first_priors`    (prior, part 2 sections 2-3): 13B from a held-out clip's real state,
  13B from noise with no condition, a draw of prior 17.1 and of prior 17.1b, each
  rendered by 13B. `data/prior17/samples17{,b}_local.npz`, `data/baseline17/baseline17.npz`.
- `flow_inversion`  (prior, section 4): a clip -> the prior run backwards -> its noise ->
  forwards -> 13B, and a walk between the noises of two clips.
  `data/prior17/noise17_local.{npz,json}` (run 2026-09-20_prior17_noise_inversion).
- `levers_and_judge` (prior, section 5): the round trip of every lever arm of 18.3-18.5
  from the run log, and the reading that retired it as a judge.
- `seed_shell`      (prior, section 6): a held-out clip's state, the same state from its
  own noise, that noise put on the Gaussian shell, and a fresh draw.
  `data/prior18/reach18_corpus_dct16_w192_lr1e3_c.{npz,json}`.
- `hex_flow`        (prior, section 10): what moved from the flow over PCA (22) to the
  flow on the lattice (23), from the run log and `data/prior23/accept23.json`.
- `bought_kurtosis` (prior, section 10): step 26's draw-time arms, kurtosis against
  radius, `data/prior23/fixdraw23.json`.
- `static_chain`    (static, section 11): what 13B makes of a 721 x 2 state with each
  crutch, and the static flow's draws against a real state and a no-flow control,
  from the run log.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import flyvis.utils.hex_utils  # noqa: E402,F401  (restyles matplotlib on import)
from fig_gh_charts import C_BAD, C_CTRL, C_FLY, C_OURS, page, plt, run_log  # noqa: E402
from gh_style import BAD, BG, GOOD, INK, LINE, MUTED, font, hex_image, honeycomb, pipeline_strip, save_gif  # noqa: E402

D = ROOT / "data"
W = 1216


def corr(a, b):
    return float(np.mean([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)]))


def grid_gif(out: Path, stage: str, title: str, lines: list[str], rows, footer: str, notes=None,
             pix: int = 5, lab_w: int = 170, fps: int = 8) -> None:
    """rows: (name, sub, [video (T, 721)], [label or None], colour). One GIF and its middle frame as PNG."""
    comb = honeycomb(pix)
    pw, ph = comb[0].shape[1], comb[0].shape[0]
    m = max(len(r[2]) for r in rows)
    gap = min(40, (W - 96 - lab_w - m * pw) // max(m - 1, 1))
    head = 24 + 34 + 26 + 44 + 26 * len(lines) + 22
    rh = ph + (72 if any(lab and "\n" in lab for r in rows for lab in (r[3] or [])) else 50)
    H = head + len(rows) * rh + (26 if notes else 0) + 40
    n = min(len(v) for r in rows for v in r[2])
    f_title, f_body, f_lab, f_small, f_num = font(30, True), font(19), font(19, True), font(16), font(17, True)
    frames = []
    for k in range(n):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        y = pipeline_strip(d, 48, 24, W - 96, stage) + 26
        d.text((48, y), title, font=f_title, fill=INK)
        y += 44
        for ln in lines:
            d.text((48, y), ln, font=f_body, fill=MUTED)
            y += 26
        y = head
        xs = [48 + lab_w + i * (pw + gap) for i in range(m)]
        d.line([xs[0] - 14, y, xs[0] - 14, y + len(rows) * rh - 50], fill=LINE, width=1)
        for name, sub, vids, labels, colour in rows:
            d.text((48, y + ph // 2 - 24), name, font=f_lab, fill=INK)
            d.text((48, y + ph // 2 + 2), sub, font=f_small, fill=MUTED)
            for i, v in enumerate(vids):
                im.paste(hex_image(v[k], comb), (xs[i], y))
                if labels and labels[i]:
                    d.multiline_text((xs[i], y + ph + 4), labels[i], font=f_num, fill=colour, spacing=2)
            y += rh
        if notes:
            d.text((48, y - 16), notes, font=f_small, fill=BAD)
        d.text((48, H - 38), f"{footer} · frame {k + 1}/{n}", font=f_small, fill=MUTED)
        frames.append(im)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, out.with_suffix(".gif"), fps)
    frames[n // 2].save(out.with_suffix(".png"))
    for s in (".gif", ".png"):
        print(f"{out.with_suffix(s)}  {out.with_suffix(s).stat().st_size / 1e6:.2f} MB")


# ----------------------------------------------------------------------------- pictures


def priors(out: Path) -> None:
    b = np.load(D / "baseline17" / "baseline17.npz")
    s1 = np.load(D / "prior17" / "samples17_local.npz")
    s2 = np.load(D / "prior17" / "samples17b_local.npz")
    g1 = json.loads((D / "prior17" / "samples17_local.json").read_text(encoding="utf-8"))["gates"]
    g2 = json.loads((D / "prior17" / "samples17b_local.json").read_text(encoding="utf-8"))["gates"]
    clips = [2208, 7885, 8086]
    rows = [("a real state", "a held-out clip's state", [s2[f"video__clip_{c}"] for c in clips],
             [f"round trip {g2['clip']['round_trip']['median']:.3f} (median)", None, None], GOOD),
            ("noise, no state", "13B's empty condition", [b["uncond"][i] for i in range(3)], None, INK),
            ("prior 17.1", "a drawn state", [s1[f"video__prior_{i}"] for i in range(3)],
             [f"round trip {g1['prior']['round_trip']['median']:.2f}", None, None], BAD),
            ("prior 17.1b", "scenes only, DCT-16", [s2[f"video__prior_{i}"] for i in range(3)],
             [f"round trip {g2['prior']['round_trip']['median']:.3f}", None, None], BAD)]
    grid_gif(out, "prior", "The first priors over brain states",
             ["Each cell is 13B's video. From a real state it renders the clip; from its empty condition, clouds; from the",
              "first prior's draws, salt and pepper; from a prior retrained on scenes in a DCT basis, texture and blobs."],
             rows, "13B generator · Sintel and procedural clips · runs 2026-09-20_prior17_*, 2026-09-20_baseline17",
             pix=6, lab_w=210)


def flow_inversion(out: Path) -> None:
    z = np.load(D / "prior17" / "noise17_local.npz")
    A, B = z["clip_A"], z["clip_B"]
    via = {k: z[f"video__{k}"] for k in ("ceiling_A", "A_through_noise", "shuffled_A", "mix_0.25", "mix_0.5",
                                         "mix_0.75", "B_through_noise")}
    half = (via["A_through_noise"] + via["B_through_noise"]) / 2
    rows = [("clip A → noise → back", "the prior run backwards",
             [A, via["ceiling_A"], via["A_through_noise"], via["shuffled_A"]],
             ["the clip", f"13B from its state\nr {corr(via['ceiling_A'], A):.2f}",
              f"via its own noise\nr {corr(via['A_through_noise'], A):.2f}",
              f"control: shuffled\nr {corr(via['shuffled_A'], A):+.2f}"], GOOD),
            ("between two noises", "a straight line",
             [via["A_through_noise"], via["mix_0.25"], via["mix_0.5"], via["mix_0.75"], via["B_through_noise"], B],
             ["A's noise", "¼", f"½\nr {corr(via['mix_0.5'], half):.2f} to (A+B)/2", "¾", "B's noise", "clip B"], INK)]
    grid_gif(out, "prior", "Every clip has its own noise",
             ["Run backwards with four fixed-point iterations per step, the prior maps a clip's state to the noise it",
              "would be drawn from, and forwards back to the clip. Halfway between two clips' noises is a double exposure."],
             rows, "prior 17.1b, 20 Euler steps · Sintel clips A and B · run 2026-09-20_prior17_noise_inversion",
             lab_w=210)


def seed_shell(out: Path) -> None:
    z = np.load(D / "prior18" / "reach18_corpus_dct16_w192_lr1e3_c.npz")
    meta = json.loads((D / "prior18" / "reach18_corpus_dct16_w192_lr1e3_c.json").read_text(encoding="utf-8"))
    rad = meta["noise"]["radius_per_clip"]
    shell = meta["typical_radius"]
    names = ["настоящее состояние", "из его шума", "тот же шум на радиусе √D", "обычный розыгрыш"]
    g = {nm: [z[f"video__{nm}|{i}"] for i in range(6)] for nm in names}
    rows = [("a real state", "a held-out clip, 13B", g[names[0]], None, GOOD),
            ("its own noise", "the flow run backwards", g[names[1]], [f"radius {r:.0f}" for r in rad], GOOD),
            ("pushed to the shell", f"radius {shell:.1f}", g[names[2]], None, BAD),
            ("a fresh draw", "N(0, I), radius ≈ 303.8", g[names[3]], None, BAD)]
    grid_gif(out, "prior", "The seed problem: scenes live inside the shell",
             ["Gaussian noise in 92,288 dimensions lies on a thin shell of radius 303.8 (sd 0.71). A clip's own noise",
              "sits far inside it. Pushed out to the shell, the same direction breaks the clip; a fresh draw is no scene."],
             rows, "prior 18.5 (width 192, lr 1e-3) · UCF101, classes never seen in training · run "
                   "2026-09-20_prior18_reach_noise_geometry", lab_w=210)


# ----------------------------------------------------------------------------- charts


def levers(out: Path) -> None:
    arms = [("17.1b, Sintel only", "2026-09-20_prior18_corpus_dct16", "control"),
            ("18.3, UCF101 corpus", "2026-09-20_prior18_corpus_dct16", "value"),
            ("compiled control", "2026-09-20_prior18_c", "value"),
            ("lr 1e-4", "2026-09-20_prior18_corpus_dct16_lr1e4", "value"),
            ("lr 6e-4", "2026-09-20_prior18_lr6e4", "value"),
            ("lr 1e-3", "2026-09-20_prior18_lr1e3", "value"),
            ("60,000 steps", "2026-09-20_prior18_60k", "value"),
            ("K = 32 coefficients", "2026-09-20_prior18_dct32", "value"),
            ("class-conditional", "2026-09-20_prior18_classes", "value"),
            ("width 192", "2026-09-20_prior18_w192", "value"),
            ("width 192 + lr 1e-3", "2026-09-20_prior18_w192_lr1e3", "value")]
    vals = [run_log(r, "roundtrip_median_prior_samples")[k] for _, r, k in arms]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 5.0), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(arms))[::-1]
    colors = [C_OURS if "width" in lab else C_CTRL for lab, _, _ in arms]
    a1.barh(y, vals, color=colors, height=0.62)
    for yy, v in zip(y, vals):
        a1.text(v * 1.07, yy, f"{v:.3g}", va="center", fontsize=10.5)
    a1.axvline(0.0209, color=C_BAD, lw=1.5, ls="--")
    a1.text(0.0209, len(arms) - 0.2, " floor of the\n representation", color=C_BAD, fontsize=10, va="top")
    a1.set_yticks(y, [lab for lab, _, _ in arms])
    a1.set_xscale("log")
    a1.set_xlim(0.01, 1.5)
    a1.set_xlabel("round trip of 16 draws (median), log scale")
    a1.grid(axis="y", visible=False)
    a1.set_title("the levers", loc="left", fontsize=12, color="#1c1f24")
    r = run_log("2026-09-20_prior18_reach_noise_geometry", "noise_radius_sigma_inside_shell")
    blank = float(re.search(r"IMPROVES TO (\d+\.\d+)", r["note"]).group(1))
    real = float(re.search(r"real clip's (\d+\.\d+)", r["note"]).group(1))
    a2.bar([0, 1], [real, blank], color=[C_OURS, C_CTRL], width=0.6)
    a2.set_xticks([0, 1], ["a real clip", "a near-blank\ngrey field"])
    for xx, v in zip([0, 1], [real, blank]):
        a2.text(xx, v + 0.0003, f"{v:.4f}", ha="center", fontsize=11)
    a2.set_ylabel("round trip (lower reads as better)")
    a2.set_title("the judge that lied", loc="left", fontsize=12, color="#1c1f24")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    page(fig, "prior", "Levers of the prior, and the judge that retired",
         ["Left: one control and its arms; width was the lever, every prediction from the literature pointed elsewhere.",
          "Right: at the floor the round trip scored a near-blank field better than a real clip - so it was retired."],
         "DCT-16 prior over the UCF101 corpus, 16 draws per arm · runs 2026-09-20_prior18_*", out)


def hex_flow(out: Path) -> None:
    k22 = run_log("2026-09-21_prior22_draw_distribution", "per-coordinate kurtosis of 256 fresh draws")
    a22 = run_log("2026-09-21_prior22_draw_distribution",
                  "angle in degrees between a draw's input noise and the latent it becomes")
    acc = json.loads((D / "prior23" / "accept23.json").read_text(encoding="utf-8"))
    s22 = run_log("2026-09-21_prior22_flow_ab1536", "pixel correlation to the raw clip from a fresh draw through the whole chain")
    s23 = run_log("2026-09-21_prior23_seed", "flat-field fraction of a fresh draw on the calibrated scale where white "
                                            "noise is 0 and raw corpus video is 100")
    tr = acc["training_subsamples"]
    rad23 = 100 * (acc["draw"]["radius_mean"] / tr["radius_mean"]["mean"] - 1)
    panels = [("kurtosis of the draws", [k22["value"], acc["draw"]["kurtosis"]], tr["kurtosis"]["mean"],
               "data 8.31", acc["gaussian"]["kurtosis"], "Gaussian 2.97"),
              ("radius overshoot, %", [20.0, rad23], 0.0, "data: 0", None, None),
              ("rotation of a point, °", [a22["value"], acc["transport"]["degrees"]], 90.0, "random 90°", None, None),
              ("own seed returns its clip, r", [s22["control"], 0.884], 0.884, "ceiling 0.884", None, None),
              ("sharpness of a draw", [s23["control"], s23["value"]], 66.0, "ceiling 66", None, None)]
    fig, axes = plt.subplots(1, 5, figsize=(11.2, 3.9))
    for ax, (t, v, ref, reflab, ref2, ref2lab) in zip(axes, panels):
        ax.bar([0, 1], v, color=[C_CTRL, C_OURS], width=0.62)
        for xx, vv in zip([0, 1], v):
            ax.text(xx, vv * 0.97, f"{vv:.3g}", ha="center", va="top", fontsize=10.5, color="white", fontweight="bold")
        ax.axhline(ref, color=C_FLY, lw=1.5, ls="--")
        top = max(max(v), ref) * 1.18
        ax.set_ylim(0, top)
        ax.text(-0.55, ref + top * 0.015, reflab, color=C_FLY, fontsize=9.5, ha="left", va="bottom")
        if ref2 is not None:
            ax.axhline(ref2, color="#9aa0a8", lw=1, ls=":")
            ax.text(-0.55, ref2 + 0.1, ref2lab, color="#6e747e", fontsize=9.5, ha="left", va="bottom")
        ax.set_xticks([0, 1], ["over PCA\n(22)", "on the lattice\n(23)"], fontsize=10)
        ax.set_title(t, loc="left", fontsize=11, color="#1c1f24")
        ax.set_xlim(-0.6, 1.6)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    page(fig, "prior", "The flow on the lattice: everything measured moved",
         ["The same T4a+T4b block, the same data. Grey: a flow over PCA-1536, which keeps no lattice. Green: the flow",
          "on the hexagonal lattice itself. Every number moved toward the data at once; a fresh draw is still no scene."],
         "256 draws against 40 matched training subsamples · runs 2026-09-21_prior22_*, 2026-09-21_prior23_accept, "
         "2026-09-21_prior23_seed", out)


def bought(out: Path) -> None:
    f = json.loads((D / "prior23" / "fixdraw23.json").read_text(encoding="utf-8"))
    tr = f["training_subsamples"]
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.axhspan(tr["kurtosis"]["mean"] - tr["kurtosis"]["sd"], tr["kurtosis"]["mean"] + tr["kurtosis"]["sd"],
               color="#e8f0fe", zorder=0)
    ax.axvspan(tr["radius_mean"]["mean"] - tr["radius_mean"]["sd"], tr["radius_mean"]["mean"] + tr["radius_mean"]["sd"],
               color="#e8f0fe", zorder=0)
    ax.text(tr["radius_mean"]["mean"] + 1.5, 9.35, "data", color=C_FLY, fontsize=10.5)
    for name, arm in f["arms"].items():
        dr = arm["draw"]
        c = C_OURS if name == "control" else (C_BAD if dr["kurtosis"] > 7 else "#9aa0a8")
        ax.scatter(dr["radius_mean"], dr["kurtosis"], s=90, color=c, zorder=3)
        lab = "no correction" if name == "control" else name.replace("=", " ")
        ax.annotate(f"{lab}  ({100 * arm['energy_outside_pca']:.1f} % outside PCA)", (dr["radius_mean"], dr["kurtosis"]),
                    xytext={"radius=151.9": (10, -20), "scale=0.95": (10, 8), "radius=145.0": (10, -14),
                            "scale=0.9": (10, 6), "radius=136.2": (10, -14)}.get(name, (10, -4)),
                    textcoords="offset points", fontsize=10, color=c)
    ax.scatter([f["gaussian"]["radius_mean"]], [f["gaussian"]["kurtosis"]], s=70, marker="s", color="#6e747e", zorder=3)
    ax.annotate("Gaussian", (f["gaussian"]["radius_mean"], f["gaussian"]["kurtosis"]), xytext=(8, 4),
                textcoords="offset points", fontsize=10, color="#6e747e")
    ax.set_xlabel("radius of the drawn latent (data 35.5)")
    ax.set_ylabel("per-coordinate kurtosis (data 8.31)")
    ax.set_xlim(0, 60)
    ax.set_ylim(2, 10)
    fig.tight_layout()
    page(fig, "prior", "A metric that can be bought",
         ["Rescaling the drawn noise, no retraining: two arms lift kurtosis to 7.1-7.2, the best of the line if read",
          "alone - by collapsing the output to the corpus mean, at a third of the data's radius. Read kurtosis with radius."],
         "hex-local flow of step 23, 256 draws per arm, 100 Euler steps · run 2026-09-21_prior23_fixdraw, "
         "data/prior23/fixdraw23.json", out)


def static_chain(out: Path) -> None:
    res = run_log("2026-09-21_prior23_resfix", "pixel correlation to the picture, best of the four crutches (DC plus "
                                               "the mean residue of still states)")
    crutch = [("DC + zeros", float(re.search(r"DC plus zeros (\d+\.\d+)", res["note"]).group(1))),
              ("DC + mean residue of still states", float(re.search(r"still states (\d+\.\d+)", res["note"]).group(1))),
              ("DC + another still's residue", float(re.search(r"ANOTHER still's residue (\d+\.\d+)", res["note"]).group(1))),
              ("DC + unit noise", float(re.search(r"unit-scale noise (\d+\.\d+)", res["note"]).group(1))),
              ("ceiling: the true still state", res["control"])]
    pic = run_log("2026-09-21_prior23_pic", "pixel correlation of a linear map from the DC of T4a+T4b (1,442 numbers) "
                                            "to the clip's mean frame, on held-out clips from ten unseen classes")
    st = run_log("2026-09-21_prior29_static_ab", "flat-field fraction of a drawn picture through the least-squares renderer")
    coh = [float(x) for x in re.search(r"neighbour coherence (\d+\.\d+) against the real state's (\d+\.\d+) and the "
                                       r"control's (\d+\.\d+)", st["note"]).groups()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(crutch) + 1)[::-1]
    names = [c[0] for c in crutch] + ["least squares from the 1,442 numbers"]
    vals = [c[1] for c in crutch] + [pic["value"]]
    cols = [C_CTRL] * 4 + [C_FLY, C_OURS]
    a1.barh(y, vals, color=cols, height=0.62)
    for yy, v in zip(y, vals):
        a1.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=10.5)
    a1.set_yticks(y, names)
    a1.set_xlim(0, 1.12)
    a1.set_xlabel("r to the picture")
    a1.grid(axis="y", visible=False)
    a1.set_title("rendering a 721 × 2 state", loc="left", fontsize=12, color="#1c1f24")
    x = np.arange(2)
    w = 0.26
    draw_v, real_v, ctrl_v = [st["value"] * 100, coh[0]], [st["value"] * 100, coh[1]], [st["control"] * 100, coh[2]]
    a2b = a2.twinx()
    for off, v, c, lab in ((-w, draw_v, C_OURS, "a draw of the static flow"), (0, real_v, C_FLY, "a real held-out state"),
                           (w, ctrl_v, C_CTRL, "control: N(0, I), no flow")):
        a2.bar(x[0] + off, v[0], w, color=c, label=lab)
        a2b.bar(x[1] + off, v[1], w, color=c)
        a2.text(x[0] + off, v[0] + 0.5, f"{v[0]:.1f}", ha="center", fontsize=10)
        a2b.text(x[1] + off, v[1] + 0.015, f"{v[1]:.2f}", ha="center", fontsize=10)
    a2.set_ylim(0, 40)
    a2b.set_ylim(0, 1.15)
    a2.set_xticks(x, ["flat field, %", "neighbour coherence"])
    a2.set_ylabel("flat field, %")
    a2b.set_ylabel("coherence")
    a2b.spines["right"].set_visible(True)
    a2b.grid(False)
    a2.grid(axis="x", visible=False)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=10)
    a2.set_title("the static flow, judged", loc="left", fontsize=12, color="#1c1f24")
    fig.tight_layout()
    page(fig, "static", "A scene as one picture",
         ["Left: 13B cannot render a state constant in time, whatever is fed beside it; a least-squares map reads the",
          "picture at 0.941. Right: through that map, the static flow's draws match a real state; the control does not."],
         "T4a+T4b coefficient 0 · held-out clips of unseen classes · runs 2026-09-21_prior23_resfix, "
         "2026-09-21_prior23_pic, 2026-09-21_prior29_static_ab (64 draws)", out)


FIGS = {"priors": ("first_priors", priors), "inversion": ("flow_inversion", flow_inversion),
        "levers": ("levers_and_judge", levers), "shell": ("seed_shell", seed_shell),
        "hex": ("hex_flow", hex_flow), "bought": ("bought_kurtosis", bought), "static": ("static_chain", static_chain)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", choices=list(FIGS), default=list(FIGS))
    p.add_argument("--out-dir", default=str(ROOT / "docs" / "figures"))
    a = p.parse_args(argv)
    for k in a.only:
        name, fn = FIGS[k]
        out = Path(a.out_dir) / name
        fn(out if k in ("priors", "inversion", "shell") else out.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
