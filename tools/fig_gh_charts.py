"""GitHub/article charts for part 1: the numbers behind the pictures, in the house style.

    python tools/fig_gh_charts.py                 # all six into docs/figures/
    python tools/fig_gh_charts.py --only dsi loop

Every chart reads a saved run, nothing is recomputed on a worker:

- `dsi_by_version`     (model, part 1 section 2): T4/T5 direction selectivity per
  member and subtype, transplants v5 / v7 / v9 against the FlyVis member,
  `data/runs/2026-09-18_step2_zero_R_*/summary.json`.
- `decoder_window`     (decode, section 3): the frame component of a ridge decoder
  (pixel r minus the time-shuffled control, median per stage) against the
  decoder's window, FlyVis member 000 on Sintel, `data/decode/2026-09-18_decode_sintel_lag_*/map.csv`.
  Windows of every second lag (80, 160 ms) alias Sintel's frame hold (ISS-0006)
  and are drawn hollow.
- `inversion_chart`    (invert, sections 4 and 6): r of the video recovered by
  inversion from each layer of model zero - a Sintel clip and white noise in the
  eye - beside each run's control, `data/generate/2026-09-19_malecns_{invert,dream_eye_noise}_*/meta.json`.
- `speedups`           (invert, section 5): the batched inversion ladder against the
  sequential one and the training step's throughput by batch, from the run log
  (`2026-09-19_generate_malecns_s3_40f5_batch20`, `2026-09-18_step3_smoke_batch_t4`).
- `round_trips`        (render, section 9): 13B's round trip on conditions no clip
  caused, mean of two seeds, `data/prompts14/summary.json`.
- `closed_loop`        (render, section 9): state -> 13B -> video -> brain -> state,
  r to the starting clip and the round trip per pass, same file.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gh_style import BG, INK, MUTED, font, pipeline_strip  # noqa: E402

W = 1216
C_FLY, C_OURS, C_CTRL, C_GOOD, C_BAD = "#2563eb", "#15803d", "#b0b5bd", "#15803d", "#b91c1c"
C_CTX = ["#d5d8dd", "#a9aeb6"]
STAGE_COLORS = ["#2563eb", "#ea580c", "#15803d", "#ca8a04", "#db2777"]
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": "#c9ced6",
                     "axes.labelcolor": "#3a3f47", "xtick.color": "#6e747e", "ytick.color": "#6e747e",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e8ec", "grid.linewidth": 1, "axes.axisbelow": True,
                     "legend.frameon": False})


def page(fig, stage: str, title: str, lines: list[str], footer: str, out: Path) -> None:
    """The matplotlib chart under the house header: pipeline strip, title, subtitle; footer names the run."""
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, facecolor="white")
    plt.close(fig)
    chart = Image.open(buf).convert("RGB")
    if chart.width > W - 64:
        chart = chart.resize((W - 64, round(chart.height * (W - 64) / chart.width)), Image.LANCZOS)
    head = 24 + 34 + 26 + 44 + 26 * len(lines) + 16
    H = head + chart.height + 50
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = pipeline_strip(d, 48, 24, W - 96, stage) + 26
    d.text((48, y), title, font=font(30, True), fill=INK)
    y += 44
    for ln in lines:
        d.text((48, y), ln, font=font(19), fill=MUTED)
        y += 26
    im.paste(chart, ((W - chart.width) // 2, head))
    d.text((48, H - 38), footer, font=font(16), fill=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(f"{out}  {out.stat().st_size / 1e6:.2f} MB")


def run_log(run: str, metric: str) -> dict:
    for ln in (ROOT / "reports" / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r["run"] == run and r["metric"] == metric:
                return r
    raise KeyError((run, metric))


# ----------------------------------------------------------------------------- charts

SUBTYPES = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]


def dsi(out: Path) -> None:
    runs = [("v5", "2026-09-18_step2_zero_R_v5"), ("v7", "2026-09-18_step2_zero_R_v7_total_cap3"),
            ("v9 (current)", "2026-09-18_step2_zero_R_v9_wk50m500oc")]
    S = {k: json.loads((ROOT / "data" / "runs" / r / "summary.json").read_text()) for k, r in runs}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.6), sharey=True)
    y = np.arange(len(SUBTYPES))[::-1]
    for m, ax in enumerate(axes):
        fly = [S["v9 (current)"][m]["flyvis_dsi"][t]["dsi"] for t in SUBTYPES]
        for (k, _), c, s in zip(runs, C_CTX + [C_OURS], (40, 40, 90)):
            v = [S[k][m]["malecns_dsi"][t]["dsi"] for t in SUBTYPES]
            ax.scatter(v, y, s=s, color=c, zorder=3, label=f"MaleCNS {k}")
        ax.scatter(fly, y, s=90, marker="D", color=C_FLY, zorder=4, label="FlyVis, same member")
        ax.axhspan(-0.5, 3.5, color="#f4f5f7", zorder=0)
        ax.set_title(f"member {m:03d}", loc="left", fontsize=12, color="#1c1f24")
        ax.set_xlim(-0.02, 0.85)
        ax.set_xlabel("direction selectivity index")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, SUBTYPES)
    axes[0].text(0.84, 1.5, "OFF\n(T5)", ha="right", va="center", fontsize=10, color="#6e747e")
    axes[0].text(0.84, 5.5, "ON\n(T4)", ha="right", va="center", fontsize=10, color="#6e747e")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    page(fig, "model", "Direction selectivity after the transplant",
         ["T4/T5 selectivity on the MaleCNS wiring, transplant by transplant, against the FlyVis member it came",
          "from. Rescaling the gain lifted T4 (ON) toward FlyVis; T5 (OFF) stayed weak or pointed the wrong way."],
         "Moving-edge protocol of FlyVis, 12 directions · runs 2026-09-18_step2_zero_R_v5 / v7_total_cap3 / "
         "v9_wk50m500oc", out)


def window(out: Path) -> None:
    from flydream.decode.sweep import STAGES, load, summarise
    runs = [f"2026-09-18_decode_sintel_lag_{x}" for x in ("0", "0_1", "0_2_4", "0_2_4_6_8")]
    s = summarise(load(runs))
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.axvspan(50, 190, color="#f4f5f7", zorder=0)
    ax.text(120, 0.585, "every second lag:\naliases Sintel's frame hold", ha="center", va="top",
            fontsize=10, color="#6e747e")
    starts = []
    for i, (st, _) in enumerate(STAGES):
        d = s[s.stage == st].sort_values("window_ms")
        c = STAGE_COLORS[i]
        clean, alias = d[d.window_ms <= 20], d[d.window_ms >= 20]
        ax.plot(clean.window_ms, clean.frame, color=c, lw=2.5, zorder=3)
        ax.plot(alias.window_ms, alias.frame, color=c, lw=1.5, ls="--", zorder=3)
        ax.scatter(clean.window_ms, clean.frame, s=70, color=c, zorder=4)
        ax.scatter(d[d.window_ms > 20].window_ms, d[d.window_ms > 20].frame, s=70, facecolor="white",
                   edgecolor=c, lw=2, zorder=4)
        starts.append((float(d.frame.iloc[0]), st[2:], c))
    starts.sort()
    placed = []
    for v, lab, c in starts:  # labels at lag 0, pushed apart where two stages start together
        yy = v if not placed or v - placed[-1] >= 0.02 else placed[-1] + 0.02
        placed.append(yy)
        ax.text(-12, yy, lab, ha="right", va="center", fontsize=10.5, color=c)
    ax.set_xticks([0, 20, 80, 160], ["0\nlag 0", "20\nlags 0,1", "80\nlags 0,2,4", "160\nlags 0,2,…,8"])
    ax.set_xlim(-75, 175)
    ax.set_xlabel("decoder window after the frame, ms")
    ax.set_ylabel("frame component of r\n(r minus time-shuffled control)")
    fig.tight_layout()
    page(fig, "decode", "A linear readout depends on its window",
         ["The frame a ridge decoder reads from each stage, beyond what a time-shuffled control reads. At lag 0",
          "the motion detectors look blind; one more frame of context and they pass the lamina and Tm/TmY."],
         "FlyVis network, member 000 · Sintel, split by scene · median over the types of a stage · "
         "runs 2026-09-18_decode_sintel_lag_*", out)


LAYERS = ["R1", "L1", "L3", "Mi1", "Mi4", "Tm9", "Tm5a", "T4a", "T5a", "T4T5"]


def inversion(out: Path) -> None:
    def meta(src, t):
        tag = f"2026-09-19_malecns_{src}_{t}" + ("_s3" if src == "invert" else "")
        return json.loads((ROOT / "data" / "generate" / tag / "meta.json").read_text())
    clip = [meta("invert", t) for t in LAYERS]
    noise = [meta("dream_eye_noise", t) for t in LAYERS]
    x = np.arange(len(LAYERS))
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    b = 0.2
    ax.bar(x - 1.5 * b, [m["inversion"] for m in clip], b, color=C_OURS, label="Sintel clip: recovered video, r to the clip")
    ax.bar(x - 0.5 * b, [m["control"] for m in clip], b, color=C_CTRL,
           label="control: aimed at another clip's state, r to this clip")
    ax.bar(x + 0.5 * b, [m["inversion"] for m in noise], b, color="#7c3aed", label="white noise in the eye: r to the noise")
    ax.bar(x + 1.5 * b, [m["control"] for m in noise], b, color="#dcdfe4", label="control: shuffled state")
    ax.axhline(0, color="#9aa0a8", lw=1)
    ax.set_xticks(x, ["T4+T5" if t == "T4T5" else t for t in LAYERS])
    ax.set_ylim(-0.3, 1.08)
    ax.set_ylabel("r to what the eye saw")
    for xi, m in zip(x, noise):
        if m["inversion"] < 0.6:
            ax.text(xi + 0.5 * b, m["inversion"] + 0.03, f"{m['inversion']:.2f}", ha="center", fontsize=10, color="#7c3aed")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=2, fontsize=10.5)
    fig.tight_layout()
    page(fig, "invert", "Inversion, layer by layer",
         ["The video recovered from each layer of frozen model zero, beside its control. A natural clip comes back",
          "from every layer; white noise in the eye comes back from all but two types, Tm5a and T5a."],
         "40 frames + 5 margin, 150 steps, 20 tasks per batch on a T4 · runs 2026-09-19_malecns_invert_*_s3, "
         "2026-09-19_malecns_dream_eye_noise_*", out)


def speedups(out: Path) -> None:
    lad = run_log("2026-09-19_generate_malecns_s3_40f5_batch20", "seconds_per_ladder_t4")
    b16 = run_log("2026-09-18_step3_smoke_batch_t4", "samples_per_s_batch16")
    b32 = run_log("2026-09-18_step3_smoke_batch_t4", "samples_per_s_batch32")
    opt = float(re.search(r"150 steps in (\d+) s", lad["note"]).group(1))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    a1.barh([1, 0], [lad["control"], lad["value"]], color=["#b0b5bd", C_OURS], height=0.55)
    a1.barh([0], [opt], color="#0f5a2c", height=0.55)
    a1.set_yticks([1, 0], ["one task at a time", "20 tasks in one batch"])
    a1.text(lad["control"] + 15, 1, f"{lad['control']:.0f} s", va="center", fontsize=11)
    a1.text(lad["value"] + 15, 0, f"{lad['value']:.0f} s whole ladder, {opt:.0f} s optimisation", va="center", fontsize=11)
    a1.set_xlim(0, 1150)
    a1.set_xlabel("seconds on a T4 for one inversion ladder (10 layers + 10 controls)")
    a1.grid(axis="y", visible=False)
    a1.set_title("inversion", loc="left", fontsize=12, color="#1c1f24")
    xs, ys = [4, 16, 32], [b16["control"], b16["value"], b32["value"]]
    a2.plot(xs, ys, color=C_FLY, lw=2.5, marker="o", ms=8)
    for xx, yy in zip(xs, ys):
        a2.text(xx, yy + 0.7, f"{yy:.1f}", ha="center", fontsize=11)
    a2.set_xticks(xs)
    a2.set_ylim(0, 17)
    a2.set_xlabel("batch size")
    a2.set_ylabel("training samples per second")
    a2.set_title("training step of model zero", loc="left", fontsize=12, color="#1c1f24")
    fig.tight_layout()
    page(fig, "invert", "Where the card goes",
         ["Left: batching the independent inversions of a ladder cut it from 948 s to 176 s with the same videos",
          "(max difference 7e-4). Right: the training step saturates the T4 at batch 16; batch 64 cannot run."],
         "Tesla T4 · runs 2026-09-19_generate_malecns_s3_40f5_batch20, 2026-09-18_step3_smoke_batch_t4", out)


ROUND_TRIPS = [
    ("e_A", "a real clip's state", "clip"),
    ("p_hand_T4a_stripe", "hand-drawn T4a stripe (ignored: grey video)", "ignored"),
    ("p_window_up", "composition: upward motion in a window", "comp"),
    ("p_right_expand", "composition: rightward beside expansion", "comp"),
    ("p_left_right_conflict", "composition: right on the left, left on the right", "comp"),
    ("e_A_amp_a", "edit: one direction amplified", "edit"),
    ("e_A_timerev", "edit: time reversed", "edit"),
    ("r0_white", "random T4/T5 activity, white", "random"),
    ("e_right_swap_ab", "edit: direction channels swapped", "edit"),
    ("r0_structured", "random T4/T5 activity, structured", "random"),
    ("e_A_swap_ac", "edit: direction channels swapped (clip)", "edit"),
    ("e_right_timerev", "edit: time reversed (grating)", "edit"),
    ("e_A_rot90", "edit: channels rotated 90° (clip)", "edit"),
    ("e_right_rot90", "edit: channels rotated 90° (grating)", "edit"),
    ("c_shuffled", "control: the clip's state, cells shuffled", "control"),
]


def round_trips(out: Path) -> None:
    sc = json.loads((ROOT / "data" / "prompts14" / "summary.json").read_text())["scores"]
    col = {"clip": C_GOOD, "ignored": "#ea580c", "comp": C_FLY, "edit": "#9aa0a8", "random": "#6b7280",
           "control": C_BAD}
    vals = [np.mean([sc[f"{k}__seed{s}"]["round_trip"] for s in (0, 1)]) for k, _, _ in ROUND_TRIPS]
    y = np.arange(len(ROUND_TRIPS))[::-1]
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.barh(y, vals, color=[col[g] for _, _, g in ROUND_TRIPS], height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v * 1.08, yy, f"{v:.3g}", va="center", fontsize=10.5)
    ax.set_yticks(y, [lab for _, lab, _ in ROUND_TRIPS])
    ax.set_xscale("log")
    ax.set_xlim(0.01, 60)
    ax.set_xlabel("round trip: video → frozen brain → state, error against the condition (log scale)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    page(fig, "render", "Which conditions the generator obeys",
         ["13B's round trip on T4/T5 states that no clip caused. Low is compatible. The hand-drawn stripe scores",
          "better than every composition although the generator ignored it - the round trip is not a judge of content."],
         "13B generator, 40 frames, mean of two noise seeds · run 2026-09-20_prompts14, data/prompts14/summary.json",
         out)


def closed_loop(out: Path) -> None:
    lp = json.loads((ROOT / "data" / "prompts14" / "summary.json").read_text())["loops"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for k, lab, c in (("loop_clipA", "Sintel clip A", C_OURS), ("loop_clipB", "Sintel clip B", "#7c3aed"),
                      ("loop_grating", "grating", C_FLY)):
        it = [p["iter"] for p in lp[k]]
        a1.plot(it, [p["r_video_start"] for p in lp[k]], color=c, lw=2.5, marker="o", ms=5, label=lab)
        a2.plot(it, [p["round_trip"] for p in lp[k]], color=c, lw=2.5, marker="o", ms=5, label=lab)
    a1.set_ylim(0, 1.05)
    a1.set_xlabel("pass")
    a1.set_ylabel("r of the video to the starting clip")
    a1.legend(loc="lower left", fontsize=10.5)
    a1.set_title("content drifts", loc="left", fontsize=12, color="#1c1f24")
    a2.set_ylim(0, 0.12)
    a2.set_xlabel("pass")
    a2.set_ylabel("round trip of each pass")
    a2.set_title("while every pass stays compatible", loc="left", fontsize=12, color="#1c1f24")
    fig.tight_layout()
    page(fig, "render", "The closed loop: state → generator → video → brain → state",
         ["Each pass renders the state, runs the video through the frozen brain and renders the new state. Every",
          "step is compatible with the one before, yet the video walks away from the clip it started from."],
         "13B generator and model zero, 12 passes, noise seed 0 · run 2026-09-20_prompts14_loop, "
         "data/prompts14/summary.json", out)


CHARTS = {"dsi": ("dsi_by_version", dsi), "window": ("decoder_window", window),
          "inversion": ("inversion_chart", inversion), "speed": ("speedups", speedups),
          "roundtrip": ("round_trips", round_trips), "loop": ("closed_loop", closed_loop)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", choices=list(CHARTS), default=list(CHARTS))
    p.add_argument("--out-dir", default=str(ROOT / "docs" / "figures"))
    a = p.parse_args(argv)
    for k in a.only:
        name, fn = CHARTS[k]
        fn(Path(a.out_dir) / f"{name}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
