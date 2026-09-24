"""GitHub figure: the MaleCNS connectome brought into FlyVis's lattice.

    python tools/fig_gh_connectome.py --out docs/figures/connectome_export

Stage "connectome" (article part 1, section 1). Left: reconstructed MaleCNS
neurons of three columnar types placed on the eye's hexagonal columns by where
their synapses sit (`data/ol/neurons_R.parquet`, home column per neuron) - the
real wiring becomes a lattice. Right: every FlyVis type pair's total input
against the same pair exported from MaleCNS (`data/ol/filter_comparison_R.csv`),
with the rank correlation and sign agreement computed here; pairs present in
one connectome only sit on the axes. A PNG.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from gh_style import BG, INK, MUTED, font, pipeline_strip  # noqa: E402

TYPES = [("L1", "#2563eb"), ("Mi1", "#15803d"), ("T4a", "#b91c1c")]


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import spearmanr

    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "docs" / "figures" / "connectome_export"))
    a = p.parse_args(argv)

    neu = pd.read_parquet(ROOT / "data" / "ol" / "neurons_R.parquet")
    cmp_ = pd.read_csv(ROOT / "data" / "ol" / "filter_comparison_R.csv")
    both = cmp_[(cmp_.flyvis_total > 0) & (cmp_.malecns_total > 0)]
    rho = spearmanr(both.flyvis_total, both.malecns_total).statistic
    s = both.dropna(subset=["flyvis_sign", "malecns_sign"])
    sign = float((np.sign(s.flyvis_sign) == np.sign(s.malecns_sign)).mean())
    fv_only = cmp_[(cmp_.flyvis_total > 0) & (cmp_.malecns_total <= 0)]
    mc_only = cmp_[(cmp_.flyvis_total <= 0) & (cmp_.malecns_total > 0)]
    n_fv = int((cmp_.flyvis_total > 0).sum())
    print(f"pairs in both {len(both)} of FlyVis {n_fv}; rank {rho:.3f}; sign {sign:.3f} on {len(s)}; "
          f"FlyVis only {len(fv_only)}, MaleCNS only {len(mc_only)}")

    import json
    M = json.loads((ROOT / "data" / "ol" / "orientation_R.json").read_text(encoding="utf-8"))["chosen_M"]
    fig = plt.figure(figsize=(11.2, 5.0), dpi=110)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, width_ratios=[0.55, 0.55, 1.35], hspace=0.25, wspace=0.32)
    slots = [gs[0, 0], gs[0, 1], gs[1, 0]]
    for (t, c), slot in zip(TYPES, slots):
        a0 = fig.add_subplot(slot)
        g = neu[(neu.type == t) & neu.hex1.notna()]
        h1, h2 = g.hex1.to_numpy(float), g.hex2.to_numpy(float)
        u, v = M[0] * h1 + M[1] * h2, M[2] * h1 + M[3] * h2          # MaleCNS axes -> FlyVis (u, v)
        a0.scatter(v * np.sqrt(3) / 2, u + v / 2, s=7, marker="h", color=c, linewidths=0)
        a0.set_aspect("equal"); a0.axis("off")
        a0.set_title(f"{t}: {len(g)} neurons", fontsize=10, color="#1c1f24")
    note = fig.add_subplot(gs[1, 1]); note.axis("off")
    note.text(0.0, 0.55, "each dot: one reconstructed\nneuron at its home column,\nfound from where its\n"
                         "synapses sit", fontsize=9, color="#6e747e", va="center")

    a1 = fig.add_subplot(gs[:, 2])
    lo = 1e-3
    a1.scatter(both.flyvis_total, both.malecns_total, s=9, color="#2563eb", alpha=0.6, linewidths=0,
               label=f"in both ({len(both)})")
    a1.scatter(fv_only.flyvis_total, np.full(len(fv_only), lo), s=9, color="#b91c1c", alpha=0.7, linewidths=0,
               label=f"FlyVis only ({len(fv_only)})")
    a1.scatter(np.full(len(mc_only), lo), mc_only.malecns_total, s=9, color="#9ca3af", alpha=0.7, linewidths=0,
               label=f"MaleCNS only ({len(mc_only)})")
    a1.set_xscale("log"); a1.set_yscale("log")
    lim = [lo * 0.7, max(cmp_.flyvis_total.max(), cmp_.malecns_total.max()) * 1.5]
    a1.plot(lim, lim, color="#d6dae0", lw=1, zorder=0)
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("FlyVis connectome: total input per type pair", fontsize=9)
    a1.set_ylabel("MaleCNS export: same pair", fontsize=9)
    a1.set_title(f"Type pairs: rank r = {rho:.2f}, signs agree {sign:.0%}", fontsize=11, loc="left",
                 color="#1c1f24")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plot = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    W = 1216
    plot = plot.resize((W - 60, int(plot.height * (W - 60) / plot.width)), Image.LANCZOS)
    H = 190 + plot.height + 50
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = pipeline_strip(d, 48, 24, W - 96, "connectome") + 26
    d.text((48, y), "The connectome, brought into a trainable network", font=font(30, True), fill=INK)
    y += 44
    d.text((48, y), "MaleCNS gives individual neurons and synapses; FlyVis needs cell types on a hexagonal lattice. "
                    "Each neuron gets its home column", font=font(19), fill=MUTED)
    y += 26
    d.text((48, y), "from its synapses, and connections become type → type → column offset. The two connectomes "
                    "agree broadly, not exactly.", font=font(19), fill=MUTED)
    im.paste(plot, (30, 190))
    d.text((48, H - 38), "MaleCNS v1.0 (Janelia FlyEM, CC-BY 4.0), right optic lobe · home column = MaleCNS's own "
                         "tag in 96.9 % of 13,267 cells · export w5wk50m500oc",
           font=font(16), fill=MUTED)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out.with_suffix(".png"))
    print(f"{out.with_suffix('.png')}  {out.with_suffix('.png').stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
