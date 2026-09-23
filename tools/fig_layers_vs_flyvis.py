"""Evidence figure for ISS-0015: activity maps of seven cell types on one frame,
FlyVis on top, model zero below. Reads data/figures/act_s3_malecns_flyvis.npz
(tools/cmp_layers_flyvis.py). Writes reports/figures/2026-09-24_malecns_layers_vs_flyvis.png.

    python tools/fig_layers_vs_flyvis.py
"""
import sys, numpy as np
sys.path.insert(0, 'tools'); sys.path.insert(0, '.')
import matplotlib; matplotlib.use("Agg"); from matplotlib import colormaps
from PIL import Image, ImageDraw
from gh_style import honeycomb, hex_image, font
A = np.load("data/figures/act_s3_malecns_flyvis.npz")
comb = honeycomb(5); pw, ph = comb[0].shape[1], comb[0].shape[0]
types = ["L1", "L3", "Mi1", "Tm1", "Tm9", "T4a", "T5a"]
im = Image.new("RGB", (60 + (len(types)) * (pw + 12), 2 * (ph + 30) + 40), (255, 255, 255)); d = ImageDraw.Draw(im)
cm = colormaps["RdBu_r"]
for r, m in enumerate(["flow/00", "malecns"]):
    d.text((4, 30 + r * (ph + 30) + ph // 2), m[:4], font=font(14, True), fill=(0, 0, 0))
    for c, t in enumerate(types):
        x = A[f"{m}_{t}"][20]; x = x - np.nanmedian(x); s = np.nanpercentile(abs(x), 99)
        im.paste(hex_image(0.5 + 0.5 * np.clip(x / s, -1, 1), comb, cmap=cm), (50 + c * (pw + 12), 30 + r * (ph + 30)))
        if r == 0: d.text((50 + c * (pw + 12), 8), t, font=font(15, True), fill=(0, 0, 0))
im.save("reports/figures/2026-09-24_malecns_layers_vs_flyvis.png")
