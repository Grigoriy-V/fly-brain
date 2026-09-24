"""Contact sheet of held-out clips' window-mean pictures, grouped by class, to pick
recognisable examples for docs/figures/static_draws by eye.

    python tools/pick_static_examples.py --per-class 8 --out data/figures/static_candidates.png

Each cell is labelled with its corpus index; the chosen indices are passed to
tools/fig_gh_static.py --clips. Grey on a fixed 0..1 scale.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from gh_style import BG, INK, font, hex_image, honeycomb  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=8)
    p.add_argument("--out", default=str(ROOT / "data" / "figures" / "static_candidates.png"))
    a = p.parse_args(argv)
    v = np.load(ROOT / "data" / "corpus18" / "videos.npz", mmap_mode="r")
    vids, meta = v["videos"], v["meta"]
    idx = np.load(ROOT / "data" / "prior19" / "pca_ab2048_latent.npz")["index_test"]
    by = collections.defaultdict(list)
    for i in idx:
        by[json.loads(str(meta[i]))["label"]].append(int(i))
    comb = honeycomb(3)
    pw, ph = comb[0].shape[1], comb[0].shape[0]
    rows = sorted(by)
    im = Image.new("RGB", (220 + a.per_class * (pw + 8), len(rows) * (ph + 22) + 10), BG)
    d = ImageDraw.Draw(im)
    f = font(13)
    rng = np.random.default_rng(0)
    for r, lab in enumerate(rows):
        pick = sorted(rng.choice(by[lab], min(a.per_class, len(by[lab])), replace=False))
        y = 5 + r * (ph + 22)
        d.text((5, y + ph // 2), f"{lab} ({len(by[lab])})", font=f, fill=INK)
        for c, i in enumerate(pick):
            x = 220 + c * (pw + 8)
            im.paste(hex_image(vids[i, :40].mean(0), comb), (x, y))
            d.text((x, y + ph + 2), str(i), font=f, fill=INK)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    im.save(a.out)
    print(a.out, {k: len(v) for k, v in by.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
