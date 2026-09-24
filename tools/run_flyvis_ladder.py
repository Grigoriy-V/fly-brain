"""FlyVis inversion ladder at the same settings as model zero's (40 frames + 5 margin,
150 steps; default Sintel clip 3, control clip 10): the data behind docs/figures/two_brains;
with --sample 27 --control 75 --prefix 2026-09-24_flyvis_s27_ the inversion shown beside decoding.
Local CPU, ~8 min. Writes data/generate/2026-09-24_flyvis_invert_<type>_s3.

    python tools/run_flyvis_ladder.py
"""
import argparse, sys, time
sys.path.insert(0, '.')
from pathlib import Path
from flydream.generate import invert as I
ap = argparse.ArgumentParser()
ap.add_argument("--sample", type=int, default=3)
ap.add_argument("--control", type=int, default=10)
ap.add_argument("--prefix", default="2026-09-24_flyvis_")
a = ap.parse_args()
g = I.settings()
t0 = time.time()
I.run_ladder("flow/0000/000", a.sample, [["R1"], ["L1"], ["Mi1"], ["Tm9"], ["T4a"], ["T5a"]],
             frames=int(g.get("frames", 40)), steps=int(g.get("steps", 150)), lr=float(g.get("lr", 0.05)),
             tv=float(g.get("tv", 0.02)), dt=float(g.get("dt", 0.02)), t_pre=float(g.get("t_pre", 1.0)),
             margin=int(g.get("margin", 5)), control_sample=a.control, out_root=Path("data/generate"),
             tag_prefix=a.prefix, batch=0)
print("TOTAL", round(time.time() - t0), "s", flush=True)
