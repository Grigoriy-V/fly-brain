"""FlyVis inversion ladder at the same settings as model zero's (40 frames + 5 margin,
150 steps, Sintel clip 3, control clip 10): the data behind docs/figures/two_brains.
Local CPU, ~8 min. Writes data/generate/2026-09-24_flyvis_invert_<type>_s3.

    python tools/run_flyvis_ladder.py
"""
import sys, time
sys.path.insert(0, '.')
from pathlib import Path
from flydream.generate import invert as I
g = I.settings()
t0 = time.time()
I.run_ladder("flow/0000/000", 3, [["R1"], ["L1"], ["Mi1"], ["Tm9"], ["T4a"], ["T5a"]],
             frames=int(g.get("frames", 40)), steps=int(g.get("steps", 150)), lr=float(g.get("lr", 0.05)),
             tv=float(g.get("tv", 0.02)), dt=float(g.get("dt", 0.02)), t_pre=float(g.get("t_pre", 1.0)),
             margin=int(g.get("margin", 5)), control_sample=10, out_root=Path("data/generate"),
             tag_prefix="2026-09-24_flyvis_", batch=0)
print("TOTAL", round(time.time() - t0), "s", flush=True)
