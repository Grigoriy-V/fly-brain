"""Model zero against FlyVis, layer by layer, on one Sintel clip: roughness of each
type's activity map (mean |cell - mean of its six neighbours| / spatial sd, frame 20)
and its correlation with the frame. Evidence for ISS-0015. Local, ~40 s.

    python tools/cmp_layers_flyvis.py
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, '.')
from flydream.generate import invert as I
from flydream.decode import pairs as P
from flyvis.utils.hex_utils import get_hex_coords, get_hextent
from scipy.spatial import cKDTree
s = I.settings(); dt = 0.02
U, V = get_hex_coords(get_hextent(721)); pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(U, V))}
x = V * np.sqrt(3) / 2; y = U + V / 2
tree = cKDTree(np.stack([x, y], 1)); nb = [tree.query_ball_point(p, 1.01) for p in np.stack([x, y], 1)]
video = I.clip_from_sintel(3, 40, dt, 5)
out = {}
Path('data/figures').mkdir(parents=True, exist_ok=True)
for model in ("malecns", "flow/0000/000"):
    net = I.load_network(model)
    types, index = P.type_index(net.connectome)
    st = net.steady_state(1.0, dt, batch_size=1, value=0.5)
    with torch.no_grad():
        act = I.simulate(net, torch.as_tensor(video[None]), dt, st)[0].numpy()[:40]
    u_all = np.asarray(net.connectome.nodes.u[:]); v_all = np.asarray(net.connectome.nodes.v[:])
    for t in ["R1", "L1", "L3", "Mi1", "Tm9", "Tm1", "T4a", "T5a"]:
        if t not in index: continue
        a = np.full((40, 721), np.nan, np.float32)
        for k in index[t]:
            j = pos.get((int(u_all[k]), int(v_all[k])))
            if j is not None: a[:, j] = act[:, k]
        # roughness: mean |cell - mean of its 6 neighbours| relative to spatial sd, frame 20
        f = a[20] - np.nanmedian(a[20])
        rough = np.nanmean([abs(f[i] - np.nanmean(f[[j for j in nb[i] if j != i]])) for i in range(721)]) / (np.nanstd(f) + 1e-9)
        rv = np.nanmean([np.corrcoef(a[k], video[k])[0, 1] for k in range(5, 40)])
        print(f"{model[:7]:8s} {t:4s} cells {len(index[t]):4d}  roughness {rough:.2f}  corr w/ frame {rv:+.2f}")
        out[f"{model[:7]}_{t}"] = a
out["video"] = video[:40]
np.savez("data/figures/act_s3_malecns_flyvis.npz", **out)
