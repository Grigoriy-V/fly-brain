"""Model zero: FlyVis dynamics on the MaleCNS export, parameters transplanted by type.

    python -m flydream.model.zero [--models 0 1 2] [--protocols flash edge]

For each FlyVis ensemble member named, build a Network on
data/ol/filters_<side>.json, copy its resting potentials and time constants by
cell type and its synapse strengths by (source type, target type), medians for
what has no match, and run the flash and moving-edge protocols on both the
FlyVis network and the MaleCNS network. Writes per-model results under
data/runs/<run>/ and prints the comparison table.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import tomllib

from flydream.model import ROOT, configure_flyvis_root

configure_flyvis_root()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402

import flyvis  # noqa: E402
from flyvis import Network, NetworkView  # noqa: E402
from flyvis.analysis.flash_responses import flash_response_index  # noqa: E402
from flyvis.analysis.moving_bar_responses import direction_selectivity_index, preferred_direction  # noqa: E402
from flyvis.datasets.flashes import Flashes  # noqa: E402
from flyvis.datasets.moving_bar import MovingEdge  # noqa: E402
from flyvis.utils import groundtruth_utils as gt  # noqa: E402
from flyvis.utils.config_utils import Namespace  # noqa: E402

DATA = ROOT / "data"


def settings() -> dict:
    return tomllib.loads((ROOT / "config.toml").read_text())


# ----------------------------------------------------------------------------- build


def content_addressed(connectome_file: pathlib.Path) -> pathlib.Path:
    """datamate caches a built connectome by its config (the file PATH), not by the
    file's content, so a re-exported file would be served stale. Copy the file to a
    name carrying its content hash and build from that."""
    import hashlib
    import shutil
    digest = hashlib.sha256(connectome_file.read_bytes()).hexdigest()[:10]
    target = connectome_file.parent / "cache" / f"{connectome_file.stem}_{digest}.json"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(connectome_file, target)
    return target


def build_network(connectome_file: pathlib.Path, extent: int) -> Network:
    """A Network with FlyVis's default dynamics and parameter shapes on our file."""
    connectome_file = content_addressed(connectome_file)
    net = Network(connectome=Namespace(type="ConnectomeFromAvgFilters", file=str(connectome_file),
                                       extent=extent, n_syn_fill=0))   # MaleCNS filters are complete: no hull filling
    net.eval()
    return net


def shuffle_strengths(net: Network, seed: int = 0) -> None:
    """Control: permute the per-type-pair synapse strengths among the pairs."""
    g = torch.Generator().manual_seed(seed)
    p = net.edge_params["syn_strength"].raw_values
    with torch.no_grad():
        p.copy_(p[torch.randperm(len(p), generator=g)])


def mean_n_syn(net: Network) -> dict:
    """Mean synapse count per edge of each (source type, target type) pair.

    Diagnostic only, and the wrong basis for a transplant: the MaleCNS export
    keeps many weak far-offset edges (`min_filter_syn` 0.25) that FlyVis's
    integer filters never have, so the per-edge mean is dragged down by edges
    that carry almost nothing, and a gain rescaled by it inflates the strong
    central edges of the same pair. Measured 2026-09-18: Tm9 to T5a came out
    4.1x by this statistic and 1.7x by the one that matters (`total_n_syn`)."""
    e = net.connectome.edges
    src = e.source_type[:].astype(str)
    tar = e.target_type[:].astype(str)
    n = np.asarray(e.n_syn[:], dtype=float)
    total, count = {}, {}
    for s, t, c in zip(src, tar, n):
        k = (s, t)
        total[k] = total.get(k, 0.0) + float(c)
        count[k] = count.get(k, 0) + 1
    return {k: total[k] / count[k] for k in total}


def total_n_syn(net: Network) -> dict:
    """Synapses a cell of the target type receives from the source type, averaged
    over target cells: the pair's total drive, whatever offsets it is spread over.

    This is the invariant a transplant must preserve. The edge weight is
    `sign * n_syn * syn_strength` with one `syn_strength` per type pair, so the
    total input a cell gets from a source type is `syn_strength * total_n_syn`;
    keep that equal between connectomes and the pathway is as strong as FlyVis
    learned it, however differently the two connectomes spread it over offsets."""
    e = net.connectome.edges
    src = e.source_type[:].astype(str)
    tar = e.target_type[:].astype(str)
    n = np.asarray(e.n_syn[:], dtype=float)
    ti = np.asarray(e.target_index[:])
    per_cell: dict = {}
    for s, t, c, i in zip(src, tar, n, ti):
        d = per_cell.setdefault((s, t), {})
        d[int(i)] = d.get(int(i), 0.0) + float(c)
    return {k: float(np.mean(list(d.values()))) for k, d in per_cell.items()}


def transplant(src: Network, dst: Network, rescale: bool = True,
               rescale_cap: float | None = 3.0) -> dict:
    """Copy src's per-type and per-type-pair parameters into dst by key.

    `syn_strength` is NOT a connectome-independent number and must not be copied
    raw. FlyVis initialises it as `scale / <n_syn>` for that pair in that
    connectome (`flyvis/network/initialization.py:501`) and forms the edge weight
    as `sign * n_syn * syn_strength` (`dynamics.py:165`), so the parameter is a
    gain already divided by the connectome's own synapse count. Copying the raw
    value into a connectome whose counts differ rescales every pathway by that
    difference. Measured 2026-09-18 on the raw copy: T5's main excitatory drive
    (Tm9) came out 1.5 to 1.7 times too weak and its TmY15 inhibition about
    twice too strong, and step 2's DSI of 0.020 against FlyVis's 0.391 was at
    least partly that: member 000 went to 0.161 once corrected.

    With `rescale=True` the transplanted value is `syn_strength_src *
    total_src / total_dst` on the `total_n_syn` basis, which preserves the mean
    total input a target cell receives from the source type.

    `rescale_cap` bounds the factor to [1/cap, cap], and the pairs that hit it
    come back in the report as `capped_pairs`. A thirtyfold difference in total
    input (R1-R6 to Am, 36 against 1.2 synapses per cell; Am to T1, 63 against
    3.3) is not something to correct with gain: it is an export defect to fix
    at the source, and multiplying a gain by thirty is exactly what sent two of
    three members to infinity on 2026-09-18. `rescale=False` reproduces the
    original raw copy for comparison.

    Returns counts of matched and defaulted keys per parameter, the factor
    statistics, and the capped pairs."""
    report = {}
    n_src = total_n_syn(src) if rescale else {}
    n_dst = total_n_syn(dst) if rescale else {}
    with torch.no_grad():
        for name in ("bias", "time_const"):
            s, d = src.node_params[name], dst.node_params[name]
            s_vals = {k: v for k, v in zip(s.keys, s.raw_values.detach().cpu())}
            med = torch.stack(list(s_vals.values())).median()
            hit = 0
            for i, k in enumerate(d.keys):
                if k in s_vals:
                    d.raw_values[i] = s_vals[k]
                    hit += 1
                else:
                    d.raw_values[i] = med
            report[name] = {"matched": hit, "defaulted": len(d.keys) - hit}
        s, d = src.edge_params["syn_strength"], dst.edge_params["syn_strength"]
        s_vals = {k: v for k, v in zip(s.keys, s.raw_values.detach().cpu())}
        med = torch.stack(list(s_vals.values())).median()
        hit, scaled, factors, capped = 0, 0, [], []
        for i, k in enumerate(d.keys):
            if k in s_vals:
                v = s_vals[k]
                if rescale:
                    key = tuple(k) if isinstance(k, (list, tuple)) else k
                    a, b = n_src.get(key), n_dst.get(key)
                    if a and b:
                        f = a / b
                        if rescale_cap is not None and (f > rescale_cap or f < 1.0 / rescale_cap):
                            capped.append({"pair": list(key), "factor": round(float(f), 3),
                                           "total_src": round(float(a), 2),
                                           "total_dst": round(float(b), 2)})
                            f = float(np.clip(f, 1.0 / rescale_cap, rescale_cap))
                        v = v * f
                        scaled += 1
                        factors.append(f)
                d.raw_values[i] = v
                hit += 1
            else:
                d.raw_values[i] = med
        report["syn_strength"] = {"matched": hit, "defaulted": len(d.keys) - hit,
                                  "rescaled": scaled, "cap": rescale_cap,
                                  "factor_median": float(np.median(factors)) if factors else None,
                                  "factor_max_applied": float(np.max(factors)) if factors else None,
                                  "capped": len(capped), "capped_pairs": capped}
    return report


# ----------------------------------------------------------------------------- protocols


def responses(net: Network, dataset, t_pre: float, t_fade_in: float, batch_size: int = 4) -> xr.Dataset:
    """The xarray shape of flyvis.analysis.stimulus_responses, for a bare Network."""
    idx = net.connectome.central_cells_index[:]
    stim, resp = [], []
    for s, r in net.stimulus_response(dataset, dataset.dt, t_pre=t_pre, t_fade_in=t_fade_in, batch_size=batch_size):
        stim.append(s)
        resp.append(np.take(r, idx, axis=-1))
    stim = np.concatenate(stim, 0)
    resp = np.concatenate(resp, 0)[None]
    ds = xr.Dataset(
        {"stimulus": (["sample", "frame", "channel", "hex_pixel"], stim),
         "responses": (["network_id", "sample", "frame", "neuron"], resp)},
        coords={"sample": np.arange(len(dataset)),
                **{c: ("sample", dataset.arg_df[c].values) for c in dataset.arg_df.columns}})
    types = net.connectome.nodes.type[:].astype(str)[idx]
    cfg = dataset.config.to_dict()
    cfg.pop("type", None)
    n_frames = ds.frame.size
    ds.coords.update({
        "frame": np.arange(n_frames), "neuron": np.arange(len(idx)),
        "time": ("frame", np.arange(n_frames) * dataset.dt - t_pre),
        "cell_type": ("neuron", types), "network_id": [0],
    })
    ds.attrs["config"] = cfg
    return ds


def flash_protocol(net: Network, dt: float) -> xr.Dataset:
    ds = Flashes(dynamic_range=[0, 1], t_stim=1, t_pre=1.0, dt=dt, radius=(-1, 6), alternations=(0, 1, 0))
    return responses(net, ds, t_pre=1.0, t_fade_in=0.0)


def edge_protocol(net: Network, dt: float, speeds) -> xr.Dataset:
    ds = MovingEdge(offsets=(-10, 11), intensities=[0, 1], speeds=list(speeds), height=80,
                    post_pad_mode="continue", dt=dt, device=flyvis.device, t_pre=1.0, t_post=1.0)
    return responses(net, ds, t_pre=1.0, t_fade_in=0.0)


# ----------------------------------------------------------------------------- metrics


def diverged(ds: xr.Dataset) -> bool:
    """NaN padding is normal for samples of unequal length; divergence is inf, a huge
    value, or NaN already in the first frame (which is never padding)."""
    r = ds["responses"].values
    return bool(np.isinf(r).any() or np.nanmax(np.abs(r)) > 1e6)


def fri_table(ds: xr.Dataset) -> pd.Series:
    if diverged(ds):
        # a diverged network (NaN/inf activity) is a result, not a crash: every FRI is NaN
        return pd.Series(np.nan, index=ds.cell_type.values)
    fri = flash_response_index(ds, radius=6)
    return pd.Series(fri.values.squeeze(), index=fri.cell_type.values)


def dsi_table(ds: xr.Dataset) -> pd.DataFrame:
    if diverged(ds):
        out = pd.DataFrame({"cell_type": ds.cell_type.values})
        for inten in (0, 1):
            out[f"dsi_i{inten}"] = np.nan
            out[f"pd_i{inten}"] = np.nan
        return out.set_index("cell_type")
    dsi = direction_selectivity_index(ds)
    pdir = preferred_direction(ds)
    out = pd.DataFrame({"cell_type": ds.cell_type.values})
    # dsi/pd have dims (network_id, intensity, neuron) after squeeze -> pick per intensity
    for inten in (0, 1):
        out[f"dsi_i{inten}"] = np.asarray(dsi.sel(intensity=inten)).squeeze()
        out[f"pd_i{inten}"] = np.degrees(np.asarray(pdir.sel(intensity=inten)).squeeze()) % 360
    return out.set_index("cell_type")


def score_fri(fri: pd.Series) -> dict:
    known = gt.known_preferred_contrasts
    common = [t for t in known if t in fri.index]
    if not common or fri.isna().all():
        return {"n_known": len(common), "polarity_agreement": float("nan"), "fri_corr": float("nan"), "diverged": True}
    sign_ok = np.mean([np.sign(fri[t]) == known[t] for t in common])
    corr = np.corrcoef([fri[t] for t in common], [known[t] for t in common])[0, 1] if len(common) > 2 else np.nan
    return {"n_known": len(common), "polarity_agreement": float(sign_ok), "fri_corr": float(corr)}


def score_dsi(dsi: pd.DataFrame) -> dict:
    out = {}
    for t, pd_known in gt.preferred_directions.items():
        if t not in dsi.index:
            continue
        inten = 1 if t.startswith("T4") else 0
        d = abs((dsi.loc[t, f"pd_i{inten}"] - pd_known + 180) % 360 - 180)
        out[t] = {"dsi": float(dsi.loc[t, f"dsi_i{inten}"]), "pd": float(dsi.loc[t, f"pd_i{inten}"]),
                  "pd_known": pd_known, "pd_error_deg": float(d)}
    return out


# ----------------------------------------------------------------------------- main


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", type=int, default=[0])
    p.add_argument("--protocols", nargs="+", default=["flash", "edge"])
    p.add_argument("--dt", type=float, default=1 / 100)
    p.add_argument("--speeds", nargs="+", type=float, default=[4.8, 13, 25])
    p.add_argument("--run", default=None)
    p.add_argument("--control", action="store_true", help="also run MaleCNS with shuffled synapse strengths")
    p.add_argument("--no-rescale", action="store_true",
                   help="transplant syn_strength raw, as before 2026-09-18; reproduces the scaling bug")
    p.add_argument("--rescale-cap", type=float, default=None,
                   help="bound on the transplant's rescale factor; default config.toml [model] rescale_cap")
    a = p.parse_args(argv)
    s = settings()
    side = s["data"]["side"]
    extent = int(s["data"]["extent"])
    cap = a.rescale_cap if a.rescale_cap is not None else float(s.get("model", {}).get("rescale_cap", 3.0))
    run = a.run or f"{time.strftime('%Y-%m-%d')}_step2_zero_{side}"
    out_dir = DATA / "runs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    from flydream.data.export import filters_path

    connectome_file = filters_path(s)          # filters_<side>_<tag>.json from config.toml [data]
    print(f"connectome export: {connectome_file.name}")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    summary = []
    for m in a.models:
        t0 = time.time()
        nv = NetworkView(flyvis.results_dir / f"flow/0000/{m:03d}")
        fv = nv.init_network()
        mc = build_network(connectome_file, extent)
        rep = transplant(fv, mc, rescale=not a.no_rescale, rescale_cap=cap)
        brief = {k: ({kk: vv for kk, vv in v.items() if kk != "capped_pairs"} if isinstance(v, dict) else v)
                 for k, v in rep.items()}
        print(f"model {m:03d}: FlyVis {fv.n_nodes:,} nodes/{fv.n_edges:,} edges; MaleCNS {mc.n_nodes:,}/{mc.n_edges:,}; transplant {brief}; build {time.time()-t0:.0f}s")
        capped = rep["syn_strength"].get("capped_pairs") or []
        if capped:
            shown = ", ".join(f"{c['pair'][0]}->{c['pair'][1]} x{c['factor']}" for c in capped[:12])
            print(f"  {len(capped)} pairs capped at x{cap} (export defects, not corrected by gain): {shown}"
                  + (" ..." if len(capped) > 12 else ""))
        row = {"model": m, "transplant": rep}
        nets = [("flyvis", fv), ("malecns", mc)]
        if a.control:
            ctl = build_network(connectome_file, extent)
            transplant(fv, ctl, rescale=not a.no_rescale, rescale_cap=cap)
            shuffle_strengths(ctl, seed=m)
            nets.append(("malecns_shuffled", ctl))
        for label, net in nets:
            if "flash" in a.protocols:
                t1 = time.time()
                ds = flash_protocol(net, a.dt)
                import pickle
                pickle.dump(ds, open(out_dir / f"flash_{label}_{m:03d}.pkl", "wb"), protocol=4)
                fri = fri_table(ds)
                fri.to_csv(out_dir / f"fri_{label}_{m:03d}.csv")
                row[f"{label}_fri"] = score_fri(fri)
                print(f"  {label} flash: {row[f'{label}_fri']}  {time.time()-t1:.0f}s")
            if "edge" in a.protocols:
                t1 = time.time()
                ds = edge_protocol(net, a.dt, a.speeds)
                import pickle
                pickle.dump(ds, open(out_dir / f"edge_{label}_{m:03d}.pkl", "wb"), protocol=4)
                dsi = dsi_table(ds)
                dsi.to_csv(out_dir / f"dsi_{label}_{m:03d}.csv")
                row[f"{label}_dsi"] = score_dsi(dsi)
                mean_err = np.mean([v["pd_error_deg"] for v in row[f"{label}_dsi"].values()])
                mean_dsi = np.mean([v["dsi"] for v in row[f"{label}_dsi"].values()])
                print(f"  {label} edge: mean T4/T5 DSI {mean_dsi:.3f}, mean PD error {mean_err:.1f} deg  {time.time()-t1:.0f}s")
        summary.append(row)
        json.dump(summary, open(out_dir / "summary.json", "w"), indent=2, default=float)
    print("written", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
