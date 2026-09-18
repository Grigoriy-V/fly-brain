"""Type-pair filters from the MaleCNS optic lobe in FlyVis's connectome shape.

For every (source type, target type, column offset) the synapse count per
target neuron, computed from data/ol/neurons_<side>.parquet and edges. The hex
axes of MaleCNS are aligned to FlyVis's (u, v) by `flydream.data.orientation`
(an integer unimodular map chosen by cosine similarity of the filters). Writes:

  data/ol/filters_<side>.json           FlyVis-shaped nodes/edges (offsets, sign)
  data/ol/filter_comparison_<side>.csv  per type pair: FlyVis vs MaleCNS
"""
from __future__ import annotations

import json
import pathlib
import sys
import tomllib
from collections import defaultdict

import flyvis
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BRIDGE = DATA / "bridge/types.csv"
NT = DATA / "malecns/body-neurotransmitters-male-cns-v1.0.feather"
FLYVIS_JSON = pathlib.Path(flyvis.__file__).parent / "connectome/fib25-fib19_v2.2.json"

# FlyVis node names that MaleCNS holds as one type (the filter is compared as a sum)
POOLED = {"R1-R6": ["R1", "R2", "R3", "R4", "R5", "R6"], "R7": ["R7"], "R8": ["R8"],
          "CT1": ["CT1(Lo1)", "CT1(M10)"], "TmY9": ["TmY9"], "Am": ["Am"]}
SINGLE_CELL_NODES = ["CT1"]
# FlyVis rule for the first four; the rest as Shiu et al. 2024
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
        "dopamine": 1, "octopamine": 1, "serotonin": 1}


def settings() -> dict:
    cfg = ROOT / "config.toml"
    return tomllib.loads(cfg.read_text()) if cfg.exists() else {}


def hex_symmetries():
    """The 12 symmetries of axial hex coordinates (q, r): 6 rotations x reflection."""
    def rot(q, r):
        return -r, q + r

    out = []
    for refl in (False, True):
        for k in range(6):
            def f(q, r, k=k, refl=refl):
                if refl:
                    q, r = r, q
                for _ in range(k):
                    q, r = rot(q, r)
                return q, r
            out.append((("refl" if refl else "rot") + str(k), f))
    return out


def malecns_to_node(bridge: pd.DataFrame) -> dict[str, str]:
    """MaleCNS type -> node name used in the export (FlyVis name, or the pooled name)."""
    m = {}
    for _, r in bridge.iterrows():
        if r["kind"] == "unmatched":
            continue
        for t in str(r["malecns_types"]).split(";"):
            if r["kind"] == "pooled" and t.startswith("R7"):
                m[t] = "R7"
            elif r["kind"] == "pooled" and t.startswith("R8"):
                m[t] = "R8"
            elif r["kind"] == "pooled":
                m[t] = "R1-R6"
            elif r["kind"] == "compartment":
                m[t] = "CT1"
            elif r["kind"] == "renamed" and r["flyvis_type"] == "TmY9":
                m[t] = "TmY9"
            elif r["kind"] == "renamed" and r["flyvis_type"] == "Am":
                m[t] = "Am"
            else:
                m[t] = r["flyvis_type"]
    return m


def flyvis_filters() -> tuple[dict, dict, dict]:
    d = json.load(open(FLYVIS_JSON))
    node_of = {}
    for pooled, members in POOLED.items():
        for mname in members:
            node_of[mname] = pooled
    filt = defaultdict(float)   # (src, tar, du, dv) -> n_syn (summed over pooled members)
    sign = {}
    for e in d["edges"]:
        s, t = node_of.get(e["src"], e["src"]), node_of.get(e["tar"], e["tar"])
        for (du, dv), n in e["offsets"]:
            filt[(s, t, du, dv)] += n
        sign[(s, t)] = e["alpha"]
    return d, dict(filt), sign


def malecns_filters(neurons: pd.DataFrame, edges: pd.DataFrame, node_of: dict, s_cfg: dict | None = None) -> tuple[dict, dict]:
    s_cfg = s_cfg if s_cfg is not None else settings().get("data", {})
    n = neurons[neurons["hex1"].notna()].copy()
    n["node"] = n["type"].map(node_of)
    n = n[n["node"].notna()]
    idx = n.set_index("bodyId")
    e = edges[edges["body_pre"].isin(idx.index) & edges["body_post"].isin(idx.index)].copy()
    e["src"] = e["body_pre"].map(idx["node"])
    e["tar"] = e["body_post"].map(idx["node"])
    e["du"] = (e["body_post"].map(idx["hex1"]) - e["body_pre"].map(idx["hex1"])).astype(int)
    e["dv"] = (e["body_post"].map(idx["hex2"]) - e["body_pre"].map(idx["hex2"])).astype(int)
    n_src = n.groupby("node")["bodyId"].nunique()
    # single giant cells (one per side in MaleCNS, one compartment per column in FlyVis):
    # their edges are placed at offset (0, 0) and averaged per partner cell
    single = set(SINGLE_CELL_NODES)
    is_single_src = e["src"].isin(single)
    is_single_tar = e["tar"].isin(single)
    e.loc[is_single_src | is_single_tar, ["du", "dv"]] = 0
    g = e.groupby(["src", "tar", "du", "dv"])["weight"].sum()
    # Normalisation, `config.toml [data] normalisation`:
    #   "src"  synapses per source cell (default; stable under the FlyVis parameter transplant)
    #   "tar"  synapses per target cell (the physical input per cell; drove wide-field
    #          loops such as TmY15 into runaway with transplanted strengths, 2026-09-18)
    #   "pair" total / number of (source cell, target cell) pairs that EXIST at the offset,
    #          FlyVis's convention as inferred from its 7-column sample; matched FlyVis's
    #          totals best in the median but ran into NaN on the same loops
    rule = str(s_cfg.get("normalisation", "src"))
    cells_at = {node: grp.groupby(["hex1", "hex2"]).size().to_dict() for node, grp in n.groupby("node")}
    filt = {}
    for (src, tar, du, dv), v in g.items():
        if rule == "src":
            filt[(src, tar, du, dv)] = v / n_src[src]
        elif rule == "tar":
            filt[(src, tar, du, dv)] = v / n_src[tar]
        else:
            pairs = 0
            for (h1, h2), c_src in cells_at[src].items():
                c_tar = cells_at[tar].get((h1 + du, h2 + dv))
                if c_tar:
                    pairs += c_src * c_tar
            filt[(src, tar, du, dv)] = v / max(pairs, 1)
    return filt, n_src.to_dict()


def choose_orientation(fv: dict, mc: dict) -> tuple[str, list]:
    results = []
    for name, f in hex_symmetries():
        mapped = defaultdict(float)
        for (s, t, du, dv), v in mc.items():
            q, r = f(du, dv)
            mapped[(s, t, q, r)] += v
        keys = list(set(fv) | set(mapped))
        a = np.array([fv.get(k, 0.0) for k in keys])
        b = np.array([mapped.get(k, 0.0) for k in keys])
        rho = spearmanr(a, b).correlation
        overlap = float(np.mean([k in mapped for k in fv]))
        results.append({"name": name, "spearman_all": float(rho), "offset_overlap": overlap})
    results.sort(key=lambda r: (r["spearman_all"], r["offset_overlap"]), reverse=True)
    return results[0]["name"], results


STRIDES = [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (5, 5)]


def stride_for_density(density: float) -> list[int]:
    """The stride [u, v] whose tiled density 1/(u*v) is nearest (in log) to `density`;
    densities above 0.7 tile every column."""
    if density >= 0.7:
        return [1, 1]
    import math
    best = min(STRIDES, key=lambda uv: abs(math.log(density) - math.log(1.0 / (uv[0] * uv[1]))))
    return list(best)


def type_signs(neurons: pd.DataFrame, node_of: dict) -> dict:
    nt = pd.read_feather(NT, columns=["body", "consensus_nt"])
    nt = nt[nt["body"].isin(neurons["bodyId"])].copy()
    nt["node"] = nt["body"].map(neurons.set_index("bodyId")["type"]).map(node_of)
    nt = nt.dropna(subset=["node", "consensus_nt"])
    cons = nt.groupby("node")["consensus_nt"].agg(lambda s: s.value_counts().idxmax())
    return {k: (v, SIGN.get(v)) for k, v in cons.items()}


def filters_tag(s: dict) -> str:
    """The export's identity from its settings: `w5` for the plain row cut,
    `w5wk50` when pairs losing more than 50% to it keep their weak rows."""
    d = s.get("data", {})
    mw = int(d.get("min_weight", 5))
    loss = float(d.get("weak_pair_loss", 1.0))
    return f"w{mw}" + (f"wk{int(round(loss * 100))}" if loss < 1.0 else "")


def filters_path(s: dict | None = None) -> pathlib.Path:
    """Where the export the settings describe lives. Consumers (model zero, the
    decoder map) call this rather than naming the file, so a changed setting
    reads a different file and never a silently re-exported one."""
    s = s if s is not None else settings()
    side = s.get("data", {}).get("side", "R")
    return DATA / f"ol/filters_{side}_{filters_tag(s)}.json"


def keep_weak_pairs(edges_cut: pd.DataFrame, edges_all: pd.DataFrame, neurons: pd.DataFrame,
                    node_of: dict, loss: float) -> tuple[pd.DataFrame, list]:
    """Give back every row of weight >= 1 to the type pairs that the row cut guts.

    The cut exists for scattered single contacts at 42% postsynaptic completion.
    A pair such as Tm2 -> L2 (1,421 synapses over 670 rows, 234 left at
    weight >= 5) is not that: it is a consistent weak pathway, and cutting it
    leaves the export 24x below FlyVis on a pair FlyVis carries at 7 synapses
    per cell (ISSUES ISS-0005). Pairs whose kept fraction falls below
    `1 - loss` take their rows from the uncut table instead. Returns the merged
    edge table and the list of pairs given the exception, with the numbers."""
    typ = neurons.set_index("bodyId")["type"].map(node_of)

    def pair_mass(e):
        t = pd.DataFrame({"src": e["body_pre"].map(typ), "tar": e["body_post"].map(typ), "w": e["weight"]})
        t = t.dropna(subset=["src", "tar"])
        return t.groupby(["src", "tar"])["w"].sum()

    cut, full = pair_mass(edges_cut), pair_mass(edges_all)
    frac = (cut.reindex(full.index).fillna(0.0) / full).astype(float)
    weak = frac[frac < 1.0 - loss].index
    weak_set = set(weak.tolist())
    src_t, tar_t = edges_all["body_pre"].map(typ), edges_all["body_post"].map(typ)
    in_weak = pd.Series(list(zip(src_t, tar_t)), index=edges_all.index).isin(weak_set)
    src_c, tar_c = edges_cut["body_pre"].map(typ), edges_cut["body_post"].map(typ)
    cut_not_weak = ~pd.Series(list(zip(src_c, tar_c)), index=edges_cut.index).isin(weak_set)
    merged = pd.concat([edges_cut[cut_not_weak], edges_all[in_weak]], ignore_index=True)
    listing = [{"src": a, "tar": b, "kept_fraction_at_cut": round(float(frac[(a, b)]), 3),
                "synapses_cut": float(cut.get((a, b), 0.0)), "synapses_all": float(full[(a, b)])}
               for (a, b) in sorted(weak_set)]
    return merged, listing


def main() -> int:
    s = settings()
    side = s.get("data", {}).get("side", "R")
    mw = int(s.get("data", {}).get("min_weight", 5))
    loss = float(s.get("data", {}).get("weak_pair_loss", 1.0))
    neurons = pd.read_parquet(DATA / f"ol/neurons_{side}.parquet")
    edges = pd.read_parquet(DATA / f"ol/edges_{side}_w{mw}.parquet")
    bridge = pd.read_csv(BRIDGE)
    node_of = malecns_to_node(bridge)
    weak_listing = []
    if loss < 1.0:
        all_path = DATA / f"ol/edges_{side}_w1.parquet"
        if not all_path.exists():
            print(f"weak_pair_loss={loss} needs {all_path.name}: run flydream.data.optic_lobe --min-weight 1 --edges-only")
            return 1
        edges_all = pd.read_parquet(all_path)
        n_before = len(edges)
        edges, weak_listing = keep_weak_pairs(edges, edges_all, neurons, node_of, loss)
        print(f"weak-pair exception (loss > {loss}): {len(weak_listing)} type pairs keep rows >= 1; "
              f"edges {n_before:,} -> {len(edges):,}")
        for w in weak_listing[:15]:
            print(f"  {w['src']:>8} -> {w['tar']:<8} kept {w['kept_fraction_at_cut']:.2f} of {w['synapses_all']:.0f} synapses")
        if len(weak_listing) > 15:
            print(f"  ... {len(weak_listing) - 15} more")
    fv_json, fv, fv_sign = flyvis_filters()
    mc, n_src = malecns_filters(neurons, edges, node_of)
    print(f"MaleCNS filters: {len(mc):,} (src,tar,offset) entries over {len(n_src)} nodes; FlyVis: {len(fv):,}")
    from flydream.data.orientation import apply, choose
    best, table = choose(fv, mc)
    print("orientation: best M", best, "cosine", table[0]["cosine"], "| worst", table[-1]["cosine"])
    mapped = apply(best, mc)

    pairs_fv = defaultdict(float)
    pairs_mc = defaultdict(float)
    for (a, b, *_), v in fv.items():
        pairs_fv[(a, b)] += v
    for (a, b, *_), v in mapped.items():
        pairs_mc[(a, b)] += v
    signs = type_signs(neurons, node_of)
    rows = []
    for k in sorted(set(pairs_fv) | set(pairs_mc)):
        nt, sg = signs.get(k[0], (None, None))
        rows.append({"src": k[0], "tar": k[1], "flyvis_total": pairs_fv.get(k, 0.0),
                     "malecns_total": pairs_mc.get(k, 0.0), "flyvis_sign": fv_sign.get(k),
                     "malecns_nt": nt, "malecns_sign": sg})
    cmp = pd.DataFrame(rows)
    both = cmp[(cmp.flyvis_total > 0) & (cmp.malecns_total > 0)]
    fv_only = cmp[(cmp.flyvis_total > 0) & (cmp.malecns_total == 0)]
    rho = spearmanr(both.flyvis_total, both.malecns_total).correlation
    sign_ok = both.dropna(subset=["flyvis_sign", "malecns_sign"])
    agree = float(np.mean(sign_ok.flyvis_sign == sign_ok.malecns_sign)) if len(sign_ok) else float("nan")
    n_fv = int((cmp.flyvis_total > 0).sum())
    print(f"type pairs: FlyVis {n_fv}, MaleCNS {int((cmp.malecns_total > 0).sum())}, both {len(both)}, "
          f"FlyVis-only {len(fv_only)}; recovered {len(both) / max(1, n_fv):.3f}; "
          f"Spearman(total) {rho:.3f}; sign agreement {agree:.3f} on {len(sign_ok)} pairs")
    cmp.to_csv(DATA / f"ol/filter_comparison_{side}.csv", index=False)

    fv_nodes = {n["name"]: n for n in fv_json["nodes"]}
    # Lattice stride per type from its real density (cells per column). The tiled model
    # places one cell per lattice point; a type with 0.33 cells per column tiled at every
    # column has three times its real number of neighbours, and with transplanted
    # per-synapse strengths its recurrent loops (TmY4 -> TmY4) run away (2026-09-18).
    # FlyVis does the same by hand for Lawf1/Lawf2 (stride [3, 2]).
    n_columns = max(n_src[t] for t in n_src if t not in SINGLE_CELL_NODES)   # ~ one per column
    strides = {}
    for node in n_src:
        density = n_src[node] / n_columns if node not in SINGLE_CELL_NODES else 1.0
        # input units share one stimulus lattice, and photoreceptor counts are unreliable
        # (lamina incomplete, ISS-0002 report): every input tiles every column
        strides[node] = [1, 1] if node in ("R1-R6", "R7", "R8") else stride_for_density(density)
    print("strides by density:", {k: v for k, v in strides.items() if v != [1, 1]})
    nodes = []
    for node in sorted(n_src):
        tmpl = fv_nodes.get(POOLED.get(node, [node])[0], {})
        nodes.append({"name": node, "pattern": ["stride", strides[node]], "activation": "relu",
                      "density_malecns": round(n_src[node] / n_columns, 3),
                      "bias": tmpl.get("bias", 0.5), "bias_fixed": False,
                      "time_constant": tmpl.get("time_constant"), "time_constant_fixed": False,
                      "n_cells_malecns": int(n_src[node])})
    # R1-R6 is one MaleCNS type; FlyVis and its stimulus expect six nodes R1..R6 with the
    # same per-cell filter, so the pooled node is expanded to six copies.
    expand = {"R1-R6": ["R1", "R2", "R3", "R4", "R5", "R6"]}
    nodes = [nd for nd in nodes if nd["name"] not in expand]
    for pooled, members in expand.items():
        if pooled in n_src:
            tmpl = fv_nodes.get(members[0], {})
            for mname in members:
                nodes.append({"name": mname, "pattern": ["stride", [1, 1]], "activation": "relu", "density_malecns": 1.0,
                              "bias": tmpl.get("bias", 0.5), "bias_fixed": False,
                              "time_constant": tmpl.get("time_constant"), "time_constant_fixed": False,
                              "n_cells_malecns": int(n_src[pooled]), "expanded_from": pooled})
    # offsets below `min_filter_syn` mean synapses per cell are dropped: FlyVis fills the
    # convex hull of the reported offsets, and long tails of tiny weights would inflate it
    min_syn = float(s.get("data", {}).get("min_filter_syn", 1.0))
    total_mass = sum(mapped.values())
    kept_mass = sum(v for v in mapped.values() if v >= min_syn)
    print(f"offset pruning at >= {min_syn} syn/cell: {sum(v >= min_syn for v in mapped.values()):,} of {len(mapped):,} offsets, {kept_mass / total_mass:.3f} of synapse mass kept")
    by_pair = defaultdict(list)
    for (a, b, q, r), v in mapped.items():
        if v < min_syn:
            continue
        srcs = expand.get(a, [a])
        tars = expand.get(b, [b])
        for a2 in srcs:
            for b2 in tars:
                by_pair[(a2, b2)].append([[int(q), int(r)], round(float(v), 4)])
    fv_sign_expanded = dict(fv_sign)
    edges_out = []
    n_sign_flyvis = n_sign_nt = 0
    for (a, b), offs in sorted(by_pair.items()):
        base_a = next((p for p, ms in expand.items() if a in ms), a)
        nt, sg_nt = signs.get(base_a, (None, None))
        if (a, b) in fv_sign_expanded:
            sg, ref = fv_sign_expanded[(a, b)], "FlyVis fib25-fib19_v2.2 alpha"
            n_sign_flyvis += 1
        else:
            sg, ref = sg_nt, f"MaleCNS consensus_nt={nt}"
            n_sign_nt += 1
        edges_out.append({"src": a, "tar": b, "offsets": sorted(offs), "alpha": sg, "alpha_fixed": True,
                          "alpha_references": [ref], "malecns_nt": nt, "time_constant": None,
                          "time_constant_fixed": False, "lambda_mult": 1.0, "edge_type": "chem"})
    edges_out = [e for e in edges_out if e["alpha"] is not None]
    print(f"signs: {n_sign_flyvis} pairs from FlyVis, {n_sign_nt} from MaleCNS NT")
    node_names = {nd["name"] for nd in nodes}
    out = {"source": f"MaleCNS v1.0 side {side}, weight>={mw}"
                     + (f", weak pairs (loss>{loss}) at weight>=1" if loss < 1.0 else "")
                     + f", columns roi/tagged/inferred, orientation M={list(best)}",
           "nodes": nodes, "edges": edges_out, "receptors": [],
           "weak_pairs": weak_listing,
           "input_units": [r for r in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"] if r in node_names],
           "output_units": [n for n in fv_json["output_units"] if n in node_names]}
    target = filters_path(s)
    json.dump(out, open(target, "w"))
    print(f"written {target.name}: {len(nodes)} nodes, {len(edges_out)} type-pair edges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
