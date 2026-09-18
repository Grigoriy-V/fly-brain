"""The optic-lobe subgraph of MaleCNS v1.0 with a column for every neuron.

Produces data/ol/neurons.parquet (bodyId, type, side, superclass, hex1, hex2,
hex_source) and data/ol/edges.parquet (body_pre, body_post, weight) for the
neurons of one side, and prints the hold-out score of the column inference.

Column for a neuron, in order of preference:
  roi       the column ROI holding most of its synapses (data/ol/roi_columns.parquet,
            from `columns_roi.py`); anatomical, independent of partners
  tagged    MaleCNS `assignedOlHex1/2` (15 types), used to validate `roi`
  inferred  synapse-weighted median of the hex of roi/tagged partners (fallback)
  none      nothing above applies

Settings are read from config.toml [data]; defaults below match it.
"""
from __future__ import annotations

import pathlib
import sys
import tomllib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ANN = DATA / "malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = DATA / "malecns/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
OUT = DATA / "ol"
OL_SUPERCLASSES = ("ol_intrinsic", "ol_sensory", "visual_projection", "visual_centrifugal")
PHOTORECEPTOR_PREFIX = ("R1-R6", "R7", "R8")


def settings() -> dict:
    cfg = ROOT / "config.toml"
    if cfg.exists():
        return tomllib.loads(cfg.read_text()).get("data", {})
    return {}


def load_ol_neurons() -> pd.DataFrame:
    ann = pd.read_feather(ANN, columns=["bodyId", "type", "somaSide", "superclass", "statusLabel",
                                         "assignedOlHex1", "assignedOlHex2"])
    ol = ann[ann["superclass"].isin(OL_SUPERCLASSES) & ann["type"].notna()].copy()
    ol = ol[~ol["statusLabel"].isin(["Out of scope", "Orphan", "Orphan-artifact", "Unimportant"])]
    return ol.reset_index(drop=True)


def load_edges(bodies: np.ndarray, min_weight: int) -> pd.DataFrame:
    tbl = pf.read_table(WEIGHTS)
    tbl = tbl.filter(pc.greater_equal(tbl["weight"], min_weight))
    keep = pc.and_(pc.is_in(tbl["body_pre"], value_set=pa.array(bodies)),
                   pc.is_in(tbl["body_post"], value_set=pa.array(bodies)))
    return tbl.filter(keep).to_pandas()


def infer_side_for_sensory(ol: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Photoreceptors have no soma side; take the majority side of their post partners."""
    side = ol.set_index("bodyId")["somaSide"]
    missing = ol["somaSide"].isna()
    e = edges[edges["body_pre"].isin(ol.loc[missing, "bodyId"])].copy()
    e["side"] = e["body_post"].map(side)
    e = e.dropna(subset=["side"])
    vote = e.groupby(["body_pre", "side"])["weight"].sum().unstack(fill_value=0)
    inferred = vote.idxmax(axis=1)
    ol = ol.copy()
    ol.loc[missing, "somaSide"] = ol.loc[missing, "bodyId"].map(inferred)
    return ol


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


def infer_columns(neurons: pd.DataFrame, edges: pd.DataFrame, tagged_mask: pd.Series) -> pd.DataFrame:
    """Column = synapse-weighted median hex of tagged partners (both directions)."""
    hex1 = neurons.set_index("bodyId")["assignedOlHex1"]
    hex2 = neurons.set_index("bodyId")["assignedOlHex2"]
    tagged_ids = set(neurons.loc[tagged_mask, "bodyId"])
    # partner table: (body, partner, weight) in both directions
    a = edges.rename(columns={"body_pre": "body", "body_post": "partner"})
    b = edges.rename(columns={"body_post": "body", "body_pre": "partner"})
    p = pd.concat([a, b], ignore_index=True)
    p = p[p["partner"].isin(tagged_ids)]
    p["h1"] = p["partner"].map(hex1).astype(float)
    p["h2"] = p["partner"].map(hex2).astype(float)
    out = {}
    for body, g in p.groupby("body"):
        w = g["weight"].to_numpy(dtype=float)
        out[body] = (weighted_median(g["h1"].to_numpy(), w), weighted_median(g["h2"].to_numpy(), w), int(len(g)))
    res = pd.DataFrame.from_dict(out, orient="index", columns=["inf_hex1", "inf_hex2", "n_tagged_partners"])
    res.index.name = "bodyId"
    return res.reset_index()


def main() -> int:
    s = settings()
    min_weight = int(s.get("min_weight", 5))
    side = s.get("side", "R")
    ol = load_ol_neurons()
    print(f"OL neurons with a type: {len(ol):,}; types: {ol['type'].nunique()}")
    edges_all = load_edges(ol["bodyId"].to_numpy(), min_weight)
    print(f"edges among them at weight >= {min_weight}: {len(edges_all):,}")
    ol = infer_side_for_sensory(ol, edges_all)
    print("side counts:", ol["somaSide"].value_counts(dropna=False).to_dict())
    n = ol[ol["somaSide"] == side].reset_index(drop=True)
    ids = set(n["bodyId"])
    e = edges_all[edges_all["body_pre"].isin(ids) & edges_all["body_post"].isin(ids)].reset_index(drop=True)
    print(f"side {side}: {len(n):,} neurons, {len(e):,} edges")

    tagged = n["assignedOlHex1"].notna()
    print(f"tagged neurons: {tagged.sum():,} in {n.loc[tagged, 'type'].nunique()} types")

    # hold-out: infer every tagged neuron from the OTHER tagged neurons only
    inf = infer_columns(n, e, tagged)
    chk = n.loc[tagged, ["bodyId", "type", "assignedOlHex1", "assignedOlHex2"]].merge(inf, on="bodyId", how="left")
    # exclude self (a neuron is not its own partner, so no leak); partners of the same type are legitimate
    ok = chk["inf_hex1"].notna()
    d = np.abs(chk.loc[ok, "inf_hex1"] - chk.loc[ok, "assignedOlHex1"]) + np.abs(chk.loc[ok, "inf_hex2"] - chk.loc[ok, "assignedOlHex2"])
    print(f"hold-out on tagged: {ok.sum():,} inferable of {len(chk):,}; exact {np.mean(d == 0):.4f}; within 1 {np.mean(d <= 1):.4f}; within 2 {np.mean(d <= 2):.4f}")
    per_type = chk.loc[ok].assign(exact=(d == 0).to_numpy()).groupby("type")["exact"].mean().round(3)
    print("exact by type:", per_type.to_dict())

    n = n.merge(inf, on="bodyId", how="left")
    roi_path = OUT / "roi_columns.parquet"
    if roi_path.exists():
        roi = pd.read_parquet(roi_path)[["bodyId", "hex1", "hex2", "np", "syn"]].rename(
            columns={"hex1": "roi_hex1", "hex2": "roi_hex2", "np": "roi_np", "syn": "roi_syn"})
        n = n.merge(roi, on="bodyId", how="left")
        both = tagged & n["roi_hex1"].notna()
        d = (np.abs(n.loc[both, "roi_hex1"] - n.loc[both, "assignedOlHex1"])
             + np.abs(n.loc[both, "roi_hex2"] - n.loc[both, "assignedOlHex2"]))
        print(f"roi vs tag on tagged neurons: {both.sum():,}; exact {np.mean(d == 0):.4f}; within 1 {np.mean(d <= 1):.4f}")
        per = n.loc[both].assign(exact=(d == 0).to_numpy()).groupby("type")["exact"].mean().round(3)
        print("roi exact by type:", per.to_dict())
        has_roi = n["roi_hex1"].notna()
        n["hex1"] = np.where(has_roi, n["roi_hex1"], np.where(tagged, n["assignedOlHex1"], n["inf_hex1"]))
        n["hex2"] = np.where(has_roi, n["roi_hex2"], np.where(tagged, n["assignedOlHex2"], n["inf_hex2"]))
        n["hex_source"] = np.where(has_roi, "roi", np.where(tagged, "tagged", np.where(n["inf_hex1"].notna(), "inferred", "none")))
    else:
        print("no roi_columns.parquet: run flydream.data.columns_roi for anatomical columns")
        n["hex1"] = np.where(tagged, n["assignedOlHex1"], n["inf_hex1"])
        n["hex2"] = np.where(tagged, n["assignedOlHex2"], n["inf_hex2"])
        n["hex_source"] = np.where(tagged, "tagged", np.where(n["inf_hex1"].notna(), "inferred", "none"))
    print("hex_source:", n["hex_source"].value_counts().to_dict())
    OUT.mkdir(parents=True, exist_ok=True)
    n[["bodyId", "type", "somaSide", "superclass", "hex1", "hex2", "hex_source", "n_tagged_partners"]].rename(
        columns={"somaSide": "side"}).to_parquet(OUT / f"neurons_{side}.parquet", index=False)
    e.to_parquet(OUT / f"edges_{side}_w{min_weight}.parquet", index=False)
    print("written", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
