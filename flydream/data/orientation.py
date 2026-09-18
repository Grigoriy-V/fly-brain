"""Align MaleCNS hex axes (hex1, hex2) to FlyVis's (u, v).

The map is an integer 2x2 matrix M with det +-1 and entries in [-2, 2] (104
candidates: the 12 lattice symmetries and every small basis change). It is
chosen by the weighted cosine similarity between FlyVis's (src, tar, offset)
filters and MaleCNS's mapped ones, over the type pairs both have. The
direction-selective inputs (Mi1/Mi9 -> T4a-d, Tm9 -> T5a-d) are reported as
centroids so a reader can see the alignment, not only a score.

    python -m flydream.data.orientation      writes data/ol/orientation_<side>.json
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from flydream.data.export import DATA, BRIDGE, flyvis_filters, malecns_filters, malecns_to_node, settings

DS_INPUTS = [("Mi1", "T4a"), ("Mi1", "T4b"), ("Mi1", "T4c"), ("Mi1", "T4d"),
             ("Mi9", "T4a"), ("Mi9", "T4b"), ("Mi9", "T4c"), ("Mi9", "T4d"),
             ("Tm9", "T5a"), ("Tm9", "T5b"), ("Tm9", "T5c"), ("Tm9", "T5d")]


def candidates() -> list[tuple[int, int, int, int]]:
    return [(a, b, c, d) for a, b, c, d in itertools.product(range(-2, 3), repeat=4) if abs(a * d - b * c) == 1]


def apply(M, filt: dict, pairs: set | None = None) -> dict:
    a, b, c, d = M
    out = defaultdict(float)
    for (s, t, du, dv), v in filt.items():
        if pairs is None or (s, t) in pairs:
            out[(s, t, a * du + b * dv, c * du + d * dv)] += v
    return out


def cosine(fv: dict, mapped: dict) -> float:
    keys = list(set(fv) | set(mapped))
    x = np.array([fv.get(k, 0.0) for k in keys])
    y = np.array([mapped.get(k, 0.0) for k in keys])
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def centroid(filt: dict, s: str, t: str):
    w = [(du, dv, v) for (a, b, du, dv), v in filt.items() if a == s and b == t]
    tot = sum(v for *_, v in w)
    if not tot:
        return None
    return (round(sum(du * v for du, _, v in w) / tot, 2), round(sum(dv * v for _, dv, v in w) / tot, 2), round(tot, 1))


def choose(fv: dict, mc: dict) -> tuple[tuple[int, int, int, int], list[dict]]:
    pairs = set((s, t) for (s, t, _, _) in fv)
    scored = sorted(((cosine(fv, apply(M, mc, pairs)), M) for M in candidates()), reverse=True)
    table = [{"M": list(M), "cosine": round(c, 4)} for c, M in scored]
    return scored[0][1], table


def main() -> int:
    s = settings()
    side = s.get("data", {}).get("side", "R")
    mw = int(s.get("data", {}).get("min_weight", 5))
    neurons = pd.read_parquet(DATA / f"ol/neurons_{side}.parquet")
    edges = pd.read_parquet(DATA / f"ol/edges_{side}_w{mw}.parquet")
    node_of = malecns_to_node(pd.read_csv(BRIDGE))
    _, fv, _ = flyvis_filters()
    mc, _ = malecns_filters(neurons, edges, node_of)
    best, table = choose(fv, mc)
    print("best M:", best, "cosine", table[0]["cosine"], "| worst", table[-1])
    print("top 6:", table[:6])
    mapped = apply(best, mc)
    rows = []
    for a, b in DS_INPUTS:
        rows.append({"pair": f"{a}->{b}", "flyvis": centroid(fv, a, b), "malecns_raw": centroid(mc, a, b), "malecns_mapped": centroid(mapped, a, b)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    json.dump({"chosen_M": list(best), "cosine": table[0]["cosine"], "candidates": table,
               "ds_centroids": rows, "hex_source_counts": neurons["hex_source"].value_counts().to_dict()},
              open(DATA / f"ol/orientation_{side}.json", "w"), indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
