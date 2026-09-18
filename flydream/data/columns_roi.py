"""Home column of every optic-lobe neuron from neuPrint's column ROIs.

MaleCNS v1.0 carries, per neuron, synapse counts inside column ROIs named
`<NP>_<side>_col_<hex1>_<hex2>` for NP in ME, LO, LOP. The home column of a
neuron is the column ROI holding the most synapses (pre + post), per neuropil
and overall. This is independent of the neuron's partners, unlike the
partner-median inference in `optic_lobe.py`.

Writes data/ol/roi_columns.parquet: bodyId, np (neuropil of the overall
maximum), side, hex1, hex2, syn, and per-neuropil maxima me_hex1, me_hex2,
me_syn, lo_*, lop_*.  Needs NEUPRINT_TOKEN in .env.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ANN = DATA / "malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather"
OUT = DATA / "ol/roi_columns.parquet"
OL_SUPERCLASSES = ("ol_intrinsic", "ol_sensory", "visual_projection", "visual_centrifugal")
COL = re.compile(r"^(ME|LO|LOP)_([LR])_col_(\d+)_(\d+)$")
BATCH = 1000


def client() -> Client:
    load_dotenv(ROOT / ".env")
    return Client("neuprint.janelia.org", dataset="male-cns:v1.0", token=os.environ["NEUPRINT_TOKEN"])


def parse(body: int, info: dict) -> dict | None:
    best = {}
    for name, cnt in info.items():
        m = COL.match(name)
        if not m:
            continue
        np_, side, h1, h2 = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        syn = int(cnt.get("pre", 0)) + int(cnt.get("post", 0))
        key = (np_, side)
        if key not in best or syn > best[key][2]:
            best[key] = (h1, h2, syn)
    if not best:
        return None
    (np_, side), (h1, h2, syn) = max(best.items(), key=lambda kv: kv[1][2])
    row = {"bodyId": body, "np": np_, "side": side, "hex1": h1, "hex2": h2, "syn": syn}
    for k in ("ME", "LO", "LOP"):
        hit = [v for (n, s), v in best.items() if n == k and s == side]
        v = hit[0] if hit else (None, None, 0)
        row[f"{k.lower()}_hex1"], row[f"{k.lower()}_hex2"], row[f"{k.lower()}_syn"] = v
    return row


def main() -> int:
    ann = pd.read_feather(ANN, columns=["bodyId", "superclass", "type", "statusLabel"])
    ol = ann[ann["superclass"].isin(OL_SUPERCLASSES) & ann["type"].notna()]
    ol = ol[~ol["statusLabel"].isin(["Out of scope", "Orphan", "Orphan-artifact", "Unimportant"])]
    ids = ol["bodyId"].tolist()
    print(f"neurons to fetch: {len(ids):,}")
    c = client()
    rows, t0 = [], time.time()
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        q = f"MATCH (n:Neuron) WHERE n.bodyId IN {chunk} RETURN n.bodyId AS bodyId, n.roiInfo AS roiInfo"
        df = c.fetch_custom(q)
        for body, info in zip(df["bodyId"], df["roiInfo"]):
            info = json.loads(info) if isinstance(info, str) else (info or {})
            r = parse(int(body), info)
            if r:
                rows.append(r)
        if (i // BATCH) % 10 == 0:
            print(f"  {i + len(chunk):,}/{len(ids):,}  {time.time() - t0:.0f}s  with column: {len(rows):,}", flush=True)
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"written {OUT}: {len(out):,} neurons with a column ROI; by neuropil {out['np'].value_counts().to_dict()}; {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
