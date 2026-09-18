"""The cell-type bridge FlyVis (FIB-25/FIB-19 names) -> MaleCNS v1.0 (Nern et al. names).

Writes data/bridge/types.csv with one row per FlyVis node: the MaleCNS types it
maps to, how the match was made, the cell counts per side, and a note. Rows
with kind "unmatched" are the residue and are listed in the step's report.

Sources of a match, in order of trust:
  exact         the same name in MaleCNS `type`
  pooled        several MaleCNS types that FlyVis treats as one (R1-R6 pooled by
                MaleCNS itself; R7/R8 subtypes by spectral class)
  renamed       a rename recorded in flyconnectome/ol_annotations olmatching.tsv
                (Schlegel_type column) or in MaleCNS `flywireType`
  compartment   one MaleCNS neuron split by FlyVis into compartments (CT1)
  unmatched     no MaleCNS type found in any table
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, asdict

import flyvis
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ANN = DATA / "malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather"
OLMATCH = DATA / "bridge_olmatching_raw.tsv"
OUT = DATA / "bridge/types.csv"
OL_SUPERCLASSES = ("ol_intrinsic", "ol_sensory", "visual_projection", "visual_centrifugal")

# Hand-recorded matches, each with its evidence (checked 2026-09-18).
MANUAL: dict[str, tuple[list[str], str, str]] = {
    "R1": (["R1-R6"], "pooled", "MaleCNS types outer photoreceptors as one class R1-R6; FlyVis gives R1-R6 identical input"),
    "R2": (["R1-R6"], "pooled", "as R1"),
    "R3": (["R1-R6"], "pooled", "as R1"),
    "R4": (["R1-R6"], "pooled", "as R1"),
    "R5": (["R1-R6"], "pooled", "as R1"),
    "R6": (["R1-R6"], "pooled", "as R1"),
    "R7": (["R7y", "R7p", "R7d", "R7_unclear"], "pooled", "MaleCNS flywireType=R7 for all four; spectral subtypes y/p/d"),
    "R8": (["R8y", "R8p", "R8d", "R8_unclear"], "pooled", "MaleCNS flywireType=R8 for all four"),
    "Am": (["Lai"], "renamed", "olmatching.tsv: OL_type Lai = Schlegel_type Am; MaleCNS flywireType=Lai; 'many OL Lai are fragmented' (table note)"),
    "CT1(Lo1)": (["CT1"], "compartment", "one CT1 neuron per side; FlyVis splits M10 and Lo1 compartments"),
    "CT1(M10)": (["CT1"], "compartment", "as CT1(Lo1)"),
    "TmY9": (["TmY9a", "TmY9b"], "renamed", "olmatching.tsv: TmY9a/TmY9b both Schlegel_type TmY9"),
    "Mi3": ([], "unmatched", "FIB-25 (Takemura 2013) name; absent from MaleCNS, olmatching.tsv, flywireType, hemibrainType"),
    "Mi11": ([], "unmatched", "as Mi3"),
    "Mi12": ([], "unmatched", "as Mi3"),
    "Tm28": ([], "unmatched", "as Mi3"),
}


@dataclass
class Row:
    flyvis_type: str
    malecns_types: str      # ';'-joined
    kind: str
    n_left: int
    n_right: int
    note: str


def flyvis_types() -> list[str]:
    path = pathlib.Path(flyvis.__file__).parent / "connectome/fib25-fib19_v2.2.json"
    return [n["name"] for n in json.load(open(path))["nodes"]]


def build() -> pd.DataFrame:
    ann = pd.read_feather(ANN)
    ol = ann[ann["superclass"].isin(OL_SUPERCLASSES)]
    counts = ol.groupby(["type", "somaSide"]).size().unstack(fill_value=0)
    rows = []
    for t in flyvis_types():
        if t in MANUAL:
            targets, kind, note = MANUAL[t]
        elif t in counts.index:
            targets, kind, note = [t], "exact", ""
        else:
            targets, kind, note = [], "unmatched", "no exact name and no manual rule"
        n_l = int(sum(counts.loc[x, "L"] for x in targets if x in counts.index))
        n_r = int(sum(counts.loc[x, "R"] for x in targets if x in counts.index))
        rows.append(Row(t, ";".join(targets), kind, n_l, n_r, note))
    df = pd.DataFrame([asdict(r) for r in rows])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build()
    pd.set_option("display.width", 200, "display.max_rows", 100)
    print(df.to_string(index=False))
    print("\nkinds:", df["kind"].value_counts().to_dict())
    print("written", OUT)
