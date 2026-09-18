"""Every external file the project fetches, with its source, size and licence.

A row here is the only place a URL is written. `flydream.data.fetch` downloads
what is listed and records it in `data/manifest.json`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str            # file name on disk
    url: str
    dest: str            # directory under data/
    licence: str
    expected_bytes: int | None = None   # from the publisher's page, informative only
    note: str = ""


_MCNS = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"

MALECNS_CORE = [
    Source(
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        _MCNS + "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "malecns", "CC-BY 4.0", 13_000_000,
        "cell types, sides, classes; no neurotransmitter",
    ),
    Source(
        "body-neurotransmitters-male-cns-v1.0.feather",
        _MCNS + "body-neurotransmitters-male-cns-v1.0.feather",
        "malecns", "CC-BY 4.0", 42_000_000,
        "per-neuron aggregated neurotransmitter predictions",
    ),
    Source(
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        _MCNS + "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "malecns", "CC-BY 4.0", 1_100_000_000,
        "segment-to-segment synapse-count weights, confidence >= 0.5",
    ),
]

# Not fetched by default: 780 MB stats, 12.7 GB synapse points, 6.8 GB partners,
# 2.7 GB per-T-bar neurotransmitters. Listed so the gate names the number.
MALECNS_LARGE = [
    Source("body-stats-male-cns-v1.0-minconf-0.5.feather",
           _MCNS + "body-stats-male-cns-v1.0-minconf-0.5.feather",
           "malecns", "CC-BY 4.0", 780_000_000, "synapse counts per segment"),
    Source("syn-points-male-cns-v1.0-minconf-0.5.feather",
           _MCNS + "syn-points-male-cns-v1.0-minconf-0.5.feather",
           "malecns", "CC-BY 4.0", 12_700_000_000, "pre/post locations, 8 nm voxels"),
    Source("syn-partners-male-cns-v1.0-minconf-0.5.feather",
           _MCNS + "syn-partners-male-cns-v1.0-minconf-0.5.feather",
           "malecns", "CC-BY 4.0", 6_800_000_000, "pre-post pairs with confidence"),
    Source("tbar-neurotransmitters-male-cns-v1.0.feather",
           _MCNS + "tbar-neurotransmitters-male-cns-v1.0.feather",
           "malecns", "CC-BY 4.0", 2_700_000_000, "per-presynapse NT probabilities"),
]

SOURCES = {s.name: s for s in MALECNS_CORE + MALECNS_LARGE}
