"""Fetch the MPI-Sintel dataset through flyvis's own downloader and record it.

    python -m flydream.data.fetch_sintel [--keep-zip]

flyvis trains and evaluates on Sintel (Lappalainen et al. 2024), so the
stimulus distribution and the optic-flow ground truth must be the ones it
expects: this delegates to `flyvis.datasets.sintel_utils.download_sintel`,
which downloads https://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
(5,627,783,629 bytes) into `flyvis.sintel_dir` and extracts it. Afterwards
the archive is deleted unless --keep-zip, and an entry is written to
data/manifest.json recording the url, the archive size, the date and the
directory tree that must exist.

A human gate: the download is over 1 GB (AGENTS.md). Approved 2026-09-18.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date

from flydream.data.fetch import load_manifest, save_manifest
from flydream.model import configure_flyvis_root

configure_flyvis_root()

import flyvis  # noqa: E402
from flyvis.datasets.sintel_utils import download_sintel  # noqa: E402

URL = "https://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"
ARCHIVE_BYTES = 5_627_783_629
REQUIRED = ["training", "test", "training/final", "training/flow"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--keep-zip", action="store_true", help="keep the 5.6 GB archive after extraction")
    p.add_argument("--depth", action="store_true", help="also fetch the depth maps (a second archive)")
    a = p.parse_args(argv)

    t0 = time.time()
    root = download_sintel(depth=a.depth)
    print(f"sintel at {root} after {time.time() - t0:.0f}s")

    missing = [r for r in REQUIRED if not (root / r).exists()]
    if missing:
        print(f"MISSING after extraction: {missing}")
        return 1
    n_final = len(list((root / "training/final").iterdir()))
    n_flow = len(list((root / "training/flow").iterdir()))
    extracted = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and f.suffix != ".zip")
    print(f"training/final sequences: {n_final}; training/flow: {n_flow}; extracted bytes: {extracted:,}")

    zips = list(root.glob("*.zip"))
    if not a.keep_zip:
        for z in zips:
            size = z.stat().st_size
            z.unlink()
            print(f"removed archive {z.name} ({size:,} bytes)")

    m = load_manifest()
    m["flyvis/SintelDataSet"] = {
        "url": URL,
        "archive_bytes": ARCHIVE_BYTES,
        "extracted_bytes": extracted,
        "sequences_final": n_final,
        "sequences_flow": n_flow,
        "licence": "MPI-Sintel terms, research use",
        "fetched": date.today().isoformat(),
        "seconds": round(time.time() - t0, 1),
        "note": "fetched by flyvis.datasets.sintel_utils.download_sintel; archive removed after extraction"
                if not a.keep_zip else "archive kept",
        "required_paths": REQUIRED,
    }
    save_manifest(m)
    print("manifest updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
