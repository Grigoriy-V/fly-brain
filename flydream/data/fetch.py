"""Download listed sources into data/ and record them in data/manifest.json.

    python -m flydream.data.fetch <name> [<name> ...]   fetch by file name
    python -m flydream.data.fetch --core                the three MaleCNS core files

A file already present with a recorded hash is not fetched again.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import requests

from flydream.data.sources import MALECNS_CORE, SOURCES, Source

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MANIFEST = DATA / "manifest.json"
CHUNK = 1 << 20


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def save_manifest(m: dict) -> None:
    DATA.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(src: Source, manifest: dict) -> Path:
    dest = DATA / src.dest / src.name
    key = f"{src.dest}/{src.name}"
    if dest.exists() and key in manifest and manifest[key].get("sha256"):
        print(f"present  {key}  {dest.stat().st_size:,} bytes")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    with requests.get(src.url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with tmp.open("wb") as f:
            for block in r.iter_content(CHUNK):
                f.write(block)
                done += len(block)
                if total and done % (64 * CHUNK) < CHUNK:
                    print(f"  {key}: {done/1e6:,.0f}/{total/1e6:,.0f} MB", flush=True)
    tmp.replace(dest)
    digest = sha256(dest)
    manifest[key] = {
        "url": src.url,
        "bytes": dest.stat().st_size,
        "sha256": digest,
        "licence": src.licence,
        "fetched": date.today().isoformat(),
        "seconds": round(time.time() - t0, 1),
        "note": src.note,
    }
    save_manifest(manifest)
    print(f"fetched  {key}  {dest.stat().st_size:,} bytes  {digest[:12]}  {time.time()-t0:.0f}s")
    return dest


def main(argv: list[str]) -> int:
    names = MALECNS_CORE if "--core" in argv else [SOURCES[n] for n in argv if n in SOURCES]
    if not names:
        print(__doc__)
        return 2
    manifest = load_manifest()
    for src in names:
        fetch(src, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
