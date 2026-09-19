"""Call a function of the **deployed** flydream app — no ephemeral app per run.

    python tools/modal_call.py train17 --kw steps=20000 name=corpus_dct16 --out data/prior18/train.json

`modal run` builds a throw-away app for every invocation, which is what fills
the dashboard with `ap-…` entries and leaves no history. The app is deployed
once (`modal deploy deploy/modal/generate_app.py`); this resolves a function
inside it by name and calls it, so every run appears under the one app.
Nothing is kept warm, so a deployed app costs nothing while idle.

Values after `--kw` are coerced: `20000` → int, `3e-4` → float, `true`/`false`
→ bool, anything else stays a string. `--json` takes a whole kwargs object for
the cases that need lists or nesting.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

APP = "flydream-generate"


def coerce(v: str):
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("function")
    p.add_argument("--app", default=APP)
    p.add_argument("--kw", nargs="*", default=[], help="key=value pairs passed to the function")
    p.add_argument("--json", default="", help="kwargs as one JSON object")
    p.add_argument("--out", default="", help="write the returned summary here")
    p.add_argument("--quiet-keys", default="history,mean,std,meta,cells,type_of_cell,coef_mean,coef_std")
    a = p.parse_args(argv)
    import modal

    kwargs = json.loads(a.json) if a.json else {}
    for kv in a.kw:
        if "=" not in kv:
            raise SystemExit(f"--kw takes key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        kwargs[k] = coerce(v)
    fn = modal.Function.from_name(a.app, a.function)
    print(f"{a.app}::{a.function}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})", flush=True)
    t0 = time.time()
    r = fn.remote(**kwargs)
    seconds = time.time() - t0
    quiet = set(a.quiet_keys.split(","))
    if isinstance(r, dict):
        print(json.dumps({k: v for k, v in r.items() if k not in quiet}, indent=1, ensure_ascii=False)[:2500])
        if "history" in r and r["history"]:
            h = r["history"]
            print(f"loss {h[0]['loss']:.4f} -> {h[-1]['loss']:.4f}, val {h[-1].get('val_loss')}")
    else:
        print(r)
    print(f"done in {seconds:.0f} s (wall, including container start)")
    if a.out:
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
