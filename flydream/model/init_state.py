"""Save a transplanted network state so a Modal training run can start from it.

    python -m flydream.model.init_state --member 0 [--no-rescale] [--cap 3]

Builds the MaleCNS network on the export `config.toml [data]` names, transplants
FlyVis member `--member`'s parameters by type with the gain rescaled and capped
(`zero.transplant`, DECISIONS 2026-09-18), and writes the network's state to
`data/runs/init_<export tag>_m<member>.pt` together with the transplant report.
`deploy/modal/train_app.py::train --init <name>` loads it before training; without
it the solver starts from flyvis's own initialisation, which is the
"trained from scratch on this wiring" control.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from flydream.model import ROOT, configure_flyvis_root

configure_flyvis_root()

import torch  # noqa: E402

import flyvis  # noqa: E402
from flyvis import NetworkView  # noqa: E402

from flydream.data.export import filters_path, filters_tag  # noqa: E402
from flydream.model.zero import build_network, settings, transplant  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--member", type=int, default=0)
    p.add_argument("--no-rescale", action="store_true")
    p.add_argument("--cap", type=float, default=None, help="default config.toml [model] rescale_cap")
    a = p.parse_args(argv)
    s = settings()
    cap = a.cap if a.cap is not None else float(s.get("model", {}).get("rescale_cap", 3.0))
    export = filters_path(s)
    tag = f"{filters_tag(s)}_m{a.member:03d}" + ("_raw" if a.no_rescale else "")
    out_dir = ROOT / "data" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    fv = NetworkView(flyvis.results_dir / f"flow/0000/{a.member:03d}").init_network()
    mc = build_network(export, int(s["data"]["extent"]))
    rep = transplant(fv, mc, rescale=not a.no_rescale, rescale_cap=cap)
    state = {k: v.detach().cpu() for k, v in mc.state_dict().items()}
    path = out_dir / f"init_{tag}.pt"
    torch.save(state, path)
    meta = {"export": export.name, "member": a.member, "rescale": not a.no_rescale, "cap": cap,
            "n_nodes": int(mc.n_nodes), "n_edges": int(mc.n_edges), "state_keys": len(state),
            "transplant": {k: ({kk: vv for kk, vv in v.items() if kk != "capped_pairs"} if isinstance(v, dict) else v)
                           for k, v in rep.items()},
            "capped_pairs": rep["syn_strength"].get("capped_pairs", [])}
    (out_dir / f"init_{tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB, {len(state)} tensors) and {path.stem}.json "
          f"in {time.time() - t0:.0f}s; capped pairs: {rep['syn_strength'].get('capped', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
