"""Forward/backward equivalence only, on the synthetic fixture; no training.

Writes a standalone numerical check and figure. No download or Modal imports.
"""
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from flydream.model import ROOT
from flydream.train.optimizations import optimized_network, activity_stats


def main():
    from datamate import set_root_context
    from flyvis import Network
    from flyvis.utils.config_utils import Namespace
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = {"seed": 7, "frames": 40, "batch": 2, "dt": 0.02, "extent": 2,
              "dtype": "float32", "fixture": "tests/fixtures/mini_connectome.json",
              "bias": [-0.4, 1.2], "time_const": 0.06, "strength": 0.04,
              "input_scale": 0.4, "rtol": 1e-5, "atol": 1e-7}
    fixture = ROOT / config["fixture"]
    identity = {"config": config, "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "optimization_sha256": hashlib.sha256((ROOT / "flydream/train/optimizations.py").read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "torch": torch.__version__}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    run = f"2026-09-18_optimization_equivalence_{digest}"
    out = ROOT / "data/benchmarks" / run
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    with tempfile.TemporaryDirectory(prefix="flydream_equivalence_") as temp:
        with set_root_context(temp):
            net = Network(connectome=Namespace(type="ConnectomeFromAvgFilters", file=str(fixture),
                                               extent=config["extent"], n_syn_fill=0))
        net.train()
        with torch.no_grad():
            b = net.node_params["bias"].raw_values
            b[:] = torch.linspace(*config["bias"], len(b))
            net.node_params["time_const"].raw_values[:] = config["time_const"]
            net.edge_params["syn_strength"].raw_values[:] = config["strength"]
        data = torch.randn(config["batch"], config["frames"], net.n_nodes) * config["input_scale"]
        target = torch.randn_like(data)
        results = []
        for enabled in (False, True):
            net.zero_grad(set_to_none=True)
            x = data.clone().requires_grad_()
            with optimized_network(net, enabled):
                y = net(x, config["dt"])
                (y - target).square().mean().backward()
            values = {"trajectory": y.detach(), "input_gradient": x.grad,
                      **{name: p.grad for name, p in net.named_parameters() if p.requires_grad}}
            a = y.detach()
            values["diagnostic_statistics"] = (torch.stack(activity_stats(a)) if enabled
                else torch.stack((a.cpu().mean(), a.cpu().min(), a.cpu().max())))
            results.append({k: v.clone() for k, v in values.items()})
        rows = []
        for name, ref in results[0].items():
            candidate = results[1][name]
            error = float((candidate - ref).abs().max())
            tolerance = config["atol"] + config["rtol"] * ref.abs()
            scaled = float(((candidate - ref).abs() / tolerance).max())
            rows.append({"name": name, "max_abs_error": error, "max_tolerance_fraction": scaled,
                         "reference_max_abs": float(ref.abs().max()),
                         "candidate_max_abs": float(candidate.abs().max()), "pass": scaled <= 1})
    result = {"run": run, **identity, "rows": rows, "passed": all(r["pass"] for r in rows),
              "control": "unmodified FlyVis 1.2.0", "device": "CPU", "training_updates": 0,
              "cost_usd": 0, "claim": "synthetic numerical equivalence only; GPU speed unmeasured"}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = [r["name"].replace("nodes_", "").replace("edges_", "") for r in rows]
    vals = [r["max_tolerance_fraction"] for r in rows]
    ax.barh(names, vals, color="#247ba0")
    ax.axvline(1, color="#b33b32", linestyle="--", label="FP32 check tolerance")
    for i, (v, r) in enumerate(zip(vals, rows)):
        ax.text(v + .02, i, f"max |difference| = {r['max_abs_error']:.2g}", va="center", fontsize=9)
    ax.set_xlim(0, 1.35)
    ax.set_xlabel("Difference / allowed tolerance (smaller is better)")
    ax.set_title("Node ReLU + device statistics vs stock FlyVis\nSynthetic fixture, CPU, 40 frames; no training or speed claim")
    ax.legend(loc="lower right")
    fig.tight_layout()
    figure = ROOT / "reports/figures" / f"{run}.png"
    if figure.exists():
        raise FileExistsError(figure)
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    print(figure)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
