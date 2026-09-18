"""Summarize downloaded benchmark JSON; no Modal calls or new computation runs."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--figure", type=Path, required=True)
    a = p.parse_args()
    records = json.loads((a.directory / "records.json").read_text(encoding="utf-8"))
    comparisons = json.loads((a.directory / "comparisons.json").read_text(encoding="utf-8"))
    lookup = {r["tag"]: r for r in comparisons}
    variants = ("baseline", "stats", "relu", "stats_relu")
    rows = []
    for variant in variants:
        chosen = [r for r in records if r["variant"] == variant and not r["profile"]]
        if not chosen:
            continue
        checks = [lookup[r["tag"]] for r in chosen]
        rows.append({"variant": variant, "n": len(chosen),
                     "seconds_per_iter": [r["s_per_iter"] for r in chosen],
                     "median_s_per_iter": statistics.median(r["s_per_iter"] for r in chosen),
                     "paired_speedups": [c["speedup"] for c in checks],
                     "median_paired_speedup": statistics.median(c["speedup"] for c in checks),
                     "max_state_error": max(c["final"]["max_abs"] for c in checks),
                     "max_loss_error": max(c["losses"]["max_abs"] for c in checks),
                     "all_checks_pass": all(c["final"]["close"] and c["losses"]["close"] for c in checks),
                     "peak_mem_gb": [r["peak_mem_gb"] for r in chosen]})
    summary = {"run": json.loads((a.directory / "manifest.json").read_text())["run"],
               "timing_scope": records[0]["timing_scope"], "rows": rows,
               "note": "Two repeats; no confidence interval or convergence claim"}
    output = a.directory / "summary.json"
    if output.exists() or a.figure.exists():
        raise FileExistsError("Refusing to overwrite an existing summary or figure")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    names = [r["variant"] for r in rows]
    for index, row in enumerate(rows):
        axes[0].bar(index, row["median_s_per_iter"], color="#267f9a")
        axes[0].scatter([index] * row["n"], row["seconds_per_iter"], color="#111111", s=24, zorder=3)
        axes[1].bar(index, row["median_paired_speedup"], color="#308267" if row["all_checks_pass"] else "#bc5343")
        axes[1].scatter([index] * row["n"], row["paired_speedups"], color="#111111", s=24, zorder=3)
        axes[1].text(index, row["median_paired_speedup"] + .025,
                     "checks pass" if row["all_checks_pass"] else "CHECKS FAIL", ha="center", fontsize=8)
    for axis in axes:
        axis.set_xticks(range(len(names)), names, rotation=15)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Seconds per actual iteration (lower is better)")
    axes[1].set_ylabel("Speedup vs paired stock FlyVis")
    axes[1].axhline(1, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, max(max(r["paired_speedups"]) for r in rows) * 1.2)
    fig.suptitle("T4 / MaleCNS training benchmark — batch 16, two repeats\nIncludes solver checkpoints; separate profiler excluded", fontsize=11)
    fig.tight_layout()
    a.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figure, dpi=170)
    plt.close(fig)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
