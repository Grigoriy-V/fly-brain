"""The only writer of reports/runs.jsonl: one line per measured outcome.

    python tools/run_log.py --agent claude --run 2026-09-18_step1_export \
        --experiment step1.filters --metric recovered_type_pairs --value 0.867 \
        [--control 0.0] [--n 571] [--cost 0] [--note "..."]

Fields: run, experiment, metric, value, control, n, cost, agent, date, note.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from datetime import date

LOG = pathlib.Path(__file__).resolve().parents[1] / "reports" / "runs.jsonl"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, choices=["claude", "codex", "human"])
    p.add_argument("--run", required=True, help="<date>_<experiment>_<config hash or tag>")
    p.add_argument("--experiment", required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--value", required=True, type=float)
    p.add_argument("--control", type=float, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--cost", type=float, default=0.0, help="USD; 0 for a local run")
    p.add_argument("--note", default="")
    a = p.parse_args()
    rec = {"run": a.run, "experiment": a.experiment, "metric": a.metric, "value": a.value,
           "control": a.control, "n": a.n, "cost": a.cost, "agent": a.agent,
           "date": date.today().isoformat(), "note": a.note}
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
