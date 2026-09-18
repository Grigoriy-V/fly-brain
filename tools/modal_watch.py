"""Watch the project's Modal apps from the owner's machine.

    python tools/modal_watch.py                       # flydream apps: id, state, tasks, age
    python tools/modal_watch.py --wait <app-id>       # poll until the app stops, then save its log
    python tools/modal_watch.py --logs <app-id>       # save the app's log now
    python tools/modal_watch.py --volume /benchmarks  # list a path on flydream-runs

Wraps the modal CLI of the project venv (profile `grigoriy98smile`, see
docs/OPERATIONS_MAP.md). Logs go to data/modal_logs/<app-id>.log, which is
ignored by git; the numbers that matter are copied into reports/runs.jsonl by
hand with tools/run_log.py. Reads only; never starts a worker.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "MSYS_NO_PATHCONV": "1"}
LOGS = ROOT / "data" / "modal_logs"


def modal(*args: str, timeout: int = 120) -> str:
    p = subprocess.run([PY, "-m", "modal", *args], capture_output=True, text=True, env=ENV,
                       timeout=timeout, encoding="utf-8", errors="replace")
    return p.stdout + p.stderr


def apps(prefix: str = "flydream") -> list[dict]:
    """Parse `modal app list` (a rich table wrapped over lines) into records."""
    out = modal("app", "list")
    rows, cur = [], None
    for line in out.splitlines():
        if not line.startswith("│"):
            continue
        cells = [c.strip() for c in line.strip("│").split("│")]
        if len(cells) < 5:
            continue
        if cells[0].startswith("ap-"):
            cur = {"id": cells[0], "description": cells[1], "state": cells[2], "tasks": cells[3], "created": cells[4]}
            rows.append(cur)
        elif cur is not None:
            for k, v in zip(("id", "description", "state", "tasks", "created"), cells):
                if v:
                    cur[k] = (cur[k] + " " + v).strip()
    return [r for r in rows if r["description"].replace("…", "").startswith(prefix[: len(r["description"].replace("…", ""))])]


def save_logs(app_id: str) -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / f"{app_id}.log"
    text = modal("app", "logs", app_id, timeout=90)
    path.write_text(text, encoding="utf-8")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wait", metavar="APP_ID")
    p.add_argument("--logs", metavar="APP_ID")
    p.add_argument("--volume", metavar="PATH", help="list this path on flydream-runs")
    p.add_argument("--every", type=int, default=60, help="poll interval in seconds for --wait")
    p.add_argument("--max-minutes", type=int, default=240)
    a = p.parse_args(argv)
    if a.volume:
        print(modal("volume", "ls", "flydream-runs", a.volume))
        return 0
    if a.logs:
        path = save_logs(a.logs)
        print(f"saved {path}")
        print("\n".join(path.read_text(encoding="utf-8").splitlines()[-15:]))
        return 0
    if a.wait:
        t0 = time.time()
        while time.time() - t0 < a.max_minutes * 60:
            live = {r["id"]: r for r in apps()}
            r = live.get(a.wait)
            state = r["state"] if r else "gone"
            print(f"{time.strftime('%H:%M:%S')} {a.wait}: {state}" + (f", tasks {r['tasks']}" if r else ""), flush=True)
            if state not in ("ephemeral", "running", "deployed") or (r and r["tasks"] == "0" and state == "ephemeral"):
                break
            time.sleep(a.every)
        path = save_logs(a.wait)
        print(f"saved {path}")
        tail = [l for l in path.read_text(encoding="utf-8").splitlines() if re.search(r'^\{|Error|error|Traceback', l)]
        print("\n".join(tail[-20:]))
        return 0
    for r in apps():
        print(f"{r['id']}  {r['state']:<10} tasks {r['tasks']:<3} {r['description']}  {r['created']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
