"""One picture of the transplant's progress: T4/T5 direction selectivity per run.

    python -m flydream.model.compare_runs --runs v5 v7 v8 v9 --latest v9

For each ensemble member and each T4/T5 subtype, the direction selectivity
index of the MaleCNS model under each transplant, beside the FlyVis member it
was transplanted from. Emphasis form (dataviz method): FlyVis in slot 1 and
the latest run in slot 2 carry the story; earlier runs are context in gray.
The CSV beside the figure is the table view.

`--runs` names run ids by their suffix after `2026-09-18_step2_zero_R_`
(`v5`, `v7_total_cap3`, ...); a prefix match is enough.
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd

from flydream.model import ROOT

RUNS = ROOT / "data" / "runs"
SUBTYPES = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
SLOT1, SLOT2 = "#2a78d6", "#eb6834"                 # validated adjacent pair, light mode
GRAYS = ["#c3c2b7", "#a9a79f", "#898781"]            # context steps, muted ink
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def resolve(tag: str) -> str:
    hits = sorted(p.name for p in RUNS.iterdir() if p.is_dir() and re.match(rf"^\d{{4}}-\d{{2}}-\d{{2}}_step2_zero_R_{re.escape(tag)}", p.name))
    if not hits:
        raise SystemExit(f"no run under data/runs matches *_step2_zero_R_{tag}")
    return hits[-1]


def dsi_of(run: str, model: str, member: int) -> pd.Series | None:
    p = RUNS / run / f"dsi_{model}_{member:03d}.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p).set_index("cell_type")
    out = {}
    for s in SUBTYPES:
        if s in t.index:
            out[s] = float(t.loc[s, "dsi_i1" if s.startswith("T4") else "dsi_i0"])
    return pd.Series(out)


def table(runs: list[str], members: list[int]) -> pd.DataFrame:
    rows = []
    for m in members:
        fv = dsi_of(runs[0], "flyvis", m)
        if fv is not None:
            for s, v in fv.items():
                rows.append({"member": m, "subtype": s, "run": "FlyVis", "dsi": v})
        for r in runs:
            mc = dsi_of(r, "malecns", m)
            if mc is None:
                continue
            for s, v in mc.items():
                rows.append({"member": m, "subtype": s, "run": r.split("_step2_zero_R_")[1], "dsi": v})
    return pd.DataFrame(rows)


def figure(t: pd.DataFrame, runs_short: list[str], latest: str, path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    members = sorted(t.member.unique())
    fig, axes = plt.subplots(1, len(members), figsize=(2.9 * len(members) + 1.2, 4.4), sharey=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    context = [r for r in runs_short if r != latest]
    y = np.arange(len(SUBTYPES))
    for ax, m in zip(axes, members):
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.grid(True, axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        d = t[t.member == m]
        for i, r in enumerate(context):
            v = d[d.run == r].set_index("subtype").dsi.reindex(SUBTYPES)
            ax.scatter(v, y, s=40, color=GRAYS[min(i, len(GRAYS) - 1)], edgecolor=SURFACE, linewidth=2,
                       zorder=3, label=r if m == members[0] else None)
        v = d[d.run == latest].set_index("subtype").dsi.reindex(SUBTYPES)
        ax.scatter(v, y, s=72, color=SLOT2, edgecolor=SURFACE, linewidth=2, zorder=5,
                   label=f"{latest} (latest)" if m == members[0] else None)
        f = d[d.run == "FlyVis"].set_index("subtype").dsi.reindex(SUBTYPES)
        ax.scatter(f, y, s=72, color=SLOT1, edgecolor=SURFACE, linewidth=2, zorder=6, marker="D",
                   label="FlyVis member" if m == members[0] else None)
        ax.set_yticks(y)
        ax.set_yticklabels(SUBTYPES, color=INK2, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.set_xlim(0, max(0.05, float(np.nanmax(d.dsi)) * 1.15))
        ax.set_title(f"member {m:03d}", color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("direction selectivity index", color=INK2, fontsize=8.5)
        ax.invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run tags, oldest first, e.g. v5 v7 v8 v9")
    p.add_argument("--latest", default=None, help="the run to emphasise (default: the last of --runs)")
    p.add_argument("--members", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--tag", default="2026-09-18_model_zero_transplants")
    a = p.parse_args(argv)
    runs = [resolve(r) for r in a.runs]
    short = [r.split("_step2_zero_R_")[1] for r in runs]
    latest = a.latest or a.runs[-1]
    latest = next(s for s in short if s.startswith(latest))
    t = table(runs, a.members)
    if t.empty:
        print("nothing found"); return 1
    out = ROOT / "reports" / "figures" / f"{a.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    print("mean T4/T5 DSI per member and run:")
    print(t.pivot_table(index="run", columns="member", values="dsi", aggfunc="mean").round(3).to_string())
    figure(t, short, latest, ROOT / "reports" / "figures" / f"{a.tag}.png",
           "T4/T5 direction selectivity on MaleCNS wiring, transplant by transplant, against the FlyVis member")
    print(f"wrote reports/figures/{a.tag}.png and .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
