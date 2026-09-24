"""Which tracked working figures under reports/figures/ nothing refers to.

    python tools/orphan_figures.py --out reports/orphan_figures.md

A figure counts as referenced when its file name, or its stem followed by a
non-name character (`.png`, `.{gif,png}`, a space, a quote), appears in any
tracked text file outside reports/figures/ - reports, the articles, the run
log, docs, code. Stems a figure script builds from a prefix (`f"{prefix}_test"`)
are caught separately: the figure is then "made by a script, cited nowhere".
Sizes are the files' on disk. Read-only; writes a Markdown list.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = (".md", ".py", ".json", ".jsonl", ".toml", ".txt", ".csv")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "reports" / "orphan_figures.md"))
    a = p.parse_args(argv)
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8").stdout.split("\n")
    figs = [f for f in tracked if f.startswith("reports/figures/") and f]
    texts, code = [], []
    for f in tracked:
        if f and f.endswith(TEXT) and not f.startswith("reports/figures/") and f != "reports/orphan_figures.md":
            body = (ROOT / f).read_text(encoding="utf-8", errors="replace")
            texts.append(body)
            if f.endswith(".py"):
                code.append(body)
    corpus = "\n".join(texts)
    prefixes = set(re.findall(r'"(20\d\d-\d\d-\d\d_[A-Za-z0-9_]+)"', "\n".join(code)))
    # brace lists (`mix_{C,A,B}.{gif,png}`) and globs (`slices8*`) count as references
    patterns = []
    for ref in re.findall(r"20\d\d-\d\d-\d\d_[A-Za-z0-9_{},*-]+", corpus):
        if "{" not in ref and "*" not in ref:
            continue
        rx = ""
        for part in re.split(r"(\{[^}]*\}|\*)", ref):
            if part == "*":
                rx += "[A-Za-z0-9_]*"
            elif part.startswith("{") and part.endswith("}"):
                rx += "(?:" + "|".join(re.escape(x) for x in part[1:-1].split(",")) + ")"
            else:
                rx += re.escape(part)
        patterns.append(re.compile("^" + rx + "$"))
    # stems a script assembles from a template: `{prefix}_dream_{source}`, `--out ..._inversion_ladder`
    joined = "\n".join(code)
    templates = set(re.findall(r"\}_([a-z][a-z0-9]*)_\{", joined))
    templates |= set(re.findall(r"_(inversion_ladder|generator_two_inputs)\b", joined))

    cited, scripted, orphan = [], [], []
    for f in figs:
        name = Path(f).name
        stem = Path(f).stem
        size = (ROOT / f).stat().st_size if (ROOT / f).exists() else 0
        tail = re.sub(r"^20\d\d-\d\d-\d\d_(?:malecns_|flyvis_)?", "", stem)
        if name in corpus or re.search(re.escape(stem) + r"(?![A-Za-z0-9_])", corpus) \
                or any(px.match(stem) for px in patterns):
            cited.append((f, size))
        elif any(stem.startswith(pr + "_") for pr in prefixes) \
                or any(tail == tp or tail.startswith(tp + "_") for tp in templates):
            scripted.append((f, size))
        else:
            orphan.append((f, size))

    mb = lambda rows: sum(s for _, s in rows) / 1e6  # noqa: E731
    lines = ["# Working figures that nothing refers to", "",
             f"Made by `tools/orphan_figures.py`. Of {len(figs)} tracked files under `reports/figures/` "
             f"({mb(cited) + mb(scripted) + mb(orphan):.0f} MB):", "",
             f"- cited by a report, the run log, docs or code: {len(cited)} files, {mb(cited):.0f} MB;",
             f"- made by a figure script from a prefix, cited nowhere: {len(scripted)} files, {mb(scripted):.0f} MB;",
             f"- referred to by nothing at all: {len(orphan)} files, {mb(orphan):.0f} MB.", "",
             "Candidates for removal from history are the last two groups; the first group is linked from",
             "text that stays public.", "",
             "## Referred to by nothing", "", "| file | MB |", "|---|---|"]
    lines += [f"| `{Path(f).name}` | {s / 1e6:.1f} |" for f, s in sorted(orphan, key=lambda r: -r[1])]
    lines += ["", "## Made by a script, cited nowhere", "", "| file | MB |", "|---|---|"]
    lines += [f"| `{Path(f).name}` | {s / 1e6:.1f} |" for f, s in sorted(scripted, key=lambda r: -r[1])]
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"cited {len(cited)} ({mb(cited):.0f} MB), scripted-only {len(scripted)} ({mb(scripted):.0f} MB), "
          f"orphan {len(orphan)} ({mb(orphan):.0f} MB) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
