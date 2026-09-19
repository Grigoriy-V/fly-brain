@AGENTS.md

## Claude-specific notes

`AGENTS.md` above contains the authoritative working and safety rules.
Nothing in this file may add, weaken, or replace a rule in it.

- `.claude/settings.json` duplicates some hard rules as a mechanical backstop.
  It is not the source of any rule.
- Use `--agent claude` when writing a run-log record.
- The human writes in Russian; canonical documents (`AGENTS.md`, `ROADMAP.md`,
  `DECISIONS.md`, `ISSUES.md`, `docs/PROJECT_MAP.md`, `docs/OPERATIONS_MAP.md`,
  `docs/ARTEFACTS.md`) are kept in English, reports and research notes in the
  language they were requested in. `docs/ideas/` holds the human's own notes
  in the language they were written in and is never rewritten by an agent.
