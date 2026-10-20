---
name: hermes-lcm
description: Use, configure, diagnose, and retrieve exact evidence with the Hermes-LCM lossless context plugin.
---

# Hermes-LCM

Use this skill when a task concerns Hermes-LCM setup, operation, compaction, diagnostics, session behavior, or recall from compacted and cross-conversation history.

Start here:

1. Confirm that the `hermes-lcm` plugin is enabled and `context.engine` is `lcm`.
2. For exact historical claims, use the recall workflow instead of trusting a compacted summary.
3. Use `lcm_status`, `lcm_inspect`, and `lcm_doctor` before changing configuration or attempting repair.
4. Treat slash-command apply paths as mutations: preview first, keep backups, and require the user's authorization.
5. Load the relevant reference rather than guessing arguments or lifecycle semantics.

Reference map:

- Configuration and activation: `references/configuration.md`
- Architecture and data ownership: `references/architecture.md`
- Diagnostics and safe operator workflow: `references/diagnostics.md`
- Recall tools and routing: `references/recall-tools.md`
- `/new`, session continuity, and `/lcm rotate`: `references/session-lifecycle.md`
- Canonical runtime recall policy: `references/recall-policy.md`

Working rules:

- Raw stored messages are authoritative; summaries are bounded recall cues.
- Prefer newer source-backed evidence when it conflicts with an older summary.
- Start with the narrowest useful scope and expand only when exact detail is needed.
- Do not infer exact commands, paths, timestamps, values, counts, or causal chains from summaries alone.
- Keep current-session, cross-conversation, and Hermes history outside `lcm.db` distinct.
- Do not treat open-cardinality results as complete without product-verifiable enumeration or coverage.
- Use `lcm_compile_evidence` when a historical answer needs several named facets, exact operands, conflict handling, or latest-state selection; treat its semantic proposal as untrusted until the product returns validated evidence.
- Keep default-off assertion, query-view, adaptive-retrieval, and destructive operator paths default-off unless the user explicitly asks to enable them.
- **NEVER back up or copy `~/.hermes/lcm.db` with a raw `cp`/`mv`/`rsync`/`tar` while it is active in WAL mode** — a torn write mid-checkpoint corrupts the btree (incident 2026-09-25: raw copy at 13:23 → `disk I/O error` at 13:26 → DB unusable, ~529 messages lost). Only coherent snapshot methods:
  - `sqlite3 /root/.hermes/lcm.db ".backup /chemin/lcm-snap-$(date +%Y%m%d_%H%M%S).sqlite3"` (preferred, safe on a live DB)
  - `python3 -c "import sqlite3; s=sqlite3.connect('/root/.hermes/lcm.db'); d=sqlite3.connect('/chemin/snap.sqlite3'); s.backup(d); d.close(); s.close()"`
  - the plugin's own `/lcm backup` (uses the SQLite API, writes to the plugin backup dir)
  - `VACUUM INTO` (requires a quiescent DB — gateway stopped)
  After any snapshot: verify the snapshot with `PRAGMA integrity_check` before trusting it.
- **NEVER restore or move `~/.hermes/lcm.db` manually** (`cp`/`mv`/`rm`) while gateway/dashboard are running — even if a `systemctl stop` was issued first, a stop can silently fail (incident 2026-09-27: restore executed with the gateway still alive → torn write). The ONLY supported restore path:
  - `python3 /root/.hermes/scripts/safe_lcm_ops.py restore <backup.sqlite3>` — it stops hermes-gateway+hermes-dashboard, **verifies death via a /proc fd scan** (not just systemctl), keeps the corrupted DB as `lcm.db.corrupted-<ts>` for forensics, verifies the backup BEFORE and the target AFTER the swap, then restarts and verifies the services. If it prints `RESTORE_ABORT` or `RESTORE_FAIL`, do not continue manually.
  - `safe_lcm_ops.py check` / `status` are read-only and safe on the live DB. `status` shows which PIDs hold lcm.db open (including the live hermes CLI session — expected, flagged `[CLI]`).
- The watchdog's `ALERT_*.txt` sentinels point at `safe_lcm_ops.py restore` — follow them, never the old manual mv/cp steps.
- Full restore + OpenViking→LCM sync workflow (validated 2026-10-03): load skill `lcm-ov-recovery` — it covers backup selection, OV API pitfalls (archive 404 → use fs/tree + content/read), import mapping, and verification. This skill keeps only the safety rules (WAL mode, safe_lcm_ops.py); `lcm-ov-recovery` keeps the procedure.
