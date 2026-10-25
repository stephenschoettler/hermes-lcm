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

- Recurring corruption, `disk I/O error`, repair tooling: `references/wal-generation-corruption.md`
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
- Never unlink `lcm.db-wal`/`-shm`, never open the database read-write merely to inspect it, and never swap the live inode while handles are open: each strands every live process on a deleted WAL generation and corrupts the file. A `disk I/O error` is a degraded environment, not corruption, and "repairing" it loops. See `references/wal-generation-corruption.md`.
