# Stage 2 Bot-Pass Fix Manifest
Date: 2026-07-29
- Worktree/branch: `/Volumes/LEXAR/hermes-work/wt-stage2`; `feat/session-expand-v1`.
- Pinned base: `cb92bf40c1d4c862cb56090792113b69d88660d4`.
- Starting candidate: `eb477f81a089766205136d2ffcb590c61e01ddcb`; fixes remain uncommitted.
- Scope: nine post-cutoff PR #173 code comments; seven `bench/**` comments excluded.
- Verdict ledger: `FINDINGS-VERDICTS-S2B.md`; all nine code findings validated and fixed.
- Code: flag-gated bridge mode; guarded date-only UTC conversion; bounded 12-row anchor paging.
- Code: captured tail bound; indexed strict walk; configurable treatment cap with 512,000 absolute ceiling.
- Test: Bun bridge seam has a 120-second subprocess timeout.
- Focused proof: `103 passed` in `focused-final-2.xml`; real harness seam enabled.
- M0-M5: all Stage 2 recall/bridge probes in the focused files passed.
- Flag-OFF proof: pinned base and candidate are byte-identical at 13,742 bytes.
- Flag-OFF SHA-256: `0480a8444845ea7226861471fd00cd6d9c0fb5daa9d8600b7d0feab81d25d875`.
- Full isolated proof: `2,754 passed, 1 skipped, 12 xfailed` in `full-final.xml`.
- Isolation: explicit fresh `TMPDIR`, `HERMES_HOME`, `LCM_DATABASE_PATH`; prior venv+agent stub.
- Static proof: Ruff and `git diff --check` passed on all changed Python files.
- Evidence: `/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-r3-0/artifacts/laneS2B-logs/`.
- Remote state: no push, PR update, merge, release, deployment, runtime, or experiment.
