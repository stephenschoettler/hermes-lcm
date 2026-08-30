# OPUS-DRIVE-LOOP — continuation-agent operating system (hermes-lcm benchmark program)

_For the Opus 4.8 (or any successor) agent taking over when Fable credits exhaust. Read order on takeover:
(1) this file, (2) PROGRAM-ARCHITECTURE.md (same dir — binding strategy + decision records), (3) the plan file
`~/.claude/plans/i-don-t-see-anything-cached-axolotl.md` STATE DELTA (if absent/stale, reconstruct state from
issues #152 → #107 → the milestone issues), (4) open issues in the wave-3/H6/H7 milestones on
100yenadmin/hermes-lcm. Do NOT re-derive strategy — it is settled; your job is execution._

## Canonical paths (cold-start pins — verified 07-24)

- **hermes-lcm working checkout:** `/Volumes/LEXAR/hermes-work/hermes-lcm` — the clone with remote `fork` =
  100yenadmin/hermes-lcm. **FIRST ACTION on takeover: `git fetch fork && git checkout fork/main`** — the local
  branch may be stale and `bench/PROGRAM-ARCHITECTURE.md` / `bench/OPUS-DRIVE-LOOP.md` / `bench/specs/` exist
  only on current fork/main. (`/Volumes/LEXAR/Claude/hermes-lcm` is a DIFFERENT clone without the fork remote —
  do not use it.) Purpose-built worktrees: `/Volumes/LEXAR/hermes-work/wt-*` (v1l1, h5-recall, upstream-w1, …).
- **Official LongMemEval-V2 harness + adapter (H6/W3 official runs):**
  `/Volumes/LEXAR/hermes-work/wt-bench-h1-v2adapter` (branch `bench/h1-v2-adapter`, org fork
  100yenadmin/LongMemEval-V2) — THE authoritative adapter worktree; ignore the other longmemeval checkouts on disk.
- **Official-run batch machinery + raw results:**
  `/Volumes/LEXAR/Codex/session-notes/2026-07-23/hermes-benchprog-h4/artifacts/phase3-openrouter/` (runners+logs)
  and `.../hermes-benchprog-h4/artifacts/OFFICIAL-FULL-RAW/` (the 125/451 evidence).
- **Program packet (specs, scratchpad, artifacts incl. `h5-targets.json`, W2A manifests):**
  `/Volumes/LEXAR/Codex/session-notes/2026-07-23/hermes-benchprog-h1/` (+ `artifacts/`).
- **Keys:** macOS keychain; retrieval recipes + service names in `~/.claude/runbooks/hermes-benchmark-ops.md`.
  Never print/log key values.
- **Gate source of truth:** the ISSUE BODY, always — architecture-doc gate summaries are abbreviations
  (e.g. #142 has a 4th condition, #143 a 3rd, that §3 omits). Score against the issue text verbatim.

## The loop (each cycle)

1. **Pick**: highest-priority OPEN issue whose dependencies are done (priority: Lane S > Lane A > Lane C
   shepherding > Lane V > Lane P). Issues carry `lane:*` and `blocked-by` markers.
2. **Execute** per the issue's spec section. Route work per the routing table below — you are the brain;
   delegate typing where the table says to.
3. **Gate**: score results against the issue's PREDECLARED gate exactly as written (the issue body is the sole
   source of truth). Pass ⇒ next step in issue. Fail ⇒ execute the issue's predeclared MISS branch. NEVER relax,
   reinterpret, or re-run-until-pass. Two distinct escape hatches, do not conflate them:
   (a) **structurally unpassable** (no candidate could pass on this data — PROVE it, like blind-R2's
   control-100/100): document the proof in the issue, `needs-owner`, move on;
   (b) **miscalibrated gate** (a whole class of valid mechanisms would fail it): rule it miscalibrated IN
   WRITING in the issue AND predeclare the replacement gate at correctness level BEFORE any rerun — never
   analysis-then-quietly-proceed (feedback_gate_proxy_calibration.md is the binding precedent).
4. **Bank**: snapshot-first (cp -R raw outputs to the session-notes artifacts dir BEFORE any further run);
   ledger line to ~/.claude/routing-ledger.jsonl; comment the result on the issue; tag score-bearing milestones
   (`bench-*` tags, pattern in git history).
5. **Advance**: close the issue with a result comment, or update its state. Return to step 1 without stopping.

## What you may decide alone vs must park

**Owner reinforcements (07-24, going to sleep):** (i) TAG the repo at every major milestone — score-bearing
results, doc/spec drafts, mechanism landings — so work survives token exhaustion (`bench-*` / `program-*`
tag patterns); (ii) Codex tokens are plentiful — lean CODEX for implementation volume (codex-dispatch skill;
this loop's routing table already defaults there for well-spec'd work); (iii) full autonomous continuation.

**Alone:** dispatch routing; retry/rerun of INFRA failures (hung CI runner, transport errors); executing
predeclared MISS branches; fork-local merges after your own review of green CI; shepherding replies on
upstream PRs (repro-first, own real bugs plainly — see #423 history); fork releases for milestones the owner
already blessed in kind; spend within existing categories and caps (OpenRouter $15 total cap stands, ~$1.1
used; Voyage W3a cap $10; codex/Sol lane = existing account).
**Park as `needs-owner` (comment on #107 + the issue, continue other work):** gate REVISIONS after numbers
exist; NEW spend categories or cap raises (H6-P4 full agentic run is explicitly this); leaderboard submission
(standing HOLD until a threshold crossing — architecture §3 Lane C); strategy/lane changes; anything
irreversible outside our fork.

## Routing table (Fable-era table adapted for an Opus brain)

| Work | Lane |
|---|---|
| Strategy/gates/scoring/promotions | YOU (Opus), inline — never delegated |
| Complex repo-native implementation | `coder` agent (Opus) or codex dispatch per `codex-dispatch` skill (well-spec'd + non-urgent → codex luna·xhigh/sol·high) |
| Mechanical volume, launches, digests | `fast-worker` (Sonnet). EVERY dispatch prompt that waits on a run MUST contain verbatim: "Watch by file in the FOREGROUND: poll the report/output file in a plain bash loop in your own shell — do NOT rely on any background watcher notifying you" + the explicit artifacts-dir path |
| Cross-model review | Codex-authored code → you review; your/Claude-authored load-bearing code → codex sol·max review. Never self-review load-bearing code |
| Official-protocol runs | the frozen batch machinery in hermes-benchprog-h4/artifacts/phase3-openrouter/ (60q slices for dev, 4×60+enterprise for full); keys in keychain (OPENROUTER_API_KEY, VOYAGE_API_KEY) — never print |

## Discipline (binding; violations have all bitten this program — see memory files)

- Snapshot-first; manifests frozen + sha256 + ACHIEVABILITY CHECKED before launch (compute the control slice
  at freeze — the blind-R2 lesson).
- One-primary law: one full primary run per candidate; no reruns of completed runs; never point a relaunch at
  a dir that held a completed run.
- File-keyed FOREGROUND polling (report.json), never process-exit; dispatch prompts must mandate foreground
  polling and the artifacts dir explicitly.
- Durable crons for anything outliving the session; if a cron misses (fires only while idle), execute its
  prompt inline — both aggregation crons tonight were executed inline.
- Assert outward success: ls-remote after pushes; re-read after posts; no `2>/dev/null` on outward actions.
- Any scorer/comparator/normalizer change ⇒ rescore ALL prior comparison baselines under the identical scorer
  BEFORE publishing any delta (the +86-that-was-+24 incident; feedback_rescore_both_sides memory).
- Spend totals are RE-DERIVED, never trusted from docs: OpenRouter usage API at loop start (cap $15 total);
  W3a Voyage backfill re-metered after the sizing step AND at 50% completion before continuing (cap $10).
- Keepalive sleep-ticks during active waiting phases (session-scoped sentinels; `touch /tmp/worldos_last_tick.
  $CLAUDE_CODE_SESSION_ID; sleep 250; echo tick` as last call; self-extinguishing); dense turns — process every
  landed wake fully and do parallel plan work before ending a turn.
- Artifacts → /Volumes/LEXAR/Codex/session-notes/YYYY-MM-DD/<slug>/artifacts/, never /tmp. Scratchpad =
  implementation-notes.html in the packet dir; timestamped entries at every decision.
- ONE WORKTREE PER LANE: never run two agents (or an agent + a run) against the same checkout — a W3b builder commit switched the shared lme-v2-official HEAD mid-P3-run (07-24; harmless only by luck). Dedicated worktree or explicit lock per lane; restore expected branches after any shared-checkout use.
- GitHub API budget: 5k/hr shared — don't poll gh in tight loops (burned once tonight); use ScheduleWakeup or
  sleep-tick micro-checks.

## State at handoff (as of 07-24 ~07:05 ONLY — ★ RE-POLL EVERY ITEM with the named command before trusting it;
this list rots the moment it is written)

- ✅ H6 P0 recon DONE — protocol pin posted as a comment on #145 (verify: `gh issue view 145 --comments`).
- ✅ R1 release PUBLISHED (verify: `gh release view program-r1 -R 100yenadmin/hermes-lcm --json isDraft`).
- ✅ Wave-1 upstream PR POSTED = stephenschoettler/hermes-lcm#436 (verify: `gh pr view 436 -R
  stephenschoettler/hermes-lcm --json state`). Shepherd it per the PR-shepherding standard.
- ⏳ GEX44 fp8 cross-check running (independent; non-blocking; fold into #151 when done).
- ⏳ Fork main CI: rerun launched ~04:35 after a hung 3.13 runner cancel (5/6 legs green pre-cancel; verify:
  `gh run list -R 100yenadmin/hermes-lcm --branch main --limit 1`). If red on a REAL failure, fix before new merges.
- ⏳ #423 awaiting maintainer; #434 awaiting review; harness PR #2 open (poll each; respond per #423 history).
- AUTHORIZED and unstarted: W2A-C2 (#150) · W3a-0→W3a (#141→#142) · W3c (#144) · H6-P1 (#145). Lane priority
  in §loop applies. If any "half-done" state is found (e.g. tag without release), complete ONLY the missing half
  after instrument-verifying which half exists.

## Escalation

Real blockers (missing credential, hard external dependency, owner-taste fork) → comment `needs-owner` on the
issue + #107, continue other lanes. If ALL lanes block: write a status comment on #107 with the precise asks,
then stop cleanly. Never idle-stop with lanes open; never fabricate progress; report failures verbatim.
