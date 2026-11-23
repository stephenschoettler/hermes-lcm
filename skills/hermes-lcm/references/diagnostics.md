# Diagnostics

Use read-only product tools before changing configuration or running an apply path.

## Fast path

1. `hermes plugins`: confirm `hermes-lcm` is enabled and the selected context engine is `lcm`.
2. Send one normal message if the session has not been bound since restart.
3. `lcm_status`: inspect runtime identity, database path, context pressure, summary/store counts, filters, and lifecycle state.
4. `lcm_inspect`: inspect current-session lineage, frontiers, fresh tail, externalized-ref readability, and skip/no-op reasons without retrieving content.
5. `lcm_doctor`: run database, FTS, lifecycle, configuration, and context-pressure diagnostics.

If optional slash commands are enabled, `/lcm status` and `/lcm doctor` expose the corresponding operator views.

## Safe mutation order

For cleanup, repair, source normalization, or rotate:

1. run the read-only preview;
2. inspect exact candidates and paths;
3. create/confirm a backup;
4. obtain user authorization for the specific apply operation;
5. run one bounded apply and verify integrity afterward.

Cleanup apply is separately feature-gated. Never infer permission to enable it from a diagnosis request.

## Common states

- Unbound status after restart: send a normal message, then check again.
- Database exists but stays empty: verify plugin enablement, `context.engine`, profile, database path, and ignore/stateless patterns.
- Weak exact recall: verify source rows exist, query construction/scope is correct, summary health is sound, and embedding coverage/provenance matches the requested mode.
- Conflicting summary and raw evidence: prefer the newer exact raw evidence and inspect lineage.
- Path B/context-engine schema log: expected on hosts where plugin-registry handlers do not receive active messages; context-engine schemas and dispatch remain the healthy route.

## Corruption forensics (field-verified 2026-09-29, corruption #4 on /mnt/fastdata/lcm/lcm.db)

When `PRAGMA integrity_check` fails, run the byte-level triage BEFORE theorizing. Order matters:

1. **Header (bytes 0-112)**: magic 16B; bytes 18/19 = WAL format flags (2,2 = WAL, 0,0 = rollback — a flip means header-level damage); change counter @24; db_size_pages @28 (compare with actual file size / page size — mismatch = truncation); freelist_first/count @32/36; schema cookie @40.
2. **Zero-block scan**: count 4K pages of all-zeros. Classic failing-media/torn-write signature. 0 on a corrupt file = NOT media, look software-side.
3. **Page-type histogram**: first byte of every page. Legal values: 0x02/0x05 interiors, 0x0a/0x0d leaves, 0x0d is a LEGAL table-leaf type (do not flag it as corruption), 0x10/0x11 freelist trunk/leaf, 0x00 freed, 0x53 only valid as byte 0 of page 1. Violations where a page must be one type (freelist chain, sqlite_master children) are the real signal.
4. **Decode "garbage" values**: big-endian misreads are usually structure, not random bits. E.g. "invalid page 218103808" = 0x0D000000 = the leaf-header `0d 00 00 00`; "rowid 2748779069441" = (448<<32)|1 = an in-page cell offset. Arithmetic sequences in decoded values = pages written to the wrong location (checkpoint/copy fault), NOT bit flips.
5. **Log timeline first**: the error CLASS changes tell the story. `file is not a database` (header non-SQLite, transient) → `malformed` = header was destroyed then partially recovered = a write landed on the file mid-operation. Cross-check agent.log compaction telemetry (`failure_class`, `total_duration_ms`) and `clone_for_agent` "disk I/O error" lines.
6. **Red herrings**: ingested messages can quote integrity_check output (search hits inside page payloads are not corruption); a "fresh" rebuilt db legitimately contains old session IDs if it was re-ingested from state.db.

Deployment facts that raised risk in this incident: 4 OS processes (gateway, webui, dashboard, cron worker) each open their own connection to the one WAL file; compaction holds one long write transaction (observed 229 s, and 153 s on 2026-10-01) during which checkpoint + mmap-invalidation races can land a bad page.

2026-10-01 update (corruption #5, `file is not a database`): the 2026-09-29 `PRAGMA mmap_size=0` hardening in `db_bootstrap.configure_connection()` was ACTIVE (verified mmap_size=0 on live conns, zero main-file mmaps) and did NOT prevent further corruption: a ~153 s compaction write transaction aborted with DatabaseError and destroyed the 16-byte header (magic gone; page-1 bytes `00 00 00 00 00 00 00 0a`, db_size_pages=2885) while ~1800 data/index pages remained intact-looking. Lesson: the mmap fix is necessary but NOT sufficient; the residual race is long-write-tx vs WAL auto-checkpoint (default 1000 pages) across 4 independent connections. Same-day recovery: rebuild from the latest clean snapshot + full `REINDEX` + checkpoint (lcm.db is a derived index; state.db is source of truth, nothing lost). Two new defenses now live: (1) **lcm_monitor v2** (`~/.hermes/lcm-monitor/lcm_monitor_v2.py`, supersedes v1): same 3s header / 15s md5 watching PLUS hourly known-good snapshots in `snapshots/` and guarded auto-restore (on magic loss or read-only integrity failure: stop hermes-gateway/webui/dashboard, swap in the latest good snapshot, restart; 3 h cooldown; corrupt file preserved as `lcm.db.corrupt-<ts>`; every step in monitor.jsonl). (2) `hermes-dashboard.service` now sets `LCM_DATABASE_PATH=/mnt/fastdata/lcm/lcm.db` (it previously ran unbound and silently fell back to the built-in compressor). Candidate next mitigations (NOT yet implemented; decide with Clint): raise `PRAGMA wal_autocheckpoint`, or batch compaction writes into smaller transactions. `PRAGMA mmap_size=0` remains the minimal, reversible hardening (rollback: set back to 268435456, restart the 3 services).

2026-10-19 update (corruption #7, `database disk image is malformed`, header intact, 2186 pages byte0=0x00): the trigger was NOT a write race. A reboot killed the process holding the exclusive flock on `<db>.lock`; the orphaned cron worker re-acquired it, so every fresh Hermes process timed out loading the plugin (10s load deadline, silent built-in-compressor fallback) while the main file had already suffered the wal_generation_guard double-checkpoint failure. LESSON: after ANY reboot, check `/proc/locks` for a stale flock owner on lcm.db.lock BEFORE blaming the database; kill the orphan worker (its cron respawns it) instead of restarting services, which only queues them behind the lock. Repair = guard v4 playbook: `.recover` -> build -> FTS rebuild + REINDEX + `wal_checkpoint(TRUNCATE)` -> integrity ok -> quarantine corrupt file by rename -> install salvage as a NEW inode (cp + atomic mv; GNU cp has no -c flag) -> clear sidecars -> restart writers -> re-verify. The plugin's `wal_generation_guard.py` (in place by 2026-10-19 20:01) pins the WAL generation behind open connections going forward.
