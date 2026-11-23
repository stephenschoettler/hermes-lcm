# Recurring `disk I/O error` / malformed `lcm.db`: the WAL generation class

Load this when `lcm.db` corrupts repeatedly (hours apart), when `lcm_doctor` reports
`disk I/O error` or `vtable constructor failed: nodes_fts`, or before writing any tool that
inspects or repairs the database.

## Diagnose before touching anything

Run these first. The answer is usually in the first two.

```bash
# 1. Does any live process hold a DELETED sidecar? This is the smoking gun.
for p in /proc/[0-9]*; do pid=${p#/proc/}; for fd in $p/fd/*; do
  t=$(readlink "$fd" 2>/dev/null); case "$t" in *lcm.db*) echo "$pid $t";; esac; done; done 2>/dev/null | sort -u

# 2. Real damage, read-only so it cannot mutate sidecars.
sqlite3 "file:/mnt/fastdata/lcm/lcm.db?mode=ro" "PRAGMA integrity_check;"

# 3. Rule out the boring causes.
df -h / /mnt/fastdata; df -i / /mnt/fastdata
```

`-wal (deleted)` or `-shm (deleted)` against a live `lcm.db` inode, especially several distinct
deleted WAL inodes at once, means the WAL generation was destroyed under open connections.
`2nd reference to page N` from `integrity_check` is the resulting damage: two generations
checkpointed into one main file and a page got claimed twice.

Correlate quarantine timestamps with gateway restarts. `logs/gateway-exit-diag.log` is UTC while
`logs/agent.log` is local time; a 1:1 match across that offset is the confirmation.

## The mechanism

SQLite protects a live WAL generation with two POSIX advisory locks: SHARED on the main file's
lock range and the DMS byte of `-shm`. A sibling process can checkpoint and unlink `-wal`/`-shm`
at its close only after taking both EXCLUSIVE, so an open connection normally refuses it.

POSIX advisory locks are owned **per process and inode, not per descriptor**
(sqlite.org/howtocorrupt.html section 2.2). Any raw `open()`/`close()` of the database or its
`-shm` anywhere inside the holder process cancels every lock that process holds on the file. The
connection stays open but unprotected; the next foreign close unlinks the sidecars underneath it.
The survivor keeps committing into a deleted WAL inode while the next opener mints a fresh
generation on the same main file.

SQLite defends its *own* in-process handles by deferring descriptor closes per inode. Measured
consequence: a sqlite3 read-only probe **connection** is safe; a raw `os.open`/`os.close` is not.

Long-running sessions amplify this. A long session keeps a connection open across gateway
restarts, guaranteeing a surviving holder at every restart boundary.

## Rules for any LCM inspection or repair tool

- **Never unlink `-wal` or `-shm`.** Not while monitoring, not while repairing, not ever.
- **Never open the database read-write just to inspect it.** `mode=ro` works in every sidecar
  state; a read-write handle that happens to be the last registered connection unlinks the
  sidecars on close.
- **A SQLite exception is not corruption.** `disk I/O error` means the environment is degraded.
  Treating it as corruption and "repairing" re-poisons every live process, which is a
  self-sustaining loop with a period of a few hours.
- **Never `os.replace` the live inode while handles are open.** Systemd units are not the whole
  picture: an interactive TUI session is not a unit and survives every `systemctl stop` loop.
- Index-only damage (`wrong # of entries in index ...`) is repairable in place with `REINDEX`:
  no inode change, no unlink, no service interruption. Do not escalate it to a file swap.
- `lcm.db` is a derived index. `~/.hermes/state.db` is the source of truth.

## The installed defence

`plugins/hermes-lcm/wal_generation_guard.py` re-applies those two lock ranges as **open file
description** locks (`F_OFD_SETLK`), which belong to the description rather than the process, so a
stray `close()` cannot cancel them. It reuses core's `hermes_state_lockguard` (the same module
core uses for `state.db`) rather than reimplementing it.

- `db_bootstrap.configure_connection()` calls `arm(conn)`: the one choke point all 8 stores route
  through. The path is read off the connection via `PRAGMA database_list`.
- The 6 stores with a graceful close call `disarm(conn)` immediately before `conn.close()`.
  **This is not optional.** Without it the guard also refuses the legitimate close-time WAL reset,
  and `tests/test_crash_safe_wal.py` fails with `WAL still N bytes after close()`.
- Rollback: `LCM_DISABLE_WAL_GENERATION_GUARD=1`. No on-disk state, nothing to clean up.

Advisory locks cannot stop an external `unlink` or `os.replace`, and cannot heal a handle that is
already stranded. A poisoned process must reopen the file, which means a restart.

## Confirm the guard is armed, do not assume it

The guard needs core's `hermes_state_lockguard`. When a bare `import` misses, it falls back to
walking up from its own module path, `<hermes home>/plugins/hermes-lcm` ->
`<hermes home>/hermes-agent`. If that directory is absent the import fails, and **the failure is
deliberately silent**: `_lockguard` becomes None, `supported()` returns False, and the guard does
nothing while every call site still succeeds. The corruption class returns with no error anywhere.

After any install, move, or reorganisation, check it:

```bash
cd "$HERMES_HOME/plugins/hermes-lcm"
python3 -c "import sys; sys.path.insert(0,'.'); import wal_generation_guard as g; print(g.supported())"
```

`True` means armed. `False` means inert: the files are present and import cleanly, but nothing is
protected.

The same condition shows up in the test suite as **skips, not failures**, which is why the counts
matter more than the pass line:

| Result | Meaning |
|---|---|
| `25 passed, 0 skipped` | guard armed |
| `22 passed, 3 skipped`, reason `OFD locks unavailable on this host/runtime` | guard inert |

A skip is indistinguishable from a silent no-op, so treat those 3 skips on Linux as a failure of
the install, not a platform quirk.

Note on location: `scripts/install.sh` symlinks the checkout to
`<target root>/plugins/hermes-lcm`, and the fallback uses `abspath`, which preserves the symlink
path rather than resolving it. So a clone can live anywhere **provided it is installed through that
symlink**. Running the tests directly inside a bare clone that was never installed is the common
way to see the 3 skips and wrongly conclude the host lacks OFD support.

## Fixes that look right and are not

`mmap_size=0` and a cross-process write flock both serialise concurrent *writes*. SQLite already
serialises multi-process writers correctly through a shared WAL. Neither can stop the shared WAL
being destroyed, so neither ends the loop. If a proposed fix targets write concurrency, it is
aimed at the wrong failure.

## Running the plugin's tests

The plugin venv has no pytest; `/usr/bin/python3` does. `tests/conftest.py` registers the plugin
as the `hermes_lcm` package.

```bash
cd ~/.hermes/plugins/hermes-lcm && export TMPDIR=~/.hermes/cache/scratch
/usr/bin/python3 -m pytest tests/test_wal_generation_guard.py tests/test_crash_safe_wal.py -q --no-header
```

The full suite takes well over an hour; run it in the background. To prove a change introduced no
regression, A/B the whole suite against the kill switch and diff the failure sets rather than
assuming an isolated failure is yours.
