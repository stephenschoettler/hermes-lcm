"""Cross-process write gate regression tests.

Context: LCM runs as several independent OS processes (gateway, web UI,
dashboard, cron workers) that each open their own write connection to a single
``lcm.db``. SQLite's free-page allocator is per-connection, so two processes
inside concurrent write transactions can each conclude the same physical page
is available and hand it to different tables. The observed failure mode was a
``messages`` overflow page also claimed as a ``metadata`` b-tree child, which
``PRAGMA integrity_check`` reports only after the fact.

The gate is an advisory ``flock`` on a ``<db>.lock`` sidecar held for the
duration of every write transaction, so one process writes at a time.

These tests pin the four behaviors the fix depends on:

1. two separate OS processes never overlap inside the gate
2. the gate is re-entrant within a process (no self-deadlock)
3. ``LCM_DISABLE_CROSS_PROCESS_LOCK`` restores the unfenced behavior
4. in-memory databases are skipped (nothing cross-process to coordinate)
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_child(script: str, env_extra: dict[str, str] | None = None) -> str:
    """Run a child process with the repo importable, returning its stdout."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT.parent)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"child failed: {proc.stderr}"
    return proc.stdout


# A child that takes the gate, records when it entered and left, and holds it
# long enough that an unfenced sibling would demonstrably overlap.
_WORKER = """
    import json, os, sys, time, sqlite3
    sys.path.insert(0, {repo!r})
    import db_bootstrap as dbb

    db = {db!r}
    tag = {tag!r}
    barrier = {barrier!r}

    conn = sqlite3.connect(db)

    # Start together so a missing fence actually produces an overlap.
    open(os.path.join(barrier, tag + ".ready"), "w").close()
    while len(os.listdir(barrier)) < 2:
        time.sleep(0.01)

    with dbb.gate_for_conn(conn):
        entered = time.time()
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO t(tag) VALUES (?)", (tag,))
        conn.execute("COMMIT")
        time.sleep(0.5)
        left = time.time()

    conn.close()
    print(json.dumps({{"tag": tag, "entered": entered, "left": left}}))
"""


def test_two_processes_never_overlap_inside_the_gate(tmp_path: Path) -> None:
    db = tmp_path / "lcm.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t(tag TEXT)")
    conn.commit()
    conn.close()

    barrier = tmp_path / "barrier"
    barrier.mkdir()

    procs = []
    for tag in ("a", "b"):
        script = _WORKER.format(
            repo=str(REPO_ROOT), db=str(db), tag=tag, barrier=str(barrier)
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", textwrap.dedent(script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(REPO_ROOT),
            )
        )

    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, f"child failed: {err}"
        results.append(json.loads(out.strip().splitlines()[-1]))

    first, second = sorted(results, key=lambda r: r["entered"])

    # The critical sections must be disjoint: the first holder must leave
    # before the second one enters. Without the gate both children enter
    # immediately after the barrier and this assertion fails.
    assert first["left"] <= second["entered"], (
        "gate failed to serialize writers: "
        f"{first['tag']} held [{first['entered']}, {first['left']}], "
        f"{second['tag']} entered at {second['entered']}"
    )

    # Both writes still landed.
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()


def test_gate_is_reentrant_within_a_process(tmp_path: Path) -> None:
    """Nested acquisition must not deadlock, and must write through."""
    db = tmp_path / "lcm.db"
    sys.path.insert(0, str(REPO_ROOT))
    import db_bootstrap as dbb

    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t(x INTEGER)")
    conn.commit()

    with dbb.gate_for_conn(conn):
        with dbb.gate_for_conn(conn):
            with dbb.gate_for_conn(conn):
                conn.execute("INSERT INTO t VALUES (1)")
                conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    conn.close()


def test_escape_hatch_disables_the_gate(tmp_path: Path) -> None:
    """The env var must restore pre-fix behavior for rollback.

    Asserted behaviorally rather than by type: ``gate_for_conn`` still hands
    back a real gate object, but a disabled gate takes no flock and therefore
    never creates the sidecar. The sidecar's absence is the observable proof
    that no cross-process lock was taken.
    """
    db = tmp_path / "lcm.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t(x INTEGER)")
    conn.commit()
    conn.close()

    out = _run_child(
        f"""
        import sys, sqlite3, json, os
        sys.path.insert(0, {str(REPO_ROOT)!r})
        import db_bootstrap as dbb
        conn = sqlite3.connect({str(db)!r})
        with dbb.gate_for_conn(conn):
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
        print(json.dumps({{
            "disabled": dbb.LCM_DISABLE_CROSS_PROCESS_LOCK,
            "sidecar": os.path.exists({str(db)!r} + ".lock"),
        }}))
        """,
        env_extra={"LCM_DISABLE_CROSS_PROCESS_LOCK": "1"},
    )
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["disabled"] is True
    assert payload["sidecar"] is False, "disabled gate must not take the flock"

    # The write still landed, and no sidecar was left behind.
    assert not (tmp_path / "lcm.db.lock").exists()
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    conn.close()


def test_in_memory_database_is_skipped(tmp_path: Path, monkeypatch) -> None:
    """``:memory:`` has nothing to coordinate and must not create a sidecar."""
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(REPO_ROOT))
    import db_bootstrap as dbb

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(x INTEGER)")

    gate = dbb.gate_for_conn(conn)
    with gate:
        conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    assert list(tmp_path.glob("*.lock")) == []
    conn.close()


def test_memory_path_factory_returns_null_gate(tmp_path: Path, monkeypatch) -> None:
    """The factory itself must skip ``:memory:`` (Path, not just str).

    ``Path(":memory:").parent / name`` resolves to ``./:memory:.lock``, so a
    caller that passes a Path directly (bypassing ``gate_for_conn``'s string
    guard) would otherwise drop a ``:memory:.lock`` file in the process's
    working directory every time the full test suite runs.
    """
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(REPO_ROOT))
    import db_bootstrap as dbb
    from pathlib import Path as _Path

    gate = dbb.cross_process_write_gate(_Path(":memory:"))
    assert type(gate).__name__ == "_NullGate"
    with gate:
        pass  # must be a valid context manager either way
    assert list(tmp_path.glob("*.lock")) == []
