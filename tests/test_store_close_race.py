"""Closing a store's shared SQLite connection must never race an in-flight query.

The long-lived LCM connections are opened ``check_same_thread=False`` and shared across
threads. Closing one while another thread is inside ``execute``/``fetch`` on it is a
use-after-free inside ``_sqlite3`` and can crash the interpreter (SIGSEGV). These tests run
the racing code in a subprocess and assert on its exit status, so a crash fails the test
instead of killing the pytest process. All data is synthetic.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

if "hermes_lcm" not in sys.modules:
    package = types.ModuleType("hermes_lcm")
    package.__path__ = [str(REPO_ROOT)]
    package.__package__ = "hermes_lcm"
    sys.modules["hermes_lcm"] = package


# Prepended to every subprocess script: make ``hermes_lcm`` importable from the flat repo root.
_BOOTSTRAP = textwrap.dedent(
    """
    import sys, types
    _pkg = types.ModuleType("hermes_lcm")
    _pkg.__path__ = [sys.argv.pop(1)]
    _pkg.__package__ = "hermes_lcm"
    sys.modules["hermes_lcm"] = _pkg
    try:
        from hermes_lcm.sqlite_util import StoreClosed
    except ImportError:  # tree without the fix: no typed error, so any exception counts as bad
        class StoreClosed(Exception):
            pass
    """
)

_LIFECYCLE_HAMMER = _BOOTSTRAP + textwrap.dedent(
    """
    import threading, time
    from hermes_lcm.lifecycle_state import LifecycleStateStore

    db = sys.argv[1]
    bad = []

    def binder(store, stop, i):
        n = 0
        while not stop.is_set():
            try:
                store.bind_session(f"s{i}-{n}")
                store.get_by_session(f"s{i}-{n}")
                store.get_by_conversation(f"s{i}-{n}")
            except StoreClosed:
                return
            except Exception as exc:
                bad.append(repr(exc))
                return
            n += 1

    for _round in range(int(sys.argv[2])):
        store = LifecycleStateStore(db)
        stop = threading.Event()
        threads = [threading.Thread(target=binder, args=(store, stop, i)) for i in range(8)]
        for t in threads:
            t.start()
        time.sleep(0.02)
        closer = threading.Thread(target=lambda: [store.close() for _ in range(50)])
        closer.start()
        closer.join()
        stop.set()
        for t in threads:
            t.join(10)
        if bad:
            break
    print("\\n".join(bad[:5]))
    sys.exit(1 if bad else 0)
    """
)

# Every long-lived LCM connection, reached the way diagnostics reach it (``store.connection``),
# raced against the owning store's close(). Exit 0 = only StoreClosed was raised.
_RAW_HANDLE_HAMMER = _BOOTSTRAP + textwrap.dedent(
    """
    import threading, time

    kind, db, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])

    def make():
        if kind == "dag":
            from hermes_lcm.dag import SummaryDAG
            return SummaryDAG(db)
        if kind == "rollup":
            from hermes_lcm.rollup_store import RollupStore
            return RollupStore(db)
        if kind == "vector":
            from hermes_lcm.vector_store import VectorStore
            return VectorStore(db)
        if kind == "query_views":
            from hermes_lcm.query_view_store import QueryViewStore
            return QueryViewStore(db)
        if kind == "assertions":
            from hermes_lcm.assertion_store import AssertionStore
            return AssertionStore(db)
        if kind == "lifecycle":
            from hermes_lcm.lifecycle_state import LifecycleStateStore
            return LifecycleStateStore(db)
        if kind == "messages":
            from hermes_lcm.store import MessageStore
            return MessageStore(db)
        raise SystemExit(f"unknown kind {kind}")

    # Dependent stores (query views, assertions) need the base schema the message store creates.
    from hermes_lcm.store import MessageStore as _MS
    _MS(db).close()

    bad = []

    def reader(conn, stop):
        while not stop.is_set():
            try:
                for _row in conn.execute("SELECT name FROM sqlite_master"):
                    pass
                conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
            except StoreClosed:
                return
            except Exception as exc:
                bad.append(repr(exc))
                return

    for _round in range(rounds):
        store = make()
        conn = store.connection
        stop = threading.Event()
        threads = [threading.Thread(target=reader, args=(conn, stop)) for _ in range(6)]
        for t in threads:
            t.start()
        time.sleep(0.01)
        store.close()
        stop.set()
        for t in threads:
            t.join(10)
        if bad:
            break
    print("\\n".join(bad[:5]))
    sys.exit(1 if bad else 0)
    """
)


def _run_hammer(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script, str(REPO_ROOT), *args],
        cwd=str(REPO_ROOT),
        env=dict(os.environ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )


def test_lifecycle_close_racing_binds_never_crashes_or_leaks_untyped_errors(tmp_path):
    proc = _run_hammer(_LIFECYCLE_HAMMER, str(tmp_path / "lcm.db"), "40")
    assert proc.returncode != -11, (
        f"SIGSEGV: lifecycle close() freed the connection under a live execute\n{proc.stderr[-2000:]}"
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}; an exception other than StoreClosed reached a lifecycle caller:\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )


def test_closed_lifecycle_store_raises_typed_error(tmp_path):
    from hermes_lcm.lifecycle_state import LifecycleStateStore
    from hermes_lcm.sqlite_util import StoreClosed

    store = LifecycleStateStore(tmp_path / "lcm.db")
    store.bind_session("s1")
    raw = store.connection
    store.close()
    store.close()  # idempotent
    assert store.connection is None
    for call in (
        lambda: store.get_by_session("s1"),
        lambda: store.get_by_conversation("s1"),
        lambda: store.bind_session("s2"),
        lambda: store.row_count(),
        lambda: store.clear_debt("s1"),
        lambda: store.delete_safe_rows_for_sessions({"s1"}),
        lambda: raw.execute("SELECT 1"),  # a handle taken before close() never reaches the freed connection
    ):
        with pytest.raises(StoreClosed):
            call()


def test_store_closed_is_a_sqlite_error():
    """Existing fail-open ``except sqlite3.Error`` callers keep catching use-after-close."""
    import sqlite3

    from hermes_lcm.sqlite_util import StoreClosed

    assert issubclass(StoreClosed, sqlite3.Error)


@pytest.mark.parametrize(
    "kind", ["dag", "rollup", "vector", "query_views", "assertions", "lifecycle", "messages"]
)
def test_raw_connection_handle_racing_store_close_is_memory_safe(tmp_path, kind):
    proc = _run_hammer(_RAW_HANDLE_HAMMER, kind, str(tmp_path / "lcm.db"), "15")
    assert proc.returncode != -11, (
        f"SIGSEGV: {kind} close() freed the connection under a live execute\n{proc.stderr[-2000:]}"
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}; {kind}: a raw-handle use racing close() raised something other "
        f"than StoreClosed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )


# -- source contract: no new unguarded long-lived connection ---------------------------------

# Connections opened, used and closed inside ONE function call on ONE thread: no other thread can
# reach them, so close() cannot race an execute. Anything stored on ``self`` or returned to a
# caller is shared and must be wrapped by ``sqlite_util.guard_connection``. Adding a row here is a
# review decision: say why the handle cannot escape its function.
_FUNCTION_LOCAL_CONNECTIONS = {
    ("aux_session.py", "_is_live_auxiliary_child_session"): "read-only state.db probe, closed in finally",
    ("aux_session.py", "_session_has_auxiliary_ancestor"): "read-only state.db probe, closed in finally",
    ("command.py", "_classify_schema_stamp"): "doctor read, closed before return",
    ("command.py", "_doctor_repair_schema_stamp_apply_text"): "doctor repair, closed before return",
    ("command.py", "_embedding_read_connection"): "context-managed read handle for one doctor command",
    ("db_bootstrap.py", "verify_temporal_rollup_schema"): "in-memory scratch DB",
    ("db_bootstrap.py", "_expected_assertion_schema_contract"): "in-memory scratch DB",
    ("db_bootstrap.py", "_run_background_integrity_scan"): "owned by the one background scan thread",
    ("db_bootstrap.py", "_dispatch_background_integrity_scan"): "claim row write, closed before return",
    ("engine.py", "_session_has_foreground_branch_marker"): "read-only state.db probe, closed in finally",
    ("lifecycle_state.py", "get_fragmentation_stats"): "read-only state.db probe, closed in finally",
    ("maintenance.py", "backup_database"): "backup destination, closed before return",
    ("maintenance.py", "rotate_backup_database"): "backup destination, closed before return",
    ("retrieval_core.py", "hydrate_chunk_hits"): "deadline-bounded read, closed in finally",
    ("retrieval_core.py", "hydrate_semantic_nodes"): "deadline-bounded read, closed in finally",
    ("tools.py", "invoke"): "deadline-bounded lcm_grep readers, closed in finally",
    ("tools.py", "_lcm_recall_summary_source_hits"): "deadline-bounded read, closed in finally",
    ("trajectory_store.py", "backup_to"): "backup destination, closed before return",
    ("vector_store.py", "_load_binary_with_deadline"): "per-call reader copy, closed in finally",
}


def _unguarded_connects(root: Path | None = None) -> list[str]:
    root = REPO_ROOT if root is None else root
    found = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        def is_guard(node):
            return isinstance(node, ast.Call) and getattr(
                node.func, "id", getattr(node.func, "attr", "")
            ) == "guard_connection"

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"
                and getattr(node.func.value, "id", "") == "sqlite3"
            ):
                continue
            if is_guard(parents.get(node)):
                continue
            func = node
            while func in parents and not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = parents[func]
            fname = getattr(func, "name", "<module>")
            # ``conn = sqlite3.connect(...)`` ... ``return guard_connection(conn)`` in the same function
            target = parents.get(node)
            name = (
                target.targets[0].id
                if isinstance(target, ast.Assign) and isinstance(target.targets[0], ast.Name)
                else None
            )
            if name and any(
                isinstance(r, ast.Return)
                and is_guard(r.value)
                and any(isinstance(a, ast.Name) and a.id == name for a in r.value.args)
                for r in ast.walk(func)
            ):
                continue
            rel = path.name
            if (rel, fname) in _FUNCTION_LOCAL_CONNECTIONS:
                continue
            found.append(f"{rel}:{node.lineno} in {fname}()")
    return found


def test_every_shared_lcm_connection_is_guarded():
    """A ``sqlite3.connect(`` in the engine modules is wrapped by ``guard_connection`` (its
    close() and every execute share one lock; use after close raises StoreClosed), or it is a
    reviewed function-local handle in ``_FUNCTION_LOCAL_CONNECTIONS``."""
    assert _unguarded_connects() == []


def test_source_contract_catches_an_unguarded_connect(tmp_path):
    """The contract is not vacuous: an unguarded shared connection in a new module is reported."""
    (tmp_path / "new_store.py").write_text(
        "import sqlite3\nclass S:\n    def __init__(self, p):\n"
        "        self._conn = sqlite3.connect(p, check_same_thread=False)\n",
        encoding="utf-8",
    )
    assert _unguarded_connects(tmp_path) == ["new_store.py:4 in __init__()"]
