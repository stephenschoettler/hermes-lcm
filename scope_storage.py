"""Per-item scope storage, Teams setup backfill, and verification.

Stage one deliberately stores a nullable owner scope.  A NULL value is the
legacy/un-stamped state; the Teams setup hook is the only path that turns that
state into an owner attribution for historical rows.
"""

from __future__ import annotations

import ast
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .db_bootstrap import (
    SCOPE_MIGRATION_STEP,
    add_column_if_missing,
    ensure_metadata_table,
    mark_migration_step_complete,
)

ACCESS_SCOPE_COLUMN = "access_scope"
# Kept as a compatibility alias for callers of the first staging build.  All
# SQL and schema work uses the unambiguous name above.
SCOPE_COLUMN = ACCESS_SCOPE_COLUMN

# Rollup ``scope`` columns pre-date this staging and remain the durable
# partition key.  Each rollup table receives a separate nullable
# ``access_scope`` column.
SCOPE_BEARING_TABLES = (
    "messages",
    "summary_nodes",
    "lcm_rollups",
    "lcm_rollup_invalidations",
    "lcm_rollup_state",
    "lcm_embedding_meta",
    "lcm_embedding_vectors",
    "lcm_embedding_binary",
    "lcm_chunk_meta",
    "lcm_chunk_vectors",
    "lcm_chunk_binary",
)

_ACCESS_SCOPE_TABLES = (
    "messages",
    "summary_nodes",
    "lcm_rollups",
    "lcm_rollup_invalidations",
    "lcm_rollup_state",
    "lcm_embedding_meta",
    "lcm_embedding_vectors",
    "lcm_embedding_binary",
    "lcm_chunk_meta",
    "lcm_chunk_vectors",
    "lcm_chunk_binary",
)

_SESSION_SCOPE_TABLES = ("messages", "summary_nodes")
_OPTIONAL_SCOPE_TABLES = {
    "lcm_embedding_meta",
    "lcm_embedding_vectors",
    "lcm_embedding_binary",
    "lcm_chunk_meta",
    "lcm_chunk_vectors",
    "lcm_chunk_binary",
}

ScopeResolver = Callable[[str], str | None]


class ScopeBackfillIncompleteError(RuntimeError):
    """At least one table could not be fully stamped.

    Raised only AFTER every table has been attempted, so the report names what
    succeeded as well as what failed. Isolation is about not cancelling the
    remaining tables; it is not about letting a caller proceed as though the
    enable had worked.
    """

    def __init__(self, report: dict[str, object]) -> None:
        self.report = report
        failures = report.get("failures") or {}
        super().__init__(
            "scope backfill incomplete for "
            + ", ".join(f"{table} ({error})" for table, error in failures.items())
        )
_TEAMS_ENABLED_ATTRIBUTE = "lcm_" + "teams_enabled"


def teams_enabled(engine: object) -> bool:
    """Read the host's explicit Teams flag for storage-only decisions."""

    return bool(getattr(engine, _TEAMS_ENABLED_ATTRIBUTE, False))


def mark_teams_enabled(engine: object) -> None:
    """Publish the Teams setup completion flag after backfill succeeds."""

    setattr(engine, _TEAMS_ENABLED_ATTRIBUTE, True)


# The DURABLE record of the operator's decision. Deliberately not a
# lcm_migration_state step: those are append-only completion markers, and this
# has to be revocable by disable_teams. Deliberately not ``scope_v1`` either --
# that records "the access_scope columns exist", which ordinary bootstrap writes
# whether or not anyone enabled Teams.
TEAMS_ENABLED_METADATA_KEY = "teams_enabled_v1"


def read_persisted_teams_enabled(conn: sqlite3.Connection) -> bool | None:
    """Return the recorded decision, or None when none was ever recorded.

    Deliberately does NOT create the metadata table. This runs on the doctor
    path and on every storage bind, and a verification that mutates schema is
    not a verification -- it would materialise `metadata` on a header-only
    database and report a shape it had just produced itself.
    """

    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
    ).fetchone()
    if exists is None:
        return None
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?", (TEAMS_ENABLED_METADATA_KEY,)
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0]).strip().lower() in {"1", "true", "yes", "on"}


def persist_teams_enabled(conn: sqlite3.Connection, enabled: bool) -> None:
    """Record the operator's enable/disable decision so a restart keeps it."""

    ensure_metadata_table(conn)
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (TEAMS_ENABLED_METADATA_KEY, "true" if enabled else "false"),
    )
    conn.commit()


def access_scope_stamps_exist(conn: sqlite3.Connection) -> bool:
    """True when any row carries a non-NULL access_scope."""

    for table in _ACCESS_SCOPE_TABLES:
        try:
            row = conn.execute(
                f"SELECT 1 FROM {table} WHERE {ACCESS_SCOPE_COLUMN} IS NOT NULL LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            # Table or column absent on this store; nothing stamped there.
            continue
        if row is not None:
            return True
    return False


def resolve_startup_teams_state(conn: sqlite3.Connection) -> tuple[bool, str]:
    """Decide the Teams flag a freshly-bound engine must start with.

    Returns ``(teams_enabled, reason)``.

    The third case is the one that matters. ``enable_teams`` stamps rows in
    committed batches and only records the flag once the whole backfill
    succeeds, so an enable that dies partway -- one unresolvable owner is
    enough -- leaves per-owner stamps behind with no recorded decision. Reading
    that as "Teams is off" hands a permissive policy real scoped data, with no
    restart required and nothing in the logs.

    Reporting it as ENABLED is what makes it safe: with no context accessor
    wired, ``policy_for_engine`` resolves enabled-but-unwired to FailClosedPolicy
    rather than to the permissive default. The store refuses work until an
    operator finishes the enable or explicitly disables, which is the direction
    to fail in when the alternative is silently serving one principal's memory
    to another.
    """

    persisted = read_persisted_teams_enabled(conn)
    if persisted is True:
        return True, "enabled"
    if persisted is False:
        return False, "disabled"
    if access_scope_stamps_exist(conn):
        return True, "stamped-without-marker"
    return False, "never-enabled"


@dataclass(frozen=True)
class ScopeWriter:
    """One source-level INSERT into an access-scope-bearing table."""

    table: str
    path: str
    line: int
    function: str
    populates_access_scope: bool

    @property
    def populates_scope(self) -> bool:
        """Compatibility view for the first staging API."""

        return self.populates_access_scope

    @property
    def name(self) -> str:
        location = f"{self.path}:{self.line}"
        return f"{location}:{self.function}" if self.function else location


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _tables_missing_scope_column(
    conn: sqlite3.Connection,
    tables: Iterable[str],
) -> list[str]:
    names = tuple(dict.fromkeys(str(table) for table in tables))
    if not names:
        return []
    placeholders = ", ".join("?" for _ in names)
    rows = conn.execute(
        f"""
        SELECT table_schema.name
        FROM sqlite_master AS table_schema
        WHERE table_schema.type = 'table'
          AND table_schema.name IN ({placeholders})
          AND NOT EXISTS (
              SELECT 1
              FROM pragma_table_info(table_schema.name) AS column_info
              WHERE column_info.name = ?
          )
        """,
        (*names, ACCESS_SCOPE_COLUMN),
    ).fetchall()
    return [str(row[0]) for row in rows]


def ensure_scope_columns(
    conn: sqlite3.Connection,
    *,
    tables: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Add nullable ``access_scope`` to every materialized item table.

    The migration intentionally does not create optional feature tables.  A
    disabled store therefore keeps exactly the same table set; a later lazy
    feature initializer calls this function again after creating its tables.
    ``add_column_if_missing`` is the repository's race-tolerant ALTER idiom.
    The core pass is recorded as ``scope_v1`` after successful materialization;
    ``tables`` is reserved for targeted repairs of a table created after that
    marker (for example, a legacy chunk source table).
    """

    materialized_tables = _ACCESS_SCOPE_TABLES if tables is None else tuple(tables)
    try:
        marker = conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name = ? LIMIT 1",
            (SCOPE_MIGRATION_STEP,),
        ).fetchone()
    except sqlite3.OperationalError:
        marker = None
    if marker is not None and tables is None:
        # The marker is written only below, after every materialized table has
        # been checked/altered successfully. A marker-absent older database is
        # therefore still fully verified rather than assumed to be current.
        return {"added": [], "existing": [], "absent": []}

    if tables is None:
        # Current table definitions already declare the nullable column. One
        # correlated pragma query confirms that all materialized tables have it;
        # only an older/partial shape pays the detailed ALTER/rename sweep.
        try:
            missing_tables = _tables_missing_scope_column(conn, materialized_tables)
        except sqlite3.Error:
            missing_tables = list(materialized_tables)
        if not missing_tables:
            mark_migration_step_complete(conn, SCOPE_MIGRATION_STEP)
            return {"added": [], "existing": [], "absent": []}

    added: list[str] = []
    existing: list[str] = []
    absent: list[str] = []
    rollup_tables = {
        "lcm_rollups", "lcm_rollup_invalidations", "lcm_rollup_state"
    }
    for table in materialized_tables:
        if not _table_exists(conn, table):
            absent.append(table)
            continue
        columns = _table_columns(conn, table)
        # Databases created by the first, defective staging build used
        # ``scope`` on the non-rollup tables.  Rename that column in place so
        # its values survive the corrective migration.  Rollup ``scope`` is a
        # different, NOT NULL partition key and must never be renamed.
        if table not in rollup_tables and "scope" in columns and ACCESS_SCOPE_COLUMN not in columns:
            conn.execute(f'ALTER TABLE "{table}" RENAME COLUMN scope TO {ACCESS_SCOPE_COLUMN}')
            columns.remove("scope")
            columns.add(ACCESS_SCOPE_COLUMN)
            existing.append(table)
            continue
        if ACCESS_SCOPE_COLUMN in columns:
            existing.append(table)
            continue
        add_column_if_missing(
            conn,
            columns,
            ACCESS_SCOPE_COLUMN,
            f'ALTER TABLE "{table}" ADD COLUMN {ACCESS_SCOPE_COLUMN} TEXT',
        )
        added.append(table)
    if tables is None:
        mark_migration_step_complete(conn, SCOPE_MIGRATION_STEP)
    return {"added": added, "existing": existing, "absent": absent}


def _resolve_scope(resolver: ScopeResolver, session_id: object) -> str:
    normalized_session = str(session_id or "")
    if not normalized_session:
        raise ValueError("cannot stamp a scope-bearing row without session_id")
    value = resolver(normalized_session)
    if value is None or not str(value).strip():
        raise ValueError(f"pre-Teams owner is unresolved for session_id={normalized_session}")
    return str(value)


def compose_scope_resolver(
    owner_for_session: ScopeResolver,
    *,
    overrides: Mapping[str, str] | None = None,
    fallback_owner: str | None = None,
) -> ScopeResolver:
    """Wrap the host resolver with operator-supplied answers.

    The host resolver raises for any session it cannot attribute, and there was
    no way to answer for it: a single unattributable session was enough to make
    an enable impossible. The override map answers named sessions, the fallback
    answers the rest, and both are operator input rather than a guess made here.
    """

    named = {
        str(key): str(value)
        for key, value in (overrides or {}).items()
        if str(value).strip()
    }
    default = str(fallback_owner).strip() if fallback_owner else ""

    def resolve(session_id: str) -> str | None:
        if session_id in named:
            return named[session_id]
        resolved = owner_for_session(session_id)
        if resolved is not None and str(resolved).strip():
            return resolved
        return default or None

    return resolve


def preflight_teams_scope(
    conn: sqlite3.Connection,
    owner_for_session: ScopeResolver,
    *,
    overrides: Mapping[str, str] | None = None,
    fallback_owner: str | None = None,
) -> dict[str, object]:
    """Resolve every owner an enable would need, WITHOUT writing anything.

    Enable stamps in committed batches, so discovering an unattributable
    session halfway through leaves the store partly stamped -- the exact state
    Phase 1 has to fail closed on. This answers the question first: run it, read
    ``unresolvable``, supply an override map or a fallback owner for whatever it
    names, and only then enable.

    Writes nothing and creates nothing -- not the access_scope columns, not the
    metadata table. A preflight that mutates the store it is inspecting is not a
    preflight.
    """

    resolver = compose_scope_resolver(
        owner_for_session, overrides=overrides, fallback_owner=fallback_owner
    )
    tables: dict[str, dict[str, object]] = {}
    unresolvable: set[str] = set()
    blank_key_tables: list[str] = []

    keyed = [(table, "session_id") for table in _SESSION_SCOPE_TABLES]
    keyed += [
        (table, "scope")
        for table in ("lcm_rollups", "lcm_rollup_invalidations", "lcm_rollup_state")
    ]

    for table, key_column in keyed:
        if not _table_exists(conn, table):
            continue
        columns = _table_columns(conn, table)
        if key_column not in columns:
            continue
        # An un-migrated store has no access_scope column at all, and every row
        # then needs attribution -- not zero rows.
        predicate = (
            f"WHERE {ACCESS_SCOPE_COLUMN} IS NULL"
            if ACCESS_SCOPE_COLUMN in columns
            else ""
        )
        keys = [
            row[0]
            for row in conn.execute(
                f'SELECT DISTINCT "{key_column}" FROM "{table}" {predicate}'
            ).fetchall()
        ]
        blank = [key for key in keys if not str(key or "").strip()]
        if blank:
            blank_key_tables.append(table)
        table_unresolvable = sorted(
            {
                str(key)
                for key in keys
                if str(key or "").strip()
                and not str(resolver(str(key)) or "").strip()
            }
        )
        unresolvable.update(table_unresolvable)
        tables[table] = {
            "distinct_keys": len([k for k in keys if str(k or "").strip()]),
            "unresolvable": table_unresolvable,
            "blank_keys": len(blank),
        }

    ready = not unresolvable and not blank_key_tables
    return {
        "ready": ready,
        "unresolvable": sorted(unresolvable),
        "blank_key_tables": sorted(blank_key_tables),
        "tables": tables,
        "message": (
            "every owner resolves; enable can proceed"
            if ready
            else "supply an override map or fallback owner for the names above "
            "before enabling"
        ),
    }


def _backfill_session_table(
    conn: sqlite3.Connection,
    table: str,
    resolver: ScopeResolver,
    batch_size: int,
) -> int:
    updated = 0
    while True:
        rows = conn.execute(
            f'SELECT rowid, session_id FROM "{table}" '
            f"WHERE {ACCESS_SCOPE_COLUMN} IS NULL ORDER BY rowid LIMIT ?",
            (batch_size,),
        ).fetchall()
        if not rows:
            return updated
        values = [(_resolve_scope(resolver, row[1]), int(row[0])) for row in rows]
        conn.executemany(
            f'UPDATE "{table}" SET {ACCESS_SCOPE_COLUMN}=? WHERE rowid=? '
            f'AND {ACCESS_SCOPE_COLUMN} IS NULL', values
        )
        conn.commit()
        updated += len(values)


def _backfill_joined_table(
    conn: sqlite3.Connection,
    *,
    table: str,
    source_table: str,
    join_sql: str,
    batch_size: int,
) -> int:
    """Copy a source row's already-stamped scope to a derived item table."""

    updated = 0
    while True:
        rows = conn.execute(
            f"""
            SELECT target.rowid, source.access_scope
            FROM {table} AS target
            JOIN {source_table} AS source ON {join_sql}
            WHERE target.access_scope IS NULL AND source.access_scope IS NOT NULL
            ORDER BY target.rowid LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
        if not rows:
            return updated
        conn.executemany(
            f"UPDATE {table} SET {ACCESS_SCOPE_COLUMN}=? WHERE rowid=? "
            f"AND {ACCESS_SCOPE_COLUMN} IS NULL",
            [(str(row[1]), int(row[0])) for row in rows],
        )
        conn.commit()
        updated += len(rows)


def _backfill_rollup_table(
    conn: sqlite3.Connection,
    table: str,
    resolver: ScopeResolver,
    batch_size: int,
) -> int:
    """Attribute rollup bookkeeping from its unchanged partition key."""

    updated = 0
    while True:
        rows = conn.execute(
            f'SELECT rowid, scope FROM "{table}" '
            f"WHERE {ACCESS_SCOPE_COLUMN} IS NULL ORDER BY rowid LIMIT ?",
            (batch_size,),
        ).fetchall()
        if not rows:
            return updated
        values = [(_resolve_scope(resolver, row[1]), int(row[0])) for row in rows]
        conn.executemany(
            f'UPDATE "{table}" SET {ACCESS_SCOPE_COLUMN}=? WHERE rowid=? '
            f'AND {ACCESS_SCOPE_COLUMN} IS NULL',
            values,
        )
        conn.commit()
        updated += len(values)


def backfill_scopes(
    conn: sqlite3.Connection,
    owner_for_session: ScopeResolver | None = None,
    *,
    batch_size: int = 256,
    overrides: Mapping[str, str] | None = None,
    fallback_owner: str | None = None,
) -> dict[str, object]:
    """Stamp all pre-existing rows, leaving already-stamped rows untouched.

    The NULL predicate makes repeated runs idempotent.  Each batch commits on
    its own, so a crash after any batch leaves a deterministic NULL remainder
    that a later run resumes.  Derived rows copy their leaf's scope only after
    messages/summary nodes have been processed.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ensure_scope_columns(conn)
    if owner_for_session is None:
        raise ValueError(
            "owner_for_session is required to attribute pre-Teams rows to their owner"
        )
    resolver = compose_scope_resolver(
        owner_for_session, overrides=overrides, fallback_owner=fallback_owner
    )
    updated: dict[str, int] = {}
    failures: dict[str, str] = {}
    attempted: list[str] = []

    def _isolated(table: str, work: Callable[[], int]) -> None:
        """Run one table's backfill so its failure cannot silence the rest.

        The loop had no exception handling, so the first unattributable session
        aborted the whole run and every LATER table was never started -- not
        merely truncated. The operator saw one error and had no way to tell
        which tables had been attempted. This also matches the ratified
        converge rule: one failure must not abort the others.
        """

        attempted.append(table)
        try:
            updated[table] = work()
        except Exception as exc:  # noqa: BLE001 - recorded, then reported
            failures[table] = f"{type(exc).__name__}: {exc}"

    for table in _SESSION_SCOPE_TABLES:
        if _table_exists(conn, table):
            _isolated(
                table,
                lambda t=table: _backfill_session_table(conn, t, resolver, batch_size),
            )

    # Chunk metadata points at a raw message; vectors/binary point at metadata.
    if _table_exists(conn, "lcm_chunk_meta") and _table_exists(conn, "messages"):
        _isolated(
            "lcm_chunk_meta",
            lambda: _backfill_joined_table(
                conn,
                table="lcm_chunk_meta",
                source_table="messages",
                join_sql="source.store_id = target.store_id",
                batch_size=batch_size,
            ),
        )
    for table in ("lcm_chunk_vectors", "lcm_chunk_binary"):
        if _table_exists(conn, table) and _table_exists(conn, "lcm_chunk_meta"):
            _isolated(
                table,
                lambda t=table: _backfill_joined_table(
                    conn,
                    table=t,
                    source_table="lcm_chunk_meta",
                    join_sql="source.chunk_id = target.chunk_id AND source.identity_hash = target.identity_hash",
                    batch_size=batch_size,
                ),
            )

    # Rollups retain their session partition key in ``scope``.  Their access
    # scope is attributed from that key only during the explicit Teams setup.
    for table in ("lcm_rollups", "lcm_rollup_invalidations", "lcm_rollup_state"):
        if _table_exists(conn, table):
            _isolated(
                table,
                lambda t=table: _backfill_rollup_table(conn, t, resolver, batch_size),
            )

    # Summary embedding metadata points at a summary node; vector/binary rows
    # point at that metadata.  ``embedded_id`` is the persisted node id.
    if _table_exists(conn, "lcm_embedding_meta") and _table_exists(conn, "summary_nodes"):
        _isolated(
            "lcm_embedding_meta",
            lambda: _backfill_joined_table(
                conn,
                table="lcm_embedding_meta",
                source_table="summary_nodes",
                join_sql="source.node_id = CAST(target.embedded_id AS INTEGER)",
                batch_size=batch_size,
            ),
        )
    for table in ("lcm_embedding_vectors", "lcm_embedding_binary"):
        if _table_exists(conn, table) and _table_exists(conn, "lcm_embedding_meta"):
            _isolated(
                table,
                lambda t=table: _backfill_joined_table(
                    conn,
                    table=t,
                    source_table="lcm_embedding_meta",
                    join_sql="source.embedded_id = target.embedded_id AND source.identity_hash = target.identity_hash",
                    batch_size=batch_size,
                ),
            )

    conn.commit()
    report = {
        "updated": updated,
        "total_updated": sum(updated.values()),
        "attempted": attempted,
        "failures": failures,
        "complete": not failures,
    }
    if failures:
        # Every table was attempted first, so the report is complete -- but the
        # run still fails loudly. A caller that ignored a returned
        # complete=False would proceed as though the enable had worked, and the
        # whole point of isolating the tables was to make the failure MORE
        # legible, not optional.
        raise ScopeBackfillIncompleteError(report)
    return report


def setup_teams_scope(
    conn: sqlite3.Connection,
    owner_for_session: ScopeResolver | None = None,
    *,
    batch_size: int = 256,
    overrides: Mapping[str, str] | None = None,
    fallback_owner: str | None = None,
) -> dict[str, object]:
    """Run the additive schema stage and Teams-only historical backfill."""

    columns = ensure_scope_columns(conn)
    result = backfill_scopes(
        conn,
        owner_for_session,
        batch_size=batch_size,
        overrides=overrides,
        fallback_owner=fallback_owner,
    )
    result["columns"] = columns
    return result


def _constant_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else "{expr}"
            for value in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_text(node.left)
        right = _constant_text(node.right)
        if left is not None and right is not None:
            return left + right
    return None


_INSERT_RE = re.compile(
    r"\bINSERT(?:\s+OR\s+[A-Z]+)?\s+INTO\s+([\"`]?[A-Za-z_][A-Za-z0-9_]*[\"`]?)"
    r"\s*(?:\(([^)]*)\))?",
    re.IGNORECASE | re.DOTALL,
)


class _WriterVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, targets: set[str]) -> None:
        self.path = path
        self.targets = targets
        self.stack: list[str] = []
        self.writers: list[ScopeWriter] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Constant(self, node: ast.Constant) -> None:
        text = node.value if isinstance(node.value, str) else None
        if text:
            self._record(str(text), node.lineno)
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        text = _constant_text(node)
        if text:
            self._record(text, node.lineno)
        self.generic_visit(node)

    def _record(self, sql: str, line: int) -> None:
        for match in _INSERT_RE.finditer(sql):
            table = match.group(1).strip('"`').lower()
            if table not in self.targets:
                continue
            columns = match.group(2) or ""
            self.writers.append(
                ScopeWriter(
                    table=table,
                    path=self.path.as_posix(),
                    line=line,
                    function=".".join(self.stack),
                    populates_access_scope=bool(
                        re.search(r"\baccess_scope\b", columns, re.IGNORECASE)
                    ),
                )
            )


def _source_files(source_root: Path) -> Iterable[Path]:
    # "benchmarking" joins its two siblings; the -ing spelling was missed when
    # they were added. Benchmark harnesses are excluded because they build their
    # OWN throwaway databases -- scripts/benchmark_fast_scan.py literally
    # `CREATE TABLE messages` with a hand-rolled six-column schema that has no
    # access_scope column to populate, in a tempfile. Scanning them reports a
    # scope-bearing writer that can never touch a customer store.
    excluded = {"tests", "bench", "benchmarks", "benchmarking", "__pycache__"}
    for path in sorted(source_root.rglob("*.py")):
        if any(part.startswith(".venv") or part in excluded for part in path.parts):
            continue
        if path.name.startswith("benchmark_"):
            continue
        yield path


def enumerate_scope_writers(source_root: str | Path | None = None) -> list[ScopeWriter]:
    """Discover scope-bearing INSERT writers from current source text."""

    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parent
    writers: list[ScopeWriter] = []
    for path in _source_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        visitor = _WriterVisitor(path, set(SCOPE_BEARING_TABLES))
        visitor.visit(tree)
        writers.extend(visitor.writers)
    return writers


def verify_scope_storage(
    conn: sqlite3.Connection | None,
    *,
    teams_enabled: bool = True,
) -> dict[str, object]:
    """Doctor-style storage verification with a non-vacuity sentinel."""

    tables: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    observed_rows = 0
    if conn is None:
        return {
            "status": "fail",
            "message": "scope storage connection is not initialized",
            "tables": tables,
            "observed_rows": 0,
        }

    for table in SCOPE_BEARING_TABLES:
        if not _table_exists(conn, table):
            tables[table] = {
                "exists": False,
                "stamped": 0,
                "unstamped": 0,
                "total": 0,
            }
            continue
        columns = _table_columns(conn, table)
        if ACCESS_SCOPE_COLUMN not in columns:
            errors.append(f"{table}: missing access_scope column")
            tables[table] = {
                "exists": True,
                "stamped": 0,
                "unstamped": 0,
                "total": 0,
                "error": "missing access_scope column",
            }
            continue
        total = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        unstamped = int(
            conn.execute(
                f'SELECT COUNT(*) FROM "{table}" '
                f"WHERE {ACCESS_SCOPE_COLUMN} IS NULL"
            ).fetchone()[0]
        )
        stamped = total - unstamped
        observed_rows += total
        tables[table] = {
            "exists": True,
            "stamped": stamped,
            "unstamped": unstamped,
            "total": total,
        }

    unstamped_total = sum(int(item.get("unstamped", 0)) for item in tables.values())
    if teams_enabled and unstamped_total:
        errors.append(f"{unstamped_total} unstamped access-scope row(s)")

    if errors:
        status = "fail"
        message = "; ".join(errors)
    elif not teams_enabled:
        # ``teams_enabled`` is the CALLER's belief, and after an aborted enable
        # that belief is wrong in the dangerous direction. Ask the database
        # instead, or the doctor reports a reassuring "legacy-compatible" on a
        # store full of real per-owner stamps -- green in exactly the state you
        # would run the doctor to detect.
        _, persisted_reason = resolve_startup_teams_state(conn)
        if persisted_reason == "stamped-without-marker":
            status = "stamped-without-marker"
            message = (
                "access_scope stamps exist but no enable decision is recorded: "
                "an enable aborted partway. Complete the enable or disable "
                "Teams explicitly; until then the store fails closed."
            )
        elif persisted_reason == "disabled":
            status = "not-enabled"
            message = (
                "Teams explicitly disabled; existing access_scope stamps are "
                "retained so a later re-enable keeps its attribution"
            )
        else:
            status = "not-enabled"
            message = (
                "Teams not enabled: NULL access_scope values remain "
                "legacy-compatible"
            )
    elif observed_rows == 0:
        status = "nothing-to-verify"
        message = "nothing to verify: no scope-bearing rows were observed"
    else:
        status = "verified"
        message = "all observed scope-bearing rows are stamped"
    return {
        "status": status,
        "message": message,
        "tables": tables,
        "observed_rows": observed_rows,
        "unstamped_rows": unstamped_total,
    }


# Names used by callers that describe the operation as an explicit migration
# rather than a backfill.  Keep one implementation so idempotence/resumability
# cannot drift between entry points.
backfill_scope = backfill_scopes
verify_scope = verify_scope_storage
