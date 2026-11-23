"""Same-database query-derived evidence views for demand-shaped V4 memory.

Raw messages remain authoritative.  This store persists only typed intent,
exact evidence dependencies, coverage watermarks, and deterministic traces.
It never stores a final narrative answer and never exposes derived rows through
ordinary message retrieval.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Iterator, Literal, NamedTuple, Sequence
import uuid

from .db_bootstrap import (
    configure_connection,
    mark_migration_step_complete,
    refuse_schema_version_too_new,
    run_versioned_migrations,
)


QUERY_VIEW_MIGRATION_STEP = "query_views_v1"
_MAX_INTENT_JSON_CHARS = 16_000
_MAX_MANIFEST_JSON_CHARS = 32_000
_MAX_TRACE_JSON_CHARS = 32_000
_MAX_DEPENDENCIES = 40
_MAX_QUOTE_CHARS = 2_400
_MAX_TEXT_CHARS = 512
_DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60
_BUILD_LEASE_SECONDS = 300
_ALLOWED_OPERATIONS = frozenset({
    "evidence_only",
    "date_interval",
    "date_filter",
    "count_distinct",
    "sum",
    "difference",
    "order",
    "latest_fact",
})
_ALLOWED_MANIFEST_KEYS = frozenset({
    "closed_slots",
    "open_slots",
    "operands",
    "retrieval_calls",
    "evidence_refs",
    "coverage",
})
_ALLOWED_TRACE_KEYS = frozenset({
    "operation",
    "result",
    "result_value",
    "unit",
    "citations",
    "entities",
    "evidence_dates",
    "steps",
})


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any, *, field: str, max_chars: int) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite JSON") from exc
    if len(encoded) > max_chars:
        raise ValueError(f"{field} exceeds {max_chars} characters")
    return encoded


def _bounded_key(value: Any, field: str, *, required: bool = False) -> str:
    text = " ".join(str(value or "").strip().casefold().split())
    if required and not text:
        raise ValueError(f"{field} is required")
    if len(text) > _MAX_TEXT_CHARS:
        raise ValueError(f"{field} exceeds {_MAX_TEXT_CHARS} characters")
    return text


def _bounded_text(value: Any, field: str) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) > _MAX_TEXT_CHARS:
        raise ValueError(f"{field} exceeds {_MAX_TEXT_CHARS} characters")
    return text


def _iso_day(value: Any, field: str) -> str:
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date") from exc


@dataclass(frozen=True)
class QueryViewIdentity:
    """Strict typed identity. Any material semantic change produces a miss."""

    intent_type: str
    operation: str = "evidence_only"
    subject_key: str = ""
    predicate_key: str = ""
    role_key: str = ""
    scope_key: str = ""
    conversation_id: str = ""
    unit: str = ""
    distinct_policy: str = ""
    requirements_digest: str = ""
    time_mode: Literal["none", "absolute", "relative"] = "none"
    question_anchor: str = ""
    window_start: str = ""
    window_end: str = ""
    policy_version: str = "v1"

    def normalized(self) -> "QueryViewIdentity":
        intent_type = _bounded_key(self.intent_type, "intent_type", required=True)
        operation = _bounded_key(self.operation, "operation", required=True)
        if operation not in _ALLOWED_OPERATIONS:
            raise ValueError(f"unsupported query-view operation: {operation}")
        time_mode = str(self.time_mode or "").strip().casefold()
        if time_mode not in {"none", "absolute", "relative"}:
            raise ValueError("time_mode must be none, absolute, or relative")
        anchor = (
            _iso_day(self.question_anchor, "question_anchor")
            if self.question_anchor
            else ""
        )
        start = _iso_day(self.window_start, "window_start") if self.window_start else ""
        end = _iso_day(self.window_end, "window_end") if self.window_end else ""
        if bool(start) != bool(end):
            raise ValueError("window_start and window_end must be supplied together")
        if start and date.fromisoformat(end) <= date.fromisoformat(start):
            raise ValueError("window_end must be after window_start")
        if time_mode == "relative" and (not anchor or not start or not end):
            raise ValueError("relative intent requires anchor and resolved window")
        if time_mode == "absolute" and (anchor or not start or not end):
            raise ValueError(
                "absolute intent requires a resolved window and no relative anchor"
            )
        if time_mode == "none" and (anchor or start or end):
            raise ValueError("time_mode=none cannot carry an anchor or window")
        digest = str(self.requirements_digest or "").strip().casefold()
        if digest and (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("requirements_digest must be a 64-character SHA-256 value")
        return QueryViewIdentity(
            intent_type=intent_type,
            operation=operation,
            subject_key=_bounded_key(self.subject_key, "subject_key"),
            predicate_key=_bounded_key(self.predicate_key, "predicate_key"),
            role_key=_bounded_key(self.role_key, "role_key"),
            scope_key=_bounded_key(self.scope_key, "scope_key"),
            conversation_id=_bounded_text(self.conversation_id, "conversation_id"),
            unit=_bounded_key(self.unit, "unit"),
            distinct_policy=_bounded_key(self.distinct_policy, "distinct_policy"),
            requirements_digest=digest,
            time_mode=time_mode,  # type: ignore[arg-type]
            question_anchor=anchor,
            window_start=start,
            window_end=end,
            policy_version=_bounded_key(
                self.policy_version, "policy_version", required=True
            ),
        )

    def as_dict(self) -> dict[str, str]:
        normalized = self.normalized()
        return {
            "intent_type": normalized.intent_type,
            "operation": normalized.operation,
            "subject_key": normalized.subject_key,
            "predicate_key": normalized.predicate_key,
            "role_key": normalized.role_key,
            "scope_key": normalized.scope_key,
            "conversation_id": normalized.conversation_id,
            "unit": normalized.unit,
            "distinct_policy": normalized.distinct_policy,
            "requirements_digest": normalized.requirements_digest,
            "time_mode": normalized.time_mode,
            "question_anchor": normalized.question_anchor,
            "window_start": normalized.window_start,
            "window_end": normalized.window_end,
            "policy_version": normalized.policy_version,
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(
            self.as_dict(), field="identity", max_chars=_MAX_INTENT_JSON_CHARS
        )

    @property
    def view_id(self) -> str:
        return _sha256_text(self.canonical_json)


@dataclass(frozen=True)
class QueryEvidenceDependency:
    source_store_id: int
    source_content_sha256: str
    source_session_id: str
    source_conversation_id: str
    source_name: str
    source_role: str
    source_timestamp: float
    span_start: int
    span_end: int
    quote: str
    assertion_id: str = ""

    @property
    def citation(self) -> str:
        return f"lcm:{self.source_store_id}:{self.span_start}-{self.span_end}"


class CorpusSnapshot(NamedTuple):
    generation: int
    max_store_id: int
    row_count: int


class QueryViewBuildToken(NamedTuple):
    view_id: str
    generation: int
    nonce: str
    base_version: int
    corpus_generation: int
    coverage_store_id: int


@dataclass(frozen=True)
class QueryViewLookup:
    status: Literal[
        "hit",
        "miss",
        "delta_required",
        "expired",
        "building",
        "failed",
        "incomplete",
    ]
    reason: str = ""
    view: dict[str, Any] | None = None
    delta_events: tuple[dict[str, Any], ...] = ()
    delta_truncated: bool = False


class QueryViewBuildInProgressError(RuntimeError):
    """Raised when another live builder already owns an intent lease."""


_REQUIRED_SCHEMA: dict[str, frozenset[str]] = {
    "lcm_query_corpus_state": frozenset({
        "singleton", "generation", "max_store_id", "row_count", "changed_at"
    }),
    "lcm_query_corpus_events": frozenset({
        "generation", "store_id", "mutation", "session_id", "conversation_id",
        "source", "role", "source_timestamp", "changed_at"
    }),
    "lcm_query_views": frozenset({
        "view_id", "identity_json", "status", "generation", "current_version",
        "build_nonce", "lease_expires_at", "stale_reason", "hit_count",
        "promotion_status", "created_at", "updated_at", "expires_at"
    }),
    "lcm_query_view_versions": frozenset({
        "view_id", "version", "manifest_json", "trace_json", "completeness",
        "search_policy_version", "corpus_generation", "coverage_store_id",
        "published_at", "expires_at", "supersedes_version"
    }),
    "lcm_query_view_sources": frozenset({
        "view_id", "version", "source_store_id", "source_content_sha256",
        "source_session_id", "source_conversation_id", "source_role",
        "source_name", "source_timestamp", "span_start", "span_end", "quote",
        "assertion_id"
    }),
}


def _ensure_query_view_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS lcm_query_corpus_state (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
            max_store_id INTEGER NOT NULL DEFAULT 0 CHECK(max_store_id >= 0),
            row_count INTEGER NOT NULL DEFAULT 0 CHECK(row_count >= 0),
            changed_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lcm_query_corpus_events (
            generation INTEGER PRIMARY KEY CHECK(generation > 0),
            store_id INTEGER NOT NULL CHECK(store_id > 0),
            mutation TEXT NOT NULL CHECK(mutation IN ('insert', 'update', 'delete')),
            session_id TEXT NOT NULL DEFAULT '',
            conversation_id TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT '',
            source_timestamp REAL,
            changed_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lcm_query_views (
            view_id TEXT PRIMARY KEY CHECK(length(view_id) = 64),
            identity_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN (
                'ready', 'stale', 'building', 'failed', 'expired'
            )),
            generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
            current_version INTEGER NOT NULL DEFAULT 0 CHECK(current_version >= 0),
            build_nonce TEXT NOT NULL DEFAULT '',
            lease_expires_at REAL,
            stale_reason TEXT NOT NULL DEFAULT '',
            hit_count INTEGER NOT NULL DEFAULT 0 CHECK(hit_count >= 0),
            promotion_status TEXT NOT NULL DEFAULT 'probationary'
                CHECK(promotion_status IN ('probationary', 'promoted')),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            expires_at REAL
        );

        CREATE TABLE IF NOT EXISTS lcm_query_view_versions (
            view_id TEXT NOT NULL,
            version INTEGER NOT NULL CHECK(version > 0),
            manifest_json TEXT NOT NULL,
            trace_json TEXT NOT NULL DEFAULT '',
            completeness TEXT NOT NULL CHECK(completeness IN ('complete', 'partial')),
            search_policy_version TEXT NOT NULL,
            corpus_generation INTEGER NOT NULL CHECK(corpus_generation >= 0),
            coverage_store_id INTEGER NOT NULL CHECK(coverage_store_id >= 0),
            published_at REAL NOT NULL,
            expires_at REAL,
            supersedes_version INTEGER,
            PRIMARY KEY(view_id, version)
        );

        CREATE TABLE IF NOT EXISTS lcm_query_view_sources (
            view_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            source_store_id INTEGER NOT NULL CHECK(source_store_id > 0),
            source_content_sha256 TEXT NOT NULL CHECK(length(source_content_sha256) = 64),
            source_session_id TEXT NOT NULL,
            source_conversation_id TEXT NOT NULL DEFAULT '',
            source_name TEXT NOT NULL DEFAULT '',
            source_role TEXT NOT NULL,
            source_timestamp REAL NOT NULL,
            span_start INTEGER NOT NULL CHECK(span_start >= 0),
            span_end INTEGER NOT NULL CHECK(span_end > span_start),
            quote TEXT NOT NULL,
            assertion_id TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(view_id, version, source_store_id, span_start, span_end)
        );

        CREATE INDEX IF NOT EXISTS idx_lcm_query_view_sources_store
            ON lcm_query_view_sources(source_store_id, view_id, version);
        CREATE INDEX IF NOT EXISTS idx_lcm_query_view_versions_coverage
            ON lcm_query_view_versions(corpus_generation, view_id, version);
        CREATE INDEX IF NOT EXISTS idx_lcm_query_views_lifecycle
            ON lcm_query_views(status, expires_at, updated_at);

        CREATE TRIGGER IF NOT EXISTS lcm_query_corpus_message_insert
        AFTER INSERT ON messages
        BEGIN
            UPDATE lcm_query_corpus_state
               SET generation = generation + 1,
                   max_store_id = max(max_store_id, NEW.store_id),
                   row_count = row_count + 1,
                   changed_at = CAST(strftime('%s','now') AS REAL)
             WHERE singleton = 1;
            INSERT INTO lcm_query_corpus_events(
                generation, store_id, mutation, session_id, conversation_id,
                source, role, source_timestamp, changed_at
            ) SELECT generation, NEW.store_id, 'insert', NEW.session_id,
                     NEW.conversation_id, NEW.source, NEW.role, NEW.timestamp,
                     changed_at
                FROM lcm_query_corpus_state WHERE singleton = 1;
        END;

        CREATE TRIGGER IF NOT EXISTS lcm_query_corpus_message_update
        AFTER UPDATE OF content, session_id, conversation_id, source, role, timestamp
        ON messages
        BEGIN
            UPDATE lcm_query_corpus_state
               SET generation = generation + 1,
                   max_store_id = max(max_store_id, NEW.store_id),
                   changed_at = CAST(strftime('%s','now') AS REAL)
             WHERE singleton = 1;
            INSERT INTO lcm_query_corpus_events(
                generation, store_id, mutation, session_id, conversation_id,
                source, role, source_timestamp, changed_at
            ) SELECT generation, NEW.store_id, 'update', NEW.session_id,
                     NEW.conversation_id, NEW.source, NEW.role, NEW.timestamp,
                     changed_at
                FROM lcm_query_corpus_state WHERE singleton = 1;
            UPDATE lcm_query_views
               SET status = 'stale', generation = generation + 1,
                   stale_reason = 'positive_source_updated',
                   build_nonce = '', lease_expires_at = NULL,
                   updated_at = CAST(strftime('%s','now') AS REAL)
             WHERE status IN ('ready', 'building')
               AND view_id IN (
                   SELECT dependency.view_id
                     FROM lcm_query_view_sources AS dependency
                    WHERE dependency.source_store_id = OLD.store_id
                      AND dependency.version = lcm_query_views.current_version
               );
        END;

        CREATE TRIGGER IF NOT EXISTS lcm_query_corpus_message_delete
        AFTER DELETE ON messages
        BEGIN
            UPDATE lcm_query_corpus_state
               SET generation = generation + 1,
                   row_count = max(0, row_count - 1),
                   changed_at = CAST(strftime('%s','now') AS REAL)
             WHERE singleton = 1;
            INSERT INTO lcm_query_corpus_events(
                generation, store_id, mutation, session_id, conversation_id,
                source, role, source_timestamp, changed_at
            ) SELECT generation, OLD.store_id, 'delete', OLD.session_id,
                     OLD.conversation_id, OLD.source, OLD.role, OLD.timestamp,
                     changed_at
                FROM lcm_query_corpus_state WHERE singleton = 1;
            UPDATE lcm_query_views
               SET status = 'stale', generation = generation + 1,
                   stale_reason = 'positive_source_deleted',
                   build_nonce = '', lease_expires_at = NULL,
                   updated_at = CAST(strftime('%s','now') AS REAL)
             WHERE status IN ('ready', 'building')
               AND view_id IN (
                   SELECT dependency.view_id
                     FROM lcm_query_view_sources AS dependency
                    WHERE dependency.source_store_id = OLD.store_id
                      AND dependency.version = lcm_query_views.current_version
               );
        END;
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO lcm_query_corpus_state(
            singleton, generation, max_store_id, row_count, changed_at
        )
        SELECT 1, 0, coalesce(max(store_id), 0), count(*), ? FROM messages
        """,
        (time.time(),),
    )


def _verify_query_view_schema(conn: sqlite3.Connection) -> list[str]:
    missing: list[str] = []
    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE 'lcm_query%'"
        )
    }
    missing.extend(
        f"unexpected-table:{table}"
        for table in sorted(tables - set(_REQUIRED_SCHEMA))
    )
    for table, columns in _REQUIRED_SCHEMA.items():
        actual = {
            str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if not actual:
            missing.append(f"table:{table}")
        else:
            missing.extend(
                f"column:{table}.{column}" for column in sorted(columns - actual)
            )
            missing.extend(
                f"unexpected-column:{table}.{column}"
                for column in sorted(actual - columns)
            )
    required_objects = {
        "index": {
            "idx_lcm_query_view_sources_store",
            "idx_lcm_query_view_versions_coverage",
            "idx_lcm_query_views_lifecycle",
        },
        "trigger": {
            "lcm_query_corpus_message_insert",
            "lcm_query_corpus_message_update",
            "lcm_query_corpus_message_delete",
        },
    }
    for object_type, names in required_objects.items():
        actual = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = ?", (object_type,)
            )
        }
        missing.extend(
            f"{object_type}:{name}" for name in sorted(names - actual)
        )
        # Real index names carry the idx_ prefix (idx_lcm_query_*); gating on
        # the bare table prefix would make the index half of this fail-closed
        # check unable to match any legitimately-named extra index.
        gate_prefix = "idx_lcm_query" if object_type == "index" else "lcm_query"
        gated = {name for name in actual if name.startswith(gate_prefix)}
        missing.extend(
            f"unexpected-{object_type}:{name}"
            for name in sorted(gated - names)
        )
    return missing


class QueryViewStore:
    """Versioned, exact-provenance materialized evidence views."""

    def __init__(self, db_path: str | Path, *, db_lock: Any | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path), timeout=5.0, check_same_thread=False
        )
        self._write_lock = db_lock or threading.RLock()
        try:
            refuse_schema_version_too_new(self._conn)
            configure_connection(self._conn)
            self._conn.row_factory = sqlite3.Row
            run_versioned_migrations(self._conn)
            _ensure_query_view_schema(self._conn)
            missing = _verify_query_view_schema(self._conn)
            if missing:
                raise RuntimeError(
                    "query-view schema incomplete after ensure: " + ", ".join(missing)
                )
            mark_migration_step_complete(self._conn, QUERY_VIEW_MIGRATION_STEP)
            self._conn.commit()
        except Exception as exc:
            self._conn.close()
            self._conn = None
            if isinstance(exc, sqlite3.Error):
                raise RuntimeError(f"query-view schema ensure failed: {exc}") from exc
            raise

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        with self._write_lock:
            if self._conn is None:
                raise RuntimeError("query-view store is closed")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _now() -> float:
        return time.time()

    def corpus_snapshot(self) -> CorpusSnapshot:
        row = self._conn.execute(
            "SELECT generation, max_store_id, row_count "
            "FROM lcm_query_corpus_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("query corpus state is unavailable")
        return CorpusSnapshot(
            int(row["generation"]),
            int(row["max_store_id"]),
            int(row["row_count"]),
        )

    def snapshot_dependency(
        self,
        source_store_id: int,
        span_start: int,
        span_end: int,
        quote: str,
        *,
        assertion_id: str = "",
    ) -> QueryEvidenceDependency:
        store_id = int(source_store_id)
        start = int(span_start)
        end = int(span_end)
        exact_quote = str(quote or "")
        if not exact_quote or len(exact_quote) > _MAX_QUOTE_CHARS:
            raise ValueError(
                f"dependency quote must contain 1..{_MAX_QUOTE_CHARS} characters"
            )
        row = self._conn.execute(
            "SELECT store_id, session_id, conversation_id, source, role, "
            "content, timestamp FROM messages WHERE store_id = ?",
            (store_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"source store_id {store_id} does not exist")
        content = str(row["content"] or "")
        if start < 0 or end <= start or end > len(content):
            raise ValueError("dependency span is outside the exact source row")
        if content[start:end] != exact_quote:
            raise ValueError("dependency quote does not match the exact source span")
        normalized_assertion = str(assertion_id or "").strip().casefold()
        if normalized_assertion:
            if len(normalized_assertion) != 64 or any(
                character not in "0123456789abcdef" for character in normalized_assertion
            ):
                raise ValueError("assertion_id must be a 64-character SHA-256 value")
            assertion = self._conn.execute(
                "SELECT source_store_id, source_span_start, source_span_end, source_quote "
                "FROM lcm_assertions WHERE assertion_id = ?",
                (normalized_assertion,),
            ).fetchone()
            if assertion is None or (
                int(assertion["source_store_id"]) != store_id
                or int(assertion["source_span_start"]) != start
                or int(assertion["source_span_end"]) != end
                or str(assertion["source_quote"]) != exact_quote
            ):
                raise ValueError("assertion_id does not match the exact dependency")
        timestamp = float(row["timestamp"])
        if not math.isfinite(timestamp):
            raise ValueError("source timestamp must be finite")
        return QueryEvidenceDependency(
            source_store_id=store_id,
            source_content_sha256=_sha256_text(content),
            source_session_id=str(row["session_id"]),
            source_conversation_id=str(row["conversation_id"] or ""),
            source_name=str(row["source"] or ""),
            source_role=str(row["role"]),
            source_timestamp=timestamp,
            span_start=start,
            span_end=end,
            quote=exact_quote,
            assertion_id=normalized_assertion,
        )

    def claim_build(
        self,
        identity: QueryViewIdentity,
        *,
        corpus_snapshot: CorpusSnapshot | None = None,
    ) -> QueryViewBuildToken:
        canonical = identity.canonical_json
        view_id = identity.view_id
        snapshot = corpus_snapshot or self.corpus_snapshot()
        nonce = uuid.uuid4().hex
        now = self._now()
        lease = now + _BUILD_LEASE_SECONDS
        with self._write_transaction():
            self._conn.execute(
                """
                INSERT INTO lcm_query_views(
                    view_id, identity_json, status, generation, current_version,
                    build_nonce, lease_expires_at, created_at, updated_at
                ) VALUES(?, ?, 'building', 0, 0, ?, ?, ?, ?)
                ON CONFLICT(view_id) DO UPDATE SET
                    status = 'building',
                    generation = lcm_query_views.generation + 1,
                    build_nonce = excluded.build_nonce,
                    lease_expires_at = excluded.lease_expires_at,
                    stale_reason = '',
                    updated_at = excluded.updated_at
                WHERE lcm_query_views.status <> 'building'
                   OR coalesce(lcm_query_views.lease_expires_at, 0) <= ?
                """,
                (view_id, canonical, nonce, lease, now, now, now),
            )
            row = self._conn.execute(
                "SELECT identity_json, generation, current_version, build_nonce "
                "FROM lcm_query_views WHERE view_id = ?",
                (view_id,),
            ).fetchone()
            if str(row["identity_json"]) != canonical:
                raise RuntimeError("query-view hash collision or identity corruption")
            if str(row["build_nonce"]) != nonce:
                raise QueryViewBuildInProgressError(
                    "query-view build already has an active lease"
                )
        return QueryViewBuildToken(
            view_id,
            int(row["generation"]),
            str(row["build_nonce"]),
            int(row["current_version"]),
            snapshot.generation,
            snapshot.max_store_id,
        )

    @staticmethod
    def _validated_payloads(
        manifest: Any,
        computation_trace: Any,
    ) -> tuple[str, str]:
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        forbidden_keys = {"answer", "final_answer", "response", "prose"}

        def _forbidden(value: Any) -> bool:
            if isinstance(value, dict):
                if forbidden_keys & {str(key).casefold() for key in value}:
                    return True
                return any(_forbidden(item) for item in value.values())
            if isinstance(value, list):
                return any(_forbidden(item) for item in value)
            return False

        if _forbidden(manifest):
            raise ValueError("manifest may not cache final prose or response fields")
        unknown = set(manifest) - _ALLOWED_MANIFEST_KEYS
        if unknown:
            raise ValueError(
                "manifest contains unsupported or prose-bearing fields: "
                + ", ".join(sorted(str(value) for value in unknown))
            )
        manifest_json = _canonical_json(
            manifest, field="manifest", max_chars=_MAX_MANIFEST_JSON_CHARS
        )
        if computation_trace in (None, {}):
            return manifest_json, ""
        if not isinstance(computation_trace, dict):
            raise ValueError("computation_trace must be an object")
        unknown_trace = set(computation_trace) - _ALLOWED_TRACE_KEYS
        if unknown_trace:
            raise ValueError(
                "computation_trace contains non-trace fields: "
                + ", ".join(sorted(str(value) for value in unknown_trace))
            )
        trace_json = _canonical_json(
            computation_trace,
            field="computation_trace",
            max_chars=_MAX_TRACE_JSON_CHARS,
        )
        return manifest_json, trace_json

    def _validate_dependency_current(
        self, dependency: QueryEvidenceDependency
    ) -> None:
        row = self._conn.execute(
            "SELECT session_id, conversation_id, source, role, content, timestamp "
            "FROM messages WHERE store_id = ?",
            (int(dependency.source_store_id),),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"dependency source {dependency.source_store_id} was deleted"
            )
        content = str(row["content"] or "")
        if _sha256_text(content) != dependency.source_content_sha256:
            raise ValueError(
                f"dependency source {dependency.source_store_id} hash changed"
            )
        if content[dependency.span_start:dependency.span_end] != dependency.quote:
            raise ValueError(
                f"dependency source {dependency.source_store_id} span changed"
            )
        if (
            str(row["session_id"]) != dependency.source_session_id
            or str(row["conversation_id"] or "")
            != dependency.source_conversation_id
            or str(row["source"] or "") != dependency.source_name
            or str(row["role"]) != dependency.source_role
            or float(row["timestamp"]) != dependency.source_timestamp
        ):
            raise ValueError(
                f"dependency source {dependency.source_store_id} metadata changed"
            )
        if dependency.assertion_id:
            assertion = self._conn.execute(
                "SELECT 1 FROM lcm_assertions AS assertion "
                "JOIN lcm_assertion_sources AS source "
                "ON source.source_store_id = assertion.source_store_id "
                "AND source.extraction_version = assertion.extraction_version "
                "AND source.source_content_sha256 = assertion.source_content_sha256 "
                "WHERE assertion.assertion_id = ? AND source.invalidated_at IS NULL",
                (dependency.assertion_id,),
            ).fetchone()
            if assertion is None:
                raise ValueError(
                    f"dependency assertion {dependency.assertion_id} is not source-valid"
                )

    def publish_ready(
        self,
        token: QueryViewBuildToken,
        *,
        dependencies: Sequence[QueryEvidenceDependency],
        manifest: dict[str, Any],
        computation_trace: dict[str, Any] | None = None,
        completeness: Literal["complete", "partial"] = "complete",
        search_policy_version: str = "v1",
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> bool:
        unique_dependencies = list(dict.fromkeys(dependencies))
        if not 1 <= len(unique_dependencies) <= _MAX_DEPENDENCIES:
            raise ValueError(
                f"dependencies must contain between 1 and {_MAX_DEPENDENCIES} exact refs"
            )
        if completeness not in {"complete", "partial"}:
            raise ValueError("completeness must be complete or partial")
        policy = _bounded_key(
            search_policy_version, "search_policy_version", required=True
        )
        ttl = int(ttl_seconds)
        if not 1 <= ttl <= 30 * 24 * 60 * 60:
            raise ValueError("ttl_seconds must be between 1 and 2592000")
        manifest_json, trace_json = self._validated_payloads(
            manifest, computation_trace
        )
        expected_refs = {dependency.citation for dependency in unique_dependencies}
        supplied_refs = manifest.get("evidence_refs")
        if not isinstance(supplied_refs, list) or {
            str(value) for value in supplied_refs
        } != expected_refs:
            raise ValueError("manifest evidence_refs must exactly match dependencies")
        open_slots = manifest.get("open_slots", [])
        if not isinstance(open_slots, list):
            raise ValueError("manifest open_slots must be an array")
        if completeness == "complete" and open_slots:
            raise ValueError("complete view cannot retain open evidence slots")
        if computation_trace:
            trace_refs = computation_trace.get("citations", [])
            if not isinstance(trace_refs, list) or not {
                str(value) for value in trace_refs
            }.issubset(expected_refs):
                raise ValueError(
                    "computation trace citations must be exact view dependencies"
                )
        now = self._now()
        expires_at = now + ttl
        with self._write_transaction():
            owner = self._conn.execute(
                "SELECT status, generation, current_version, build_nonce "
                "FROM lcm_query_views WHERE view_id = ?",
                (token.view_id,),
            ).fetchone()
            if owner is None:
                raise ValueError(f"unknown query view: {token.view_id}")
            if (
                str(owner["status"]) != "building"
                or int(owner["generation"]) != int(token.generation)
                or str(owner["build_nonce"]) != str(token.nonce)
            ):
                return False
            current_corpus = self.corpus_snapshot()
            if current_corpus.generation != int(token.corpus_generation):
                self._conn.execute(
                    "UPDATE lcm_query_views SET status='stale', generation=generation+1, "
                    "stale_reason='corpus_changed_during_build', build_nonce='', "
                    "lease_expires_at=NULL, updated_at=? WHERE view_id=? "
                    "AND generation=? AND build_nonce=? AND status='building'",
                    (now, token.view_id, token.generation, token.nonce),
                )
                return False
            for dependency in unique_dependencies:
                self._validate_dependency_current(dependency)
            base_version = int(owner["current_version"])
            if base_version != int(token.base_version):
                return False
            version = base_version + 1
            self._conn.execute(
                """
                INSERT INTO lcm_query_view_versions(
                    view_id, version, manifest_json, trace_json, completeness,
                    search_policy_version, corpus_generation, coverage_store_id,
                    published_at, expires_at, supersedes_version
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token.view_id,
                    version,
                    manifest_json,
                    trace_json,
                    completeness,
                    policy,
                    current_corpus.generation,
                    current_corpus.max_store_id,
                    now,
                    expires_at,
                    base_version or None,
                ),
            )
            self._conn.executemany(
                """
                INSERT INTO lcm_query_view_sources(
                    view_id, version, source_store_id, source_content_sha256,
                    source_session_id, source_conversation_id, source_name, source_role,
                    source_timestamp, span_start, span_end, quote, assertion_id
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        token.view_id,
                        version,
                        dependency.source_store_id,
                        dependency.source_content_sha256,
                        dependency.source_session_id,
                        dependency.source_conversation_id,
                        dependency.source_name,
                        dependency.source_role,
                        dependency.source_timestamp,
                        dependency.span_start,
                        dependency.span_end,
                        dependency.quote,
                        dependency.assertion_id,
                    )
                    for dependency in unique_dependencies
                ),
            )
            cur = self._conn.execute(
                """
                UPDATE lcm_query_views
                   SET status='ready', current_version=?, build_nonce='',
                       lease_expires_at=NULL, stale_reason='', updated_at=?,
                       expires_at=?
                 WHERE view_id=? AND generation=? AND build_nonce=?
                   AND status='building'
                """,
                (
                    version,
                    now,
                    expires_at,
                    token.view_id,
                    token.generation,
                    token.nonce,
                ),
            )
            if int(cur.rowcount or 0) != 1:
                raise RuntimeError("query-view publication ownership changed mid-transaction")
        return True

    def mark_failed(self, token: QueryViewBuildToken, error: str) -> bool:
        bounded_error = str(error or "").strip()[:1_000]
        with self._write_transaction():
            cur = self._conn.execute(
                """
                UPDATE lcm_query_views
                   SET status='failed', stale_reason=?, build_nonce='',
                       lease_expires_at=NULL, updated_at=?
                 WHERE view_id=? AND generation=? AND build_nonce=?
                   AND status='building'
                """,
                (
                    bounded_error,
                    self._now(),
                    token.view_id,
                    token.generation,
                    token.nonce,
                ),
            )
        return int(cur.rowcount or 0) == 1

    def reclaim_expired_builds(self, now: float | None = None) -> int:
        cutoff = self._now() if now is None else float(now)
        with self._write_transaction():
            cur = self._conn.execute(
                """
                UPDATE lcm_query_views
                   SET status='stale', generation=generation+1,
                       stale_reason='build_lease_expired', build_nonce='',
                       lease_expires_at=NULL, updated_at=?
                 WHERE status='building' AND lease_expires_at < ?
                """,
                (cutoff, cutoff),
            )
        return int(cur.rowcount or 0)

    def _dependencies_for(self, view_id: str, version: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM lcm_query_view_sources "
            "WHERE view_id=? AND version=? "
            "ORDER BY source_store_id, span_start, span_end",
            (view_id, int(version)),
        ).fetchall()
        return [dict(row) for row in rows]

    def delta_events(
        self, since_generation: int, *, limit: int = 256
    ) -> tuple[tuple[dict[str, Any], ...], bool]:
        bounded = min(1_000, max(1, int(limit)))
        rows = self._conn.execute(
            "SELECT * FROM lcm_query_corpus_events WHERE generation > ? "
            "ORDER BY generation LIMIT ?",
            (int(since_generation), bounded + 1),
        ).fetchall()
        return tuple(dict(row) for row in rows[:bounded]), len(rows) > bounded

    def lookup(
        self,
        identity: QueryViewIdentity,
        *,
        now: float | None = None,
        delta_limit: int = 256,
        record_hit: bool = True,
    ) -> QueryViewLookup:
        view_id = identity.view_id
        canonical = identity.canonical_json
        row = self._conn.execute(
            "SELECT * FROM lcm_query_views WHERE view_id = ?", (view_id,)
        ).fetchone()
        if row is None:
            return QueryViewLookup("miss", "typed intent has no materialized view")
        if str(row["identity_json"]) != canonical:
            return QueryViewLookup("miss", "typed intent identity mismatch")
        status = str(row["status"])
        if status == "building":
            return QueryViewLookup("building", "view refresh is in progress")
        if status == "failed":
            return QueryViewLookup("failed", str(row["stale_reason"] or "build failed"))
        version = int(row["current_version"])
        generation = int(row["generation"])
        if version <= 0:
            return QueryViewLookup("miss", "view has no published evidence version")
        version_row = self._conn.execute(
            "SELECT * FROM lcm_query_view_versions WHERE view_id=? AND version=?",
            (view_id, version),
        ).fetchone()
        if version_row is None:
            return QueryViewLookup("failed", "current view version is missing")
        current_time = self._now() if now is None else float(now)
        expires_at = version_row["expires_at"]
        if expires_at is not None and float(expires_at) <= current_time:
            with self._write_transaction():
                self._conn.execute(
                    "UPDATE lcm_query_views SET status='expired', generation=generation+1, "
                    "stale_reason='ttl_expired', updated_at=? "
                    "WHERE view_id=? AND current_version=? AND status<>'expired'",
                    (current_time, view_id, version),
                )
            return QueryViewLookup("expired", "view TTL expired")

        dependencies = self._dependencies_for(view_id, version)
        positive_error = ""
        for dependency in dependencies:
            try:
                self._validate_dependency_current(QueryEvidenceDependency(
                    source_store_id=int(dependency["source_store_id"]),
                    source_content_sha256=str(dependency["source_content_sha256"]),
                    source_session_id=str(dependency["source_session_id"]),
                    source_conversation_id=str(dependency["source_conversation_id"]),
                    source_name=str(dependency["source_name"]),
                    source_role=str(dependency["source_role"]),
                    source_timestamp=float(dependency["source_timestamp"]),
                    span_start=int(dependency["span_start"]),
                    span_end=int(dependency["span_end"]),
                    quote=str(dependency["quote"]),
                    assertion_id=str(dependency["assertion_id"] or ""),
                ))
            except ValueError as exc:
                positive_error = str(exc)
                break
        corpus = self.corpus_snapshot()
        covered_generation = int(version_row["corpus_generation"])
        negative_stale = corpus.generation != covered_generation
        if status in {"stale", "expired"} or positive_error or negative_stale:
            reason = positive_error or str(row["stale_reason"] or "")
            if negative_stale and not reason:
                reason = "corpus advanced beyond the negative-space watermark"
            with self._write_transaction():
                self._conn.execute(
                    "UPDATE lcm_query_views SET status='stale', generation=generation+1, "
                    "stale_reason=?, build_nonce='', lease_expires_at=NULL, updated_at=? "
                    "WHERE view_id=? AND status='ready' AND current_version=?",
                    (reason, current_time, view_id, version),
                )
            row = self._conn.execute(
                "SELECT * FROM lcm_query_views WHERE view_id=?", (view_id,)
            ).fetchone()
            events, truncated = self.delta_events(
                covered_generation, limit=delta_limit
            )
            return QueryViewLookup(
                "delta_required",
                reason or "view is stale",
                self._shape_view(row, version_row, dependencies),
                events,
                truncated,
            )
        if str(version_row["completeness"]) != "complete":
            return QueryViewLookup(
                "incomplete",
                "view has not closed every required evidence slot",
                self._shape_view(row, version_row, dependencies),
            )
        if record_hit:
            with self._write_transaction():
                cur = self._conn.execute(
                    "UPDATE lcm_query_views SET hit_count=hit_count+1, "
                    "promotion_status=CASE WHEN hit_count+1 >= 2 THEN 'promoted' "
                    "ELSE promotion_status END, updated_at=? "
                    "WHERE view_id=? AND status='ready' AND current_version=? "
                    "AND generation=?",
                    (current_time, view_id, version, generation),
                )
            row = self._conn.execute(
                "SELECT * FROM lcm_query_views WHERE view_id=?", (view_id,)
            ).fetchone()
            confirmed = (
                int(cur.rowcount or 0) == 1
                and row is not None
                and str(row["status"]) == "ready"
                and int(row["current_version"]) == version
                and int(row["generation"]) == generation
            )
            if not confirmed:
                events, truncated = self.delta_events(
                    covered_generation, limit=delta_limit
                )
                current_view = (
                    self._shape_view(row, version_row, dependencies)
                    if row is not None and int(row["current_version"]) == version
                    else None
                )
                return QueryViewLookup(
                    "delta_required",
                    (
                        str(row["stale_reason"] or "")
                        if row is not None
                        else ""
                    )
                    or "view changed during hit confirmation",
                    current_view,
                    events,
                    truncated,
                )
        return QueryViewLookup(
            "hit",
            "exact typed intent and fresh positive/negative dependencies",
            self._shape_view(row, version_row, dependencies),
        )

    @staticmethod
    def _shape_view(
        row: sqlite3.Row,
        version_row: sqlite3.Row,
        dependencies: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "view_id": str(row["view_id"]),
            "identity": json.loads(str(row["identity_json"])),
            "status": str(row["status"]),
            "generation": int(row["generation"]),
            "version": int(version_row["version"]),
            "manifest": json.loads(str(version_row["manifest_json"])),
            "computation_trace": (
                json.loads(str(version_row["trace_json"]))
                if str(version_row["trace_json"] or "")
                else None
            ),
            "completeness": str(version_row["completeness"]),
            "search_policy_version": str(version_row["search_policy_version"]),
            "corpus_generation": int(version_row["corpus_generation"]),
            "coverage_store_id": int(version_row["coverage_store_id"]),
            "published_at": float(version_row["published_at"]),
            "expires_at": (
                float(version_row["expires_at"])
                if version_row["expires_at"] is not None
                else None
            ),
            "supersedes_version": version_row["supersedes_version"],
            "hit_count": int(row["hit_count"]),
            "promotion_status": str(row["promotion_status"]),
            "dependencies": list(dependencies),
        }

    def expire_views(self, now: float | None = None, *, limit: int = 256) -> int:
        cutoff = self._now() if now is None else float(now)
        bounded = min(1_000, max(1, int(limit)))
        with self._write_transaction():
            rows = self._conn.execute(
                "SELECT view_id FROM lcm_query_views WHERE expires_at <= ? "
                "AND status <> 'expired' ORDER BY expires_at LIMIT ?",
                (cutoff, bounded),
            ).fetchall()
            ids = [str(row["view_id"]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self._conn.execute(
                    f"UPDATE lcm_query_views SET status='expired', "
                    f"generation=generation+1, stale_reason='ttl_expired', "
                    f"updated_at=? WHERE view_id IN ({placeholders})",
                    (cutoff, *ids),
                )
        return len(ids)

    def purge_expired(self, *, older_than: float, limit: int = 256) -> int:
        bounded = min(1_000, max(1, int(limit)))
        with self._write_transaction():
            rows = self._conn.execute(
                "SELECT view_id FROM lcm_query_views WHERE status='expired' "
                "AND updated_at < ? ORDER BY updated_at LIMIT ?",
                (float(older_than), bounded),
            ).fetchall()
            ids = [str(row["view_id"]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self._conn.execute(
                    f"DELETE FROM lcm_query_view_sources WHERE view_id IN ({placeholders})",
                    ids,
                )
                self._conn.execute(
                    f"DELETE FROM lcm_query_view_versions WHERE view_id IN ({placeholders})",
                    ids,
                )
                self._conn.execute(
                    f"DELETE FROM lcm_query_views WHERE view_id IN ({placeholders})",
                    ids,
                )
        return len(ids)

    def prune_corpus_events(self, *, retain: int = 10_000) -> int:
        keep = max(1, int(retain))
        current = self.corpus_snapshot().generation
        floor_row = self._conn.execute(
            "SELECT min(version.corpus_generation) AS floor "
            "FROM lcm_query_views AS view "
            "JOIN lcm_query_view_versions AS version "
            "ON version.view_id=view.view_id AND version.version=view.current_version "
            "WHERE view.status IN ('ready', 'stale', 'building')"
        ).fetchone()
        active_floor = (
            int(floor_row["floor"])
            if floor_row and floor_row["floor"] is not None
            else current
        )
        safe_floor = min(active_floor, max(0, current - keep))
        with self._write_transaction():
            cur = self._conn.execute(
                "DELETE FROM lcm_query_corpus_events WHERE generation <= ?",
                (safe_floor,),
            )
        return int(cur.rowcount or 0)

    def commit(self) -> None:
        with self._write_lock:
            self._conn.commit()

    @property
    def connection(self) -> sqlite3.Connection | None:
        return getattr(self, "_conn", None)

    def close(self) -> None:
        with self._write_lock:
            conn = getattr(self, "_conn", None)
            if conn is not None:
                try:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.Error:
                    pass
                conn.close()
                self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass


def _synchronized(method):
    """Serialize operations that share one SQLite connection across clones."""
    def locked(self, *args, **kwargs):
        with self._write_lock:
            return method(self, *args, **kwargs)

    return locked


for _method_name in (
    "corpus_snapshot",
    "snapshot_dependency",
    "claim_build",
    "publish_ready",
    "mark_failed",
    "reclaim_expired_builds",
    "delta_events",
    "lookup",
    "expire_views",
    "purge_expired",
    "prune_corpus_events",
):
    setattr(QueryViewStore, _method_name, _synchronized(getattr(QueryViewStore, _method_name)))
