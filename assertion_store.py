"""Rebuildable, provenance-linked V4 assertions over immutable LCM messages.

The assertion family is logically separate but physically colocated in the
profile's existing ``lcm.db``. Raw ``messages`` remain authoritative. This
module deliberately contains no extractor, provider, embedding, or benchmark
logic: callers may plan bounded work, publish already-derived candidates with a
late source-hash compare-and-swap, and query only source-valid assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Iterable, Sequence
from urllib.parse import quote

from .db_bootstrap import (
    ASSERTION_MIGRATION_STEP,
    configure_connection,
    ensure_assertion_tables,
    mark_migration_step_complete,
    refuse_schema_version_too_new,
    run_versioned_migrations,
    verify_assertion_schema,
)


CURRENT_EXTRACTION_VERSION = "assertions-v1"

ASSERTION_KINDS = frozenset({
    "fact",
    "event",
    "preference",
    "recommendation",
    "commitment",
    "action",
    "status",
    "quotation",
})
ASSERTION_POLARITIES = frozenset({"positive", "negative", "unknown"})
ASSERTION_RELATION_TYPES = frozenset({
    "confirms",
    "supersedes",
    "contradicts",
    "narrows",
    "weakens",
    "reverses",
    "cancels",
    "fulfills",
    "quotes",
})

_MAX_BATCH_SOURCES = 500
_MAX_CANDIDATES_PER_SOURCE = 500
_MAX_KEY_CHARS = 512
_MAX_VALUE_TEXT_CHARS = 16_384
_MAX_OBJECT_JSON_CHARS = 32_768
_MAX_VERSION_CHARS = 128


class AssertionStoreError(RuntimeError):
    """Base class for assertion-store failures."""


class AssertionSchemaUnavailableError(AssertionStoreError):
    """Raised when a read-only caller opens a DB without the assertion schema."""


class AssertionSourceStaleError(AssertionStoreError):
    """Raised when a source changed after a snapshot was planned."""


class AssertionPublicationConflictError(AssertionStoreError):
    """Raised when one extraction version emits two results for one source hash."""


@dataclass(frozen=True)
class SourceSnapshot:
    store_id: int
    session_id: str
    source: str
    role: str
    content: str
    timestamp: float
    content_sha256: str


@dataclass(frozen=True)
class AssertionCandidate:
    source_span_start: int
    source_span_end: int
    subject_key: str
    predicate_key: str
    object_value: Any = None
    value_text: str = ""
    kind: str = "fact"
    polarity: str = "positive"
    strength: float | None = None
    scope_key: str = ""
    event_at: float | None = None
    valid_from: float | None = None
    valid_to: float | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class AssertionRelationCandidate:
    source_span_start: int
    source_span_end: int
    from_assertion_id: str
    relation_type: str
    to_assertion_id: str
    confidence: float = 1.0


@dataclass(frozen=True)
class RebuildPlan:
    extraction_version: str
    pending_count: int
    selected_sources: tuple[SourceSnapshot, ...]
    limit: int

    @property
    def remaining_count(self) -> int:
        return max(0, self.pending_count - len(self.selected_sources))


@dataclass(frozen=True)
class PublishResult:
    source_store_id: int
    extraction_version: str
    source_content_sha256: str
    candidate_digest: str
    assertion_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    already_current: bool

    @property
    def assertions_written(self) -> int:
        return 0 if self.already_current else len(self.assertion_ids)

    @property
    def relations_written(self) -> int:
        return 0 if self.already_current else len(self.relation_ids)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _canonical_json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("assertion object_value must be finite JSON") from exc
    if len(encoded) > _MAX_OBJECT_JSON_CHARS:
        raise ValueError(
            f"assertion object_value exceeds {_MAX_OBJECT_JSON_CHARS} characters"
        )
    return encoded


def _validated_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    text = str(value or "").strip()
    if not text and not allow_empty:
        raise ValueError(f"{field} must not be empty")
    if len(text) > _MAX_KEY_CHARS:
        raise ValueError(f"{field} exceeds {_MAX_KEY_CHARS} characters")
    return text


def _validated_version(version: str) -> str:
    normalized = str(version or "").strip()
    if not normalized:
        raise ValueError("extraction_version must not be empty")
    if len(normalized) > _MAX_VERSION_CHARS:
        raise ValueError(
            f"extraction_version exceeds {_MAX_VERSION_CHARS} characters"
        )
    return normalized


def _finite_number(value: float | None, field: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _bounded_probability(value: float | None, field: str) -> float | None:
    result = _finite_number(value, field)
    if result is not None and not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return result


class AssertionStore:
    """SQLite assertion store bound to the same physical DB as ``MessageStore``."""

    def __init__(self, db_path: str | Path, *, read_only: bool = False):
        self.db_path = Path(db_path)
        self.read_only = bool(read_only)
        self._write_lock = threading.RLock()
        self._conn = self._open_connection()
        try:
            self._init_db()
        except Exception:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
            raise

    def _open_connection(self) -> sqlite3.Connection:
        if self.read_only:
            uri = f"file:{quote(str(self.db_path), safe='/')}?mode=ro"
            conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=5.0,
                check_same_thread=False,
                isolation_level=None,
            )
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA busy_timeout=30000")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=5.0,
                check_same_thread=False,
                isolation_level=None,
            )
            refuse_schema_version_too_new(conn)
            configure_connection(conn)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        if self.read_only:
            findings = verify_assertion_schema(self._conn)
            if findings:
                raise AssertionSchemaUnavailableError(
                    "assertion schema unavailable for read-only planning: "
                    + "; ".join(findings)
                )
            return

        run_versioned_migrations(self._conn)
        ensure_assertion_tables(self._conn)
        findings = verify_assertion_schema(self._conn)
        if findings:
            ensure_assertion_tables(self._conn)
            findings = verify_assertion_schema(self._conn)
            if findings:
                raise sqlite3.OperationalError(
                    "assertion schema incompatible after ensure: "
                    + "; ".join(findings)
                )
        marker = self._conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name = ?",
            (ASSERTION_MIGRATION_STEP,),
        ).fetchone()
        if marker is None:
            mark_migration_step_complete(self._conn, ASSERTION_MIGRATION_STEP)
        self._conn.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        with self._write_lock:
            conn = self._conn
            if conn is None:
                return
            try:
                if not self.read_only:
                    conn.commit()
            finally:
                conn.close()
                self._conn = None  # type: ignore[assignment]

    def commit(self) -> None:
        """Flush completed assertion work without splitting an active publish."""
        with self._write_lock:
            if not self.read_only and self._conn is not None:
                self._conn.commit()

    def _require_writable(self) -> None:
        if self.read_only:
            raise AssertionStoreError("read-only AssertionStore cannot publish or invalidate")

    def _begin_write(self) -> None:
        self._require_writable()
        self._conn.execute("BEGIN IMMEDIATE")

    def _source_row(self, store_id: int) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT store_id, session_id, source, role, content, timestamp,
                   observed_at
            FROM messages
            WHERE store_id = ?
            """,
            (int(store_id),),
        ).fetchone()

    @staticmethod
    def _snapshot_from_row(row: sqlite3.Row) -> SourceSnapshot:
        content = str(row["content"] or "")
        return SourceSnapshot(
            store_id=int(row["store_id"]),
            session_id=str(row["session_id"] or ""),
            source=str(row["source"] or ""),
            role=str(row["role"] or ""),
            content=content,
            timestamp=float(
                row["observed_at"]
                if row["observed_at"] is not None
                else row["timestamp"]
            ),
            content_sha256=_sha256_text(content),
        )

    def snapshot_source(self, store_id: int) -> SourceSnapshot:
        row = self._source_row(store_id)
        if row is None:
            raise KeyError(f"message store_id {store_id} does not exist")
        return self._snapshot_from_row(row)

    def has_current_receipt(
        self,
        snapshot: SourceSnapshot,
        *,
        extraction_version: str = CURRENT_EXTRACTION_VERSION,
    ) -> bool:
        """Return whether this exact source hash already has an active receipt."""
        version = _validated_version(extraction_version)
        return self._conn.execute(
            """
            SELECT 1
            FROM lcm_assertion_sources
            WHERE source_store_id = ? AND extraction_version = ?
              AND source_content_sha256 = ? AND invalidated_at IS NULL
            """,
            (snapshot.store_id, version, snapshot.content_sha256),
        ).fetchone() is not None

    def plan_rebuild(
        self,
        *,
        extraction_version: str = CURRENT_EXTRACTION_VERSION,
        limit: int = 100,
    ) -> RebuildPlan:
        """Return a bounded, read-only plan of sources without a current receipt."""
        version = _validated_version(extraction_version)
        bounded_limit = int(limit)
        if not 1 <= bounded_limit <= _MAX_BATCH_SOURCES:
            raise ValueError(f"limit must be between 1 and {_MAX_BATCH_SOURCES}")
        join = (
            "LEFT JOIN lcm_assertion_sources AS s "
            "ON s.source_store_id = m.store_id "
            "AND s.extraction_version = ? "
            "AND s.invalidated_at IS NULL"
        )
        pending_count = int(
            self._conn.execute(
                f"SELECT COUNT(*) FROM messages AS m {join} "
                "WHERE s.source_store_id IS NULL",
                (version,),
            ).fetchone()[0]
        )
        rows = self._conn.execute(
            f"""
            SELECT m.store_id, m.session_id, m.source, m.role, m.content,
                   m.timestamp, m.observed_at
            FROM messages AS m
            {join}
            WHERE s.source_store_id IS NULL
            ORDER BY m.store_id ASC
            LIMIT ?
            """,
            (version, bounded_limit),
        ).fetchall()
        return RebuildPlan(
            extraction_version=version,
            pending_count=pending_count,
            selected_sources=tuple(self._snapshot_from_row(row) for row in rows),
            limit=bounded_limit,
        )

    def _prepare_assertion(
        self,
        snapshot: SourceSnapshot,
        candidate: AssertionCandidate,
        version: str,
    ) -> dict[str, Any]:
        start = int(candidate.source_span_start)
        end = int(candidate.source_span_end)
        if start < 0 or end <= start or end > len(snapshot.content):
            raise ValueError(
                f"invalid source span [{start}, {end}) for {len(snapshot.content)} characters"
            )
        subject = _validated_text(candidate.subject_key, "subject_key")
        predicate = _validated_text(candidate.predicate_key, "predicate_key")
        kind = str(candidate.kind or "").strip().lower()
        if kind not in ASSERTION_KINDS:
            raise ValueError(f"unsupported assertion kind: {kind}")
        polarity = str(candidate.polarity or "").strip().lower()
        if polarity not in ASSERTION_POLARITIES:
            raise ValueError(f"unsupported assertion polarity: {polarity}")
        value_text = str(candidate.value_text or "")
        if len(value_text) > _MAX_VALUE_TEXT_CHARS:
            raise ValueError(
                f"value_text exceeds {_MAX_VALUE_TEXT_CHARS} characters"
            )
        scope_key = _validated_text(
            candidate.scope_key, "scope_key", allow_empty=True
        )
        strength = _bounded_probability(candidate.strength, "strength")
        confidence = _bounded_probability(candidate.confidence, "confidence")
        if confidence is None:
            raise ValueError("confidence must not be null")
        event_at = _finite_number(candidate.event_at, "event_at")
        valid_from = _finite_number(candidate.valid_from, "valid_from")
        valid_to = _finite_number(candidate.valid_to, "valid_to")
        if valid_from is not None and valid_to is not None and valid_to <= valid_from:
            raise ValueError("valid_to must be greater than valid_from")
        payload = {
            "source_store_id": snapshot.store_id,
            "extraction_version": version,
            "source_content_sha256": snapshot.content_sha256,
            "subject_key": subject,
            "predicate_key": predicate,
            "object_json": _canonical_json(candidate.object_value),
            "value_text": value_text,
            "kind": kind,
            "polarity": polarity,
            "strength": strength,
            "scope_key": scope_key,
            "speaker_role": snapshot.role,
            "observed_at": snapshot.timestamp,
            "event_at": event_at,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "source_span_start": start,
            "source_span_end": end,
            "source_quote": snapshot.content[start:end],
            "confidence": confidence,
        }
        payload["assertion_id"] = _sha256_text(_canonical_json(payload))
        return payload

    def assertion_id_for(
        self,
        snapshot: SourceSnapshot,
        candidate: AssertionCandidate,
        *,
        extraction_version: str = CURRENT_EXTRACTION_VERSION,
    ) -> str:
        version = _validated_version(extraction_version)
        return str(self._prepare_assertion(snapshot, candidate, version)["assertion_id"])

    def _prepare_relation(
        self,
        snapshot: SourceSnapshot,
        candidate: AssertionRelationCandidate,
        version: str,
    ) -> dict[str, Any]:
        start = int(candidate.source_span_start)
        end = int(candidate.source_span_end)
        if start < 0 or end <= start or end > len(snapshot.content):
            raise ValueError(
                f"invalid relation source span [{start}, {end}) for "
                f"{len(snapshot.content)} characters"
            )
        relation_type = str(candidate.relation_type or "").strip().lower()
        if relation_type not in ASSERTION_RELATION_TYPES:
            raise ValueError(f"unsupported assertion relation: {relation_type}")
        from_id = str(candidate.from_assertion_id or "").strip().lower()
        to_id = str(candidate.to_assertion_id or "").strip().lower()
        if not _is_sha256_hex(from_id) or not _is_sha256_hex(to_id):
            raise ValueError("relation assertion IDs must be 64-character SHA-256 hex")
        if from_id == to_id:
            raise ValueError("an assertion relation cannot point to itself")
        confidence = _bounded_probability(candidate.confidence, "confidence")
        if confidence is None:
            raise ValueError("confidence must not be null")
        payload = {
            "source_store_id": snapshot.store_id,
            "extraction_version": version,
            "source_content_sha256": snapshot.content_sha256,
            "from_assertion_id": from_id,
            "relation_type": relation_type,
            "to_assertion_id": to_id,
            "source_span_start": start,
            "source_span_end": end,
            "source_quote": snapshot.content[start:end],
            "confidence": confidence,
        }
        payload["relation_id"] = _sha256_text(_canonical_json(payload))
        return payload

    def publish_source(
        self,
        snapshot: SourceSnapshot,
        candidates: Sequence[AssertionCandidate],
        *,
        relations: Sequence[AssertionRelationCandidate] = (),
        extraction_version: str = CURRENT_EXTRACTION_VERSION,
    ) -> PublishResult:
        """Atomically publish one already-derived source generation.

        The raw row is reread under ``BEGIN IMMEDIATE`` and must still match the
        planned snapshot. Identical re-publication is a zero-write no-op. A
        different result for the same source hash and extraction version fails
        closed as extractor nondeterminism; changing semantics requires a new
        extraction version.
        """
        self._require_writable()
        version = _validated_version(extraction_version)
        if len(candidates) > _MAX_CANDIDATES_PER_SOURCE:
            raise ValueError(
                f"at most {_MAX_CANDIDATES_PER_SOURCE} assertions may be published per source"
            )
        if len(relations) > _MAX_CANDIDATES_PER_SOURCE:
            raise ValueError(
                f"at most {_MAX_CANDIDATES_PER_SOURCE} relations may be published per source"
            )
        assertion_rows = [
            self._prepare_assertion(snapshot, candidate, version)
            for candidate in candidates
        ]
        relation_rows = [
            self._prepare_relation(snapshot, candidate, version)
            for candidate in relations
        ]
        assertion_rows.sort(key=lambda row: str(row["assertion_id"]))
        relation_rows.sort(key=lambda row: str(row["relation_id"]))
        assertion_id_set = {str(row["assertion_id"]) for row in assertion_rows}
        relation_id_set = {str(row["relation_id"]) for row in relation_rows}
        if len(assertion_id_set) != len(assertion_rows):
            raise ValueError("duplicate assertion candidates are not allowed")
        if len(relation_id_set) != len(relation_rows):
            raise ValueError("duplicate assertion relations are not allowed")
        candidate_digest = _sha256_text(
            _canonical_json({"assertions": assertion_rows, "relations": relation_rows})
        )
        assertion_ids = tuple(str(row["assertion_id"]) for row in assertion_rows)
        relation_ids = tuple(str(row["relation_id"]) for row in relation_rows)

        with self._write_lock:
            self._begin_write()
            try:
                current_row = self._source_row(snapshot.store_id)
                if current_row is None:
                    raise AssertionSourceStaleError(
                        f"source store_id {snapshot.store_id} was deleted before publication"
                    )
                current = self._snapshot_from_row(current_row)
                if current != snapshot:
                    raise AssertionSourceStaleError(
                        f"source store_id {snapshot.store_id} changed before publication"
                    )
                existing = self._conn.execute(
                    """
                    SELECT source_content_sha256, candidate_digest, assertion_count, relation_count
                    FROM lcm_assertion_sources
                    WHERE source_store_id = ? AND extraction_version = ?
                      AND invalidated_at IS NULL
                    """,
                    (snapshot.store_id, version),
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing["source_content_sha256"]) == snapshot.content_sha256
                        and str(existing["candidate_digest"]) == candidate_digest
                        and int(existing["assertion_count"]) == len(assertion_rows)
                        and int(existing["relation_count"]) == len(relation_rows)
                    ):
                        self._conn.execute("COMMIT")
                        return PublishResult(
                            source_store_id=snapshot.store_id,
                            extraction_version=version,
                            source_content_sha256=snapshot.content_sha256,
                            candidate_digest=candidate_digest,
                            assertion_ids=assertion_ids,
                            relation_ids=relation_ids,
                            already_current=True,
                        )
                    raise AssertionPublicationConflictError(
                        "different assertion output already exists for this source hash "
                        "and extraction version; bump the extraction version"
                    )

                historical = self._conn.execute(
                    """
                    SELECT candidate_digest, assertion_count, relation_count
                    FROM lcm_assertion_sources
                    WHERE source_store_id = ? AND extraction_version = ?
                      AND source_content_sha256 = ? AND invalidated_at IS NOT NULL
                    """,
                    (snapshot.store_id, version, snapshot.content_sha256),
                ).fetchone()
                if historical is not None:
                    if (
                        str(historical["candidate_digest"]) != candidate_digest
                        or int(historical["assertion_count"]) != len(assertion_rows)
                        or int(historical["relation_count"]) != len(relation_rows)
                    ):
                        raise AssertionPublicationConflictError(
                            "historical output differs for this source hash and extraction "
                            "version; bump the extraction version"
                        )
                    self._conn.execute(
                        """
                        UPDATE lcm_assertion_sources
                           SET source_session_id = ?, source_role = ?, source_name = ?,
                               source_timestamp = ?, processed_at = ?, invalidated_at = NULL,
                               invalidation_reason = NULL
                         WHERE source_store_id = ? AND extraction_version = ?
                           AND source_content_sha256 = ?
                        """,
                        (
                            snapshot.session_id,
                            snapshot.role,
                            snapshot.source,
                            snapshot.timestamp,
                            time.time(),
                            snapshot.store_id,
                            version,
                            snapshot.content_sha256,
                        ),
                    )
                    self._conn.execute("COMMIT")
                    return PublishResult(
                        source_store_id=snapshot.store_id,
                        extraction_version=version,
                        source_content_sha256=snapshot.content_sha256,
                        candidate_digest=candidate_digest,
                        assertion_ids=assertion_ids,
                        relation_ids=relation_ids,
                        already_current=True,
                    )

                self._conn.execute(
                    """
                    INSERT INTO lcm_assertion_sources(
                        source_store_id, extraction_version, source_content_sha256,
                        source_session_id, source_role, source_name, source_timestamp,
                        candidate_digest, assertion_count, relation_count, processed_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.store_id,
                        version,
                        snapshot.content_sha256,
                        snapshot.session_id,
                        snapshot.role,
                        snapshot.source,
                        snapshot.timestamp,
                        candidate_digest,
                        len(assertion_rows),
                        len(relation_rows),
                        time.time(),
                    ),
                )
                for row in assertion_rows:
                    self._conn.execute(
                        """
                        INSERT INTO lcm_assertions(
                            assertion_id, source_store_id, extraction_version,
                            source_content_sha256, subject_key, predicate_key,
                            object_json, value_text, kind, polarity, strength,
                            scope_key, speaker_role, observed_at, event_at,
                            valid_from, valid_to, source_span_start, source_span_end,
                            source_quote, confidence, created_at
                        ) VALUES(
                            :assertion_id, :source_store_id, :extraction_version,
                            :source_content_sha256, :subject_key, :predicate_key,
                            :object_json, :value_text, :kind, :polarity, :strength,
                            :scope_key, :speaker_role, :observed_at, :event_at,
                            :valid_from, :valid_to, :source_span_start, :source_span_end,
                            :source_quote, :confidence, :created_at
                        )
                        """,
                        {**row, "created_at": time.time()},
                    )
                for row in relation_rows:
                    endpoints = (row["from_assertion_id"], row["to_assertion_id"])
                    for endpoint in endpoints:
                        active_endpoint = self._conn.execute(
                            """
                            SELECT 1
                            FROM lcm_assertions AS a
                            JOIN lcm_assertion_sources AS s
                              ON s.source_store_id = a.source_store_id
                             AND s.extraction_version = a.extraction_version
                             AND s.source_content_sha256 = a.source_content_sha256
                             AND s.invalidated_at IS NULL
                            WHERE a.assertion_id = ?
                            """,
                            (endpoint,),
                        ).fetchone()
                        if active_endpoint is None:
                            raise ValueError(
                                f"relation endpoint {endpoint} is missing or invalidated"
                            )
                    self._conn.execute(
                        """
                        INSERT INTO lcm_assertion_relations(
                            relation_id, source_store_id, extraction_version,
                            source_content_sha256, from_assertion_id, relation_type,
                            to_assertion_id, source_span_start, source_span_end,
                            source_quote, confidence, created_at
                        ) VALUES(
                            :relation_id, :source_store_id, :extraction_version,
                            :source_content_sha256, :from_assertion_id, :relation_type,
                            :to_assertion_id, :source_span_start, :source_span_end,
                            :source_quote, :confidence, :created_at
                        )
                        """,
                        {**row, "created_at": time.time()},
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

        return PublishResult(
            source_store_id=snapshot.store_id,
            extraction_version=version,
            source_content_sha256=snapshot.content_sha256,
            candidate_digest=candidate_digest,
            assertion_ids=assertion_ids,
            relation_ids=relation_ids,
            already_current=False,
        )

    @staticmethod
    def _in_clause(values: Iterable[str]) -> tuple[str, list[str]]:
        items = [str(value) for value in values]
        return ",".join("?" for _ in items), items

    @staticmethod
    def _validate_source_provenance(
        *,
        content: Any,
        expected_sha256: Any,
        span_start: Any,
        span_end: Any,
        quote: Any,
        label: str,
    ) -> None:
        current_content = str(content or "")
        if _sha256_text(current_content) != str(expected_sha256):
            raise AssertionSourceStaleError(f"active {label} hash changed")
        start = int(span_start)
        end = int(span_end)
        if current_content[start:end] != quote:
            raise AssertionSourceStaleError(f"active {label} span changed")

    def query_assertions(
        self,
        *,
        extraction_version: str = CURRENT_EXTRACTION_VERSION,
        subject_key: str | None = None,
        predicate_key: str | None = None,
        kinds: Iterable[str] | None = None,
        scope_key: str | None = None,
        speaker_role: str | None = None,
        source_store_id: int | None = None,
        assertion_id: str | None = None,
        as_of: float | None = None,
        include_invalidated: bool = False,
        access_scope: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query source-valid assertions with bitemporal historical filtering.

        ``access_scope`` is the per-principal owner predicate. Absent -- which is
        every default-off and non-Teams caller -- the query is byte-identical to
        what it was before.

        An assertion is DERIVED from a source message and carries that message's
        text in ``source_quote``, so it inherits the source's owner. The
        assertion tables carry no ``access_scope`` column of their own, but they
        do not need one here: ``messages`` is already joined on
        ``source_store_id``, so the owner is one predicate away. Without it,
        ``lcm_query_state`` returned assertions extracted from EVERY principal's
        messages, each carrying the source quote verbatim plus the foreign
        session_id and store_id.
        """
        version = _validated_version(extraction_version)
        bounded_limit = int(limit)
        if not 1 <= bounded_limit <= _MAX_BATCH_SOURCES:
            raise ValueError(f"limit must be between 1 and {_MAX_BATCH_SOURCES}")
        where = ["a.extraction_version = ?"]
        args: list[Any] = [version]
        if not include_invalidated:
            where.extend(["s.invalidated_at IS NULL", "m.store_id IS NOT NULL"])
        if subject_key is not None:
            where.append("a.subject_key = ?")
            args.append(str(subject_key))
        if predicate_key is not None:
            where.append("a.predicate_key = ?")
            args.append(str(predicate_key))
        if scope_key is not None:
            where.append("a.scope_key = ?")
            args.append(str(scope_key))
        if speaker_role is not None:
            where.append("a.speaker_role = ?")
            args.append(str(speaker_role))
        if source_store_id is not None:
            where.append("a.source_store_id = ?")
            args.append(int(source_store_id))
        if access_scope is not None:
            # The owner of the SOURCE message owns what was derived from it.
            # `messages` is LEFT JOINed, so this also excludes an assertion
            # whose source row has gone -- the fail-closed direction, and the
            # right one for a scoped read: an orphaned assertion still carries
            # its source_quote verbatim.
            where.append("m.access_scope = ?")
            args.append(str(access_scope))
        if assertion_id is not None:
            normalized_assertion_id = str(assertion_id).strip().lower()
            if not _is_sha256_hex(normalized_assertion_id):
                raise ValueError("assertion_id must be a 64-character SHA-256 hex value")
            where.append("a.assertion_id = ?")
            args.append(normalized_assertion_id)
        if kinds is not None:
            normalized_kinds = [str(kind).strip().lower() for kind in kinds]
            if not normalized_kinds or any(kind not in ASSERTION_KINDS for kind in normalized_kinds):
                raise ValueError("kinds must contain supported assertion kinds")
            placeholders, kind_args = self._in_clause(normalized_kinds)
            where.append(f"a.kind IN ({placeholders})")
            args.extend(kind_args)
        if as_of is not None:
            boundary = _finite_number(as_of, "as_of")
            where.extend([
                "a.observed_at <= ?",
                "(a.valid_from IS NULL OR a.valid_from <= ?)",
                "(a.valid_to IS NULL OR a.valid_to > ?)",
            ])
            args.extend([boundary, boundary, boundary])
        args.append(bounded_limit)
        rows = self._conn.execute(
            f"""
            SELECT a.*, s.source_session_id, s.source_role, s.source_name,
                   s.source_timestamp, s.invalidated_at, s.invalidation_reason,
                   m.content AS current_source_content,
                   m.session_id AS current_source_session_id
            FROM lcm_assertions AS a
            JOIN lcm_assertion_sources AS s
              ON s.source_store_id = a.source_store_id
             AND s.extraction_version = a.extraction_version
             AND s.source_content_sha256 = a.source_content_sha256
            LEFT JOIN messages AS m ON m.store_id = a.source_store_id
            WHERE {' AND '.join(where)}
            ORDER BY a.observed_at DESC, a.assertion_id ASC
            LIMIT ?
            """,
            args,
        ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            content = item.pop("current_source_content", None)
            if item.get("invalidated_at") is None:
                self._validate_source_provenance(
                    content=content,
                    expected_sha256=item["source_content_sha256"],
                    span_start=item["source_span_start"],
                    span_end=item["source_span_end"],
                    quote=item["source_quote"],
                    label=f"assertion source {item['source_store_id']}",
                )
            try:
                item["object_value"] = json.loads(str(item.pop("object_json")))
            except (TypeError, json.JSONDecodeError) as exc:
                raise AssertionStoreError("stored assertion object_json is invalid") from exc
            output.append(item)
        return output

    def query_relations(
        self,
        *,
        extraction_version: str = CURRENT_EXTRACTION_VERSION,
        assertion_id: str | None = None,
        assertion_ids: Iterable[str] | None = None,
        relation_types: Iterable[str] | None = None,
        as_of: float | None = None,
        include_invalidated: bool = False,
        access_scope: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        version = _validated_version(extraction_version)
        bounded_limit = int(limit)
        if not 1 <= bounded_limit <= _MAX_BATCH_SOURCES:
            raise ValueError(f"limit must be between 1 and {_MAX_BATCH_SOURCES}")
        where = ["r.extraction_version = ?"]
        args: list[Any] = [version]
        if access_scope is not None:
            # A relation discloses BOTH endpoints -- the projection returns
            # `from_source_quote` and `to_source_quote` verbatim -- so all three
            # joined source messages must belong to the acting principal.
            # Filtering only the relation's own source would still hand over a
            # foreign quote whenever a relation points across the boundary.
            where.extend([
                "relation_message.access_scope = ?",
                "from_message.access_scope = ?",
                "to_message.access_scope = ?",
            ])
            args.extend([str(access_scope)] * 3)
        if not include_invalidated:
            where.extend([
                "s.invalidated_at IS NULL",
                "from_source.invalidated_at IS NULL",
                "to_source.invalidated_at IS NULL",
                "relation_message.store_id IS NOT NULL",
                "from_message.store_id IS NOT NULL",
                "to_message.store_id IS NOT NULL",
            ])
        if assertion_id is not None:
            if assertion_ids is not None:
                raise ValueError("use assertion_id or assertion_ids, not both")
            normalized_assertion_id = str(assertion_id).strip().lower()
            if not _is_sha256_hex(normalized_assertion_id):
                raise ValueError("assertion_id must be a 64-character SHA-256 hex value")
            where.append("(r.from_assertion_id = ? OR r.to_assertion_id = ?)")
            args.extend([normalized_assertion_id, normalized_assertion_id])
        if assertion_ids is not None:
            normalized_ids = sorted({str(value).strip().lower() for value in assertion_ids})
            if not normalized_ids or len(normalized_ids) > _MAX_BATCH_SOURCES:
                raise ValueError(
                    f"assertion_ids must contain between 1 and {_MAX_BATCH_SOURCES} values"
                )
            if any(not _is_sha256_hex(value) for value in normalized_ids):
                raise ValueError(
                    "assertion_ids must contain 64-character SHA-256 hex values"
                )
            placeholders, id_args = self._in_clause(normalized_ids)
            where.append(
                f"(r.from_assertion_id IN ({placeholders}) "
                f"OR r.to_assertion_id IN ({placeholders}))"
            )
            args.extend(id_args)
            args.extend(id_args)
        if relation_types is not None:
            normalized = [str(value).strip().lower() for value in relation_types]
            if not normalized or any(value not in ASSERTION_RELATION_TYPES for value in normalized):
                raise ValueError("relation_types contains an unsupported relation")
            placeholders, relation_args = self._in_clause(normalized)
            where.append(f"r.relation_type IN ({placeholders})")
            args.extend(relation_args)
        if as_of is not None:
            boundary = _finite_number(as_of, "as_of")
            where.extend([
                "s.source_timestamp <= ?",
                "from_assertion.observed_at <= ?",
                "to_assertion.observed_at <= ?",
                "(from_assertion.valid_from IS NULL OR from_assertion.valid_from <= ?)",
                "(from_assertion.valid_to IS NULL OR from_assertion.valid_to > ?)",
                "(to_assertion.valid_from IS NULL OR to_assertion.valid_from <= ?)",
                "(to_assertion.valid_to IS NULL OR to_assertion.valid_to > ?)",
            ])
            args.extend([boundary] * 7)
        args.append(bounded_limit)
        rows = self._conn.execute(
            f"""
            SELECT r.*, s.source_session_id, s.source_timestamp,
                   s.invalidated_at, s.invalidation_reason,
                   relation_message.content AS current_relation_source_content,
                   from_assertion.source_store_id AS from_source_store_id,
                   from_assertion.source_content_sha256 AS from_source_sha256,
                   from_assertion.source_span_start AS from_source_span_start,
                   from_assertion.source_span_end AS from_source_span_end,
                   from_assertion.source_quote AS from_source_quote,
                   from_message.content AS current_from_source_content,
                   to_assertion.source_store_id AS to_source_store_id,
                   to_assertion.source_content_sha256 AS to_source_sha256,
                   to_assertion.source_span_start AS to_source_span_start,
                   to_assertion.source_span_end AS to_source_span_end,
                   to_assertion.source_quote AS to_source_quote,
                   to_message.content AS current_to_source_content
            FROM lcm_assertion_relations AS r
            JOIN lcm_assertion_sources AS s
              ON s.source_store_id = r.source_store_id
             AND s.extraction_version = r.extraction_version
             AND s.source_content_sha256 = r.source_content_sha256
            LEFT JOIN messages AS relation_message
              ON relation_message.store_id = r.source_store_id
            JOIN lcm_assertions AS from_assertion
              ON from_assertion.assertion_id = r.from_assertion_id
            JOIN lcm_assertion_sources AS from_source
              ON from_source.source_store_id = from_assertion.source_store_id
             AND from_source.extraction_version = from_assertion.extraction_version
             AND from_source.source_content_sha256 = from_assertion.source_content_sha256
            LEFT JOIN messages AS from_message
              ON from_message.store_id = from_assertion.source_store_id
            JOIN lcm_assertions AS to_assertion
              ON to_assertion.assertion_id = r.to_assertion_id
            JOIN lcm_assertion_sources AS to_source
              ON to_source.source_store_id = to_assertion.source_store_id
             AND to_source.extraction_version = to_assertion.extraction_version
             AND to_source.source_content_sha256 = to_assertion.source_content_sha256
            LEFT JOIN messages AS to_message
              ON to_message.store_id = to_assertion.source_store_id
            WHERE {' AND '.join(where)}
            ORDER BY r.created_at ASC, r.relation_id ASC
            LIMIT ?
            """,
            args,
        ).fetchall()
        output: list[dict[str, Any]] = []
        internal_fields = {
            "current_relation_source_content",
            "from_source_store_id",
            "from_source_sha256",
            "from_source_span_start",
            "from_source_span_end",
            "from_source_quote",
            "current_from_source_content",
            "to_source_store_id",
            "to_source_sha256",
            "to_source_span_start",
            "to_source_span_end",
            "to_source_quote",
            "current_to_source_content",
        }
        for row in rows:
            item = dict(row)
            if not include_invalidated:
                self._validate_source_provenance(
                    content=item["current_relation_source_content"],
                    expected_sha256=item["source_content_sha256"],
                    span_start=item["source_span_start"],
                    span_end=item["source_span_end"],
                    quote=item["source_quote"],
                    label=f"relation source {item['source_store_id']}",
                )
                for endpoint in ("from", "to"):
                    self._validate_source_provenance(
                        content=item[f"current_{endpoint}_source_content"],
                        expected_sha256=item[f"{endpoint}_source_sha256"],
                        span_start=item[f"{endpoint}_source_span_start"],
                        span_end=item[f"{endpoint}_source_span_end"],
                        quote=item[f"{endpoint}_source_quote"],
                        label=(
                            f"relation {endpoint} endpoint "
                            f"{item[f'{endpoint}_source_store_id']}"
                        ),
                    )
            output.append({key: value for key, value in item.items() if key not in internal_fields})
        return output

    def invalidate_source(
        self,
        store_id: int,
        *,
        reason: str = "manual",
        invalidated_at: float | None = None,
    ) -> int:
        self._require_writable()
        normalized_reason = _validated_text(reason, "invalidation reason")
        timestamp = _finite_number(invalidated_at, "invalidated_at")
        if timestamp is None:
            timestamp = time.time()
        with self._write_lock:
            self._begin_write()
            try:
                cursor = self._conn.execute(
                    """
                    UPDATE lcm_assertion_sources
                       SET invalidated_at = ?, invalidation_reason = ?
                     WHERE source_store_id = ? AND invalidated_at IS NULL
                    """,
                    (timestamp, normalized_reason, int(store_id)),
                )
                changed = int(cursor.rowcount or 0)
                self._conn.execute("COMMIT")
                return changed
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def remove_version(self, extraction_version: str) -> dict[str, int]:
        """Remove one rebuildable derived generation; raw messages are untouched."""
        self._require_writable()
        version = _validated_version(extraction_version)
        with self._write_lock:
            self._begin_write()
            try:
                counts = {
                    "sources": int(self._conn.execute(
                        "SELECT COUNT(*) FROM lcm_assertion_sources WHERE extraction_version = ?",
                        (version,),
                    ).fetchone()[0]),
                    "assertions": int(self._conn.execute(
                        "SELECT COUNT(*) FROM lcm_assertions WHERE extraction_version = ?",
                        (version,),
                    ).fetchone()[0]),
                    "relations": int(self._conn.execute(
                        "SELECT COUNT(*) FROM lcm_assertion_relations WHERE extraction_version = ?",
                        (version,),
                    ).fetchone()[0]),
                }
                # The receipt-delete trigger removes owned assertion/relation rows.
                self._conn.execute(
                    "DELETE FROM lcm_assertion_sources WHERE extraction_version = ?",
                    (version,),
                )
                self._conn.execute("COMMIT")
                return counts
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
