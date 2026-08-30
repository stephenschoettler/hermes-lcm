"""Immutable trajectory sources and bounded exact retrieval in one ``lcm.db``.

The store is intentionally provider-free.  It models agent trajectories as
first-class source material instead of flattening them into chat messages.
One database owns one corpus identity; normalized states and image manifests
remain traceable to protected canonical source JSON.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import struct
import threading
import time
from typing import Any, Callable, Iterable, Protocol, Sequence
from urllib.parse import quote, unquote

from .db_bootstrap import (
    configure_connection,
    get_fts_shadow_table_names,
    mark_migration_step_complete,
    refuse_schema_version_too_new,
    run_versioned_migrations,
)
from .ingest_protection import redact_sensitive_value
from .search_query import extract_search_terms


TRAJECTORY_MIGRATION_STEP = "trajectory_store_v1"
TRAJECTORY_SCHEMA_VERSION = 1
TRAJECTORY_SEMANTIC_DOCUMENT_VERSION = "trajectory-semantic-document-v1"
# Per-STATE semantic index (issue #142, Lane S / W3a). A separate, additive
# embedding space keyed by state_id (the source-level index above is one coarse
# vector per trajectory). Bumping this string invalidates a state backfill the
# same way the source version does.
TRAJECTORY_STATE_SEMANTIC_DOCUMENT_VERSION = "trajectory-state-semantic-document-v1"
# Voyage single-request caps (mirrors embedding_provider's budgets, applied with
# the same 0.9 safety factor the provider uses): one document <= 27K tokens, one
# request <= 80K tokens. The state backfill packs 32 items / 72K tokens per
# request (matches the #141 sizing simulation); a state above the per-document
# budget takes the chunked path (token-window split, mean-pooled).
_STATE_EMBED_MAX_BATCH_ITEMS = 32
_STATE_EMBED_DOCUMENT_TOKEN_BUDGET = int(27_000 * 0.9)
_STATE_EMBED_BATCH_TOKEN_BUDGET = int(80_000 * 0.9)

_TRAJECTORY_BASE_SCHEMA: dict[str, frozenset[str]] = {
    "lcm_trajectory_corpora": frozenset({
        "singleton", "identity_digest", "identity_json", "schema_version",
        "corpus_uid", "haystack_digest", "source_manifest_digest",
        "trajectory_count", "ingest_cursor", "status", "created_at",
        "completed_at",
    }),
    "lcm_trajectory_sources": frozenset({
        "source_id", "trajectory_id", "ordinal", "source_json", "source_sha256",
        "goal", "start_url", "outcome", "state_count", "inserted_at",
    }),
    "lcm_trajectory_states": frozenset({
        "state_id", "source_id", "state_index", "sequence_ordinal", "step",
        "url", "incoming_action", "thoughts", "text", "search_text",
        "observed_at", "observed_at_source", "occurred_at",
        "occurred_at_source", "ingested_at",
    }),
    "lcm_trajectory_assets": frozenset({
        "asset_id", "state_id", "relative_path", "sha256", "byte_size",
    }),
    "lcm_trajectory_ingest_receipts": frozenset({
        "ordinal", "trajectory_id", "source_sha256", "committed_at",
    }),
    "lcm_trajectory_transitions": frozenset({
        "transition_id", "source_id", "sequence_ordinal", "pre_state_id",
        "post_state_id", "incoming_action",
    }),
}
_TRAJECTORY_OPTIONAL_SCHEMAS: tuple[dict[str, frozenset[str]], ...] = (
    {
        "lcm_trajectory_embedding_profiles": frozenset({
            "profile_digest", "provider", "model_name", "dim",
            "document_version", "source_manifest_digest", "document_count",
            "index_digest", "active", "created_at",
        }),
        "lcm_trajectory_embeddings": frozenset({
            "source_id", "profile_digest", "document_sha256", "vector",
            "embedded_at",
        }),
    },
    {
        "lcm_trajectory_state_embedding_profiles": frozenset({
            "profile_digest", "provider", "model_name", "dim",
            "document_version", "source_manifest_digest", "state_count",
            "active", "created_at",
        }),
        "lcm_trajectory_state_embeddings": frozenset({
            "state_id", "profile_digest", "document_sha256", "vector",
            "embedded_at",
        }),
    },
)


def _verify_trajectory_schema(conn: sqlite3.Connection) -> list[str]:
    """Return any non-current trajectory shape without mutating source data."""
    findings: list[str] = []
    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE 'lcm_trajectory%'"
        )
    }
    expected = dict(_TRAJECTORY_BASE_SCHEMA)
    expected_fts = "lcm_trajectory_states_fts"
    allowed = set(expected) | {expected_fts}
    allowed.update(get_fts_shadow_table_names(expected_fts))
    for optional in _TRAJECTORY_OPTIONAL_SCHEMAS:
        if tables.intersection(optional):
            expected.update(optional)
            allowed.update(optional)

    findings.extend(
        f"unexpected-table:{table}" for table in sorted(tables - allowed)
    )
    for table, columns in expected.items():
        actual = {
            str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")
        }
        if not actual:
            findings.append(f"table:{table}")
            continue
        findings.extend(
            f"column:{table}.{column}" for column in sorted(columns - actual)
        )
        findings.extend(
            f"unexpected-column:{table}.{column}"
            for column in sorted(actual - columns)
        )
    if expected_fts not in tables:
        findings.append(f"table:{expected_fts}")

    required_objects = {
        "index": {
            "lcm_trajectory_states_source_sequence",
            *(
                {
                    "lcm_trajectory_embedding_one_active",
                    "lcm_trajectory_embeddings_profile",
                }
                if "lcm_trajectory_embeddings" in expected
                else set()
            ),
            *(
                {
                    "lcm_trajectory_state_embedding_one_active",
                    "lcm_trajectory_state_embeddings_profile",
                }
                if "lcm_trajectory_state_embeddings" in expected
                else set()
            ),
        },
        "trigger": {
            "lcm_trajectory_fts_insert",
            "lcm_trajectory_fts_delete",
            "lcm_trajectory_fts_update",
        },
    }
    for object_type, names in required_objects.items():
        actual = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = ? "
                "AND name LIKE 'lcm_trajectory%'",
                (object_type,),
            )
        }
        findings.extend(
            f"{object_type}:{name}" for name in sorted(names - actual)
        )
        findings.extend(
            f"unexpected-{object_type}:{name}" for name in sorted(actual - names)
        )
    return findings
_MAX_CANDIDATES = 128
_MAX_RESULTS = 24
_MAX_IMAGES = 8
_MAX_QUERY_TEXT_CHARS = 8_000
_MAX_ADJACENCY_RADIUS = 8
_MAX_DIVERSITY_CAP = 24
_MAX_SHARP_TOKEN_BUDGET = 4_000
# Knob G (HERMES_LCM_ANTIBOILERPLATE): additive re-weighting inside the C1
# per-trajectory MMR survivor selection. A candidate is penalized by how much
# it lexically resembles the OTHER pooled states of its own trajectory
# (boilerplate looks like its siblings) and rewarded for query-term density, so
# a trajectory's allotted seats go to query-relevant states rather than repeated
# task headers / page furniture. Both terms are in [0, 1]; the weights keep the
# base position-relevance signal dominant while giving the two signals real
# re-ranking influence.
_ANTIBOILERPLATE_BOILERPLATE_WEIGHT = 0.30
_ANTIBOILERPLATE_DENSITY_WEIGHT = 0.30
# Knob H (HERMES_LCM_TITLE_BOOST): contiguous question n-gram sizes matched
# against a candidate's normalized title/heading/field-label text at the lexical
# candidate stage.
_TITLE_BOOST_MIN_GRAM = 2
_TITLE_BOOST_MAX_GRAM = 4
_MAX_SOURCE_JSON_CHARS = 16_000_000
_MAX_TEXT_CHARS = 2_000_000
_MAX_SEMANTIC_DOCUMENT_CHARS = 48_000
_MAX_SEMANTIC_STATE_CHARS = 900
_MAX_SEMANTIC_STATES = 64
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "before", "did", "do",
    "does", "for", "from", "happen", "happened", "how", "i", "in", "is",
    "it", "of", "on", "or", "should", "the", "then", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would",
})
_TEMPORAL_TERMS = frozenset({
    "after", "before", "current", "date", "day", "earliest", "first",
    "initial", "last", "latest", "month", "newest", "previous", "recent",
    "time", "timestamp", "today", "week", "when", "year", "yesterday",
})
_ACTION_TERMS = frozenset({
    "add", "apply", "assign", "buy", "change", "choose", "click", "compare",
    "configure", "create", "delete", "edit", "export", "filter", "insert",
    "navigate", "open", "order", "remove", "retry", "save", "search",
    "select", "set", "sort", "submit", "update", "view",
})
_EXACT_REF_RE = re.compile(
    r"^trajectory://(?P<corpus>[0-9a-f]{64})/"
    r"(?P<trajectory>[^/]+)/state/(?P<state>[0-9]+)$"
)


class TrajectoryStoreError(RuntimeError):
    """Base class for trajectory-store failures."""


class CorpusIdentityError(TrajectoryStoreError):
    """Raised when a database belongs to another immutable corpus."""


class TrajectoryAssetError(TrajectoryStoreError):
    """Raised when an image asset is missing, changed, or escapes its root."""


class ExactTrajectoryRefError(TrajectoryStoreError):
    """Raised when an exact trajectory reference is invalid or unresolved."""


class TrajectorySchemaUnavailableError(TrajectoryStoreError):
    """Raised when a read-only database lacks the trajectory schema."""


class TrajectoryEmbeddingProvider(Protocol):
    """Provider-neutral surface needed by the trajectory semantic index."""

    provider_id: str
    model_id: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class CorpusIdentity:
    dataset_name: str
    dataset_revision: str
    harness_commit: str
    tier: str
    domain: str
    ingest_config_digest: str = ""

    def to_dict(self) -> dict[str, str]:
        values = {
            "dataset_name": self.dataset_name,
            "dataset_revision": self.dataset_revision,
            "harness_commit": self.harness_commit,
            "tier": self.tier,
            "domain": self.domain,
            "ingest_config_digest": self.ingest_config_digest,
        }
        normalized = {key: str(value or "").strip() for key, value in values.items()}
        missing = [key for key, value in normalized.items() if key != "ingest_config_digest" and not value]
        if missing:
            raise ValueError(f"corpus identity fields must not be empty: {missing}")
        return normalized

    @property
    def digest(self) -> str:
        return _sha256_text(_canonical_json(self.to_dict()))


@dataclass(frozen=True)
class TrajectoryState:
    state_index: int
    step: int
    url: str
    incoming_action: str | None
    thoughts: str | None
    text: str
    screenshot_path: str | Path
    observed_at: float | None = None
    observed_at_source: str | None = None
    occurred_at: float | None = None
    occurred_at_source: str | None = None


@dataclass(frozen=True)
class TrajectorySource:
    trajectory_id: str
    ordinal: int
    goal: str
    start_url: str
    outcome: str | None
    states: tuple[TrajectoryState, ...]
    source_payload: Any


@dataclass(frozen=True)
class TrajectoryInsertResult:
    trajectory_id: str
    source_sha256: str
    state_count: int
    already_current: bool


@dataclass(frozen=True)
class TrajectorySemanticAttempt:
    """Typed record of one semantic-retrieval attempt on the query path.

    Replaces the historical bare ``except Exception: fallbacks += 1`` that
    discarded the failure class/status entirely (the defect that hid the
    client-side spend-guard rate-limit behind an undifferentiated counter in
    the frozen V2 run). ``outcome`` is ``"success"`` or ``"fallback"``; failure
    fields are populated by best-effort duck-typing off the raised exception so
    this record stays decoupled from the embedding_provider exception classes.
    """

    provider: str
    model: str
    outcome: str
    exception_class: str | None = None
    http_status: int | None = None
    retry_after: float | None = None
    latency_ms: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class TrajectoryHit:
    exact_ref: str
    trajectory_id: str
    goal: str
    outcome: str | None
    state_index: int
    sequence_ordinal: int
    step: int
    url: str
    incoming_action: str | None
    thoughts: str | None
    text: str
    text_offset: int
    text_truncated: bool
    observed_at: float | None
    observed_at_source: str | None
    occurred_at: float | None
    occurred_at_source: str | None
    screenshot_path: str | None
    screenshot_sha256: str | None
    score: float
    match_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_ref": self.exact_ref,
            "trajectory_id": self.trajectory_id,
            "goal": self.goal,
            "outcome": self.outcome,
            "state_index": self.state_index,
            "sequence_ordinal": self.sequence_ordinal,
            "step": self.step,
            "url": self.url,
            "incoming_action": self.incoming_action,
            "thoughts": self.thoughts,
            "text": self.text,
            "text_offset": self.text_offset,
            "text_truncated": self.text_truncated,
            "observed_at": self.observed_at,
            "observed_at_source": self.observed_at_source,
            "occurred_at": self.occurred_at,
            "occurred_at_source": self.occurred_at_source,
            "screenshot_path": self.screenshot_path,
            "screenshot_sha256": self.screenshot_sha256,
            "score": round(float(self.score), 8),
            "match_kind": self.match_kind,
        }


class _ProtectionConfig:
    sensitive_patterns_enabled = True
    sensitive_patterns = (
        "api_key",
        "bearer_token",
        "password_assignment",
        "private_key",
    )
    sensitive_patterns_source = "trajectory_store"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError("trajectory source must be finite JSON") from exc
    if len(encoded) > _MAX_SOURCE_JSON_CHARS:
        raise ValueError(
            f"trajectory source exceeds {_MAX_SOURCE_JSON_CHARS} characters"
        )
    return encoded


def _normalized_vector(values: Sequence[float], *, expected_dim: int | None = None) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory embedding vector must be numeric") from exc
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("trajectory embedding vector must be finite and non-empty")
    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError("trajectory embedding vector dimension changed")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("trajectory embedding vector must have a finite nonzero norm")
    return tuple(value / norm for value in vector)


def _pack_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(payload: bytes, dim: int) -> tuple[float, ...]:
    expected_bytes = dim * 4
    if len(payload) != expected_bytes:
        raise TrajectoryStoreError("stored trajectory embedding dimension is invalid")
    return tuple(struct.unpack(f"<{dim}f", payload))


def create_trajectory_embedding_provider(
    provider_name: str,
    model_name: str,
    *,
    timeout_seconds: float,
    for_backfill: bool = False,
) -> TrajectoryEmbeddingProvider:
    """Resolve an existing LCM provider without persisting its credential.

    The provider reads credentials from its normal environment-backed secret
    seam. Only the provider/model identifiers belong in saved memory config.
    """
    from .config import LCMConfig
    from .embedding_provider import resolve_provider

    timeout = max(0.1, float(timeout_seconds))
    config = LCMConfig(
        embedding_provider=str(provider_name).strip(),
        embedding_model=str(model_name).strip(),
        embedding_query_timeout_s=timeout,
        embedding_backfill_timeout_s=timeout,
    )
    provider = resolve_provider(config, for_backfill=for_backfill)
    if provider is None:
        raise TrajectoryStoreError("trajectory embedding provider is not configured")
    return provider


def _finite_optional(value: float | None, field: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _time_with_source(
    value: float | None,
    source: str | None,
    field: str,
) -> tuple[float | None, str | None]:
    timestamp = _finite_optional(value, field)
    normalized_source = (
        _bounded_text(source, f"{field}_source", allow_empty=False).strip()
        if source is not None
        else None
    )
    if timestamp is None and normalized_source is not None:
        raise ValueError(f"{field}_source requires {field}")
    if timestamp is not None and normalized_source is None:
        raise ValueError(f"{field} requires explicit {field}_source provenance")
    return timestamp, normalized_source


def _bounded_text(value: Any, field: str, *, allow_empty: bool = True) -> str:
    text = str(value if value is not None else "")
    if not allow_empty and not text.strip():
        raise ValueError(f"{field} must not be empty")
    if len(text) > _MAX_TEXT_CHARS:
        raise ValueError(f"{field} exceeds {_MAX_TEXT_CHARS} characters")
    return text


class TrajectoryStore:
    """SQLite trajectory store bound to the same physical file as LCM core."""

    def __init__(
        self,
        db_path: str | Path,
        identity: CorpusIdentity,
        *,
        asset_root: str | Path,
        read_only: bool = False,
        protect_sensitive: bool = True,
        embedding_provider: TrajectoryEmbeddingProvider | None = None,
        semantic_top_trajectories: int = 12,
    ) -> None:
        self.db_path = Path(db_path)
        self.identity = identity
        self.identity_payload = identity.to_dict()
        self.identity_digest = identity.digest
        self.asset_root = Path(asset_root).expanduser().resolve()
        self.read_only = bool(read_only)
        self.protect_sensitive = bool(protect_sensitive)
        self.embedding_provider = embedding_provider
        self.semantic_top_trajectories = min(
            max(1, int(semantic_top_trajectories)),
            32,
        )
        self._semantic_usage: dict[str, int] = {
            "document_calls": 0,
            "document_tokens": 0,
            "query_calls": 0,
            "query_tokens": 0,
            "fallbacks": 0,
        }
        # Typed per-run instrument state (additive; the existing
        # ``_semantic_usage`` counters above are unchanged for back-compat).
        # The attempt log is BOUNDED (recent-window ring) so a long run cannot
        # grow it without bound; the funnel counters are tracked SEPARATELY and
        # stay cumulative regardless of the ring's cap.
        self._semantic_attempts: deque[TrajectorySemanticAttempt] = deque(maxlen=1024)
        self._semantic_attempt_totals: dict[str, Any] = {
            "attempts": 0,
            "successes": 0,
            "fallbacks": 0,
            "fallbacks_by_reason": {},
        }
        self._last_semantic_attempt: TrajectorySemanticAttempt | None = None
        self._last_query_telemetry: dict[str, Any] | None = None
        # Lazily-populated per-STATE semantic matrix cache (issue #142): the
        # tuple is ``(profile_digest, freshness, state_ids, matrix)`` where
        # ``freshness`` is the profile row count/latest-write marker and
        # ``matrix`` is a normalized float32 array (numpy when available, else a
        # list of tuples). The marker also catches supported same-profile
        # rewrites that occur outside this instance's explicit cache reset.
        self._state_semantic_cache: (
            tuple[str, tuple[int, float], list[int], Any] | None
        ) = None
        self._lock = threading.RLock()
        self._conn = self._open_connection()
        try:
            self._validate_existing_schema_version()
            self._init_schema()
            self._bind_identity()
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
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _validate_existing_schema_version(self) -> None:
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'lcm_trajectory_corpora'"
        ).fetchone()
        if exists is None:
            return
        try:
            row = self._conn.execute(
                "SELECT schema_version FROM lcm_trajectory_corpora "
                "WHERE singleton = 1"
            ).fetchone()
        except sqlite3.OperationalError:
            # Two very different tables reach this branch, and only one of them
            # should be tolerated:
            #
            #   LEGACY   -- a corpora table written before the ``schema_version``
            #               column existed. It still carries the core identity
            #               columns, and failing the open on it would brick a
            #               store that is merely old. Treat it as unversioned,
            #               the same way the table-absent branch above does.
            #   MALFORMED -- a table that is missing the core columns as well.
            #               Nothing can be recovered from it, and swallowing the
            #               error here lets the open continue until some later
            #               INSERT fails on an arbitrary column, reporting the
            #               wrong cause and doing so AFTER FTS repair has run.
            #
            # Distinguish them by a column the legacy table certainly has.
            columns = {
                str(row[1])
                for row in self._conn.execute(
                    "PRAGMA table_info(lcm_trajectory_corpora)"
                )
            }
            if "identity_digest" not in columns:
                raise
            return

        if row is not None and int(row["schema_version"]) != TRAJECTORY_SCHEMA_VERSION:
            raise CorpusIdentityError(
                "trajectory database corpus identity does not match requested identity"
            )

    def _init_schema(self) -> None:
        required = {
            "lcm_trajectory_corpora",
            "lcm_trajectory_sources",
            "lcm_trajectory_states",
            "lcm_trajectory_assets",
            "lcm_trajectory_ingest_receipts",
            "lcm_trajectory_transitions",
            "lcm_trajectory_states_fts",
        }
        if self.read_only:
            existing = {
                str(row[0])
                for row in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE name LIKE 'lcm_trajectory%'"
                )
            }
            missing = sorted(required - existing)
            if missing:
                raise TrajectorySchemaUnavailableError(
                    f"trajectory schema unavailable for read-only query: {missing}"
                )
            findings = _verify_trajectory_schema(self.connection)
            if findings:
                raise TrajectorySchemaUnavailableError(
                    "trajectory schema unavailable for read-only query: "
                    f"{findings}"
                )
            return

        run_versioned_migrations(self._conn)
        fts_preexisting = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'lcm_trajectory_states_fts'"
        ).fetchone() is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lcm_trajectory_corpora (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                identity_digest TEXT NOT NULL UNIQUE,
                identity_json TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                corpus_uid TEXT,
                haystack_digest TEXT,
                source_manifest_digest TEXT,
                trajectory_count INTEGER,
                ingest_cursor INTEGER NOT NULL DEFAULT 0 CHECK(ingest_cursor >= 0),
                status TEXT NOT NULL CHECK(status IN ('building', 'complete', 'invalid')),
                created_at REAL NOT NULL,
                completed_at REAL
            );

            CREATE TABLE IF NOT EXISTS lcm_trajectory_sources (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trajectory_id TEXT NOT NULL UNIQUE,
                ordinal INTEGER NOT NULL UNIQUE,
                source_json TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                goal TEXT NOT NULL,
                start_url TEXT NOT NULL,
                outcome TEXT,
                state_count INTEGER NOT NULL CHECK(state_count > 0),
                inserted_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lcm_trajectory_states (
                state_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES lcm_trajectory_sources(source_id) ON DELETE CASCADE,
                state_index INTEGER NOT NULL CHECK(state_index >= 0),
                sequence_ordinal INTEGER NOT NULL CHECK(sequence_ordinal >= 0),
                step INTEGER NOT NULL CHECK(step >= 0),
                url TEXT NOT NULL,
                incoming_action TEXT,
                thoughts TEXT,
                text TEXT NOT NULL,
                search_text TEXT NOT NULL,
                observed_at REAL,
                observed_at_source TEXT,
                occurred_at REAL,
                occurred_at_source TEXT,
                ingested_at REAL NOT NULL,
                UNIQUE(source_id, state_index),
                UNIQUE(source_id, sequence_ordinal)
            );

            CREATE TABLE IF NOT EXISTS lcm_trajectory_assets (
                asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_id INTEGER NOT NULL UNIQUE REFERENCES lcm_trajectory_states(state_id) ON DELETE CASCADE,
                relative_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                byte_size INTEGER NOT NULL CHECK(byte_size >= 0)
            );

            CREATE TABLE IF NOT EXISTS lcm_trajectory_ingest_receipts (
                ordinal INTEGER PRIMARY KEY CHECK(ordinal >= 0),
                trajectory_id TEXT NOT NULL UNIQUE,
                source_sha256 TEXT NOT NULL,
                committed_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lcm_trajectory_transitions (
                transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES lcm_trajectory_sources(source_id) ON DELETE CASCADE,
                sequence_ordinal INTEGER NOT NULL CHECK(sequence_ordinal >= 1),
                pre_state_id INTEGER NOT NULL REFERENCES lcm_trajectory_states(state_id) ON DELETE CASCADE,
                post_state_id INTEGER NOT NULL REFERENCES lcm_trajectory_states(state_id) ON DELETE CASCADE,
                incoming_action TEXT,
                UNIQUE(source_id, sequence_ordinal),
                UNIQUE(source_id, post_state_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS lcm_trajectory_states_fts
            USING fts5(search_text, content='lcm_trajectory_states', content_rowid='state_id');

            CREATE TRIGGER IF NOT EXISTS lcm_trajectory_fts_insert
            AFTER INSERT ON lcm_trajectory_states BEGIN
                INSERT INTO lcm_trajectory_states_fts(rowid, search_text)
                VALUES (new.state_id, new.search_text);
            END;

            CREATE TRIGGER IF NOT EXISTS lcm_trajectory_fts_delete
            AFTER DELETE ON lcm_trajectory_states BEGIN
                INSERT INTO lcm_trajectory_states_fts(lcm_trajectory_states_fts, rowid, search_text)
                VALUES ('delete', old.state_id, old.search_text);
            END;

            CREATE TRIGGER IF NOT EXISTS lcm_trajectory_fts_update
            AFTER UPDATE OF search_text ON lcm_trajectory_states BEGIN
                INSERT INTO lcm_trajectory_states_fts(lcm_trajectory_states_fts, rowid, search_text)
                VALUES ('delete', old.state_id, old.search_text);
                INSERT INTO lcm_trajectory_states_fts(rowid, search_text)
                VALUES (new.state_id, new.search_text);
            END;

            CREATE INDEX IF NOT EXISTS lcm_trajectory_states_source_sequence
            ON lcm_trajectory_states(source_id, sequence_ordinal);
            """
        )
        state_count = int(
            self._conn.execute("SELECT COUNT(*) FROM lcm_trajectory_states").fetchone()[0]
        )
        fts_count = int(
            self._conn.execute("SELECT COUNT(*) FROM lcm_trajectory_states_fts").fetchone()[0]
        )
        fts_needs_rebuild = not fts_preexisting or state_count != fts_count
        if not fts_needs_rebuild:
            try:
                self._conn.execute(
                    "INSERT INTO lcm_trajectory_states_fts(lcm_trajectory_states_fts, rank) "
                    "VALUES ('integrity-check', 1)"
                )
            except sqlite3.DatabaseError:
                fts_needs_rebuild = True
        if fts_needs_rebuild:
            self._conn.execute(
                "INSERT INTO lcm_trajectory_states_fts(lcm_trajectory_states_fts) VALUES ('rebuild')"
            )
        marker = self._conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name = ?",
            (TRAJECTORY_MIGRATION_STEP,),
        ).fetchone()
        if marker is None:
            mark_migration_step_complete(self._conn, TRAJECTORY_MIGRATION_STEP)
        self._conn.commit()

    def _bind_identity(self) -> None:
        row = self._conn.execute(
            "SELECT * FROM lcm_trajectory_corpora WHERE singleton = 1"
        ).fetchone()
        if row is None:
            if self.read_only:
                raise CorpusIdentityError("trajectory database has no corpus identity")
            self._conn.execute(
                """
                INSERT INTO lcm_trajectory_corpora(
                    singleton, identity_digest, identity_json, schema_version,
                    status, created_at
                ) VALUES (1, ?, ?, ?, 'building', ?)
                """,
                (
                    self.identity_digest,
                    _canonical_json(self.identity_payload),
                    TRAJECTORY_SCHEMA_VERSION,
                    time.time(),
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM lcm_trajectory_corpora WHERE singleton = 1"
            ).fetchone()
        if (
            str(row["identity_digest"]) != self.identity_digest
            or int(row["schema_version"]) != TRAJECTORY_SCHEMA_VERSION
        ):
            raise CorpusIdentityError(
                "trajectory database corpus identity does not match requested identity"
            )

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    @property
    def status(self) -> str:
        row = self._conn.execute(
            "SELECT status FROM lcm_trajectory_corpora WHERE singleton = 1"
        ).fetchone()
        return str(row[0])

    @property
    def corpus_uid(self) -> str | None:
        row = self._conn.execute(
            "SELECT corpus_uid FROM lcm_trajectory_corpora WHERE singleton = 1"
        ).fetchone()
        return str(row[0]) if row and row[0] else None

    def _require_writable(self) -> None:
        if self.read_only:
            raise TrajectoryStoreError("read-only TrajectoryStore cannot write")

    def _protect(self, value: Any) -> Any:
        if not self.protect_sensitive:
            return value
        return redact_sensitive_value(
            value,
            _ProtectionConfig(),
            parse_json_strings=True,
        )

    def _validated_asset(self, path_value: str | Path) -> tuple[str, str, int]:
        try:
            candidate = Path(path_value).expanduser().resolve(strict=True)
            root = self.asset_root.resolve(strict=True)
            relative = candidate.relative_to(root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise TrajectoryAssetError(
                "trajectory screenshot must exist inside the configured asset root"
            ) from exc
        if not candidate.is_file():
            raise TrajectoryAssetError(f"trajectory screenshot is not a file: {candidate}")
        return relative.as_posix(), _sha256_file(candidate), candidate.stat().st_size

    def _protected_source(
        self, source: TrajectorySource
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[tuple[str, str, int], ...]]:
        trajectory_id = _bounded_text(
            source.trajectory_id, "trajectory_id", allow_empty=False
        ).strip()
        if source.ordinal < 0:
            raise ValueError("trajectory ordinal must be non-negative")
        if not source.states:
            raise ValueError("trajectory must contain at least one state")
        state_indexes = [state.state_index for state in source.states]
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in state_indexes
        ):
            raise ValueError("state indexes must be non-negative integers")
        if len(state_indexes) != len(set(state_indexes)):
            raise ValueError("trajectory state indexes must be unique")

        protected_states: list[dict[str, Any]] = []
        assets: list[tuple[str, str, int]] = []
        for sequence_ordinal, state in enumerate(source.states):
            if (
                not isinstance(state.step, int)
                or isinstance(state.step, bool)
                or state.step < 0
            ):
                raise ValueError("trajectory state step must be a non-negative integer")
            observed_at, observed_at_source = _time_with_source(
                state.observed_at,
                state.observed_at_source,
                "observed_at",
            )
            occurred_at, occurred_at_source = _time_with_source(
                state.occurred_at,
                state.occurred_at_source,
                "occurred_at",
            )
            relative_path, asset_sha, asset_size = self._validated_asset(
                state.screenshot_path
            )
            assets.append((relative_path, asset_sha, asset_size))
            protected = self._protect({
                "state_index": int(state.state_index),
                "sequence_ordinal": sequence_ordinal,
                "step": int(state.step),
                "url": _bounded_text(state.url, "url", allow_empty=False),
                "incoming_action": (
                    _bounded_text(state.incoming_action, "incoming_action")
                    if state.incoming_action is not None
                    else None
                ),
                "thoughts": (
                    _bounded_text(state.thoughts, "thoughts")
                    if state.thoughts is not None
                    else None
                ),
                "text": _bounded_text(state.text, "text"),
                "screenshot": relative_path,
                "screenshot_sha256": asset_sha,
                "observed_at": observed_at,
                "observed_at_source": observed_at_source,
                "occurred_at": occurred_at,
                "occurred_at_source": occurred_at_source,
            })
            protected_states.append(dict(protected))

        protected_source = self._protect({
            "source_payload": source.source_payload,
            "normalized": {
                "trajectory_id": trajectory_id,
                "ordinal": int(source.ordinal),
                "goal": _bounded_text(source.goal, "goal"),
                "start_url": _bounded_text(
                    source.start_url, "start_url", allow_empty=False
                ),
                "outcome": (
                    _bounded_text(source.outcome, "outcome")
                    if source.outcome is not None
                    else None
                ),
                "states": protected_states,
            },
        })
        return dict(protected_source), tuple(protected_states), tuple(assets)

    @staticmethod
    def _search_text(
        *,
        goal: str,
        outcome: str | None,
        url: str,
        incoming_action: str | None,
        thoughts: str | None,
        text: str,
    ) -> str:
        return "\n".join((
            f"Goal: {goal}",
            f"Outcome: {outcome or ''}",
            f"URL: {url}",
            f"Incoming action: {incoming_action or ''}",
            f"Thought: {thoughts or ''}",
            f"Visible state: {text}",
        ))

    def insert(self, source: TrajectorySource) -> TrajectoryInsertResult:
        self._require_writable()
        protected_source, protected_states, assets = self._protected_source(source)
        source_json = _canonical_json(protected_source)
        source_sha = _sha256_text(source_json)
        normalized = protected_source["normalized"]
        trajectory_id = str(normalized["trajectory_id"])
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._conn.execute(
                    """
                    SELECT source_sha256, state_count
                    FROM lcm_trajectory_sources
                    WHERE trajectory_id = ?
                    """,
                    (trajectory_id,),
                ).fetchone()
                if existing is not None:
                    if str(existing["source_sha256"]) != source_sha:
                        raise TrajectoryStoreError(
                            f"trajectory {trajectory_id} already exists with a different digest"
                        )
                    self._conn.commit()
                    return TrajectoryInsertResult(
                        trajectory_id=trajectory_id,
                        source_sha256=source_sha,
                        state_count=int(existing["state_count"]),
                        already_current=True,
                    )
                corpus_row = self._conn.execute(
                    """
                    SELECT status, ingest_cursor
                    FROM lcm_trajectory_corpora WHERE singleton = 1
                    """
                ).fetchone()
                if str(corpus_row["status"]) == "complete":
                    raise TrajectoryStoreError("complete trajectory corpus is immutable")
                expected_ordinal = int(corpus_row["ingest_cursor"])
                if int(normalized["ordinal"]) != expected_ordinal:
                    raise TrajectoryStoreError(
                        f"trajectory ordinal {normalized['ordinal']} does not match contiguous ingest cursor {expected_ordinal}"
                    )

                now = time.time()
                cursor = self._conn.execute(
                    """
                    INSERT INTO lcm_trajectory_sources(
                        trajectory_id, ordinal, source_json, source_sha256,
                        goal, start_url, outcome, state_count, inserted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trajectory_id,
                        int(normalized["ordinal"]),
                        source_json,
                        source_sha,
                        str(normalized["goal"]),
                        str(normalized["start_url"]),
                        normalized["outcome"],
                        len(protected_states),
                        now,
                    ),
                )
                source_id = int(cursor.lastrowid)
                inserted_state_ids: list[int] = []
                for state, asset in zip(protected_states, assets):
                    search_text = self._search_text(
                        goal=str(normalized["goal"]),
                        outcome=normalized["outcome"],
                        url=str(state["url"]),
                        incoming_action=state["incoming_action"],
                        thoughts=state["thoughts"],
                        text=str(state["text"]),
                    )
                    state_cursor = self._conn.execute(
                        """
                        INSERT INTO lcm_trajectory_states(
                            source_id, state_index, sequence_ordinal, step, url,
                            incoming_action, thoughts, text, search_text,
                            observed_at, observed_at_source,
                            occurred_at, occurred_at_source, ingested_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            source_id,
                            int(state["state_index"]),
                            int(state["sequence_ordinal"]),
                            int(state["step"]),
                            str(state["url"]),
                            state["incoming_action"],
                            state["thoughts"],
                            str(state["text"]),
                            search_text,
                            state["observed_at"],
                            state["observed_at_source"],
                            state["occurred_at"],
                            state["occurred_at_source"],
                            now,
                        ),
                    )
                    inserted_state_ids.append(int(state_cursor.lastrowid))
                    relative_path, asset_sha, asset_size = asset
                    self._conn.execute(
                        """
                        INSERT INTO lcm_trajectory_assets(
                            state_id, relative_path, sha256, byte_size
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            int(state_cursor.lastrowid),
                            relative_path,
                            asset_sha,
                            asset_size,
                        ),
                    )
                for sequence_ordinal in range(1, len(inserted_state_ids)):
                    state = protected_states[sequence_ordinal]
                    self._conn.execute(
                        """
                        INSERT INTO lcm_trajectory_transitions(
                            source_id, sequence_ordinal, pre_state_id,
                            post_state_id, incoming_action
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            source_id,
                            sequence_ordinal,
                            inserted_state_ids[sequence_ordinal - 1],
                            inserted_state_ids[sequence_ordinal],
                            state["incoming_action"],
                        ),
                    )
                self._conn.execute(
                    """
                    INSERT INTO lcm_trajectory_ingest_receipts(
                        ordinal, trajectory_id, source_sha256, committed_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (int(normalized["ordinal"]), trajectory_id, source_sha, now),
                )
                self._conn.execute(
                    """
                    UPDATE lcm_trajectory_corpora
                    SET ingest_cursor = ingest_cursor + 1
                    WHERE singleton = 1
                    """
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return TrajectoryInsertResult(
            trajectory_id=trajectory_id,
            source_sha256=source_sha,
            state_count=len(protected_states),
            already_current=False,
        )

    def finalize(self, ordered_trajectory_ids: Sequence[str]) -> str:
        self._require_writable()
        ordered = tuple(str(value or "").strip() for value in ordered_trajectory_ids)
        if not ordered or any(not value for value in ordered):
            raise ValueError("ordered trajectory ids must be non-empty")
        if len(ordered) != len(set(ordered)):
            raise ValueError("ordered trajectory ids must be unique")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                rows = self._conn.execute(
                    """
                    SELECT trajectory_id, source_sha256
                    FROM lcm_trajectory_sources
                    ORDER BY ordinal, trajectory_id
                    """
                ).fetchall()
                actual = tuple(str(row["trajectory_id"]) for row in rows)
                if actual != ordered:
                    raise CorpusIdentityError(
                        "inserted trajectory order does not match final ordered haystack"
                    )
                corpus_row = self._conn.execute(
                    """
                    SELECT status, corpus_uid, ingest_cursor
                    FROM lcm_trajectory_corpora WHERE singleton = 1
                    """
                ).fetchone()
                if int(corpus_row["ingest_cursor"]) != len(ordered):
                    raise CorpusIdentityError(
                        "trajectory ingest cursor does not cover the final ordered haystack"
                    )
                haystack_digest = _sha256_text(_canonical_json(list(ordered)))
                source_manifest_digest = _sha256_text(_canonical_json([
                    [str(row["trajectory_id"]), str(row["source_sha256"])]
                    for row in rows
                ]))
                corpus_uid = _sha256_text(
                    f"{self.identity_digest}\0{haystack_digest}\0{source_manifest_digest}"
                )
                if str(corpus_row["status"]) == "complete":
                    if str(corpus_row["corpus_uid"]) != corpus_uid:
                        raise CorpusIdentityError(
                            "complete corpus source or haystack identity mismatch"
                        )
                    self._conn.commit()
                    return corpus_uid
                self._conn.execute(
                    """
                    UPDATE lcm_trajectory_corpora
                    SET corpus_uid = ?, haystack_digest = ?, source_manifest_digest = ?,
                        trajectory_count = ?, status = 'complete', completed_at = ?
                    WHERE singleton = 1
                    """,
                    (
                        corpus_uid,
                        haystack_digest,
                        source_manifest_digest,
                        len(ordered),
                        time.time(),
                    ),
                )
                self._conn.commit()
                return corpus_uid
            except Exception:
                self._conn.rollback()
                raise

    def set_embedding_provider(
        self,
        provider: TrajectoryEmbeddingProvider | None,
    ) -> None:
        """Replace the ephemeral provider without changing saved corpus state."""
        self.embedding_provider = provider

    def _ensure_semantic_schema(self) -> None:
        self._require_writable()
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lcm_trajectory_embedding_profiles (
                profile_digest TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL CHECK(dim > 0),
                document_version TEXT NOT NULL,
                source_manifest_digest TEXT NOT NULL,
                document_count INTEGER NOT NULL CHECK(document_count >= 0),
                index_digest TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 0 CHECK(active IN (0, 1)),
                created_at REAL NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS lcm_trajectory_embedding_one_active
            ON lcm_trajectory_embedding_profiles(active) WHERE active = 1;

            CREATE TABLE IF NOT EXISTS lcm_trajectory_embeddings (
                source_id INTEGER PRIMARY KEY
                    REFERENCES lcm_trajectory_sources(source_id) ON DELETE CASCADE,
                profile_digest TEXT NOT NULL
                    REFERENCES lcm_trajectory_embedding_profiles(profile_digest)
                    ON DELETE CASCADE,
                document_sha256 TEXT NOT NULL,
                vector BLOB NOT NULL,
                embedded_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS lcm_trajectory_embeddings_profile
            ON lcm_trajectory_embeddings(profile_digest, source_id);
            """
        )
        self._conn.commit()

    def _semantic_profile(self) -> sqlite3.Row | None:
        exists = self._conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'lcm_trajectory_embedding_profiles'
            """
        ).fetchone()
        if exists is None:
            return None
        return self._conn.execute(
            """
            SELECT * FROM lcm_trajectory_embedding_profiles
            WHERE active = 1
            """
        ).fetchone()

    @staticmethod
    def _sample_state_rows(rows: Sequence[sqlite3.Row]) -> list[sqlite3.Row]:
        if len(rows) <= _MAX_SEMANTIC_STATES:
            return list(rows)
        last = len(rows) - 1
        indexes = {
            round(position * last / (_MAX_SEMANTIC_STATES - 1))
            for position in range(_MAX_SEMANTIC_STATES)
        }
        return [rows[index] for index in sorted(indexes)]

    def _semantic_documents(self) -> list[tuple[int, str, str]]:
        sources = self._conn.execute(
            """
            SELECT source_id, trajectory_id, goal, start_url, outcome
            FROM lcm_trajectory_sources
            ORDER BY ordinal, trajectory_id
            """
        ).fetchall()
        documents: list[tuple[int, str, str]] = []
        for source in sources:
            states = self._conn.execute(
                """
                SELECT state_index, sequence_ordinal, step, url,
                       incoming_action, thoughts, text
                FROM lcm_trajectory_states
                WHERE source_id = ?
                ORDER BY sequence_ordinal
                """,
                (int(source["source_id"]),),
            ).fetchall()
            lines = [
                f"Trajectory: {source['trajectory_id']}",
                f"Goal: {source['goal']}",
                f"Start URL: {source['start_url']}",
                f"Outcome: {source['outcome'] or '<unknown>'}",
            ]
            for state in self._sample_state_rows(states):
                state_text = " | ".join(
                    part
                    for part in (
                        f"State {state['state_index']} sequence {state['sequence_ordinal']} step {state['step']}",
                        f"URL {state['url']}",
                        (
                            f"Action {state['incoming_action']}"
                            if state["incoming_action"] is not None
                            else ""
                        ),
                        (
                            f"Thought {state['thoughts']}"
                            if state["thoughts"] is not None
                            else ""
                        ),
                        f"Visible {state['text']}",
                    )
                    if part
                )
                lines.append(state_text[:_MAX_SEMANTIC_STATE_CHARS])
            document = "\n".join(lines)[:_MAX_SEMANTIC_DOCUMENT_CHARS]
            documents.append(
                (int(source["source_id"]), document, _sha256_text(document))
            )
        return documents

    def build_semantic_index(
        self,
        provider: TrajectoryEmbeddingProvider | None = None,
    ) -> dict[str, Any]:
        """Build one deterministic source-derived vector per trajectory."""
        if self.status != "complete":
            raise CorpusIdentityError("trajectory corpus must be finalized before indexing")
        active_provider = provider or self.embedding_provider
        if active_provider is None:
            raise TrajectoryStoreError("trajectory embedding provider is not configured")
        self._ensure_semantic_schema()
        corpus = self._conn.execute(
            """
            SELECT source_manifest_digest, trajectory_count
            FROM lcm_trajectory_corpora WHERE singleton = 1
            """
        ).fetchone()
        provider_name = str(getattr(active_provider, "provider_id", "unknown"))
        model_name = str(getattr(active_provider, "model_id", "")).strip()
        if not model_name:
            raise ValueError("trajectory embedding model_id must not be empty")
        current = self._semantic_profile()
        expected_count = int(corpus["trajectory_count"])
        if current is not None and (
            str(current["provider"]) == provider_name
            and str(current["model_name"]) == model_name
            and str(current["document_version"]) == TRAJECTORY_SEMANTIC_DOCUMENT_VERSION
            and str(current["source_manifest_digest"]) == str(corpus["source_manifest_digest"])
            and int(current["document_count"]) == expected_count
        ):
            actual_count = int(self._conn.execute(
                "SELECT COUNT(*) FROM lcm_trajectory_embeddings WHERE profile_digest = ?",
                (str(current["profile_digest"]),),
            ).fetchone()[0])
            if actual_count == expected_count:
                return {
                    "status": "current",
                    "profile_digest": str(current["profile_digest"]),
                    "document_count": actual_count,
                    "dim": int(current["dim"]),
                    "index_digest": str(current["index_digest"]),
                }

        documents = self._semantic_documents()
        texts = [document for _source_id, document, _digest in documents]
        embed_batches = getattr(active_provider, "embed_document_batches", None)
        vectors: list[list[float]] = []
        if callable(embed_batches):
            indexed_vectors: dict[int, list[float]] = {}
            for batch in embed_batches(texts):
                self._semantic_usage["document_calls"] += 1
                self._semantic_usage["document_tokens"] += max(
                    0,
                    int(getattr(active_provider, "last_usage_tokens", 0) or 0),
                )
                for index, vector in zip(batch.indexes, batch.vectors):
                    indexed_vectors[int(index)] = list(vector)
            vectors = [indexed_vectors[index] for index in range(len(texts))]
        else:
            self._semantic_usage["document_calls"] += 1
            vectors = active_provider.embed_documents(texts)
            self._semantic_usage["document_tokens"] += max(
                0,
                int(getattr(active_provider, "last_usage_tokens", 0) or 0),
            )
        if len(vectors) != len(documents):
            raise ValueError("trajectory embedding count does not match source count")
        normalized: list[tuple[float, ...]] = []
        dim: int | None = None
        for vector in vectors:
            normalized_vector = _normalized_vector(vector, expected_dim=dim)
            if dim is None:
                dim = len(normalized_vector)
            normalized.append(normalized_vector)
        if dim is None:
            raise ValueError("trajectory semantic index cannot be empty")
        profile_digest = _sha256_text(_canonical_json({
            "provider": provider_name,
            "model": model_name,
            "dim": dim,
            "document_version": TRAJECTORY_SEMANTIC_DOCUMENT_VERSION,
            "source_manifest_digest": str(corpus["source_manifest_digest"]),
        }))
        packed = [_pack_vector(vector) for vector in normalized]
        index_digest = _sha256_text(_canonical_json([
            [source_id, document_sha, hashlib.sha256(vector).hexdigest()]
            for (source_id, _document, document_sha), vector in zip(documents, packed)
        ]))
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "UPDATE lcm_trajectory_embedding_profiles SET active = 0 WHERE active = 1"
                )
                self._conn.execute("DELETE FROM lcm_trajectory_embeddings")
                self._conn.execute(
                    """
                    INSERT INTO lcm_trajectory_embedding_profiles(
                        profile_digest, provider, model_name, dim,
                        document_version, source_manifest_digest,
                        document_count, index_digest, active, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(profile_digest) DO UPDATE SET
                        document_count = excluded.document_count,
                        index_digest = excluded.index_digest,
                        active = 1,
                        created_at = excluded.created_at
                    """,
                    (
                        profile_digest,
                        provider_name,
                        model_name,
                        dim,
                        TRAJECTORY_SEMANTIC_DOCUMENT_VERSION,
                        str(corpus["source_manifest_digest"]),
                        len(documents),
                        index_digest,
                        now,
                    ),
                )
                self._conn.executemany(
                    """
                    INSERT INTO lcm_trajectory_embeddings(
                        source_id, profile_digest, document_sha256, vector, embedded_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (source_id, profile_digest, document_sha, vector, now)
                        for (source_id, _document, document_sha), vector
                        in zip(documents, packed)
                    ],
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {
            "status": "built",
            "profile_digest": profile_digest,
            "document_count": len(documents),
            "dim": dim,
            "index_digest": index_digest,
        }

    def semantic_metrics(self) -> dict[str, int]:
        return dict(self._semantic_usage)

    # ------------------------------------------------------------------
    # Per-STATE semantic index (issue #142, Lane S / W3a).
    #
    # An ADDITIVE second embedding space keyed by ``state_id`` -- the coarse
    # per-trajectory index above cannot surface a lexically-invisible answer
    # state, so the recall lane needs a per-state vector. The tables are created
    # lazily/idempotently exactly like ``_ensure_semantic_schema`` (no change to
    # any existing table); the backfill is resumable (skip embedded states) and
    # drives its own request packing so a bulk run stays inside Voyage's caps.
    # ------------------------------------------------------------------

    def _ensure_state_semantic_schema(self) -> None:
        self._require_writable()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS lcm_trajectory_state_embedding_profiles (
                        profile_digest TEXT PRIMARY KEY,
                        provider TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        dim INTEGER NOT NULL CHECK(dim > 0),
                        document_version TEXT NOT NULL,
                        source_manifest_digest TEXT NOT NULL,
                        state_count INTEGER NOT NULL CHECK(state_count >= 0),
                        active INTEGER NOT NULL DEFAULT 0 CHECK(active IN (0, 1)),
                        created_at REAL NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS
                    lcm_trajectory_state_embedding_one_active
                    ON lcm_trajectory_state_embedding_profiles(active)
                    WHERE active = 1
                    """
                )
                self._migrate_state_semantic_embeddings_to_profile_scope()
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS lcm_trajectory_state_embeddings (
                        state_id INTEGER NOT NULL
                            REFERENCES lcm_trajectory_states(state_id)
                            ON DELETE CASCADE,
                        profile_digest TEXT NOT NULL
                            REFERENCES lcm_trajectory_state_embedding_profiles(
                                profile_digest
                            )
                            ON DELETE CASCADE,
                        document_sha256 TEXT NOT NULL,
                        vector BLOB NOT NULL,
                        embedded_at REAL NOT NULL,
                        PRIMARY KEY(profile_digest, state_id)
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    lcm_trajectory_state_embeddings_profile
                    ON lcm_trajectory_state_embeddings(profile_digest, state_id)
                    """
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _migrate_state_semantic_embeddings_to_profile_scope(self) -> None:
        """Replace the legacy ``state_id`` primary key with a profile-scoped key.

        Legacy rows belong to the active profile only when every stored digest
        matches it. If there is no active profile, a sole stored digest is
        unambiguous and is adopted. Any mixed digests describe an interrupted
        legacy rebuild whose overwritten vectors cannot be reconstructed, so the
        migration fails closed instead of assigning mixed vectors to a profile.

        The caller owns the transaction. Detecting the composite primary key
        makes repeat calls idempotent.
        """
        table_exists = self._conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'lcm_trajectory_state_embeddings'
            """
        ).fetchone()
        if table_exists is None:
            return

        table_info = self._conn.execute(
            "PRAGMA table_info(lcm_trajectory_state_embeddings)"
        ).fetchall()
        primary_key = tuple(
            str(row[1])
            for row in sorted(table_info, key=lambda row: int(row[5]))
            if int(row[5]) > 0
        )
        if primary_key == ("profile_digest", "state_id"):
            return
        if primary_key != ("state_id",):
            raise TrajectoryStoreError(
                "unsupported state embedding primary key: "
                f"{primary_key!r}"
            )

        active = self._conn.execute(
            """
            SELECT profile_digest
            FROM lcm_trajectory_state_embedding_profiles
            WHERE active = 1
            """
        ).fetchone()
        stored_digests = [
            str(row[0])
            for row in self._conn.execute(
                """
                SELECT DISTINCT profile_digest
                FROM lcm_trajectory_state_embeddings
                ORDER BY profile_digest
                """
            ).fetchall()
        ]
        adopted_digest = str(active[0]) if active is not None else None
        if adopted_digest is None and len(stored_digests) == 1:
            adopted_digest = stored_digests[0]
        if adopted_digest is None and len(stored_digests) > 1:
            raise TrajectoryStoreError(
                "cannot migrate legacy state embeddings with multiple profile "
                "digests and no active profile; discard the legacy state "
                "embeddings and run a fresh backfill to recover"
            )
        if adopted_digest is not None and any(
            digest != adopted_digest for digest in stored_digests
        ):
            raise TrajectoryStoreError(
                "cannot migrate because legacy state embeddings do not all "
                "match the active profile; discard the legacy state embeddings "
                "and run a fresh backfill to recover"
            )

        self._conn.execute(
            """
            CREATE TABLE lcm_trajectory_state_embeddings_profile_scoped (
                state_id INTEGER NOT NULL
                    REFERENCES lcm_trajectory_states(state_id) ON DELETE CASCADE,
                profile_digest TEXT NOT NULL
                    REFERENCES lcm_trajectory_state_embedding_profiles(
                        profile_digest
                    )
                    ON DELETE CASCADE,
                document_sha256 TEXT NOT NULL,
                vector BLOB NOT NULL,
                embedded_at REAL NOT NULL,
                PRIMARY KEY(profile_digest, state_id)
            )
            """
        )
        if adopted_digest is not None:
            self._conn.execute(
                """
                INSERT INTO lcm_trajectory_state_embeddings_profile_scoped(
                    state_id, profile_digest, document_sha256, vector, embedded_at
                )
                SELECT state_id, ?, document_sha256, vector, embedded_at
                FROM lcm_trajectory_state_embeddings
                """,
                (adopted_digest,),
            )
        self._conn.execute("DROP TABLE lcm_trajectory_state_embeddings")
        self._conn.execute(
            """
            ALTER TABLE lcm_trajectory_state_embeddings_profile_scoped
            RENAME TO lcm_trajectory_state_embeddings
            """
        )
        self._state_semantic_cache = None

    def _state_semantic_profile_exists(self) -> bool:
        return self._conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'lcm_trajectory_state_embedding_profiles'
            """
        ).fetchone() is not None

    def active_state_semantic_profile(self) -> sqlite3.Row | None:
        if not self._state_semantic_profile_exists():
            return None
        return self._conn.execute(
            "SELECT * FROM lcm_trajectory_state_embedding_profiles WHERE active = 1"
        ).fetchone()

    @staticmethod
    def _state_semantic_profile_digest(
        provider_name: str, model_name: str, dim: int, source_manifest_digest: str
    ) -> str:
        return _sha256_text(_canonical_json({
            "provider": provider_name,
            "model": model_name,
            "dim": int(dim),
            "document_version": TRAJECTORY_STATE_SEMANTIC_DOCUMENT_VERSION,
            "source_manifest_digest": source_manifest_digest,
        }))

    @staticmethod
    def _state_embed_document(text: str, url: str, state_id: int) -> str:
        """The per-state embedding document. ``states.text`` is the FTS-indexed
        visible content (what the #141 sizing counted); an empty text falls back
        to the URL, then a stable state marker, so the provider never receives an
        empty input (Voyage rejects those)."""
        candidate = str(text or "").strip()
        if candidate:
            return str(text)
        candidate = str(url or "").strip()
        if candidate:
            return str(url)
        return f"state-{int(state_id)}"

    def _state_token_chunks(self, document: str, token_budget: int) -> list[str]:
        """Split ``document`` into contiguous pieces each <= ``token_budget``
        tokens (the chunked path for the ~228 states over Voyage's per-document
        cap). Uses the shared cl100k encoder -- the same tokenizer the provider's
        packing uses to gate the caps -- and falls back to a conservative
        character window if the encoder is unavailable."""
        from .tokens import _fallback_token_estimate, _get_encoder

        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        encoder = _get_encoder()
        if encoder is None:
            # Use the shared estimator itself rather than ``budget * 4``:
            # the estimator includes a rounding token and deliberately assigns
            # denser budgets to non-ASCII text, so a flat character window can
            # exceed the same limit that selected this path.
            chunks: list[str] = []
            start = 0
            while start < len(document):
                low = 1
                high = min(len(document) - start, max(1, token_budget * 4))
                accepted = 0
                while low <= high:
                    middle = (low + high) // 2
                    piece = document[start:start + middle]
                    if _fallback_token_estimate(piece) <= token_budget:
                        accepted = middle
                        low = middle + 1
                    else:
                        high = middle - 1
                # A one-character piece always estimates to one token, but keep
                # forward progress explicit if that estimator contract changes.
                accepted = max(1, accepted)
                chunks.append(document[start:start + accepted])
                start += accepted
            return chunks or [document]
        token_ids = encoder.encode(document)
        encoded = bytearray()
        token_byte_offsets = [0]
        for token_id in token_ids:
            encoded.extend(encoder.decode_single_token_bytes(token_id))
            token_byte_offsets.append(len(encoded))

        chunks: list[str] = []
        start = 0
        while start < len(token_ids):
            end = min(len(token_ids), start + token_budget)
            while end > start:
                piece_bytes = encoded[
                    token_byte_offsets[start]:token_byte_offsets[end]
                ]
                try:
                    piece = piece_bytes.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    end -= 1
                    continue
                # Re-encoding a boundary-safe piece can differ from slicing the
                # full-document token stream. Keep the provider-facing budget
                # authoritative rather than assuming the sliced count survives.
                if len(encoder.encode(piece)) <= token_budget:
                    chunks.append(piece)
                    start = end
                    break
                end -= 1
            else:
                raise ValueError(
                    "token_budget is too small to preserve a Unicode character boundary"
                )
        return chunks or [document]

    def build_state_semantic_index(
        self,
        provider: TrajectoryEmbeddingProvider | None = None,
        *,
        resume: bool = True,
        batch_max_items: int = _STATE_EMBED_MAX_BATCH_ITEMS,
        batch_token_budget: int = _STATE_EMBED_BATCH_TOKEN_BUDGET,
        document_token_budget: int = _STATE_EMBED_DOCUMENT_TOKEN_BUDGET,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Resumable per-state embedding backfill (issue #142).

        Embeds one vector per state (``states.text``) into
        ``lcm_trajectory_state_embeddings`` under a profile keyed by
        provider/model/dim/document_version/source_manifest_digest. A new profile
        remains inactive while staging, then becomes active only after every
        state is present. Requests are packed to ``batch_max_items`` items /
        ``batch_token_budget`` tokens; a state whose document exceeds
        ``document_token_budget`` is split into token windows, each embedded,
        and the pieces mean-pooled + normalized into a single state vector
        (same billed tokens, more requests).

        Resumable + idempotent: with ``resume=True`` (default) states that
        already carry a row under the target profile are skipped, so a re-run
        after an interruption embeds only the remainder and a fully-embedded
        store makes ZERO provider calls. ``progress_callback`` is invoked after
        each dispatched request with a cumulative-stats dict (for a live ledger).
        """
        from .tokens import count_tokens

        if self.status != "complete":
            raise CorpusIdentityError(
                "trajectory corpus must be finalized before state indexing"
            )
        active_provider = provider or self.embedding_provider
        if active_provider is None:
            raise TrajectoryStoreError("trajectory embedding provider is not configured")
        self._ensure_state_semantic_schema()
        corpus = self._conn.execute(
            """
            SELECT source_manifest_digest, trajectory_count
            FROM lcm_trajectory_corpora WHERE singleton = 1
            """
        ).fetchone()
        source_manifest_digest = str(corpus["source_manifest_digest"])
        provider_name = str(getattr(active_provider, "provider_id", "unknown"))
        model_name = str(getattr(active_provider, "model_id", "")).strip()
        if not model_name:
            raise ValueError("trajectory embedding model_id must not be empty")

        batch_max_items = max(1, int(batch_max_items))
        batch_token_budget = max(1, int(batch_token_budget))
        document_token_budget = max(1, int(document_token_budget))

        # Discover the embedding dimension without a wasted call when possible:
        # the source-level active profile shares this provider/model, so its
        # dim is authoritative; otherwise probe one state.
        source_profile = self._semantic_profile()
        dim: int | None = None
        probe_provider_calls = 0
        probe_billed_tokens = 0
        if (
            source_profile is not None
            and str(source_profile["provider"]) == provider_name
            and str(source_profile["model_name"]) == model_name
        ):
            dim = int(source_profile["dim"])

        all_states = self._conn.execute(
            """
            SELECT s.state_id, s.text, s.url
            FROM lcm_trajectory_states s
            ORDER BY s.state_id
            """
        ).fetchall()
        total_states = len(all_states)

        if not resume:
            # Reject an in-place rebuild before a provider probe can incur cost
            # or turn a deterministic local refusal into a provider failure.
            # Prefer the caller's resolved dimension; only infer from the
            # serving profile when the requested dimension is genuinely
            # unknown, so a distinct-dimension rebuild is not falsely refused.
            with self._lock:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    serving_profile = self.active_state_semantic_profile()
                    if serving_profile is not None:
                        requested_digest = self._state_semantic_profile_digest(
                            provider_name,
                            model_name,
                            int(dim) if dim is not None
                            else int(serving_profile["dim"]),
                            source_manifest_digest,
                        )
                        if (
                            str(serving_profile["profile_digest"])
                            == requested_digest
                        ):
                            raise TrajectoryStoreError(
                                "cannot rebuild the active state semantic profile "
                                "in place; use a distinct profile identity so the "
                                "serving vectors remain intact"
                            )
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise

        if dim is None:
            if not all_states:
                raise ValueError("trajectory corpus has no states to index")
            probe_doc = self._state_embed_document(
                all_states[0]["text"], all_states[0]["url"], int(all_states[0]["state_id"])
            )
            probe_doc = self._state_token_chunks(
                probe_doc, document_token_budget
            )[0]
            probe_vector = _normalized_vector(active_provider.embed_query(probe_doc))
            dim = len(probe_vector)
            probe_provider_calls = 1
            probe_billed_tokens = max(
                0, int(getattr(active_provider, "last_usage_tokens", 0) or 0)
            )

        profile_digest = self._state_semantic_profile_digest(
            provider_name, model_name, dim, source_manifest_digest
        )
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                serving_profile = self.active_state_semantic_profile()
                if (
                    not resume
                    and serving_profile is not None
                    and str(serving_profile["profile_digest"]) == profile_digest
                ):
                    raise TrajectoryStoreError(
                        "cannot rebuild the active state semantic profile in place; "
                        "use a distinct profile identity so the serving vectors "
                        "remain intact"
                    )
                # The profile-scoped embedding key keeps the serving profile
                # complete and untouched while this inactive profile is staged.
                # Readers remain on the predecessor until the atomic flag flip
                # after every state for this profile has been persisted.
                self._conn.execute(
                    """
                    INSERT INTO lcm_trajectory_state_embedding_profiles(
                        profile_digest, provider, model_name, dim,
                        document_version, source_manifest_digest,
                        state_count, active, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                    ON CONFLICT(profile_digest) DO UPDATE SET
                        state_count = excluded.state_count
                    """,
                    (
                        profile_digest,
                        provider_name,
                        model_name,
                        int(dim),
                        TRAJECTORY_STATE_SEMANTIC_DOCUMENT_VERSION,
                        source_manifest_digest,
                        total_states,
                        now,
                    ),
                )
                if not resume:
                    self._conn.execute(
                        "DELETE FROM lcm_trajectory_state_embeddings "
                        "WHERE profile_digest = ?",
                        (profile_digest,),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        already: set[int] = set()
        if resume:
            already = {
                int(row[0])
                for row in self._conn.execute(
                    "SELECT state_id FROM lcm_trajectory_state_embeddings "
                    "WHERE profile_digest = ?",
                    (profile_digest,),
                )
            }
        pending = [row for row in all_states if int(row["state_id"]) not in already]

        stats: dict[str, Any] = {
            "profile_digest": profile_digest,
            "dim": int(dim),
            "total_states": total_states,
            "already_embedded": len(already),
            "pending": len(pending),
            "states_embedded": 0,
            "chunked_states": 0,
            "provider_calls": probe_provider_calls,
            "billed_tokens": probe_billed_tokens,
        }

        def _emit_progress() -> None:
            if progress_callback is not None:
                progress_callback(dict(stats))

        if probe_provider_calls:
            _emit_progress()

        # Partition pending states into single-request documents and the
        # oversize (chunked) minority, both packed to the same item/token caps.
        request_document_token_budget = min(
            document_token_budget, batch_token_budget
        )
        normal: list[tuple[int, str, str, int]] = []  # (state_id, doc, sha, tokens)
        oversize: list[tuple[int, str, str]] = []  # (state_id, doc, sha)
        for row in pending:
            state_id = int(row["state_id"])
            document = self._state_embed_document(row["text"], row["url"], state_id)
            document_sha = _sha256_text(document)
            tokens = count_tokens(document)
            if tokens > request_document_token_budget:
                oversize.append((state_id, document, document_sha))
            else:
                normal.append((state_id, document, document_sha, tokens))

        def _persist(rows: list[tuple[int, str, bytes]]) -> None:
            with self._lock:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    target = self._conn.execute(
                        """
                        SELECT active
                        FROM lcm_trajectory_state_embedding_profiles
                        WHERE profile_digest = ?
                        """,
                        (profile_digest,),
                    ).fetchone()
                    if target is not None and int(target["active"]) == 1:
                        raise TrajectoryStoreError(
                            "state semantic profile became active during backfill; "
                            "refusing to rewrite serving vectors"
                        )
                    self._conn.executemany(
                        """
                        INSERT INTO lcm_trajectory_state_embeddings(
                            state_id, profile_digest, document_sha256, vector, embedded_at
                        ) VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(profile_digest, state_id) DO UPDATE SET
                            document_sha256 = excluded.document_sha256,
                            vector = excluded.vector,
                            embedded_at = excluded.embedded_at
                        """,
                        [
                            (sid, profile_digest, sha, vec, time.time())
                            for sid, sha, vec in rows
                        ],
                    )
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise

        # --- normal single-request states -------------------------------------
        batch_ids: list[int] = []
        batch_docs: list[str] = []
        batch_shas: list[str] = []
        batch_tokens = 0

        def _flush_normal() -> None:
            nonlocal batch_ids, batch_docs, batch_shas, batch_tokens
            if not batch_docs:
                return
            vectors = active_provider.embed_documents(batch_docs)
            stats["provider_calls"] += 1
            stats["billed_tokens"] += max(
                0, int(getattr(active_provider, "last_usage_tokens", 0) or 0)
            )
            _emit_progress()
            if len(vectors) != len(batch_docs):
                raise ValueError("state embedding count does not match batch size")
            packed = []
            for sid, sha, vector in zip(batch_ids, batch_shas, vectors):
                normalized = _normalized_vector(vector, expected_dim=dim)
                packed.append((sid, sha, _pack_vector(normalized)))
            _persist(packed)
            stats["states_embedded"] += len(packed)
            batch_ids, batch_docs, batch_shas, batch_tokens = [], [], [], 0
            _emit_progress()

        for state_id, document, document_sha, tokens in normal:
            if batch_docs and (
                len(batch_docs) >= batch_max_items
                or batch_tokens + tokens > batch_token_budget
            ):
                _flush_normal()
            batch_ids.append(state_id)
            batch_docs.append(document)
            batch_shas.append(document_sha)
            batch_tokens += tokens
        _flush_normal()

        # --- oversize (chunked) states ----------------------------------------
        for state_id, document, document_sha in oversize:
            chunks = self._state_token_chunks(
                document, request_document_token_budget
            )
            chunk_vectors: list[tuple[float, ...]] = []
            start = 0
            while start < len(chunks):
                sub: list[str] = []
                sub_tokens = 0
                while start < len(chunks) and len(sub) < batch_max_items:
                    chunk = chunks[start]
                    chunk_tokens = count_tokens(chunk)
                    if sub and sub_tokens + chunk_tokens > batch_token_budget:
                        break
                    sub.append(chunk)
                    sub_tokens += chunk_tokens
                    start += 1
                vectors = active_provider.embed_documents(sub)
                if len(vectors) != len(sub):
                    raise ValueError("chunk embedding count does not match batch size")
                for vector in vectors:
                    chunk_vectors.append(_normalized_vector(vector, expected_dim=dim))
                stats["provider_calls"] += 1
                stats["billed_tokens"] += max(
                    0, int(getattr(active_provider, "last_usage_tokens", 0) or 0)
                )
                _emit_progress()
            pooled = [sum(values) / len(chunk_vectors) for values in zip(*chunk_vectors)]
            normalized = _normalized_vector(pooled, expected_dim=dim)
            _persist([(state_id, document_sha, _pack_vector(normalized))])
            stats["states_embedded"] += 1
            stats["chunked_states"] += 1
            _emit_progress()

        # A rewrite invalidates any cached query-time matrix.
        self._state_semantic_cache = None
        embedded_count = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM lcm_trajectory_state_embeddings "
                "WHERE profile_digest = ?",
                (profile_digest,),
            ).fetchone()[0]
        )
        if embedded_count != total_states:
            raise TrajectoryStoreError(
                "state semantic profile is incomplete after backfill"
            )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO lcm_trajectory_state_embedding_profiles(
                        profile_digest, provider, model_name, dim,
                        document_version, source_manifest_digest,
                        state_count, active, created_at
                    )
                    SELECT
                        profile_digest, provider, model_name, dim,
                        document_version, source_manifest_digest,
                        state_count,
                        CASE WHEN profile_digest = ? THEN 1 ELSE 0 END,
                        created_at
                    FROM lcm_trajectory_state_embedding_profiles
                    WHERE active = 1 OR profile_digest = ?
                    ORDER BY CASE WHEN profile_digest = ? THEN 1 ELSE 0 END
                    ON CONFLICT(profile_digest) DO UPDATE SET
                        active = excluded.active
                    """,
                    (profile_digest, profile_digest, profile_digest),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        stats["status"] = "current" if not pending else "built"
        return stats

    @staticmethod
    def _safe_getattr(obj: Any, name: str) -> Any:
        """getattr that swallows EVERY exception, not just AttributeError.

        A hostile/exotic exception can expose ``kind``/``status_code``/
        ``retry_after`` as *properties that raise*; plain getattr(obj, name,
        default) only absorbs AttributeError and would let that propagate out
        of telemetry recording and crash the query path. Introspection must
        never be able to turn a semantic fallback into a total failure.
        """
        try:
            return getattr(obj, name, None)
        except Exception:
            return None

    @staticmethod
    def _classify_semantic_reason(exception: BaseException) -> str:
        """Best-effort failure category for the fallbacks-by-reason counter.

        Duck-typed so trajectory_store need not import the provider exception
        classes: client-side spend guard, circuit breaker, and Voyage's
        classified ``kind`` (auth/rate_limit/bad_request/server_error/...) all
        surface distinctly instead of collapsing to one opaque bucket.
        """
        name = type(exception).__name__
        kind = str(TrajectoryStore._safe_getattr(exception, "kind") or "")
        if name == "ProviderRateLimited":
            return "client_rate_guard"
        if name == "ProviderCircuitOpen":
            return "circuit_open"
        if isinstance(exception, TimeoutError) or kind == "timeout":
            return "timeout"
        if kind:
            return kind
        return "other"

    def _bump_attempt_totals(self, attempt: TrajectorySemanticAttempt) -> None:
        totals = self._semantic_attempt_totals
        totals["attempts"] += 1
        if attempt.outcome == "success":
            totals["successes"] += 1
        elif attempt.outcome == "fallback":
            totals["fallbacks"] += 1
            key = attempt.reason or "other"
            by_reason = totals["fallbacks_by_reason"]
            by_reason[key] = by_reason.get(key, 0) + 1

    def _store_attempt(self, attempt: TrajectorySemanticAttempt) -> TrajectorySemanticAttempt:
        self._semantic_attempts.append(attempt)
        self._last_semantic_attempt = attempt
        self._bump_attempt_totals(attempt)
        return attempt

    def _record_semantic_attempt(
        self,
        *,
        outcome: str,
        latency_ms: float,
        exception: BaseException | None,
    ) -> TrajectorySemanticAttempt:
        provider = self.embedding_provider
        provider_id = str(self._safe_getattr(provider, "provider_id") or "unknown") if provider else "none"
        model_id = str(self._safe_getattr(provider, "model_id") or "") if provider else ""
        exception_class: str | None = None
        http_status: int | None = None
        retry_after: float | None = None
        reason: str | None = None
        if exception is not None:
            exception_class = type(exception).__name__
            raw_status = self._safe_getattr(exception, "status_code")
            http_status = (
                int(raw_status)
                if isinstance(raw_status, int) and not isinstance(raw_status, bool)
                else None
            )
            raw_retry = self._safe_getattr(exception, "retry_after")
            retry_after = (
                float(raw_retry)
                if isinstance(raw_retry, (int, float)) and not isinstance(raw_retry, bool)
                else None
            )
            reason = self._classify_semantic_reason(exception)
        attempt = TrajectorySemanticAttempt(
            provider=provider_id,
            model=model_id,
            outcome=str(outcome),
            exception_class=exception_class,
            http_status=http_status,
            retry_after=retry_after,
            latency_ms=round(float(latency_ms), 3),
            reason=reason,
        )
        return self._store_attempt(attempt)

    def _record_minimal_fallback_attempt(
        self, *, latency_ms: float
    ) -> TrajectorySemanticAttempt:
        """Last-resort record when even best-effort introspection failed. Keeps
        the funnel honest (a fallback still counts) without touching the
        offending exception again."""
        provider = self.embedding_provider
        attempt = TrajectorySemanticAttempt(
            provider="unknown" if provider is not None else "none",
            model="",
            outcome="fallback",
            reason="attempt_record_introspection_failed",
            latency_ms=round(float(latency_ms), 3),
        )
        return self._store_attempt(attempt)

    def last_semantic_attempt(self) -> dict[str, Any] | None:
        """The typed attempt record from the most recent ``query()`` call."""
        if self._last_semantic_attempt is None:
            return None
        return asdict(self._last_semantic_attempt)

    def semantic_attempt_counters(self) -> dict[str, Any]:
        """Per-run semantic funnel counters (attempts / successes / fallbacks by
        reason). Tracked cumulatively and independently of the bounded attempt
        ring, so counts stay exact across an arbitrarily long run."""
        totals = self._semantic_attempt_totals
        return {
            "attempts": totals["attempts"],
            "successes": totals["successes"],
            "fallbacks": totals["fallbacks"],
            "fallbacks_by_reason": dict(totals["fallbacks_by_reason"]),
        }

    def last_query_telemetry(self) -> dict[str, Any] | None:
        """Side-channel telemetry for the most recent ``query()``: the semantic
        attempt record, the ranked source-candidate list, the pre-selection
        state-candidate pool, and the delivered-evidence refs. Written after the
        call returns and never affects the returned hits (byte-identical
        evidence)."""
        if self._last_query_telemetry is None:
            return None
        return dict(self._last_query_telemetry)

    def _semantic_source_ranks(self, query: str) -> list[tuple[int, float]]:
        provider = self.embedding_provider
        profile = self._semantic_profile()
        if provider is None or profile is None:
            return []
        if (
            str(profile["provider"]) != str(getattr(provider, "provider_id", "unknown"))
            or str(profile["model_name"]) != str(getattr(provider, "model_id", ""))
        ):
            return []
        self._semantic_usage["query_calls"] += 1
        query_vector = _normalized_vector(
            provider.embed_query(query),
            expected_dim=int(profile["dim"]),
        )
        self._semantic_usage["query_tokens"] += max(
            0,
            int(getattr(provider, "last_usage_tokens", 0) or 0),
        )
        rows = self._conn.execute(
            """
            SELECT source_id, vector
            FROM lcm_trajectory_embeddings
            WHERE profile_digest = ?
            """,
            (str(profile["profile_digest"]),),
        ).fetchall()
        ranked = []
        for row in rows:
            vector = _unpack_vector(bytes(row["vector"]), int(profile["dim"]))
            similarity = sum(left * right for left, right in zip(query_vector, vector))
            ranked.append((int(row["source_id"]), float(similarity)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[: self.semantic_top_trajectories]

    def _load_state_semantic_matrix(
        self, profile_digest: str, dim: int
    ) -> tuple[list[int], Any]:
        """Cache and return ``(state_ids, matrix)`` for the active state profile.

        ``matrix`` is a normalized ``float32`` numpy array when numpy is present
        (a single mat-vec ranks all states) and otherwise a list of vector
        tuples for a pure-Python fallback -- the state vectors were normalized at
        backfill, so a query-vector dot product is cosine similarity either way.
        """
        with self._lock:
            freshness_row = self._conn.execute(
                """
                SELECT COUNT(*), COALESCE(MAX(embedded_at), 0.0)
                FROM lcm_trajectory_state_embeddings
                WHERE profile_digest = ?
                """,
                (profile_digest,),
            ).fetchone()
            freshness = (int(freshness_row[0]), float(freshness_row[1]))
            cache = self._state_semantic_cache
            if (
                cache is not None
                and cache[0] == profile_digest
                and cache[1] == freshness
            ):
                return cache[2], cache[3]
            rows = self._conn.execute(
                """
                SELECT state_id, vector FROM lcm_trajectory_state_embeddings
                WHERE profile_digest = ?
                ORDER BY state_id
                """,
                (profile_digest,),
            ).fetchall()
        state_ids = [int(row["state_id"]) for row in rows]
        try:
            import numpy as _np

            if rows:
                matrix: Any = _np.frombuffer(
                    b"".join(bytes(row["vector"]) for row in rows), dtype="<f4"
                ).reshape(len(rows), int(dim))
            else:
                matrix = _np.zeros((0, int(dim)), dtype="<f4")
        except Exception:
            matrix = [_unpack_vector(bytes(row["vector"]), int(dim)) for row in rows]
        self._state_semantic_cache = (
            profile_digest, freshness, state_ids, matrix
        )
        return state_ids, matrix

    def _semantic_state_ranks(
        self, query: str, top_k: int
    ) -> list[tuple[int, float]]:
        """Top-``top_k`` ``(state_id, similarity)`` for the query against the
        active per-state semantic index (issue #142). Returns ``[]`` when no
        provider or no active state profile is present (the arm is then inert)."""
        provider = self.embedding_provider
        profile = self.active_state_semantic_profile()
        if provider is None or profile is None or top_k <= 0:
            return []
        if (
            str(profile["provider"]) != str(getattr(provider, "provider_id", "unknown"))
            or str(profile["model_name"]) != str(getattr(provider, "model_id", ""))
        ):
            return []
        dim = int(profile["dim"])
        state_ids, matrix = self._load_state_semantic_matrix(
            str(profile["profile_digest"]), dim
        )
        if not state_ids:
            return []
        self._semantic_usage["query_calls"] += 1
        query_vector = _normalized_vector(provider.embed_query(query), expected_dim=dim)
        self._semantic_usage["query_tokens"] += max(
            0,
            int(getattr(provider, "last_usage_tokens", 0) or 0),
        )
        try:
            import numpy as _np

            query_array = _np.asarray(query_vector, dtype="<f4")
            scores = matrix @ query_array
            limit = min(int(top_k), scores.shape[0])
            # argpartition for the top-k, then order that slice by score desc,
            # state_id asc (a stable, deterministic tie-break).
            top_idx = _np.argpartition(-scores, limit - 1)[:limit]
            ordered = sorted(
                (int(i) for i in top_idx),
                key=lambda i: (-float(scores[i]), state_ids[i]),
            )
            return [(state_ids[i], float(scores[i])) for i in ordered]
        except Exception:
            ranked = [
                (state_ids[i], sum(a * b for a, b in zip(query_vector, vector)))
                for i, vector in enumerate(matrix)
            ]
            ranked.sort(key=lambda item: (-item[1], item[0]))
            return ranked[: int(top_k)]

    @staticmethod
    def _fts_expression(query: str) -> str:
        terms: list[str] = []
        seen: set[str] = set()
        for raw in extract_search_terms(query):
            normalized = raw.casefold().strip()
            if len(normalized) < 2 or normalized in _STOPWORDS or normalized in seen:
                continue
            safe = normalized.replace('"', '""')
            if not any(character.isalnum() for character in safe):
                continue
            seen.add(normalized)
            terms.append(f'"{safe}"')
            if len(terms) >= 16:
                break
        return " OR ".join(terms)

    def _asset_from_row(self, row: sqlite3.Row) -> tuple[str, str] | tuple[None, None]:
        relative = row["relative_path"]
        expected_sha = row["asset_sha256"]
        if relative is None or expected_sha is None:
            return None, None
        candidate = (self.asset_root / str(relative)).resolve()
        try:
            candidate.relative_to(self.asset_root.resolve())
        except ValueError as exc:
            raise TrajectoryAssetError("stored screenshot path escapes asset root") from exc
        if not candidate.is_file() or _sha256_file(candidate) != str(expected_sha):
            raise TrajectoryAssetError(
                f"stored screenshot is missing or changed: {relative}"
            )
        return str(candidate), str(expected_sha)

    def _row_to_hit(
        self,
        row: sqlite3.Row,
        *,
        score: float,
        match_kind: str,
        include_image: bool,
        query: str | None = None,
        text_char_limit: int | None = None,
        dense_excerpt: bool = False,
    ) -> TrajectoryHit:
        corpus_uid = self.corpus_uid
        if not corpus_uid:
            raise CorpusIdentityError("trajectory corpus is not complete")
        trajectory_id = str(row["trajectory_id"])
        state_index = int(row["state_index"])
        encoded_id = quote(trajectory_id, safe="")
        exact_ref = f"trajectory://{corpus_uid}/{encoded_id}/state/{state_index}"
        screenshot_path: str | None = None
        screenshot_sha: str | None = None
        if include_image:
            screenshot_path, screenshot_sha = self._asset_from_row(row)
        full_text = str(row["text"])
        if text_char_limit is None or len(full_text) <= text_char_limit:
            text = full_text
            text_offset = 0
            text_truncated = False
        else:
            excerpt_fn = (
                self._densest_exact_excerpt if dense_excerpt else self._exact_excerpt
            )
            text, text_offset = excerpt_fn(
                full_text,
                query or "",
                text_char_limit,
            )
            text_truncated = True
        return TrajectoryHit(
            exact_ref=exact_ref,
            trajectory_id=trajectory_id,
            goal=str(row["goal"]),
            outcome=str(row["outcome"]) if row["outcome"] is not None else None,
            state_index=state_index,
            sequence_ordinal=int(row["sequence_ordinal"]),
            step=int(row["step"]),
            url=str(row["url"]),
            incoming_action=(
                str(row["incoming_action"]) if row["incoming_action"] is not None else None
            ),
            thoughts=str(row["thoughts"]) if row["thoughts"] is not None else None,
            text=text,
            text_offset=text_offset,
            text_truncated=text_truncated,
            observed_at=(
                float(row["observed_at"]) if row["observed_at"] is not None else None
            ),
            observed_at_source=(
                str(row["observed_at_source"])
                if row["observed_at_source"] is not None
                else None
            ),
            occurred_at=(
                float(row["occurred_at"]) if row["occurred_at"] is not None else None
            ),
            occurred_at_source=(
                str(row["occurred_at_source"])
                if row["occurred_at_source"] is not None
                else None
            ),
            screenshot_path=screenshot_path,
            screenshot_sha256=screenshot_sha,
            score=float(score),
            match_kind=match_kind,
        )

    @staticmethod
    def _exact_excerpt(text: str, query: str, limit: int) -> tuple[str, int]:
        """Return a bounded verbatim substring centered near the first query match."""
        if len(text) <= limit:
            return text, 0
        folded = text.casefold()
        positions = [
            folded.find(term.casefold())
            for term in extract_search_terms(query)
            if len(term.strip()) >= 2 and folded.find(term.casefold()) >= 0
        ]
        first_match = min(positions) if positions else 0
        start = max(0, first_match - (limit // 3))
        end = min(len(text), start + limit)
        start = max(0, end - limit)
        return text[start:end], start

    @staticmethod
    def _densest_exact_excerpt(text: str, query: str, limit: int) -> tuple[str, int]:
        """Return the deterministic window with the densest rare query terms.

        The historical excerpt anchors on the first query match. Web AXTree
        pages often repeat generic page-title terms near the top while the
        answer-bearing field occurs below the fold. This scorer gives rarer
        query terms more weight, then uses unique-term coverage, occurrence
        count, and earliest offset as deterministic tie-breaks.
        """
        if len(text) <= limit:
            return text, 0
        folded = text.casefold()
        terms: list[str] = []
        seen: set[str] = set()
        for raw in extract_search_terms(query):
            term = raw.casefold().strip()
            if len(term) < 2 or term in _STOPWORDS or term in seen:
                continue
            seen.add(term)
            terms.append(term)
        occurrences: list[tuple[int, str, float]] = []
        for term in terms:
            positions = [
                match.start() for match in re.finditer(re.escape(term), folded)
            ]
            if not positions:
                continue
            weight = 1.0 / len(positions)
            occurrences.extend((position, term, weight) for position in positions)
        if not occurrences:
            return text[:limit], 0
        occurrences.sort(key=lambda item: (item[0], item[1]))
        best: tuple[float, int, int, int] | None = None
        right = 0
        counts: dict[str, int] = {}
        weighted = 0.0
        for left, (left_pos, _left_term, _left_weight) in enumerate(occurrences):
            while right < len(occurrences) and occurrences[right][0] - left_pos < limit:
                _position, term, weight = occurrences[right]
                if counts.get(term, 0) == 0:
                    weighted += weight
                counts[term] = counts.get(term, 0) + 1
                right += 1
            score = (weighted, len(counts), right - left, -left_pos)
            if best is None or score > best:
                best = score
                # Anchor on the rarest term inside the winning window, leaving
                # two thirds of the excerpt after it for answer values/columns.
                anchor = max(
                    occurrences[left:right],
                    key=lambda item: (item[2], item[0]),
                )[0]
                best_start = max(0, anchor - (limit // 3))
            _position, term, weight = occurrences[left]
            counts[term] -= 1
            if counts[term] == 0:
                del counts[term]
                weighted -= weight
        start = min(best_start, len(text) - limit)
        return text[start:start + limit], start

    @staticmethod
    def _candidate_term_set(row: Any) -> set[str]:
        """Bounded lexical signature used by the per-trajectory MMR pass."""
        text = " ".join(
            str(row[key] or "")
            for key in ("goal", "url", "incoming_action", "text")
        )[:12_000]
        return {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9_-]+", text.casefold())
            if len(token) >= 3 and token not in _STOPWORDS
        }

    @staticmethod
    def _query_term_set(query: str) -> frozenset[str]:
        """Tokenize a query into the same lexical signature space as states.

        Mirrors ``_candidate_term_set`` tokenization so query-term density is
        measured against the identical vocabulary the MMR signatures use.
        """
        return frozenset(
            token
            for token in re.findall(r"[a-z0-9][a-z0-9_-]+", query.casefold())
            if len(token) >= 3 and token not in _STOPWORDS
        )

    @staticmethod
    def _title_field_text(row: Any) -> str:
        """Normalized title/heading/field-label text for the Knob H boost.

        Punctuation folds to single spaces and the result is padded with a
        leading/trailing space so a matched n-gram is checked at word
        boundaries via a plain substring test.
        """
        raw = " ".join(
            str(row[key] or "")
            for key in ("goal", "url", "text")
        )[:12_000].casefold()
        collapsed = re.sub(r"[^a-z0-9]+", " ", raw)
        return f" {' '.join(collapsed.split())} "

    @staticmethod
    def _query_title_ngrams(query: str) -> list[str]:
        """Contiguous 2-4 gram question phrases for the Knob H title boost.

        Grams composed entirely of stopwords are dropped (a pure ``what is
        the`` gram carries no signal); the remaining grams are normalized the
        same way as ``_title_field_text`` and returned in stable first-seen
        order.
        """
        words = [
            word.casefold()
            for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query)
        ]
        grams: list[str] = []
        seen: set[str] = set()
        for size in range(_TITLE_BOOST_MIN_GRAM, _TITLE_BOOST_MAX_GRAM + 1):
            for start in range(0, max(0, len(words) - size + 1)):
                window = words[start:start + size]
                if all(token in _STOPWORDS for token in window):
                    continue
                normalized = re.sub(r"[^a-z0-9]+", " ", " ".join(window)).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                grams.append(normalized)
        return grams

    @classmethod
    def _apply_title_boost(
        cls,
        rows: Sequence[Any],
        query: str,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Knob H: stable-reorder the lexical pool by exact title n-gram hits.

        A candidate is boosted by the number of DISTINCT contiguous question
        2-4 grams that appear (case/punctuation-normalized) in its
        title/heading/field-label text. The reorder is stable: candidates keep
        their original relative order within an equal-boost band, so with no
        matches the pool is byte-identical to the input. Pure-lexical and
        deterministic -- no model calls.
        """
        grams = cls._query_title_ngrams(query)
        boosts: dict[int, int] = {}
        matched: list[dict[str, Any]] = []
        if grams:
            for row in rows:
                field_text = cls._title_field_text(row)
                hits = [gram for gram in grams if f" {gram} " in field_text]
                if hits:
                    state_id = int(row["state_id"])
                    boosts[state_id] = len(hits)
                    matched.append({"state_id": state_id, "phrases": hits})
        reordered = [
            row
            for _key, row in sorted(
                enumerate(rows),
                key=lambda item: (
                    -boosts.get(int(item[1]["state_id"]), 0),
                    item[0],
                ),
            )
        ]
        telemetry = {
            "ngrams": grams,
            "boosted": matched,
            "boosted_count": len(matched),
        }
        return reordered, telemetry

    @staticmethod
    def _cap_composed_pool(
        rows: Sequence[Any],
        cap: int,
        *,
        antiboilerplate: bool = False,
        query_terms: frozenset[str] | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Apply one global per-trajectory cap after all candidate arms compose.

        Survivors are chosen independently within each trajectory by a bounded
        MMR-style relevance/redundancy score. The returned pool preserves the
        original cross-trajectory order, so the cap changes only which states
        survive, not unrelated tie-breaking.

        When ``antiboilerplate`` is set (Knob G, default-off) the per-candidate
        MMR score is additionally penalized by the candidate's mean lexical
        similarity to the OTHER pooled states of its own trajectory and rewarded
        by its query-term density, so a trajectory's seats go to query-relevant
        states rather than repeated task-header boilerplate. ``query_terms`` is
        the tokenized query used for the density reward; both signals are inert
        when ``antiboilerplate`` is ``False`` (byte-identical to the base cap).
        """
        query_terms = query_terms or frozenset()
        unique_rows: list[Any] = []
        seen_state_ids: set[int] = set()
        for row in rows:
            state_id = int(row["state_id"])
            if state_id in seen_state_ids:
                continue
            seen_state_ids.add(state_id)
            unique_rows.append(row)
        grouped: dict[str, list[tuple[int, Any]]] = {}
        for position, row in enumerate(unique_rows):
            grouped.setdefault(str(row["trajectory_id"]), []).append((position, row))
        survivor_ids: set[int] = set()
        details: list[dict[str, Any]] = []
        antiboilerplate_scores: list[dict[str, Any]] = []
        for trajectory_id, candidates in grouped.items():
            if len(candidates) <= cap:
                survivor_ids.update(int(row["state_id"]) for _position, row in candidates)
                continue
            signatures = {
                int(row["state_id"]): TrajectoryStore._candidate_term_set(row)
                for _position, row in candidates
            }
            boilerplate: dict[int, float] = {}
            density: dict[int, float] = {}
            if antiboilerplate:
                for _position, row in candidates:
                    state_id = int(row["state_id"])
                    signature = signatures[state_id]
                    sibling_scores: list[float] = []
                    for _other_position, other_row in candidates:
                        other_id = int(other_row["state_id"])
                        if other_id == state_id:
                            continue
                        other_signature = signatures[other_id]
                        union = signature | other_signature
                        sibling_scores.append(
                            len(signature & other_signature) / len(union)
                            if union else 0.0
                        )
                    boilerplate[state_id] = (
                        sum(sibling_scores) / len(sibling_scores)
                        if sibling_scores else 0.0
                    )
                    density[state_id] = (
                        len(signature & query_terms) / len(signature)
                        if signature else 0.0
                    )
            selected: list[tuple[int, Any]] = []
            remaining = list(candidates)
            while remaining and len(selected) < cap:
                best_item: tuple[int, Any] | None = None
                best_key: tuple[float, int, int] | None = None
                for local_position, item in enumerate(remaining):
                    original_position, row = item
                    relevance = 1.0 - (
                        original_position / max(1, len(unique_rows) - 1)
                    )
                    state_id = int(row["state_id"])
                    signature = signatures[state_id]
                    redundancy = 0.0
                    for _selected_position, selected_row in selected:
                        selected_signature = signatures[int(selected_row["state_id"])]
                        union = signature | selected_signature
                        similarity = (
                            len(signature & selected_signature) / len(union)
                            if union else 0.0
                        )
                        redundancy = max(redundancy, similarity)
                    mmr_score = (0.72 * relevance) - (0.28 * redundancy)
                    if antiboilerplate:
                        mmr_score += (
                            _ANTIBOILERPLATE_DENSITY_WEIGHT * density[state_id]
                        ) - (
                            _ANTIBOILERPLATE_BOILERPLATE_WEIGHT
                            * boilerplate[state_id]
                        )
                    key = (mmr_score, -original_position, -state_id)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_item = item
                assert best_item is not None
                selected.append(best_item)
                remaining.remove(best_item)
            if antiboilerplate:
                antiboilerplate_scores.extend(
                    {
                        "state_id": int(row["state_id"]),
                        "boilerplate": round(boilerplate[int(row["state_id"])], 6),
                        "density": round(density[int(row["state_id"])], 6),
                        "selected": int(row["state_id"]) in {
                            int(sel_row["state_id"]) for _pos, sel_row in selected
                        },
                    }
                    for _position, row in candidates
                )
            selected_ids = [int(row["state_id"]) for _position, row in selected]
            removed_ids = [
                int(row["state_id"])
                for _position, row in candidates
                if int(row["state_id"]) not in set(selected_ids)
            ]
            survivor_ids.update(selected_ids)
            details.append({
                "trajectory_id": trajectory_id,
                "before": len(candidates),
                "after": len(selected_ids),
                "capped_out": len(removed_ids),
                "selected_state_ids": selected_ids,
                "removed_state_ids": removed_ids,
            })
        filtered = [
            row for row in unique_rows if int(row["state_id"]) in survivor_ids
        ]
        telemetry: dict[str, Any] = {
            "cap": cap,
            "pool_before": len(unique_rows),
            "pool_after": len(filtered),
            "capped_out": len(unique_rows) - len(filtered),
            "trajectories": details,
            "survivor_state_ids": [int(row["state_id"]) for row in filtered],
        }
        if antiboilerplate:
            telemetry["antiboilerplate"] = {
                "density_weight": _ANTIBOILERPLATE_DENSITY_WEIGHT,
                "boilerplate_weight": _ANTIBOILERPLATE_BOILERPLATE_WEIGHT,
                "scored": antiboilerplate_scores,
            }
        return filtered, telemetry

    @staticmethod
    def _exact_query_phrases(query: str) -> list[str]:
        phrases: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'", query):
            phrase = next(group for group in match.groups() if group is not None)
            normalized = " ".join(phrase.casefold().split())
            if len(normalized) < 3 or normalized in seen:
                continue
            seen.add(normalized)
            phrases.append(normalized)
        return phrases[:8]

    @staticmethod
    def _typed_subqueries(query: str) -> list[dict[str, str]]:
        """Deterministically decompose one question into typed FTS queries."""
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query)
        folded_words = [word.casefold() for word in words]
        candidates: list[tuple[str, str]] = [("raw_state", query)]
        exact = TrajectoryStore._exact_query_phrases(query)
        if exact:
            candidates.append(("entity", " ".join(exact)))
        entity_clauses = [
            clause.strip()
            for clause in re.split(r"\s+(?:vs\.?|versus)\s+|[,;]", query)
            if 2 <= len(clause.split()) <= 18
        ]
        if len(entity_clauses) >= 2:
            candidates.extend(("entity", clause) for clause in entity_clauses[:3])
        temporal: list[str] = []
        actions: list[str] = []
        for index, word in enumerate(folded_words):
            if word in _TEMPORAL_TERMS:
                temporal.extend(folded_words[max(0, index - 1):index + 2])
            if word in _ACTION_TERMS:
                actions.extend(folded_words[index:index + 3])
        if temporal:
            candidates.append(("time", " ".join(temporal)))
        if actions:
            candidates.append(("action", " ".join(actions)))
        result: list[dict[str, str]] = []
        seen: set[str] = set()
        for pool_type, subquery in candidates:
            normalized = " ".join(subquery.split())
            key = normalized.casefold()
            if not normalized or key in seen:
                continue
            seen.add(key)
            result.append({"pool_type": pool_type, "query": normalized})
            if len(result) >= 6:
                break
        return result

    def _sharp_fts_rows(
        self,
        query: str,
        candidate_limit: int,
    ) -> tuple[list[sqlite3.Row], dict[str, Any]]:
        subqueries = self._typed_subqueries(query)
        row_by_id: dict[int, sqlite3.Row] = {}
        score_by_id: dict[int, float] = {}
        exact_phrases = self._exact_query_phrases(query)
        weights = {"raw_state": 1.0, "entity": 1.2, "time": 1.1, "action": 1.1}
        for entry in subqueries:
            expression = self._fts_expression(entry["query"])
            if not expression:
                continue
            rows = self._fts_rows(expression, candidate_limit)
            weight = weights.get(entry["pool_type"], 1.0)
            for rank, row in enumerate(rows, start=1):
                state_id = int(row["state_id"])
                row_by_id[state_id] = row
                score_by_id[state_id] = score_by_id.get(state_id, 0.0) + (
                    weight / (60.0 + rank)
                )
        boosted: list[dict[str, Any]] = []
        for state_id, row in row_by_id.items():
            title_text = f"{row['goal']} {row['url']}".casefold()
            matches = [phrase for phrase in exact_phrases if phrase in title_text]
            if matches:
                score_by_id[state_id] += 0.05 * len(matches)
                boosted.append({"state_id": state_id, "phrases": matches})
        ordered = sorted(
            row_by_id.values(),
            key=lambda row: (
                -score_by_id[int(row["state_id"])],
                int(row["ordinal"]),
                int(row["sequence_ordinal"]),
            ),
        )[:candidate_limit]
        return ordered, {
            "subqueries": subqueries,
            "exact_title_boosts": boosted,
            "lexical_pool_size": len(ordered),
        }

    @staticmethod
    def _question_template(query: str) -> str:
        folded = query.casefold()
        if any(term in folded for term in ("protocol", "workflow", "procedure", "steps")):
            return "procedure"
        words = set(re.findall(r"[a-z0-9]+", folded))
        if words.intersection(_TEMPORAL_TERMS):
            return "temporal"
        if (
            any(character in query for character in ",;")
            or " among " in folded
            or " across " in folded
            or " both " in folded
        ):
            return "multi_entity"
        if any(
            term in folded
            for term in ("page", "url", "tab", "button", "column", "field", "link")
        ):
            return "navigation"
        return "generic"

    @staticmethod
    def _template_order(
        rows: Sequence[sqlite3.Row],
        query: str,
        template: str,
    ) -> list[sqlite3.Row]:
        """Apply small question-type priors without replacing retrieval rank."""
        query_terms = {
            term.casefold()
            for term in extract_search_terms(query)
            if len(term.strip()) >= 2 and term.casefold() not in _STOPWORDS
        }
        exact_phrases = TrajectoryStore._exact_query_phrases(query)

        def _key(item: tuple[int, sqlite3.Row]) -> tuple[float, int, int]:
            position, row = item
            goal_url = f"{row['goal']} {row['url']}".casefold()
            action = str(row["incoming_action"] or "").casefold()
            text = str(row["text"] or "")[:4_000].casefold()
            exact = sum(1 for phrase in exact_phrases if phrase in goal_url)
            goal_density = sum(1 for term in query_terms if term in goal_url)
            action_density = sum(1 for term in query_terms if term in action)
            text_density = sum(1 for term in query_terms if term in text)
            prior = exact * 8.0
            if template in {"navigation", "procedure"}:
                prior += goal_density * 0.6 + action_density * 0.4
            elif template == "temporal":
                prior += (
                    (1.0 if row["observed_at"] is not None else 0.0)
                    + (1.0 if row["occurred_at"] is not None else 0.0)
                    + text_density * 0.15
                )
            elif template == "multi_entity":
                prior += goal_density * 0.35
            return (-prior, position, int(row["state_id"]))

        return [
            row for _position, row in sorted(enumerate(rows), key=_key)
        ]

    @staticmethod
    def _adaptive_excerpt_limits(
        rows: Sequence[sqlite3.Row],
        base_limit: int,
        enabled: bool,
    ) -> tuple[dict[int, int], dict[str, Any] | None]:
        limits = {int(row["state_id"]): base_limit for row in rows}
        if not enabled or not rows:
            return limits, None
        counts: dict[str, int] = {}
        for row in rows:
            trajectory_id = str(row["trajectory_id"])
            counts[trajectory_id] = counts.get(trajectory_id, 0) + 1
        total_budget = base_limit * len(rows)
        floor = max(256, (base_limit * 3) // 4)
        for row in rows:
            if counts[str(row["trajectory_id"])] > 1:
                limits[int(row["state_id"])] = floor
        used = sum(
            min(len(str(row["text"])), limits[int(row["state_id"])])
            for row in rows
        )
        bank = max(0, total_budget - used)
        raised: list[dict[str, int]] = []
        for row in rows:
            if bank <= 0:
                break
            if counts[str(row["trajectory_id"])] != 1:
                continue
            state_id = int(row["state_id"])
            current = limits[state_id]
            ceiling = min(_MAX_QUERY_TEXT_CHARS, base_limit * 2)
            desired = min(len(str(row["text"])), ceiling)
            extra = min(bank, max(0, desired - current))
            if extra:
                limits[state_id] += extra
                bank -= extra
                raised.append({"state_id": state_id, "chars": limits[state_id]})
        return limits, {
            "base_char_limit": base_limit,
            "total_char_budget": total_budget,
            "raised_only_hits": raised,
        }

    @staticmethod
    def _rendered_hit_text(hit: TrajectoryHit) -> str:
        """Mirror the official adapter's text rendering for token budgeting."""
        lines = [
            f"[{hit.exact_ref}]",
            f"Trajectory: {hit.trajectory_id}",
            f"Goal: {hit.goal}",
            f"Outcome: {hit.outcome or '<unknown>'}",
            (
                f"State: {hit.state_index} "
                f"(sequence {hit.sequence_ordinal}, step {hit.step})"
            ),
            f"URL: {hit.url}",
            f"Incoming action: {hit.incoming_action or '<none>'}",
        ]
        if hit.thoughts:
            lines.append(f"Thought: {hit.thoughts}")
        label = (
            f"Visible state excerpt (offset {hit.text_offset})"
            if hit.text_truncated else "Visible state"
        )
        lines.append(f"{label}: {hit.text}")
        if hit.observed_at is not None:
            lines.append(
                f"Observed at: {hit.observed_at} (source: {hit.observed_at_source})"
            )
        if hit.occurred_at is not None:
            lines.append(
                f"Occurred at: {hit.occurred_at} (source: {hit.occurred_at_source})"
            )
        return "\n".join(lines)

    @staticmethod
    def _trim_hit_text(hit: TrajectoryHit, target_tokens: int) -> TrajectoryHit:
        from .tokens import count_tokens

        if count_tokens(hit.text) <= target_tokens:
            return hit
        low, high = 0, len(hit.text)
        while low < high:
            middle = (low + high + 1) // 2
            if count_tokens(hit.text[:middle]) <= target_tokens:
                low = middle
            else:
                high = middle - 1
        keep = max(0, low)
        start = max(0, (len(hit.text) - keep) // 2)
        return replace(
            hit,
            text=hit.text[start:start + keep],
            text_offset=hit.text_offset + start,
            text_truncated=True,
        )

    @staticmethod
    def _apply_sharp_token_budget(
        hits: Sequence[TrajectoryHit],
        token_budget: int,
    ) -> tuple[list[TrajectoryHit], dict[str, Any]]:
        from .tokens import count_tokens

        working = list(hits)
        original_tokens = sum(
            count_tokens(TrajectoryStore._rendered_hit_text(hit)) for hit in working
        )
        dropped: list[str] = []
        while working:
            fixed_tokens = sum(
                count_tokens(
                    TrajectoryStore._rendered_hit_text(replace(hit, text=""))
                )
                for hit in working
            )
            if fixed_tokens <= token_budget:
                break
            dropped.append(working.pop().exact_ref)
        if working:
            fixed_tokens = sum(
                count_tokens(
                    TrajectoryStore._rendered_hit_text(replace(hit, text=""))
                )
                for hit in working
            )
            available = max(0, token_budget - fixed_tokens)
            per_hit = max(0, available // len(working))
            working = [
                TrajectoryStore._trim_hit_text(hit, per_hit) for hit in working
            ]
        final_tokens = sum(
            count_tokens(TrajectoryStore._rendered_hit_text(hit)) for hit in working
        )
        while working and final_tokens > token_budget:
            largest_index = max(
                range(len(working)),
                key=lambda index: count_tokens(working[index].text),
            )
            current = count_tokens(working[largest_index].text)
            if current <= 0:
                dropped.append(working.pop().exact_ref)
            else:
                working[largest_index] = TrajectoryStore._trim_hit_text(
                    working[largest_index], max(0, current - 8)
                )
            final_tokens = sum(
                count_tokens(TrajectoryStore._rendered_hit_text(hit))
                for hit in working
            )
        return working, {
            "text_token_budget": token_budget,
            "rendered_text_tokens_before": original_tokens,
            "rendered_text_tokens_after": final_tokens,
            "dropped_evidence_refs": dropped,
        }

    @staticmethod
    def _select_diverse(
        rows: Iterable[sqlite3.Row],
        limit: int,
        max_per_trajectory: int = 5,
    ) -> list[sqlite3.Row]:
        selected: list[sqlite3.Row] = []
        per_trajectory: dict[str, int] = {}
        for row in rows:
            trajectory_id = str(row["trajectory_id"])
            if per_trajectory.get(trajectory_id, 0) >= max_per_trajectory:
                continue
            selected.append(row)
            per_trajectory[trajectory_id] = per_trajectory.get(trajectory_id, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _select_with_floor(
        fused_rows: Sequence[sqlite3.Row],
        global_rows: Sequence[sqlite3.Row],
        limit: int,
        floor_k: int,
        max_per_trajectory: int = 5,
    ) -> list[sqlite3.Row]:
        """Policy A -- reserve ``floor_k`` nucleus slots for the top pure-BM25
        states, then fill the remainder from the fused order.

        The lexical floor guarantees the strongest lexical winners a slot even
        when the semantic boost would otherwise let a few semantic-top
        trajectories monopolise the nucleus. Both the floor and the fill honour
        the same 5-per-trajectory diversity cap as ``_select_diverse``.
        """
        selected = list(TrajectoryStore._select_diverse(
            global_rows,
            floor_k,
            max_per_trajectory=max_per_trajectory,
        ))
        selected_ids = {int(row["state_id"]) for row in selected}
        per_trajectory: dict[str, int] = {}
        for row in selected:
            trajectory_id = str(row["trajectory_id"])
            per_trajectory[trajectory_id] = per_trajectory.get(trajectory_id, 0) + 1
        for row in fused_rows:
            if len(selected) >= limit:
                break
            state_id = int(row["state_id"])
            if state_id in selected_ids:
                continue
            trajectory_id = str(row["trajectory_id"])
            if per_trajectory.get(trajectory_id, 0) >= max_per_trajectory:
                continue
            selected.append(row)
            selected_ids.add(state_id)
            per_trajectory[trajectory_id] = per_trajectory.get(trajectory_id, 0) + 1
        return selected[:limit]

    @staticmethod
    def _merge_arms(
        arm_lex: Sequence[sqlite3.Row],
        arm_sem: Sequence[sqlite3.Row],
        limit: int,
        q_lex: int,
        q_sem: int,
        floor_k: int = 0,
    ) -> list[sqlite3.Row]:
        """Policy D -- round-robin a pure-lexical arm and the semantic/fused arm
        into the nucleus by a ``q_lex:q_sem`` quota.

        Strictly generalises Policy A: the lexical arm is guaranteed its quota
        (serving the SOURCE_MISS bucket) while the semantic arm keeps its own
        quota (preserving the semantic gains). Deduped by ``state_id``; a short
        arm is backfilled by the other (a skipped duplicate does not consume a
        quota slot). Arm order is the deterministic tie-break.

        ``floor_k`` composes Policy A on top (the A+D hybrid, issue #127): the
        top ``floor_k`` pure-lexical (BM25) states are reserved a nucleus slot
        FIRST -- protecting the strongest lexical incumbents as Policy A does --
        and the quota round-robin then fills the remaining slots. ``floor_k == 0``
        (default) is byte-identical to the pure Policy D round-robin. The floor
        is drawn from the head of ``arm_lex`` (already 5-per-trajectory
        diversity-capped), matching ``_select_with_floor``'s guaranteed slots.
        """
        selected: list[sqlite3.Row] = []
        seen: set[int] = set()

        def _pull(arm: Sequence[sqlite3.Row], start: int, quota: int) -> int:
            added = 0
            index = start
            while index < len(arm) and added < quota and len(selected) < limit:
                row = arm[index]
                index += 1
                state_id = int(row["state_id"])
                if state_id in seen:
                    continue
                selected.append(row)
                seen.add(state_id)
                added += 1
            return index

        lex_i = sem_i = 0
        floor_k = min(max(0, int(floor_k)), limit)
        if floor_k:
            lex_i = _pull(arm_lex, lex_i, floor_k)
        while len(selected) < limit and (lex_i < len(arm_lex) or sem_i < len(arm_sem)):
            next_lex = _pull(arm_lex, lex_i, q_lex)
            next_sem = _pull(arm_sem, sem_i, q_sem)
            if next_lex == lex_i and next_sem == sem_i:
                break
            lex_i, sem_i = next_lex, next_sem
        return selected[:limit]

    def _fts_rows(
        self,
        expression: str,
        candidate_limit: int,
        *,
        source_ids: Sequence[int] = (),
    ) -> list[sqlite3.Row]:
        source_clause = ""
        params: list[Any] = [expression]
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            source_clause = f" AND s.source_id IN ({placeholders})"
            params.extend(int(source_id) for source_id in source_ids)
        params.append(int(candidate_limit))
        return self._conn.execute(
            f"""
            SELECT s.*, src.trajectory_id, src.goal, src.outcome, src.ordinal,
                   a.relative_path, a.sha256 AS asset_sha256,
                   bm25(lcm_trajectory_states_fts) AS rank
            FROM lcm_trajectory_states_fts
            JOIN lcm_trajectory_states s
              ON s.state_id = lcm_trajectory_states_fts.rowid
            JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
            LEFT JOIN lcm_trajectory_assets a ON a.state_id = s.state_id
            WHERE lcm_trajectory_states_fts MATCH ?{source_clause}
            ORDER BY rank ASC, src.ordinal ASC, s.sequence_ordinal ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

    def _adjacency_expansion_arm(
        self,
        seed_rows: Sequence[sqlite3.Row],
        radius: int,
    ) -> list[tuple[int, int, int]]:
        """H5(b) pool-expansion arm (issue #135): sequence neighbors of the
        lexical seed hits, as ``(state_id, seed_state_id, distance)`` triples.

        Every pool row IS a lexical seed (it entered via ``global_rows`` /
        ``scoped_rows`` FTS), so the seeds are exactly ``seed_rows`` in pool
        (fused-rank) order. For each seed, the states at ``sequence_ordinal
        +/- 1..radius`` WITHIN THE SAME SOURCE are candidates, giving a
        non-lexical recall path: a target state with no query-term match of
        its own is reachable when any state of its trajectory seeds.

        Deterministic arm order is DISTANCE-major, then seed pool rank, then
        ordinal ascending (``-d`` before ``+d``): a +/-1 neighbor of any seed
        outranks a +/-2 neighbor of a stronger seed, mirroring the
        ``ORDER BY ABS(...)`` discipline of the delivery-stage adjacency
        backfill. States already in the pool are excluded; expanded neighbors
        earn NO semantic boost and no BM25 rank of their own (anti-magnet
        control -- they are admitted by the caller only through the
        quota-capped ``_merge_arms`` tail).

        Returns LIGHTWEIGHT id triples (a batched index-only probe): the full
        rows -- state text is large -- are fetched by the caller for the
        ADMITTED quota subset only, keeping the expansion inside the latency
        budget (gate iv).
        """
        pool_ids = {int(row["state_id"]) for row in seed_rows}
        positions: list[tuple[int, int]] = []
        wanted: set[tuple[int, int]] = set()
        occupied = {
            (int(row["source_id"]), int(row["sequence_ordinal"]))
            for row in seed_rows
        }
        for row in seed_rows:
            source_id = int(row["source_id"])
            ordinal = int(row["sequence_ordinal"])
            for distance in range(1, radius + 1):
                for neighbor in (ordinal - distance, ordinal + distance):
                    key = (source_id, neighbor)
                    if neighbor < 0 or key in occupied or key in wanted:
                        continue
                    wanted.add(key)
                    positions.append(key)
        id_by_position: dict[tuple[int, int], int] = {}
        chunk = 400  # 2 bound params per pair; stay far below SQLite limits
        for start in range(0, len(positions), chunk):
            batch = positions[start : start + chunk]
            values = ",".join("(?,?)" for _ in batch)
            params: list[int] = []
            for source_id, neighbor in batch:
                params.extend((source_id, neighbor))
            fetched = self._conn.execute(
                f"""
                SELECT s.state_id, s.source_id, s.sequence_ordinal
                FROM lcm_trajectory_states s
                WHERE (s.source_id, s.sequence_ordinal) IN (VALUES {values})
                """,
                params,
            ).fetchall()
            for fetched_row in fetched:
                id_by_position[
                    (int(fetched_row["source_id"]), int(fetched_row["sequence_ordinal"]))
                ] = int(fetched_row["state_id"])
        arm: list[tuple[int, int, int]] = []
        emitted: set[int] = set(pool_ids)
        for distance in range(1, radius + 1):
            for row in seed_rows:
                source_id = int(row["source_id"])
                ordinal = int(row["sequence_ordinal"])
                for neighbor in (ordinal - distance, ordinal + distance):
                    state_id = id_by_position.get((source_id, neighbor))
                    if state_id is None or state_id in emitted:
                        continue
                    emitted.add(state_id)
                    arm.append((state_id, int(row["state_id"]), distance))
        return arm

    def _state_rows_by_ids(self, state_ids: Sequence[int]) -> dict[int, sqlite3.Row]:
        """Full candidate-shaped rows (text, source join, asset join) for the
        given state ids -- the same column shape as ``_fts_rows`` with a
        placeholder rank, fetched only for the quota-admitted expansion
        states."""
        if not state_ids:
            return {}
        placeholders = ",".join("?" for _ in state_ids)
        rows = self._conn.execute(
            f"""
            SELECT s.*, src.trajectory_id, src.goal, src.outcome, src.ordinal,
                   a.relative_path, a.sha256 AS asset_sha256, 0.0 AS rank
            FROM lcm_trajectory_states s
            JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
            LEFT JOIN lcm_trajectory_assets a ON a.state_id = s.state_id
            WHERE s.state_id IN ({placeholders})
            """,
            [int(state_id) for state_id in state_ids],
        ).fetchall()
        return {int(row["state_id"]): row for row in rows}

    def query(
        self,
        query: str,
        *,
        candidate_limit: int = 128,
        limit: int = 16,
        image_limit: int = 8,
        include_adjacent: bool = True,
        text_char_limit: int = 2_000,
        lexical_floor: int = 0,
        arm_quota: tuple[int, int] | None = None,
        adjacency_radius: int = 0,
        adjacency_quota: int = 0,
        state_semantic_quota: int = 0,
        diversity_cap: int = 0,
        adaptive_excerpt: bool = False,
        sharp_token_budget: int = 0,
        antiboilerplate: bool = False,
        title_boost: bool = False,
    ) -> tuple[TrajectoryHit, ...]:
        if self.status != "complete":
            raise CorpusIdentityError("trajectory corpus must be finalized before query")
        candidate_limit = min(max(1, int(candidate_limit)), _MAX_CANDIDATES)
        limit = min(max(1, int(limit)), _MAX_RESULTS)
        image_limit = min(max(0, int(image_limit)), _MAX_IMAGES)
        lexical_floor = min(max(0, int(lexical_floor)), _MAX_RESULTS)
        adjacency_radius = min(max(0, int(adjacency_radius)), _MAX_ADJACENCY_RADIUS)
        adjacency_quota = min(max(0, int(adjacency_quota)), _MAX_CANDIDATES)
        state_semantic_quota = min(max(0, int(state_semantic_quota)), _MAX_CANDIDATES)
        diversity_cap = min(max(0, int(diversity_cap)), _MAX_DIVERSITY_CAP)
        sharp_token_budget = min(
            max(0, int(sharp_token_budget)),
            _MAX_SHARP_TOKEN_BUDGET,
        )
        adaptive_excerpt = bool(adaptive_excerpt)
        antiboilerplate = bool(antiboilerplate)
        title_boost = bool(title_boost)
        text_char_limit = min(
            max(256, int(text_char_limit)),
            _MAX_QUERY_TEXT_CHARS,
        )
        expression = self._fts_expression(query)
        if not expression and state_semantic_quota == 0:
            self._last_query_telemetry = {
                "semantic_attempt": None,
                "source_candidate_ranks": [],
                "state_candidate_pool": [],
                "delivered_evidence_refs": [],
            }
            return ()
        sharp_telemetry: dict[str, Any] | None = None
        if not expression:
            global_rows = []
        elif sharp_token_budget > 0:
            global_rows, sharp_telemetry = self._sharp_fts_rows(
                query, candidate_limit
            )
        else:
            global_rows = self._fts_rows(expression, candidate_limit)
        # Knob H (title boost, default-off): stable-reorder the lexical
        # candidate pool so states whose title/heading/field-label text exactly
        # contains a 2-4 gram question phrase rank ahead of same-band peers.
        # ``title_boost is False`` (default) leaves ``global_rows`` untouched and
        # reproduces current bytes.
        title_boost_telemetry: dict[str, Any] | None = None
        if title_boost:
            global_rows, title_boost_telemetry = self._apply_title_boost(
                global_rows, query
            )
        semantic_ranks: list[tuple[int, float]] = []
        semantic_attempt: TrajectorySemanticAttempt | None = None
        if expression:
            attempt_started = time.monotonic()
            calls_before = self._semantic_usage["query_calls"]
            try:
                semantic_ranks = self._semantic_source_ranks(query)
            except Exception as exc:
                # Restore historical semantics FIRST, unconditionally, before any
                # introspection can fail: the fallback counter must bump even if the
                # (hostile) exception explodes during telemetry recording.
                self._semantic_usage["fallbacks"] += 1
                fallback_latency_ms = (time.monotonic() - attempt_started) * 1000.0
                try:
                    # Was a bare ``except Exception: fallbacks += 1`` that discarded
                    # the failure class/status. Now the typed reason survives -- and
                    # the whole record step is itself fenced so an exotic exception
                    # (kind/status_code/retry_after as raising properties) degrades
                    # to a clean FTS fallback instead of failing the query.
                    semantic_attempt = self._record_semantic_attempt(
                        outcome="fallback",
                        latency_ms=fallback_latency_ms,
                        exception=exc,
                    )
                except Exception:
                    semantic_attempt = self._record_minimal_fallback_attempt(
                        latency_ms=fallback_latency_ms,
                    )
            else:
                # Only record a success when an embed was actually dispatched; an
                # early return (no provider / profile mismatch) is a skip, not an
                # attempt, and must not inflate the success count.
                if self._semantic_usage["query_calls"] > calls_before:
                    semantic_attempt = self._record_semantic_attempt(
                        outcome="success",
                        latency_ms=(time.monotonic() - attempt_started) * 1000.0,
                        exception=None,
                    )
        scoped_rows = self._fts_rows(
            expression,
            candidate_limit,
            source_ids=[source_id for source_id, _score in semantic_ranks],
        ) if expression and semantic_ranks else []

        if semantic_ranks and scoped_rows:
            row_by_id: dict[int, sqlite3.Row] = {}
            score_by_candidate: dict[int, float] = {}
            semantic_position = {
                source_id: position
                for position, (source_id, _score) in enumerate(semantic_ranks, start=1)
            }
            for position, row in enumerate(global_rows, start=1):
                state_id = int(row["state_id"])
                row_by_id[state_id] = row
                score_by_candidate[state_id] = score_by_candidate.get(state_id, 0.0) + (
                    1.0 / (60.0 + position)
                )
            for position, row in enumerate(scoped_rows, start=1):
                state_id = int(row["state_id"])
                row_by_id[state_id] = row
                score_by_candidate[state_id] = score_by_candidate.get(state_id, 0.0) + (
                    1.0 / (60.0 + position)
                )
                trajectory_position = semantic_position.get(int(row["source_id"]), 32)
                score_by_candidate[state_id] += 1.0 / (60.0 + trajectory_position)
            rows = sorted(
                row_by_id.values(),
                key=lambda row: (
                    -score_by_candidate[int(row["state_id"])],
                    int(row["ordinal"]),
                    int(row["sequence_ordinal"]),
                ),
            )
            candidate_kind = {
                int(row["state_id"]): "semantic_fts"
                for row in scoped_rows
            }
            candidate_score = {
                state_id: -score
                for state_id, score in score_by_candidate.items()
            }
        else:
            rows = global_rows
            candidate_kind = {}
            candidate_score = {
                int(row["state_id"]): float(row["rank"])
                for row in rows
            }

        # H5(b) lexical-seed adjacency pool-expansion (issue #135): pull the
        # sequence neighbors of the lexical seed hits INTO the state pool,
        # pre-selection, as a QUOTA-CAPPED ADDITIVE arm through the existing
        # ``_merge_arms`` machinery. The arm is appended strictly AFTER the
        # ranked pool (no semantic boost, no BM25 rank of its own), so the
        # nucleus selection -- and therefore delivery -- only changes when the
        # ranked pool alone cannot fill the nucleus; the 5-per-trajectory
        # diversity cap at selection is untouched. ``adjacency_radius == 0``
        # or ``adjacency_quota == 0`` (defaults) skip this path entirely and
        # reproduce current bytes.
        adjacency_admitted: list[dict[str, int]] = []
        if adjacency_radius > 0 and adjacency_quota > 0 and rows:
            arm_triples = self._adjacency_expansion_arm(rows, adjacency_radius)
            if arm_triples:
                seed_by_state = {
                    state_id: (seed_state_id, distance)
                    for state_id, seed_state_id, distance in arm_triples
                }
                arm_adjacent = [
                    {"state_id": state_id} for state_id, _seed, _dist in arm_triples
                ]
                merged = self._merge_arms(
                    rows,
                    arm_adjacent,  # type: ignore[arg-type]  # only "state_id" is read
                    len(rows) + adjacency_quota,
                    len(rows),
                    adjacency_quota,
                )
                admitted_ids = [int(row["state_id"]) for row in merged[len(rows):]]
                full_by_id = self._state_rows_by_ids(admitted_ids)
                expanded = list(rows)
                for state_id in admitted_ids:
                    seed_state_id, distance = seed_by_state[state_id]
                    expanded.append(full_by_id[state_id])
                    candidate_kind[state_id] = "adjacent"
                    candidate_score[state_id] = (
                        candidate_score[seed_state_id] + 0.000001 * distance
                    )
                    adjacency_admitted.append({
                        "state_id": state_id,
                        "seed_state_id": seed_state_id,
                        "distance": distance,
                    })
                rows = expanded

        # State-level semantic pool-expansion (issue #142, Lane S / W3a): rank
        # the per-state semantic index against the query and admit up to
        # ``state_semantic_quota`` states that are NOT already in the pool, as a
        # STRICTLY ADDITIVE tail through the same ``_merge_arms`` machinery as the
        # adjacency arm -- no semantic boost, no BM25 rank of their own, appended
        # AFTER the ranked pool so the nucleus (and delivery) only changes when
        # the ranked pool alone underfills. ``state_semantic_quota == 0``
        # (default), no provider, or no active state index skip this entirely and
        # reproduce current bytes. Independent of the adjacency arm above.
        state_semantic_admitted: list[dict[str, Any]] = []
        if state_semantic_quota > 0:
            pool_ids = {int(row["state_id"]) for row in rows}
            state_attempt_started = time.monotonic()
            try:
                ranked_states = self._semantic_state_ranks(
                    query, state_semantic_quota + len(pool_ids) + 16
                )
            except Exception as exc:
                self._semantic_usage["fallbacks"] += 1
                fallback_latency_ms = (
                    time.monotonic() - state_attempt_started
                ) * 1000.0
                try:
                    semantic_attempt = self._record_semantic_attempt(
                        outcome="fallback",
                        latency_ms=fallback_latency_ms,
                        exception=exc,
                    )
                except Exception:
                    semantic_attempt = self._record_minimal_fallback_attempt(
                        latency_ms=fallback_latency_ms,
                    )
                ranked_states = []
            score_by_state = {sid: score for sid, score in ranked_states}
            arm_semantic = [
                {"state_id": sid}
                for sid, _score in ranked_states
                if sid not in pool_ids
            ]
            if arm_semantic:
                merged = self._merge_arms(
                    rows,
                    arm_semantic,  # type: ignore[arg-type]  # only "state_id" is read
                    len(rows) + state_semantic_quota,
                    len(rows),
                    state_semantic_quota,
                )
                admitted_ids = [int(row["state_id"]) for row in merged[len(rows):]]
                full_by_id = self._state_rows_by_ids(admitted_ids)
                expanded = list(rows)
                for rank, state_id in enumerate(admitted_ids, start=1):
                    similarity = float(score_by_state.get(state_id, 0.0))
                    expanded.append(full_by_id[state_id])
                    candidate_kind[state_id] = "state_semantic"
                    candidate_score[state_id] = similarity
                    state_semantic_admitted.append({
                        "state_id": state_id,
                        "rank": rank,
                        "score": similarity,
                    })
                rows = expanded

        question_template: str | None = None
        if sharp_token_budget > 0:
            question_template = self._question_template(query)
            rows = self._template_order(rows, query, question_template)

        diversity_telemetry: dict[str, Any] | None = None
        diversity_survivors: set[int] | None = None
        if diversity_cap > 0:
            # C1 is deliberately applied once, after lexical, source-semantic,
            # adjacency, and state-semantic arms have composed. Filtering each
            # arm independently would allow a capped hub to re-enter through a
            # different arm.
            rows, diversity_telemetry = self._cap_composed_pool(
                rows,
                diversity_cap,
                antiboilerplate=antiboilerplate,
                query_terms=(
                    self._query_term_set(query) if antiboilerplate else None
                ),
            )
            diversity_survivors = {
                int(row["state_id"]) for row in rows
            }

        adjacent_reserve = min(6, limit // 3) if include_adjacent else 0
        nucleus_limit = max(1, limit - adjacent_reserve)
        lexical_candidates = (
            [
                row for row in global_rows
                if int(row["state_id"]) in diversity_survivors
            ]
            if diversity_survivors is not None else global_rows
        )
        per_trajectory = diversity_cap or 5
        if arm_quota is not None:
            # Policy D (candidate-composition repair, issue #127): round-robin a
            # pure-lexical arm and the semantic/fused arm into the nucleus by the
            # requested quota. Superset of Policy A; ``arm_quota is None``
            # (default) is byte-identical to the historical selection below.
            # When ``lexical_floor > 0`` is ALSO supplied this becomes the A+D
            # hybrid: the top ``lexical_floor`` pure-BM25 incumbents are reserved
            # a slot first, then the quota round-robin fills the rest
            # (``lexical_floor == 0`` reproduces the pure Policy D bytes).
            q_lex = max(0, int(arm_quota[0]))
            q_sem = max(0, int(arm_quota[1]))
            arm_lex = self._select_diverse(
                lexical_candidates,
                nucleus_limit,
                max_per_trajectory=per_trajectory,
            )
            arm_sem = self._select_diverse(
                rows,
                nucleus_limit,
                max_per_trajectory=per_trajectory,
            )
            selected = self._merge_arms(
                arm_lex, arm_sem, nucleus_limit, q_lex, q_sem,
                floor_k=lexical_floor,
            )
        elif lexical_floor > 0:
            # Policy A (candidate-composition repair, issue #127): guarantee the
            # top pure-BM25 states a nucleus slot before the fused order fills
            # the rest. ``lexical_floor == 0`` (default) is byte-identical to the
            # historical fused-only selection below.
            selected = self._select_with_floor(
                rows,
                lexical_candidates,
                nucleus_limit,
                lexical_floor,
                max_per_trajectory=per_trajectory,
            )
        else:
            selected = self._select_diverse(
                rows,
                nucleus_limit,
                max_per_trajectory=diversity_cap or 5,
            )
        selected_ids = {int(row["state_id"]) for row in selected}
        match_kind_by_id = {
            int(row["state_id"]): candidate_kind.get(int(row["state_id"]), "fts")
            for row in selected
        }
        score_by_id = {
            int(row["state_id"]): candidate_score[int(row["state_id"])]
            for row in selected
        }
        selected_per_trajectory: dict[str, int] = {}
        for row in selected:
            trajectory_id = str(row["trajectory_id"])
            selected_per_trajectory[trajectory_id] = (
                selected_per_trajectory.get(trajectory_id, 0) + 1
            )

        if include_adjacent and selected and len(selected) < limit:
            nucleus_rows = list(selected)
            adjacent_by_nucleus: list[list[sqlite3.Row]] = []
            for nucleus in nucleus_rows:
                adjacent_rows = self._conn.execute(
                    """
                    SELECT s.*, src.trajectory_id, src.goal, src.outcome, src.ordinal,
                           a.relative_path, a.sha256 AS asset_sha256, 0.0 AS rank
                    FROM lcm_trajectory_states s
                    JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
                    LEFT JOIN lcm_trajectory_assets a ON a.state_id = s.state_id
                    WHERE s.source_id = ? AND s.sequence_ordinal IN (?, ?)
                    ORDER BY ABS(s.sequence_ordinal - ?), s.sequence_ordinal
                    """,
                    (
                        int(nucleus["source_id"]),
                        int(nucleus["sequence_ordinal"]) - 1,
                        int(nucleus["sequence_ordinal"]) + 1,
                        int(nucleus["sequence_ordinal"]),
                    ),
                ).fetchall()
                adjacent_by_nucleus.append(list(adjacent_rows))
            while len(selected) < limit and any(adjacent_by_nucleus):
                made_progress = False
                for nucleus, adjacent_rows in zip(nucleus_rows, adjacent_by_nucleus):
                    while adjacent_rows:
                        row = adjacent_rows.pop(0)
                        state_id = int(row["state_id"])
                        if state_id in selected_ids:
                            continue
                        trajectory_id = str(row["trajectory_id"])
                        if (
                            diversity_cap > 0
                            and selected_per_trajectory.get(trajectory_id, 0)
                            >= diversity_cap
                        ):
                            continue
                        selected.append(row)
                        selected_ids.add(state_id)
                        selected_per_trajectory[trajectory_id] = (
                            selected_per_trajectory.get(trajectory_id, 0) + 1
                        )
                        match_kind_by_id[state_id] = "adjacent"
                        score_by_id[state_id] = score_by_id[int(nucleus["state_id"])] + 0.000001
                        made_progress = True
                        break
                    if len(selected) >= limit:
                        break
                if not made_progress:
                    break

        # The adjacency reserve is a priority carve-out, not a quota. Return
        # any unused slots to the ranked pool in its existing order.
        if include_adjacent and len(selected) < limit:
            for row in rows:
                if len(selected) >= limit:
                    break
                state_id = int(row["state_id"])
                if state_id in selected_ids:
                    continue
                trajectory_id = str(row["trajectory_id"])
                if (
                    diversity_cap > 0
                    and selected_per_trajectory.get(trajectory_id, 0)
                    >= diversity_cap
                ):
                    continue
                selected.append(row)
                selected_ids.add(state_id)
                selected_per_trajectory[trajectory_id] = (
                    selected_per_trajectory.get(trajectory_id, 0) + 1
                )
                match_kind_by_id[state_id] = candidate_kind.get(state_id, "fts")
                score_by_id[state_id] = candidate_score[state_id]

        adaptive_active = (
            adaptive_excerpt
            and str(self.identity_payload.get("domain", "")).casefold() == "web"
        )
        excerpt_limits, excerpt_telemetry = self._adaptive_excerpt_limits(
            selected[:limit],
            text_char_limit,
            adaptive_active,
        )
        hits: list[TrajectoryHit] = []
        for index, row in enumerate(selected[:limit]):
            state_id = int(row["state_id"])
            hits.append(self._row_to_hit(
                row,
                score=score_by_id[state_id],
                match_kind=match_kind_by_id[state_id],
                include_image=index < image_limit,
                query=query,
                text_char_limit=excerpt_limits[state_id],
                dense_excerpt=adaptive_active,
            ))
        budget_telemetry: dict[str, Any] | None = None
        if sharp_token_budget > 0:
            hits, budget_telemetry = self._apply_sharp_token_budget(
                hits, sharp_token_budget
            )

        # Side-channel per-query telemetry (does not affect the returned hits).
        self._last_query_telemetry = {
            "semantic_attempt": (
                asdict(semantic_attempt) if semantic_attempt is not None else None
            ),
            "source_candidate_ranks": [
                {"source_id": int(source_id), "rank": position, "score": float(score)}
                for position, (source_id, score) in enumerate(
                    semantic_ranks[:64], start=1
                )
            ],
            "state_candidate_pool": [
                {
                    "state_id": int(row["state_id"]),
                    "rank": position,
                    "score": float(candidate_score.get(int(row["state_id"]), 0.0)),
                }
                for position, row in enumerate(rows[:64], start=1)
            ],
            "delivered_evidence_refs": [hit.exact_ref for hit in hits],
        }
        if diversity_telemetry is not None:
            # Present only when C1 is active; the complete survivor ids let the
            # provider-free replay harness measure exact pool coverage without
            # relying on the historical 64-row telemetry display bound.
            self._last_query_telemetry["diversity_cap"] = diversity_telemetry
        if excerpt_telemetry is not None:
            self._last_query_telemetry["adaptive_excerpt"] = excerpt_telemetry
        if sharp_telemetry is not None:
            self._last_query_telemetry["sharp_compilation"] = {
                **sharp_telemetry,
                **(budget_telemetry or {}),
                "question_template": question_template,
            }
        if adjacency_radius > 0 and adjacency_quota > 0:
            # Present only when the H5(b) knob is active so the default
            # telemetry payload stays byte-identical (golden 451/451).
            self._last_query_telemetry["adjacency_expansion"] = {
                "radius": adjacency_radius,
                "quota": adjacency_quota,
                "admitted": adjacency_admitted,
            }
        if state_semantic_quota > 0:
            # Present only when the state-semantic knob is active so the default
            # telemetry payload stays byte-identical (golden 451/451).
            self._last_query_telemetry["state_semantic_expansion"] = {
                "quota": state_semantic_quota,
                "admitted": state_semantic_admitted,
            }
        if title_boost_telemetry is not None:
            # Knob H: present only when title boost is active so the default
            # telemetry payload stays byte-identical (golden 451/451).
            self._last_query_telemetry["title_boost"] = title_boost_telemetry
        return tuple(hits)

    def resolve_exact_ref(self, exact_ref: str) -> TrajectoryHit:
        match = _EXACT_REF_RE.fullmatch(str(exact_ref or "").strip())
        if not match or match.group("corpus") != self.corpus_uid:
            raise ExactTrajectoryRefError("exact trajectory ref does not match this corpus")
        trajectory_id = unquote(match.group("trajectory"))
        state_index = int(match.group("state"))
        row = self._conn.execute(
            """
            SELECT s.*, src.trajectory_id, src.goal, src.outcome, a.relative_path,
                   a.sha256 AS asset_sha256
            FROM lcm_trajectory_states s
            JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
            LEFT JOIN lcm_trajectory_assets a ON a.state_id = s.state_id
            WHERE src.trajectory_id = ? AND s.state_index = ?
            """,
            (trajectory_id, state_index),
        ).fetchone()
        if row is None:
            raise ExactTrajectoryRefError("exact trajectory ref does not resolve")
        return self._row_to_hit(
            row,
            score=0.0,
            match_kind="exact_ref",
            include_image=True,
        )

    @staticmethod
    def query_digest(hits: Sequence[TrajectoryHit]) -> str:
        portable_hits: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.to_dict()
            payload.pop("screenshot_path", None)
            portable_hits.append(payload)
        return _sha256_text(_canonical_json(portable_hits))

    def manifest(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM lcm_trajectory_corpora WHERE singleton = 1"
        ).fetchone()
        semantic = self._semantic_profile()
        return {
            "identity": dict(self.identity_payload),
            "identity_digest": str(row["identity_digest"]),
            "schema_version": int(row["schema_version"]),
            "corpus_uid": row["corpus_uid"],
            "haystack_digest": row["haystack_digest"],
            "source_manifest_digest": row["source_manifest_digest"],
            "trajectory_count": row["trajectory_count"],
            "ingest_cursor": int(row["ingest_cursor"]),
            "status": str(row["status"]),
            "semantic_index": (
                {
                    "profile_digest": str(semantic["profile_digest"]),
                    "provider": str(semantic["provider"]),
                    "model_name": str(semantic["model_name"]),
                    "dim": int(semantic["dim"]),
                    "document_version": str(semantic["document_version"]),
                    "document_count": int(semantic["document_count"]),
                    "index_digest": str(semantic["index_digest"]),
                }
                if semantic is not None
                else None
            ),
        }

    def backup_to(self, destination: str | Path) -> Path:
        self._require_writable()
        destination_path = Path(destination)
        if destination_path.resolve() == self.db_path.resolve():
            raise ValueError("trajectory backup destination must differ from source")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            raise FileExistsError(destination_path)
        with self._lock:
            self._conn.commit()
            target = sqlite3.connect(str(destination_path))
            try:
                self._conn.backup(target)
                target.commit()
            finally:
                target.close()
        return destination_path

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            if conn is None:
                return
            try:
                if not self.read_only:
                    conn.commit()
            finally:
                conn.close()
                self._conn = None  # type: ignore[assignment]
