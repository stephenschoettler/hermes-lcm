"""SQLite-backed storage and brute-force KNN for summary embeddings.

This module is intentionally not wired into the LCM engine yet. It owns only
the durable embedding schema operations and the dependency-optional compute
ladder used by later semantic retrieval work.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import math
import sqlite3
import struct
import sys
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from .config import LCMConfig
from .db_bootstrap import (
    configure_connection,
    ensure_chunk_tables,
    ensure_embedding_tables,
    mark_migration_step_complete,
    refuse_schema_version_too_new,
    run_versioned_migrations,
    verify_chunk_schema,
    verify_embedding_schema,
)
from .sqlite_util import _is_sqlite_locked_error

logger = logging.getLogger(__name__)

_monotonic = time.monotonic


def _prescreen_deadline_expired(deadline: float | None, scanned_rows: int) -> bool:
    """Deadline check for the binary-prescreen loop, position-aware.

    ``scanned_rows`` is how many rows have completed when the check runs (the
    batch's ``start`` before scoring, its ``end`` after). Tests patch THIS
    seam keyed on position — not the process clock, call counts, or frame
    inspection — so a loop refactor cannot silently disable the covered
    branch.
    """
    del scanned_rows
    return deadline is not None and _monotonic() >= deadline


def _caller_deadline_expired(
    caller_deadline: float | None,
    *,
    stop_deadline: float | None,
    stop_expired: bool,
) -> bool:
    """Classify an effective stop as the caller's absolute deadline only."""
    if caller_deadline is None or not stop_expired:
        return False
    return stop_deadline == caller_deadline or _monotonic() >= caller_deadline


# Vectors are float32 in native little-endian order by default. These are
# recorded as part of the canonical profile identity so a dtype/byteorder change
# is detectable rather than silently reinterpreting stored bytes.
_VECTOR_DTYPE = "float32"
_VECTOR_BYTEORDER = "little"
# Opt-in int8 storage dtype (SPEC C1): vectors are stored per-vector
# symmetrically quantized — ``dim`` signed int8 values followed by a little-endian
# float32 scale in the SAME ``vec`` BLOB (total ``dim + 4`` bytes) — and a
# sign-bit prescreen is written to the companion binary table. dtype is part of
# the profile identity hash, so int8 vectors never mix with float32 ones (a
# float32 reader's ``len(blob) == dim*4`` guard also rejects an int8 blob).
_INT8_DTYPE = "int8"
_SUPPORTED_DTYPES = frozenset({_VECTOR_DTYPE, _INT8_DTYPE})
# int8 symmetric quantization uses the full signed range except -128 (kept out so
# the range is symmetric and dequantization has no bias toward negatives).
_INT8_MAX = 127
# Per-byte set-bit counts for the binary Hamming prescreen (built once; converted
# to a NumPy lookup array at query time). Indexing a uint8 XOR result into this
# table and summing rows is the popcount without a per-element Python loop.
_POPCOUNT_TABLE = tuple(bin(value).count("1") for value in range(256))
# Revision tag that folds the float32 sign-bit prescreen opt-in
# (LCM_EMBEDDING_BINARY_PRESCREEN) into the profile identity. A prescreen corpus
# grows a companion sign-bit table that drives the two-stage KNN, so it MUST be a
# distinct identity from a legacy binary-free float32 corpus — otherwise flipping
# the flag on a populated identity leaves a mixed corpus the two-stage INNER JOIN
# silently truncates. See VectorStore._prescreen_revision / FIX 1.
_BINARY_PRESCREEN_REVISION_TAG = "binprescreen"
_DEFAULT_TASK = "summary"
_CHUNK_TASK = "chunk"
# Tasks this store can encode. Summary and chunk corpora coexist in the shared
# profile table, each keyed by its own canonical identity (task is part of the
# identity hash), so their vectors never collide.
_SUPPORTED_TASKS = frozenset({_DEFAULT_TASK, _CHUNK_TASK})

# Candidate-enumeration sentinel: SQLite reads a negative LIMIT as "no limit",
# so a full-corpus scan enumerates every live candidate for the identity.
_SCAN_ALL_ROWS = -1

# SQLite caps host parameters per statement (SQLITE_MAX_VARIABLE_NUMBER); a
# single ``WHERE id IN (?, ?, ...)`` over tens of thousands of ids overflows it
# (observed failure at ~33k ids). Candidate id resolution loads ids into a temp
# table in bounded chunks and JOINs instead, so it scales past that limit.
_ID_INSERT_CHUNK = 500
# Full scans below this measured crossover retain the pre-R1 temp-table loader.
# The R1 streaming cursor wins above the crossover while avoiding the small-store
# setup regression found by F39. See the R2-prime benchmark manifest.
_FAST_SCAN_STREAMING_MIN_ROWS = 2_500
_SOURCE_LINEAGE_WORK_LIMIT = 4096

# Full-corpus resident matrices outlive the per-query VectorStore instances that
# use them. The key is (resolved db_path, filesystem device, filesystem inode,
# data_version, identity_hash, budget_mb). The filesystem incarnation prevents a
# database replaced at the same path from inheriting the old file's resident
# rows. Each configured budget value owns an independent LRU partition;
# mixed-budget stores therefore never evict one another using whichever ceiling
# queried last.
_RESIDENT_MATRIX_REGISTRY_LOCK = threading.RLock()
_RESIDENT_MATRIX_REGISTRY: "OrderedDict[tuple[str, int, int, int, str, int], tuple[list[int], list[str], list[str], Any, Any]]" = OrderedDict()
_RESIDENT_MATRIX_INSTANCES: "weakref.WeakSet[VectorStore]" = weakref.WeakSet()
_RESIDENT_MATRIX_BUILD_LOCKS: "weakref.WeakValueDictionary[tuple[str, int, int, int, str, int], threading.Lock]" = weakref.WeakValueDictionary()


def _drop_resident_registry_entry_locked(
    key: tuple[str, int, int, int, str, int],
) -> tuple[list[int], list[str], list[str], Any, Any] | None:
    """Drop one pooled entry and every instance-local reference to it."""
    entry = _RESIDENT_MATRIX_REGISTRY.pop(key, None)
    for store in list(_RESIDENT_MATRIX_INSTANCES):
        if (
            getattr(store, "_resident_registry_db_path", None) != key[0]
            or getattr(store, "_resident_registry_incarnation", None) != key[1:3]
            or getattr(store, "knn_resident_max_mb", None) != key[5]
        ):
            continue
        local_cache = getattr(store, "_resident_matrix_cache", {})
        for local_key in list(local_cache):
            if local_key[1:] == (key[4], key[3]):
                local_cache.pop(local_key, None)
    return entry


def _drop_resident_registry_db_locked(
    db_path: str, incarnation: tuple[int, int]
) -> None:
    """Invalidate pooled residency and local references for one database."""
    for key in list(_RESIDENT_MATRIX_REGISTRY):
        if key[0] == db_path and key[1:3] == incarnation:
            _drop_resident_registry_entry_locked(key)
    for store in list(_RESIDENT_MATRIX_INSTANCES):
        if (
            getattr(store, "_resident_registry_db_path", None) == db_path
            and getattr(store, "_resident_registry_incarnation", None)
            == incarnation
        ):
            store._resident_matrix_cache.clear()
            store._resident_ineligible_cache.clear()


class _UnverifiableProvenance(RuntimeError):
    """A requested provenance filter could not be checked completely."""


class _PrescreenDeadlineExpired(TimeoutError):
    def __init__(self, scanned: int = 0) -> None:
        super().__init__("binary prescreen deadline expired")
        self.scanned = max(0, int(scanned))


def _require_supported_identity(identity: "EmbeddingIdentity") -> None:
    """Reject profile representations this implementation cannot encode."""
    unsupported = []
    if identity.dtype not in _SUPPORTED_DTYPES:
        unsupported.append(f"dtype={identity.dtype!r}")
    if identity.byteorder != _VECTOR_BYTEORDER:
        unsupported.append(f"byteorder={identity.byteorder!r}")
    if identity.task not in _SUPPORTED_TASKS:
        unsupported.append(f"task={identity.task!r}")
    if unsupported:
        raise ValueError(
            "unsupported embedding representation: "
            + ", ".join(unsupported)
            + "; supported representation is float32|int8/little/summary|chunk"
        )


@dataclass(frozen=True)
class EmbeddingIdentity:
    """Immutable canonical identity captured before provider execution."""

    provider: str
    model_name: str
    revision: str
    dim: int
    dtype: str
    byteorder: str
    task: str

    @classmethod
    def canonical(
        cls,
        provider: str,
        model_name: str,
        revision: str,
        dim: int,
        dtype: str,
        byteorder: str,
        task: str,
    ) -> "EmbeddingIdentity":
        return cls(
            provider=str(provider).strip().lower(),
            model_name=str(model_name).strip(),
            revision=str(revision).strip(),
            dim=int(dim),
            dtype=str(dtype).strip().lower(),
            byteorder=str(byteorder).strip().lower(),
            task=str(task).strip().lower(),
        )

    @property
    def identity_hash(self) -> str:
        canonical = "\x1f".join(
            [
                self.provider,
                self.model_name,
                self.revision,
                str(self.dim),
                self.dtype,
                self.byteorder,
                self.task,
            ]
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _identity_hash(
    provider: str,
    model_name: str,
    revision: str,
    dim: int,
    dtype: str,
    byteorder: str,
    task: str,
) -> str:
    """Return the stable canonical-identity hash for an embedding profile."""
    identity = EmbeddingIdentity.canonical(
        provider, model_name, revision, dim, dtype, byteorder, task
    )
    return identity.identity_hash


def _load_numpy():
    """Import NumPy lazily. Kept isolated so fallback behavior is testable."""
    import numpy

    return numpy


def _encode_int8_vector(normalized: Sequence[float]) -> bytes:
    """Symmetrically quantize a normalized vector to int8 + a float32 scale.

    Layout is ``dim`` signed bytes followed by a little-endian float32 scale, so
    dequantization is ``value_i = q_i * scale``. The scale is the per-vector max
    magnitude divided by 127; a degenerate all-zero vector (never produced by the
    non-zero-magnitude normalizer, but handled defensively) uses scale 0.
    """
    dim = len(normalized)
    amax = max((abs(float(value)) for value in normalized), default=0.0)
    scale = amax / _INT8_MAX if amax > 0.0 else 0.0
    if scale > 0.0:
        quantized = [
            max(-_INT8_MAX, min(_INT8_MAX, int(round(float(value) / scale))))
            for value in normalized
        ]
    else:
        quantized = [0] * dim
    return struct.pack(f"<{dim}b", *quantized) + struct.pack("<f", scale)


def _decode_int8_vector(blob: bytes, dim: int) -> tuple[float, ...] | None:
    """Dequantize an int8 ``vec`` BLOB (``dim`` bytes + float32 scale) to floats."""
    if len(blob) != dim + 4:
        return None
    try:
        quantized = struct.unpack(f"<{dim}b", blob[:dim])
        (scale,) = struct.unpack("<f", blob[dim:dim + 4])
    except struct.error:
        return None
    return tuple(value * scale for value in quantized)


def _pack_sign_bits(normalized: Sequence[float]) -> bytes:
    """Pack the sign bits of a vector MSB-first, ``ceil(dim/8)`` bytes.

    Bit ``i`` is 1 when ``value_i >= 0``. Matches ``numpy.packbits`` (MSB-first),
    so the query side can XOR-and-popcount against these bytes for a Hamming
    prescreen without re-deriving the bit order.
    """
    out = bytearray((len(normalized) + 7) // 8)
    for index, value in enumerate(normalized):
        if float(value) >= 0.0:
            out[index >> 3] |= 0x80 >> (index & 7)
    return bytes(out)


class KNNResult(list[tuple[str, float, str]]):
    """Ranked KNN rows with reach and score-representation metadata.

    ``coverage`` describes candidate reach, not numerical precision: ``full``
    means every requested candidate was scored. ``scoring`` separately
    discloses whether scores came from exact float32 vectors or int8-quantized
    vectors (including float32 corpora quantized for resident scoring).
    """

    def __init__(
        self,
        rows: Sequence[tuple[str, float, str]] = (),
        *,
        coverage: str,
        scoring: str | None = None,
        reason: str | None = None,
        scanned: int | None = None,
        total: int | None = None,
    ) -> None:
        super().__init__(rows)
        self.coverage = coverage
        self.scoring = scoring
        self.reason = reason
        # Coverage provenance: how many of the corpus's live vectors were
        # actually scored (``scanned``) out of the total live for the identity
        # (``total``), so a caller can surface partial-archive coverage (SCAN-1)
        # or certify a deadline-adjacent scan that nevertheless completed.
        self.scanned = scanned
        self.total = total


class EmbeddingPublishOutcome(str, Enum):
    """Typed result for the post-provider publication CAS."""

    PUBLISHED = "published"
    OWNERSHIP_LOST = "ownership_lost"
    IDENTITY_SUPERSEDED = "identity_superseded"


@dataclass(frozen=True)
class BatchPublishResult:
    """Per-row result from a batched publication.

    ``outcome`` carries the same CAS verdict a single-row publish would return;
    ``error`` is set (and ``outcome`` is ``None``) when that row's publish raised
    — its savepoint was rolled back, so the row stayed in-flight while its
    batch siblings that already succeeded remain committed. The caller reproduces
    exactly the accounting it applied on the single-row path.
    """

    embedded_id: str
    outcome: EmbeddingPublishOutcome | None
    error: Exception | None = None


class VectorStore:
    """SQLite-backed store for normalized summary embedding vectors."""

    # Opt-in marker read by the retrieval-core pool: only the genuine store is
    # long-lived/poolable (test fakes injected as ``vector_store_cls`` are not).
    _supports_pooling = True
    # Per-store matrix-cache ceiling. A bounded LRU (not clear-on-every-miss) so
    # back-to-back recalls over an unchanged corpus reuse the loaded candidate
    # matrix across BOTH arms and across calls once the store is pooled, without
    # letting distinct candidate sets grow the cache without bound (sprint-opt-6).
    _MATRIX_CACHE_MAX_ENTRIES = 4

    def __init__(
        self,
        db_path: str | Path,
        *,
        config: LCMConfig | None = None,
        bounded_scan_rows: int | None = None,
    ) -> None:
        raw_db_path = str(db_path)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._resident_registry_is_memory = raw_db_path == ":memory:"
        if self._resident_registry_is_memory:
            # Each ``:memory:`` connection owns an independent database. A
            # resolved pseudo-path would pool unrelated corpora, and an object
            # id can be reused after close while pooled entries intentionally
            # survive, so use a durable per-connection namespace token.
            self._resident_registry_db_path = f":memory:#{uuid.uuid4().hex}"
        else:
            self._resident_registry_db_path = str(self.db_path.resolve())
        resolved_config = config or LCMConfig.from_env()
        self.bounded_scan_rows = (
            resolved_config.embedding_bounded_scan_rows
            if bounded_scan_rows is None
            else int(bounded_scan_rows)
        )
        # Stage-1 prescreen breadth for the two-stage (binary Hamming -> int8
        # rescore) KNN: M = max(1, multiplier) x k survivors are rescored. Read
        # once at construction so a pooled store honors the configured value.
        self.knn_prescreen_multiplier = max(
            1, int(getattr(resolved_config, "knn_prescreen_multiplier", 4) or 4)
        )
        self.knn_resident_max_mb = max(
            0, int(getattr(resolved_config, "knn_resident_max_mb", 128) or 0)
        )
        self.knn_resident_max_bytes = self.knn_resident_max_mb * 1024 * 1024
        # Opt-in: also write the sign-bit prescreen for float32 identities (not
        # just int8). A float32-vec identity carrying the prescreen gets the
        # full-corpus two-stage path with EXACT float rescore of survivors — the
        # high-recall, low-RAM config. Default-off keeps stock float32 identities
        # byte-identical AND binary-free (legacy bounded path). Operators pick a
        # distinct identity (e.g. via revision) so prescreen rows never mix into a
        # legacy float32 identity.
        self._write_binary_prescreen = bool(
            getattr(resolved_config, "embedding_binary_prescreen", False)
        )
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        # Nesting depth for _write_transaction. The outermost entry owns the
        # BEGIN IMMEDIATE/COMMIT (one fsync); a re-entrant inner entry (e.g. a
        # single-row publish invoked from publish_*_batch_under_lease) rides the
        # open transaction under a SAVEPOINT so its failure rolls back only its
        # own row, never the sibling rows already written in the same batch.
        self._txn_depth = 0
        # Cache key: (identity_hash, data_version, candidate_ids). The durable
        # per-identity ``data_version`` counter is bumped inside every vector
        # write/delete transaction, so a cross-process write invalidates this
        # cache even when max_rowid and row_count are unchanged.
        self._matrix_cache: "OrderedDict[tuple[str, int, tuple[str, ...]], tuple[list[int], list[str], list[str], Any]]" = OrderedDict()
        # Separate cache for chunk-corpus matrices: chunk identities never
        # collide with summary identities (task is part of the hash), but the
        # loaders differ (chunk vectors join messages, summary vectors join
        # summary_nodes), so the two corpora keep independent matrix caches.
        self._chunk_matrix_cache: "OrderedDict[tuple[str, int, tuple[str, ...]], tuple[list[int], list[str], list[str], Any]]" = OrderedDict()
        # Full-corpus binary prescreen matrices for the two-stage path, keyed by
        # (identity_hash, data_version). Only the UNFILTERED full corpus is cached
        # (filtered loads vary per call); this lets a pooled store's back-to-back
        # recalls skip re-loading the whole sign-bit corpus (the dominant
        # two-stage cost) — the same amortization the float matrix cache gives.
        self._binary_matrix_cache: "OrderedDict[tuple[str, int], tuple[list[str], Any]]" = OrderedDict()
        self._chunk_binary_matrix_cache: "OrderedDict[tuple[str, int], tuple[list[str], Any]]" = OrderedDict()
        # Instance-local references into the process-wide resident registry.
        # Keys retain their pre-R2-double-prime shape for diagnostics/tests:
        # (is_chunk, identity_hash, data_version).
        self._resident_matrix_cache: "OrderedDict[tuple[bool, str, int], tuple[list[int], list[str], list[str], Any, Any]]" = OrderedDict()
        self._resident_ineligible_cache: set[tuple[bool, str, int]] = set()
        self._chunk_schema_ready = False
        self._init_db()
        if self._resident_registry_is_memory:
            self._resident_registry_incarnation = (0, 0)
        else:
            db_stat = self.db_path.stat()
            self._resident_registry_incarnation = (
                int(db_stat.st_dev),
                int(db_stat.st_ino),
            )
        with _RESIDENT_MATRIX_REGISTRY_LOCK:
            _RESIDENT_MATRIX_INSTANCES.add(self)

    def _init_db(self) -> None:
        # isolation_level=None (autocommit) so every read runs as its own
        # short read transaction rather than pinning a WAL snapshot for the
        # life of the connection. Without this, a long-lived VectorStore never
        # observes another process's committed vector writes (or the bumped
        # data_version), defeating the cross-process cache invalidation. Write
        # atomicity is preserved via the explicit BEGIN IMMEDIATE in
        # _write_transaction.
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
            isolation_level=None,
        )
        refuse_schema_version_too_new(self._conn)
        configure_connection(self._conn)
        self._conn.row_factory = sqlite3.Row
        run_versioned_migrations(self._conn)
        self._ensure_embedding_schema()
        self._conn.commit()

    def _ensure_embedding_schema(self) -> None:
        """Materialize (and verify) the opt-in embedding tables on VectorStore use.

        Constructing a VectorStore means embeddings are in use; the core
        migration path leaves these tables uncreated so a disabled install
        stays at schema_version 5 with none of them. Creation is idempotent
        (CREATE TABLE/INDEX IF NOT EXISTS).

        The ``embeddings_v1`` marker is NOT trusted on its own: it can be set
        while a table or index is absent (e.g. one was dropped after the marker
        was written). So the required tables/indexes are re-ensured and then
        VERIFIED every init — a set marker over a missing table is repaired
        rather than believed. Only a fully-materialized schema keeps the marker.
        """
        with self._write_lock:
            ensure_embedding_tables(self._conn)
            errors = verify_embedding_schema(self._conn)
            if errors:
                # Should be unreachable: the ensure above is a superset of the
                # verified set. Re-ensure once more and fail loudly if a
                # required object still cannot be created, rather than marking a
                # broken schema complete.
                ensure_embedding_tables(self._conn)
                errors = verify_embedding_schema(self._conn)
                if errors:
                    raise sqlite3.OperationalError(
                        "embedding schema incompatible after ensure: "
                        + "; ".join(errors)
                    )
            # Skip the marker write + commit when it is already stamped and the
            # schema verified clean -- re-writing it every construction is an
            # otherwise-needless write-transaction contending with real writers
            # (sprint-opt-1). CREATE-IF-NOT-EXISTS and verify above are read-only
            # no-ops on an already-materialized schema.
            if not self._migration_step_present("embeddings_v1"):
                mark_migration_step_complete(self._conn, "embeddings_v1")
                self._conn.commit()

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        # The connection is in autocommit mode (isolation_level=None), so a
        # multi-statement write must open its own explicit transaction to stay
        # atomic — the data_version bump has to commit with the vector write it
        # accompanies, never separately.
        #
        # Re-entrant: the outermost caller owns the BEGIN IMMEDIATE + COMMIT (one
        # fsync); a nested caller opens a SAVEPOINT instead so a batch of per-row
        # publications shares a single transaction. A nested failure rolls back
        # to its own savepoint and re-raises, leaving the earlier rows of the
        # batch intact for the outer COMMIT.
        try:
            with self._write_lock, self._cache_lock:
                if self._txn_depth > 0:
                    savepoint = f"lcm_pub_sp_{self._txn_depth}"
                    self._conn.execute(f"SAVEPOINT {savepoint}")
                    self._txn_depth += 1
                    try:
                        yield
                        self._conn.execute(f"RELEASE {savepoint}")
                    except BaseException:
                        self._conn.execute(f"ROLLBACK TO {savepoint}")
                        self._conn.execute(f"RELEASE {savepoint}")
                        raise
                    finally:
                        self._txn_depth -= 1
                    return
                self._conn.execute("BEGIN IMMEDIATE")
                self._txn_depth = 1
                try:
                    yield
                    self._conn.commit()
                except BaseException:
                    self._conn.rollback()
                    raise
                finally:
                    self._txn_depth = 0
                    self._matrix_cache.clear()
                    self._binary_matrix_cache.clear()
                    self._chunk_binary_matrix_cache.clear()
                    # Accepted #171 cost: every vector write drops residency;
                    # the next eligible full scan reloads the complete matrix.
                    with _RESIDENT_MATRIX_REGISTRY_LOCK:
                        _drop_resident_registry_db_locked(
                            self._resident_registry_db_path,
                            self._resident_registry_incarnation,
                        )
        except sqlite3.Error as exc:
            if _is_sqlite_locked_error(exc):
                logger.warning("Embedding write blocked by SQLite lock contention")
            raise

    @property
    def connection(self) -> sqlite3.Connection | None:
        """The live connection for read-only diagnostics, or ``None`` after close."""
        return getattr(self, "_conn", None)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalized(values: Sequence[float], *, expected_dim: int) -> list[float]:
        try:
            floats = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding vector must contain only numeric values") from exc
        if len(floats) != expected_dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {expected_dim}, got {len(floats)}"
            )
        if not all(math.isfinite(value) for value in floats):
            raise ValueError("embedding vector values must be finite")
        magnitude = math.sqrt(sum(value * value for value in floats))
        if magnitude == 0.0:
            raise ValueError("embedding vector must have non-zero magnitude")
        return [value / magnitude for value in floats]

    def _profile_by_identity(self, identity_hash: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT identity_hash, model_name, provider, revision, dim, dtype,
                   byteorder, task, registered_at, active, archived_at, data_version
            FROM lcm_embedding_profile
            WHERE identity_hash = ?
            """,
            (str(identity_hash),),
        ).fetchone()

    def _resolve_profile(
        self, model_name: str, provider: str | None = None
    ) -> sqlite3.Row | None:
        """Resolve a model name to its profile, preferring the current/active one.

        Multiple identities can share a ``model_name`` (e.g. the same model
        served by two providers). When ``provider`` is given the resolution is
        constrained to that provider too, so the read path selects vectors that
        match the *configured* provider identity rather than whichever provider
        happens to be active for the bare model name — switching provider A→B
        for the same model no longer scores a B-embedded query against A's
        vectors. With no ``provider`` the bare-name active identity is used
        (recording targets the operator's current selection).
        """
        # Task-scoped: summary and chunk profiles coexist for the same
        # (model, provider) in the shared table. Without the task filter this
        # lookup returned whichever row registered most recently — registering
        # the chunk profile silently redirected the SUMMARY knn to the chunk
        # identity (zero summary meta rows → coverage 'none'). The chunk read
        # path has its own resolver (_resolve_chunk_profile).
        model_name = str(model_name).strip()
        if provider is not None:
            provider = str(provider).strip().lower()
            return self._conn.execute(
                """
                SELECT identity_hash, model_name, provider, revision, dim, dtype,
                       byteorder, task, registered_at, active, archived_at, data_version
                FROM lcm_embedding_profile
                WHERE model_name = ? AND provider = ? AND task = ?
                ORDER BY (active = 1 AND archived_at IS NULL) DESC,
                         registered_at DESC, identity_hash DESC
                LIMIT 1
                """,
                (model_name, provider, _DEFAULT_TASK),
            ).fetchone()
        return self._conn.execute(
            """
            SELECT identity_hash, model_name, provider, revision, dim, dtype,
                   byteorder, task, registered_at, active, archived_at, data_version
            FROM lcm_embedding_profile
            WHERE model_name = ? AND task = ?
            ORDER BY (active = 1 AND archived_at IS NULL) DESC,
                     registered_at DESC, identity_hash DESC
            LIMIT 1
            """,
            (model_name, _DEFAULT_TASK),
        ).fetchone()

    @staticmethod
    def _as_node_id(embedded_id: str) -> int | None:
        """Coerce an embedded id to the integer ``summary_nodes.node_id`` PK.

        ``node_id`` is an ``INTEGER PRIMARY KEY``; binding the integer lets the
        lookup use the PK/rowid index instead of a ``CAST(node_id AS TEXT)``
        full scan. A non-integer id can never match a node, so it maps to None.
        """
        try:
            return int(str(embedded_id))
        except (TypeError, ValueError):
            return None

    def _table_columns(self, table: str) -> set[str]:
        return {
            str(row[1])
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _current_profile(self) -> sqlite3.Row | None:
        # The current SUMMARY profile. Task-scoped so a coexisting active chunk
        # profile can never be mistaken for the summary corpus's current profile.
        return self._conn.execute(
            """
            SELECT identity_hash, model_name, provider, revision, dim, dtype,
                   byteorder, task, registered_at, active, archived_at, data_version
            FROM lcm_embedding_profile
            WHERE active = 1 AND archived_at IS NULL AND task = ?
            ORDER BY registered_at DESC, identity_hash DESC
            LIMIT 1
            """,
            (_DEFAULT_TASK,),
        ).fetchone()

    def _resolve_chunk_profile(
        self, model_name: str, *, provider: str | None = None
    ) -> sqlite3.Row | None:
        """Resolve a profile by model (+optional provider) within task='chunk'."""
        base = (
            "SELECT identity_hash, model_name, provider, revision, dim, dtype, "
            "byteorder, task, registered_at, active, archived_at, data_version "
            "FROM lcm_embedding_profile WHERE model_name = ? AND task = ?"
        )
        order = (
            " ORDER BY (active = 1 AND archived_at IS NULL) DESC, "
            "registered_at DESC, identity_hash DESC LIMIT 1"
        )
        if provider is not None:
            return self._conn.execute(
                base + " AND provider = ?" + order,
                (str(model_name), _CHUNK_TASK, str(provider).strip().lower()),
            ).fetchone()
        return self._conn.execute(
            base + order, (str(model_name), _CHUNK_TASK)
        ).fetchone()

    @staticmethod
    def _identity_from_profile(profile: sqlite3.Row) -> EmbeddingIdentity:
        identity = EmbeddingIdentity.canonical(
            profile["provider"],
            profile["model_name"],
            profile["revision"],
            profile["dim"],
            profile["dtype"],
            profile["byteorder"],
            profile["task"],
        )
        if identity.identity_hash != str(profile["identity_hash"]):
            raise ValueError("embedding profile identity hash does not match its fields")
        _require_supported_identity(identity)
        return identity

    def capture_identity(
        self, model_name: str | None = None, *, provider: str | None = None
    ) -> EmbeddingIdentity:
        """Capture the full immutable identity before provider execution."""
        profile = (
            self._resolve_profile(model_name, provider=provider)
            if model_name is not None
            else self._current_profile()
        )
        if profile is None:
            target = model_name if model_name is not None else "active"
            raise ValueError(f"embedding profile is not registered: {target}")
        return self._identity_from_profile(profile)

    def _prescreen_revision(self, revision: str, *, dtype: str) -> str:
        """Fold the float32 sign-bit prescreen opt-in into the identity via revision.

        ``LCM_EMBEDDING_BINARY_PRESCREEN`` changes a float32 identity's stored
        corpus shape (it grows a companion sign-bit table that drives the
        two-stage KNN), so a prescreen corpus MUST be a distinct identity from a
        legacy binary-free one. Otherwise flipping the flag on a populated
        float32 identity leaves a mixed corpus (a few post-flip rows with
        sign-bits, the rest without) that the two-stage INNER JOIN silently
        truncates while reporting coverage='full' (FIX 1).

        Only the default (empty) revision is auto-tagged: an operator who set an
        explicit revision has already taken ownership of identity separation (the
        documented manual escape hatch), so their value is respected verbatim.
        int8 identities always carry the prescreen and are already distinguished
        by dtype, so the tag is float32-only — int8 identity hashes stay stable.
        """
        base = str(revision).strip()
        if base or str(dtype).strip().lower() == _INT8_DTYPE:
            return base
        return _BINARY_PRESCREEN_REVISION_TAG if self._write_binary_prescreen else base

    def register_profile(
        self,
        model_name: str,
        provider: str,
        dim: int,
        *,
        revision: str = "",
        dtype: str = _VECTOR_DTYPE,
        byteorder: str = _VECTOR_BYTEORDER,
        task: str = _DEFAULT_TASK,
    ) -> str:
        """Register (or reactivate) the profile for a canonical identity.

        The identity is ``(provider, model_name, revision, dim, dtype,
        byteorder, task)``. Registering a *different* identity for the same
        model_name creates a new profile row rather than clobbering the old
        provider's metadata; re-registering an already-known identity
        reactivates it (its vectors remain valid, so switching config back
        needs no re-backfill). Exactly one profile is left active. Returns the
        identity hash.
        """
        revision = self._prescreen_revision(revision, dtype=dtype)
        canonical = EmbeddingIdentity.canonical(
            provider, model_name, revision, dim, dtype, byteorder, task
        )
        if not canonical.model_name:
            raise ValueError("model_name must not be empty")
        if not canonical.provider:
            raise ValueError("provider must not be empty")
        if canonical.dim < 1 or canonical.dim > 4096:
            raise ValueError("embedding dimension must be between 1 and 4096")
        _require_supported_identity(canonical)

        identity = canonical.identity_hash
        with self._write_transaction():
            active = self._conn.execute(
                """
                SELECT identity_hash
                FROM lcm_embedding_profile
                WHERE active = 1 AND archived_at IS NULL AND task = ?
                ORDER BY registered_at DESC, identity_hash DESC
                LIMIT 1
                """,
                (canonical.task,),
            ).fetchone()
            profile_swapped = (
                active is not None and str(active["identity_hash"]) != identity
            )
            existing = self._profile_by_identity(identity)
            if existing is None:
                self._conn.execute(
                    """
                    INSERT INTO lcm_embedding_profile(
                        identity_hash, provider, model_name, revision, dim,
                        dtype, byteorder, task, registered_at, active,
                        archived_at, data_version
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, 0)
                    """,
                    (
                        identity,
                        canonical.provider,
                        canonical.model_name,
                        canonical.revision,
                        canonical.dim,
                        canonical.dtype,
                        canonical.byteorder,
                        canonical.task,
                        self._now(),
                    ),
                )
            else:
                # Reactivate a previously-registered identity in place; its
                # vectors are still valid so no re-backfill is required.
                self._conn.execute(
                    """
                    UPDATE lcm_embedding_profile
                    SET active = 1, archived_at = NULL
                    WHERE identity_hash = ?
                    """,
                    (identity,),
                )
            # Selection of the "current" profile is by identity WITHIN A TASK:
            # only the just-registered identity stays active for its task; other
            # profiles of the SAME task are deactivated (retained so their
            # vectors survive a later switch back). Profiles of a DIFFERENT task
            # are untouched, so the summary and chunk corpora coexist — each has
            # its own active profile.
            self._conn.execute(
                "UPDATE lcm_embedding_profile SET active = 0 "
                "WHERE identity_hash != ? AND task = ?",
                (identity, canonical.task),
            )
            if profile_swapped:
                self._conn.execute(
                    "UPDATE lcm_embedding_profile "
                    "SET data_version = data_version + 1 WHERE task = ?",
                    (canonical.task,),
                )
        return identity

    def record_embedding(
        self,
        embedded_id: str,
        kind: str,
        model: str,
        vec: Sequence[float],
        *,
        identity: EmbeddingIdentity,
    ) -> None:
        with self._write_transaction():
            profile = self._validate_embedding_write(identity, model=model)
            self._write_embedding_row(
                str(embedded_id), kind, vec, identity=identity, profile=profile
            )

    def publish_embedding_under_lease(
        self,
        embedded_id: str,
        kind: str,
        model: str,
        vec: Sequence[float],
        *,
        identity: EmbeddingIdentity,
        claim_key: str,
        lease_id: str,
        generation: int,
        request_id: str,
    ) -> EmbeddingPublishOutcome:
        """Atomically revalidate ownership and publish one accepted summary vector."""
        embedded_id = str(embedded_id)
        return self._publish_under_lease(
            embedded_id,
            model=model,
            identity=identity,
            claim_key=claim_key,
            lease_id=lease_id,
            generation=generation,
            request_id=request_id,
            write_row=lambda profile: self._write_embedding_row(
                embedded_id, kind, vec, identity=identity, profile=profile
            ),
        )

    def publish_embedding_batch_under_lease(
        self,
        rows: Sequence[tuple[str, str, str, Sequence[float]]],
        *,
        identity: EmbeddingIdentity,
        claim_key: str,
        lease_id: str,
        generation: int,
    ) -> list[BatchPublishResult]:
        """Publish an accepted network batch of summary vectors in ONE transaction.

        ``rows`` is a sequence of ``(embedded_id, kind, model, vec)`` in provider
        order. All rows share one ``BEGIN IMMEDIATE`` (a single fsync per network
        batch instead of one per row), yet each still runs the full per-row
        ownership/identity CAS under its own savepoint via the single-row
        ``publish_embedding_under_lease``. Publication stops at the first row that
        is not ``PUBLISHED`` (or that raises), mirroring the single-row loop it
        replaces; the rows already published in that transaction are kept.
        """
        return self._run_publish_batch(
            [
                (
                    str(embedded_id),
                    lambda embedded_id=embedded_id, kind=kind, model=model, vec=vec, request_id=request_id: self.publish_embedding_under_lease(
                        embedded_id,
                        kind,
                        model,
                        vec,
                        identity=identity,
                        claim_key=claim_key,
                        lease_id=lease_id,
                        generation=generation,
                        request_id=request_id,
                    ),
                )
                for embedded_id, kind, model, vec, request_id in rows
            ]
        )

    def _run_publish_batch(
        self, publishers: Sequence[tuple[str, "Any"]]
    ) -> list[BatchPublishResult]:
        """Run per-row publishers inside one write transaction.

        Each entry is ``(embedded_id, publish)`` where ``publish`` is a zero-arg
        call into the corpus-specific single-row publish method (kept as the
        per-row worker so its behaviour — and any test monkeypatch of it — is
        exercised unchanged, now under a savepoint). A raised publisher rolls its
        own savepoint back and is reported as an errored row; the loop then stops
        so the caller can quarantine the request exactly as the single-row path
        did, while the committed siblings survive.
        """
        results: list[BatchPublishResult] = []
        with self._write_transaction():
            for embedded_id, publish in publishers:
                try:
                    outcome = publish()
                except Exception as exc:  # noqa: BLE001 — reported to the caller
                    results.append(
                        BatchPublishResult(embedded_id, outcome=None, error=exc)
                    )
                    break
                results.append(BatchPublishResult(embedded_id, outcome=outcome))
                if outcome is not EmbeddingPublishOutcome.PUBLISHED:
                    break
        return results

    def _publish_under_lease(
        self,
        embedded_id: str,
        *,
        model: str,
        identity: EmbeddingIdentity,
        claim_key: str,
        lease_id: str,
        generation: int,
        request_id: str,
        write_row,
    ) -> EmbeddingPublishOutcome:
        """Corpus-agnostic lease publication CAS shared by summary and chunk.

        The metadata owner/generation check, active captured identity check,
        matching dispatched-row check, corpus-specific vector/meta write (via
        ``write_row``), data-version bump, and in-flight clear all share one
        ``BEGIN IMMEDIATE`` transaction. A superseded worker therefore performs
        none of these mutations. Summary and chunk publications share the one
        ``lcm_embedding_backfill_inflight`` table because their embedded_ids are
        disjoint (numeric node ids vs ``store_id:chunk_index``) and their
        identity hashes differ (task is part of the hash).
        """
        embedded_id = str(embedded_id)
        with self._write_transaction():
            owner = self._conn.execute(
                """
                SELECT 1
                FROM metadata
                WHERE key = ?
                  AND json_extract(value, '$.owner') = ?
                  AND CAST(json_extract(value, '$.generation') AS INTEGER) = ?
                """,
                (claim_key, str(lease_id), int(generation)),
            ).fetchone()
            inflight = self._conn.execute(
                """
                SELECT 1
                FROM lcm_embedding_backfill_inflight
                WHERE embedded_id = ? AND identity_hash = ?
                  AND lease_id = ? AND generation = ?
                  AND request_id = ? AND state = 'dispatched'
                """,
                (
                    embedded_id,
                    identity.identity_hash,
                    str(lease_id),
                    int(generation),
                    str(request_id),
                ),
            ).fetchone()
            if owner is None or inflight is None:
                return EmbeddingPublishOutcome.OWNERSHIP_LOST
            profile = self._validate_embedding_write(identity, model=model)
            if int(profile["active"] or 0) != 1 or profile["archived_at"] is not None:
                transitioned = self._conn.execute(
                    """
                    UPDATE lcm_embedding_backfill_inflight
                    SET state = 'uncertain', updated_at = ?,
                        last_error = 'embedding identity superseded after remote acceptance'
                    WHERE identity_hash = ? AND lease_id = ? AND generation = ?
                      AND request_id = ? AND state = 'dispatched'
                    """,
                    (
                        datetime.now(timezone.utc).timestamp(),
                        identity.identity_hash,
                        str(lease_id),
                        int(generation),
                        str(request_id),
                    ),
                )
                if int(transitioned.rowcount or 0) < 1:
                    return EmbeddingPublishOutcome.OWNERSHIP_LOST
                released = self._conn.execute(
                    """
                    DELETE FROM metadata
                    WHERE key = ?
                      AND json_extract(value, '$.owner') = ?
                      AND CAST(json_extract(value, '$.generation') AS INTEGER) = ?
                    """,
                    (claim_key, str(lease_id), int(generation)),
                )
                if int(released.rowcount or 0) != 1:
                    raise sqlite3.OperationalError(
                        "identity supersession lost lease release ownership"
                    )
                return EmbeddingPublishOutcome.IDENTITY_SUPERSEDED
            write_row(profile)
            cleared = self._conn.execute(
                """
                DELETE FROM lcm_embedding_backfill_inflight
                WHERE embedded_id = ? AND identity_hash = ?
                  AND lease_id = ? AND generation = ?
                  AND request_id = ? AND state = 'dispatched'
                """,
                (
                    embedded_id,
                    identity.identity_hash,
                    str(lease_id),
                    int(generation),
                    str(request_id),
                ),
            )
            if cleared.rowcount != 1:
                raise sqlite3.OperationalError(
                    "embedding publication lost its in-flight ownership"
                )
            return EmbeddingPublishOutcome.PUBLISHED

    def _validate_embedding_write(
        self, identity: EmbeddingIdentity, *, model: str
    ) -> sqlite3.Row:
        model = str(model)
        if not isinstance(identity, EmbeddingIdentity):
            raise TypeError("identity must be an EmbeddingIdentity captured before provider work")
        canonical_identity = EmbeddingIdentity.canonical(
            identity.provider,
            identity.model_name,
            identity.revision,
            identity.dim,
            identity.dtype,
            identity.byteorder,
            identity.task,
        )
        if identity != canonical_identity:
            raise ValueError("embedding identity fields are not canonical")
        _require_supported_identity(identity)
        expected_identity = identity.identity_hash
        profile = self._profile_by_identity(expected_identity)
        if profile is None:
            raise ValueError(f"embedding profile is not registered: {expected_identity}")
        registered_identity = self._identity_from_profile(profile)
        if registered_identity != identity:
            raise ValueError(
                "captured embedding identity does not match registered profile"
            )
        if str(profile["model_name"]) != model.strip():
            raise ValueError(
                "embedding model does not match captured identity: "
                f"{model} != {profile['model_name']}"
            )
        return profile

    def _encode_stored_vector(
        self, vec: Sequence[float], profile: sqlite3.Row
    ) -> tuple[list[float], bytes, bytes | None]:
        """Normalize (and, for a Matryoshka profile, truncate) then encode a vector.

        Returns ``(normalized, vec_blob, sign_bits)``. For a float32 profile the
        blob is the existing little-endian float32 wire format; ``sign_bits`` is
        None unless the store opts into a float32 prescreen
        (``LCM_EMBEDDING_BINARY_PRESCREEN``), which yields float32-vec + binary =
        the two-stage path with EXACT float rescore. For an int8 profile the blob
        is the quantized ``dim`` bytes + float32 scale and ``sign_bits`` is always
        the packed prescreen row. When the incoming vector is longer than the
        profile dim (Matryoshka ``LCM_EMBEDDING_STORE_DIM``), it is truncated to
        the leading ``dim`` components and renormalized before encoding.
        """
        dim = int(profile["dim"])
        source = list(vec)
        if len(source) > dim:
            source = source[:dim]
        normalized = VectorStore._normalized(source, expected_dim=dim)
        if str(profile["dtype"]) == _INT8_DTYPE:
            return (
                normalized,
                _encode_int8_vector(normalized),
                _pack_sign_bits(normalized),
            )
        # Persist an explicit little-endian float32 wire format. Native
        # ``array('f')`` bytes would be mislabeled on a big-endian host. The
        # sign-bit prescreen is written only when explicitly opted in.
        sign_bits = _pack_sign_bits(normalized) if self._write_binary_prescreen else None
        return normalized, struct.pack(f"<{len(normalized)}f", *normalized), sign_bits

    def _write_embedding_row(
        self,
        embedded_id: str,
        kind: str,
        vec: Sequence[float],
        *,
        identity: EmbeddingIdentity,
        profile: sqlite3.Row,
    ) -> None:
        if kind != "summary":
            raise ValueError("embedded kind must be 'summary'")
        identity_hash = identity.identity_hash
        normalized, packed, sign_bits = self._encode_stored_vector(vec, profile)
        numeric_id = self._as_node_id(embedded_id)
        summary = (
            self._conn.execute(
                """
                SELECT source_token_count, access_scope
                FROM summary_nodes
                WHERE node_id = ?
                """,
                (numeric_id,),
            ).fetchone()
            if numeric_id is not None
            else None
        )
        if summary is None:
            raise ValueError(f"summary node does not exist: {embedded_id}")
        access_scope = summary["access_scope"]
        embedded_at = self._now()
        self._conn.execute(
            "DELETE FROM lcm_embedding_vectors "
            "WHERE embedded_id = ? AND identity_hash = ?",
            (embedded_id, identity_hash),
        )
        self._conn.execute(
            "DELETE FROM lcm_embedding_meta WHERE embedded_id = ? "
            "AND embedded_kind = ? AND identity_hash = ?",
            (embedded_id, kind, identity_hash),
        )
        self._conn.execute(
            "DELETE FROM lcm_embedding_binary WHERE embedded_id = ? "
            "AND identity_hash = ?",
            (embedded_id, identity_hash),
        )
        self._conn.execute(
            "INSERT INTO lcm_embedding_vectors(embedded_id, identity_hash, vec, access_scope) "
            "VALUES(?, ?, ?, ?)",
            (embedded_id, identity_hash, packed, access_scope),
        )
        if sign_bits is not None:
            self._conn.execute(
                "INSERT INTO lcm_embedding_binary(embedded_id, identity_hash, bits, access_scope) "
                "VALUES(?, ?, ?, ?)",
                (embedded_id, identity_hash, sign_bits, access_scope),
            )
        self._conn.execute(
            """
            INSERT INTO lcm_embedding_meta(
                embedded_id, embedded_kind, identity_hash, embedded_at,
                source_token_count, archived, access_scope
            ) VALUES(?, ?, ?, ?, ?, 0, ?)
            """,
            (
                embedded_id,
                kind,
                identity_hash,
                embedded_at,
                int(summary["source_token_count"] or 0),
                access_scope,
            ),
        )
        self._bump_data_version(identity_hash)

    def _bump_data_version(self, identity_hash: str) -> None:
        """Bump one identity's durable data-version counter.

        Must run inside the same write transaction as the vector write/delete
        it accompanies so a cross-process reader observes the new counter and
        invalidates its NumPy matrix cache.
        """
        self._conn.execute(
            "UPDATE lcm_embedding_profile SET data_version = data_version + 1 "
            "WHERE identity_hash = ?",
            (str(identity_hash),),
        )

    def _bump_all_data_versions(self) -> None:
        self._conn.execute(
            "UPDATE lcm_embedding_profile SET data_version = data_version + 1"
        )

    @contextmanager
    def _temp_id_table(self, ids: Sequence[str]) -> Iterator[str]:
        """Load ids into a unique per-call temp table in bounded chunks for JOINs.

        Avoids the ``WHERE id IN (?, ?, ...)`` host-parameter cap
        (SQLITE_MAX_VARIABLE_NUMBER) that failed at ~33k ids: ``executemany``
        binds one row at a time, so an arbitrary number of ids can be staged.

        The table name is unique per call (the store's connection is shared
        across threads, ``check_same_thread=False``); a single reused name meant
        two concurrent ``knn()`` calls scribbled over each other's candidate set.
        The table is dropped in ``finally`` so it never outlives the call.
        """
        table = f"_lcm_id_scratch_{uuid.uuid4().hex}"
        self._conn.execute(f"CREATE TEMP TABLE {table}(id TEXT PRIMARY KEY)")
        try:
            for offset in range(0, len(ids), _ID_INSERT_CHUNK):
                rows = [
                    (str(value),)
                    for value in ids[offset:offset + _ID_INSERT_CHUNK]
                ]
                self._conn.executemany(
                    f"INSERT OR IGNORE INTO {table}(id) VALUES(?)",
                    rows,
                )
            yield table
        finally:
            self._conn.execute(f"DROP TABLE IF EXISTS {table}")

    @contextmanager
    def _optional_temp_id_table(
        self, ids: Sequence[str] | None
    ) -> Iterator[str | None]:
        if ids is None:
            yield None
            return
        normalized = [str(value) for value in ids]
        with self._temp_id_table(normalized) as table:
            yield table

    def purge_embeddings_for_nodes(self, node_ids: Sequence[int | str]) -> int:
        unique_ids = list(dict.fromkeys(str(node_id) for node_id in node_ids))
        if not unique_ids:
            return 0
        with self._write_transaction():
            with self._temp_id_table(unique_ids) as table:
                cur = self._conn.execute(
                    f"DELETE FROM lcm_embedding_vectors "
                    f"WHERE embedded_id IN (SELECT id FROM {table})"
                )
                self._conn.execute(
                    f"DELETE FROM lcm_embedding_meta "
                    f"WHERE embedded_id IN (SELECT id FROM {table})"
                )
                if self._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='lcm_embedding_binary'"
                ).fetchone() is not None:
                    self._conn.execute(
                        f"DELETE FROM lcm_embedding_binary "
                        f"WHERE embedded_id IN (SELECT id FROM {table})"
                    )
                if self._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='lcm_embedding_backfill_inflight'"
                ).fetchone() is not None:
                    self._conn.execute(
                        f"DELETE FROM lcm_embedding_backfill_inflight "
                        f"WHERE embedded_id IN (SELECT id FROM {table})"
                    )
            # Purge spans every identity keyed by these ids; bump all data
            # versions so any cached matrix that referenced a removed vector is
            # invalidated. Over-invalidation across identities is safe.
            if cur.rowcount:
                self._bump_all_data_versions()
        return int(cur.rowcount or 0)

    @staticmethod
    def purge_embedding_batch_on_connection(
        conn: sqlite3.Connection, node_ids: Sequence[int | str]
    ) -> int:
        """Purge one already-bounded node batch in the caller's transaction."""
        unique_ids = list(dict.fromkeys(str(node_id) for node_id in node_ids))
        if not unique_ids:
            return 0
        if len(unique_ids) > 256:
            raise ValueError("embedding purge batch exceeds 256 rows")
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ("
                "'lcm_embedding_vectors','lcm_embedding_meta','lcm_embedding_binary',"
                "'lcm_embedding_profile','lcm_embedding_backfill_inflight')"
            ).fetchall()
        }
        if not {
            "lcm_embedding_vectors",
            "lcm_embedding_meta",
            "lcm_embedding_profile",
        } <= tables:
            return 0
        placeholders = ",".join("?" for _ in unique_ids)
        cur = conn.execute(
            f"DELETE FROM lcm_embedding_vectors "
            f"WHERE embedded_id IN ({placeholders})",
            unique_ids,
        )
        conn.execute(
            f"DELETE FROM lcm_embedding_meta "
            f"WHERE embedded_id IN ({placeholders})",
            unique_ids,
        )
        if "lcm_embedding_binary" in tables:
            conn.execute(
                f"DELETE FROM lcm_embedding_binary "
                f"WHERE embedded_id IN ({placeholders})",
                unique_ids,
            )
        if "lcm_embedding_backfill_inflight" in tables:
            conn.execute(
                f"DELETE FROM lcm_embedding_backfill_inflight "
                f"WHERE embedded_id IN ({placeholders})",
                unique_ids,
            )
        if cur.rowcount:
            conn.execute(
                "UPDATE lcm_embedding_profile "
                "SET data_version = data_version + 1"
            )
        return int(cur.rowcount or 0)

    def _migration_step_present(self, step_name: str) -> bool:
        """Read-before-write probe: is this migration marker already stamped?

        Lets schema-ensure skip an otherwise-needless marker re-write + commit on
        every construction (sprint-opt-1); returns False if the state table is
        absent (nothing stamped yet).
        """
        try:
            row = self._conn.execute(
                "SELECT 1 FROM lcm_migration_state WHERE step_name = ? LIMIT 1",
                (str(step_name),),
            ).fetchone()
        except sqlite3.OperationalError:
            return False
        return row is not None

    def _count_embedded_vectors(self, identity_hash: str, *, chunk: bool) -> int | None:
        """Cheap single-table COUNT of embedded vectors for one identity.

        Used only to annotate a ``coverage='bounded'`` result with the total
        corpus size (SCAN-1); returns ``None`` if the table is absent. Archival
        is tracked in the meta tables, so this is a total-embedded hint (the
        scanned/total ratio signals partial-archive coverage, not an exact live
        count).
        """
        table = "lcm_chunk_vectors" if chunk else "lcm_embedding_vectors"
        try:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE identity_hash = ?",
                (str(identity_hash),),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return int(row[0]) if row is not None else None

    def _scanned_knn_result(
        self,
        rows: Sequence[tuple[str, float, str]] = (),
        *,
        coverage: str,
        scoring: str,
        scanned_rows: int,
        deadline_expired: bool,
        identity_hash: str,
        chunk: bool,
        complete_total: int | None = None,
    ) -> KNNResult:
        """Build one scan result with the shared deadline-aware total policy."""
        scanned = total = None
        if coverage == "bounded":
            scanned = scanned_rows
            if not deadline_expired:
                total = self._count_embedded_vectors(identity_hash, chunk=chunk)
        elif complete_total is not None:
            scanned = scanned_rows
            total = complete_total
        return KNNResult(
            rows,
            coverage=coverage,
            scoring=scoring,
            scanned=scanned,
            total=total,
        )

    def _data_version(self, identity_hash: str) -> int:
        row = self._conn.execute(
            "SELECT data_version FROM lcm_embedding_profile WHERE identity_hash = ?",
            (str(identity_hash),),
        ).fetchone()
        return int(row["data_version"]) if row is not None else 0

    def _live_vector_count(self, identity_hash: str, *, chunk: bool) -> int:
        if chunk:
            row = self._conn.execute(
                """
                SELECT COUNT(*)
                FROM lcm_chunk_vectors v
                JOIN lcm_chunk_meta m
                  ON m.chunk_id = v.chunk_id
                 AND m.identity_hash = v.identity_hash
                JOIN messages msg
                  ON msg.store_id = m.store_id
                WHERE v.identity_hash = ? AND m.archived = 0
                """,
                (str(identity_hash),),
            ).fetchone()
        else:
            suppressed_clause = (
                " AND sn.suppressed_at IS NULL"
                if "suppressed_at" in self._table_columns("summary_nodes")
                else ""
            )
            row = self._conn.execute(
                f"""
                SELECT COUNT(*)
                FROM lcm_embedding_vectors v
                JOIN lcm_embedding_meta m
                  ON m.embedded_id = v.embedded_id
                 AND m.identity_hash = v.identity_hash
                JOIN summary_nodes sn
                  ON sn.node_id = CAST(m.embedded_id AS INTEGER)
                WHERE v.identity_hash = ? AND m.archived = 0{suppressed_clause}
                """,
                (str(identity_hash),),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    @staticmethod
    def _resident_entry_bytes(entry: tuple[Any, ...]) -> int:
        rowids, ids, kinds, matrix, scales = entry
        metadata_bytes = sum(
            sys.getsizeof(values)
            + sum(sys.getsizeof(value) for value in values)
            for values in (rowids, ids, kinds)
        )
        return int(matrix.nbytes) + int(scales.nbytes) + metadata_bytes

    @staticmethod
    def _purge_stale_resident_versions_locked(
        registry_key: tuple[str, int, int, int, str, int],
    ) -> None:
        """Drop old profile versions in this database/budget namespace."""
        for stale_key in list(_RESIDENT_MATRIX_REGISTRY):
            if (
                stale_key[0] == registry_key[0]
                and stale_key[1:3] == registry_key[1:3]
                and stale_key[4] == registry_key[4]
                and stale_key[5] == registry_key[5]
                and stale_key[3] != registry_key[3]
            ):
                _drop_resident_registry_entry_locked(stale_key)

    def _enforce_resident_budget_locked(
        self,
        registry_key: tuple[str, int, int, int, str, int],
        *,
        incoming_bytes: int = 0,
        protected_key: tuple[str, int, int, int, str, int] | None = None,
    ) -> bool:
        """Evict LRU entries only within the caller's budget namespace.

        The budget value is part of the pooled key, so stores configured with
        different ceilings are independent partitions rather than enforcing
        whichever ceiling happened to query last.
        """
        budget_mb = registry_key[5]
        matching_keys = [
            key for key in _RESIDENT_MATRIX_REGISTRY if key[5] == budget_mb
        ]
        used = sum(
            self._resident_entry_bytes(_RESIDENT_MATRIX_REGISTRY[key])
            for key in matching_keys
        )
        while used + incoming_bytes > self.knn_resident_max_bytes:
            oldest_key = next(
                (
                    key
                    for key in _RESIDENT_MATRIX_REGISTRY
                    if key[5] == budget_mb and key != protected_key
                ),
                None,
            )
            if oldest_key is None:
                return False
            evicted = _drop_resident_registry_entry_locked(oldest_key)
            if evicted is not None:
                used -= self._resident_entry_bytes(evicted)
        return True

    def _mark_resident_ineligible(
        self, key: tuple[bool, str, int]
    ) -> None:
        with _RESIDENT_MATRIX_REGISTRY_LOCK:
            self._resident_ineligible_cache = {
                cached_key
                for cached_key in self._resident_ineligible_cache
                if cached_key[:2] != key[:2]
            }
            self._resident_ineligible_cache.add(key)

    def _resident_registry_key(
        self, identity_hash: str, data_version: int
    ) -> tuple[str, int, int, int, str, int]:
        """Derive the one pooled key used for resident lookup and build."""
        return (
            self._resident_registry_db_path,
            *self._resident_registry_incarnation,
            int(data_version),
            str(identity_hash),
            self.knn_resident_max_mb,
        )

    def _resident_int8_matrix(
        self,
        numpy: Any,
        identity_hash: str,
        dim: int,
        dtype: str,
        *,
        data_version: int,
        chunk: bool,
        deadline: float | None,
    ) -> tuple[list[int], list[str], list[str], Any, Any] | None:
        """Return a budgeted, live-row-filtered int8-scored corpus matrix.

        D-ARCH-5 deliberately keeps the 128 MiB default enabled for eligible
        float32 corpora: F44 measured net-zero recall change at 185k vectors,
        every result discloses ``scoring='int8_quantized'``, and operators can
        disable residency with ``LCM_KNN_RESIDENT_MAX_MB=0``.

        Cold builds are serialized per pooled key. The count is an admission
        estimate only: the loader guards growth and rechecks live membership
        before publication, so a concurrent commit cannot overrun a preallocated
        matrix or publish a stale snapshot.
        """
        if self.knn_resident_max_bytes <= 0:
            return None
        if _prescreen_deadline_expired(deadline, 0):
            raise _PrescreenDeadlineExpired()
        data_version = int(data_version)
        key = (bool(chunk), str(identity_hash), data_version)
        registry_key = self._resident_registry_key(identity_hash, data_version)
        with _RESIDENT_MATRIX_REGISTRY_LOCK:
            self._purge_stale_resident_versions_locked(registry_key)
            cached = _RESIDENT_MATRIX_REGISTRY.get(registry_key)
            if cached is not None:
                if self._enforce_resident_budget_locked(
                    registry_key, protected_key=registry_key
                ):
                    _RESIDENT_MATRIX_REGISTRY.move_to_end(registry_key)
                    self._resident_matrix_cache[key] = cached
                    self._resident_matrix_cache.move_to_end(key)
                    if _prescreen_deadline_expired(deadline, len(cached[1])):
                        # The matrix is loaded but ranking has not scored a row.
                        raise _PrescreenDeadlineExpired()
                    return cached
                else:
                    _drop_resident_registry_entry_locked(registry_key)
            if (
                deadline is not None
                and dtype == _INT8_DTYPE
                and self._binary_fully_synced(identity_hash, chunk=chunk)
            ):
                # Preserve the deadline-aware two-stage int8 path, whose
                # completed-batch accounting is finer than a cold resident build.
                return None
            build_lock = _RESIDENT_MATRIX_BUILD_LOCKS.setdefault(
                registry_key, threading.Lock()
            )

        with build_lock:
            # Another caller may have completed the same cold build while this
            # caller waited. The shared key already includes data_version, so a
            # hit needs no profile/cardinality SQL.
            with _RESIDENT_MATRIX_REGISTRY_LOCK:
                self._purge_stale_resident_versions_locked(registry_key)
                cached = _RESIDENT_MATRIX_REGISTRY.get(registry_key)
                if cached is not None:
                    _RESIDENT_MATRIX_REGISTRY.move_to_end(registry_key)
                    self._resident_matrix_cache[key] = cached
                    self._resident_ineligible_cache.discard(key)
                    return cached

            # Admission sizing and the row-2/row-3 correctness checks belong to
            # cold construction. Every live-membership mutation advances the
            # profile data_version, so no corpus recount is needed on a hit.
            count = self._live_vector_count(identity_hash, chunk=chunk)
            required_bytes = count * (dim + 4)
            if (
                count == 0
                or (
                    dtype != _INT8_DTYPE
                    and count < _FAST_SCAN_STREAMING_MIN_ROWS
                )
                or required_bytes > self.knn_resident_max_bytes
            ):
                self._mark_resident_ineligible(key)
                return None
            # A float32 identity with a complete sign-bit corpus explicitly
            # selects the existing two-stage approximate-reach path.
            if dtype != _INT8_DTYPE and self._binary_fully_synced(
                identity_hash, chunk=chunk
            ):
                self._mark_resident_ineligible(key)
                return None

            matrix = numpy.empty((count, dim), dtype=numpy.int8)
            scales = numpy.empty(count, dtype=numpy.float32)
            rowids: list[int] = []
            ids: list[str] = []
            kinds: list[str] = []
            loaded = 0
            for (
                batch_rowids,
                batch_ids,
                batch_kinds,
                batch_matrix,
                batch_scales,
            ) in self._iter_vectorized_batches(
                numpy,
                identity_hash,
                dim,
                None,
                dtype,
                chunk=chunk,
                batch_rows=max(1, self.bounded_scan_rows),
                deadline=deadline,
            ):
                if batch_scales is None:
                    batch_matrix, batch_scales = self._quantize_float_matrix(
                        numpy, batch_matrix
                    )
                end = loaded + len(batch_ids)
                if end > count:
                    # The count and cursor can observe adjacent snapshots. Abort
                    # safely and let exact R1 read current state.
                    return None
                matrix[loaded:end] = batch_matrix
                scales[loaded:end] = batch_scales
                rowids.extend(batch_rowids)
                ids.extend(batch_ids)
                kinds.extend(batch_kinds)
                loaded = end
                if _prescreen_deadline_expired(deadline, loaded):
                    # Loading is not scoring; KNNResult.scanned must remain zero.
                    raise _PrescreenDeadlineExpired()
            if loaded != count:
                return None
            entry = (rowids, ids, kinds, matrix, scales)

            # Profile writes, suppression, and message deletion can all change
            # the live snapshot. Never publish a matrix built across that change.
            if (
                self._data_version(identity_hash) != data_version
                or self._live_vector_count(identity_hash, chunk=chunk) != count
            ):
                return None
            entry_bytes = self._resident_entry_bytes(entry)
            if entry_bytes > self.knn_resident_max_bytes:
                self._mark_resident_ineligible(key)
                return None
            with _RESIDENT_MATRIX_REGISTRY_LOCK:
                self._purge_stale_resident_versions_locked(registry_key)
                cached = _RESIDENT_MATRIX_REGISTRY.get(registry_key)
                if cached is not None:
                    _RESIDENT_MATRIX_REGISTRY.move_to_end(registry_key)
                    self._resident_matrix_cache[key] = cached
                    self._resident_ineligible_cache.discard(key)
                    return cached
                if not self._enforce_resident_budget_locked(
                    registry_key, incoming_bytes=entry_bytes
                ):
                    self._mark_resident_ineligible(key)
                    return None
                self._resident_ineligible_cache.discard(key)
                _RESIDENT_MATRIX_REGISTRY[registry_key] = entry
                self._resident_matrix_cache[key] = entry
            return entry

    def _rank_resident_int8(
        self,
        numpy: Any,
        resident: tuple[list[int], list[str], list[str], Any, Any],
        query: Any,
        limit: int,
        *,
        deadline: float | None,
    ) -> list[tuple[str, float, str]]:
        """Score every resident row with disclosed int8-quantized arithmetic."""
        rowids, ids, kinds, matrix, scales = resident
        best: list[tuple[int, str, float, str]] = []
        batch_rows = max(1, self.bounded_scan_rows)
        for start in range(0, len(ids), batch_rows):
            if _prescreen_deadline_expired(deadline, start):
                raise _PrescreenDeadlineExpired(start)
            end = min(len(ids), start + batch_rows)
            scores = self._score_vectorized_batch(
                numpy, matrix[start:end], scales[start:end], query
            )
            best.extend(
                (int(rowid), str(vector_id), float(score), str(kind))
                for rowid, vector_id, score, kind in zip(
                    rowids[start:end], ids[start:end], scores, kinds[start:end]
                )
            )
            if len(best) > limit:
                best.sort(key=self._rank_key)
                del best[limit:]
            if _prescreen_deadline_expired(deadline, end):
                raise _PrescreenDeadlineExpired(end)
        best.sort(key=self._rank_key)
        return [
            (vector_id, score, kind)
            for _, vector_id, score, kind in best[:limit]
        ]

    def _numpy_rows(
        self,
        numpy: Any,
        identity_hash: str,
        dim: int,
        embedded_ids: Sequence[str],
        dtype: str = _VECTOR_DTYPE,
        cache: bool = True,
    ) -> tuple[list[int], list[str], list[str], Any]:
        """Load only the SQL-bounded candidate set into a NumPy matrix.

        ``cache=False`` streams the batch past the LRU entirely — see
        ``_scan_ranked`` for why a multi-batch sweep must not cache.
        """
        if not cache:
            return self._load_matrix(numpy, identity_hash, dim, embedded_ids, dtype)
        with self._cache_lock:
            data_version = self._data_version(identity_hash)
            key = (identity_hash, data_version, tuple(str(value) for value in embedded_ids))
            cached = self._matrix_cache.get(key)
            if cached is not None:
                self._matrix_cache.move_to_end(key)  # mark most-recently used
                return cached
            loaded = self._load_matrix(numpy, identity_hash, dim, embedded_ids, dtype)
            self._matrix_cache[key] = loaded
            while len(self._matrix_cache) > self._MATRIX_CACHE_MAX_ENTRIES:
                self._matrix_cache.popitem(last=False)  # evict oldest
            return loaded

    def _load_matrix(
        self,
        numpy: Any,
        identity_hash: str,
        dim: int,
        embedded_ids: Sequence[str],
        dtype: str,
    ) -> tuple[list[int], list[str], list[str], Any]:
        """Decode one candidate set through the retained pre-R1 loader."""
        rowids, loaded_ids, kinds, raw_vectors = self._load_vectors_for_ids(
            identity_hash, dim, embedded_ids, dtype
        )
        matrix = (
            numpy.asarray(raw_vectors, dtype=numpy.float32)
            if raw_vectors
            else numpy.empty((0, dim), dtype=numpy.float32)
        )
        return rowids, loaded_ids, kinds, matrix

    def _bounded_candidate_ids(
        self,
        identity_hash: str,
        *,
        since: float | None,
        until: float | None,
        conversation_ids: Sequence[str] | None,
        source: str | None,
        limit: int,
        access_scope: str | None = None,
    ) -> list[str]:
        """Enumerate at most ``limit`` live candidate ids, most-recent first.

        The candidate enumeration itself is bounded at the SQL layer: the
        column filters (recency window, conversation, suppression) are applied
        in the ``WHERE`` clause and a hard ``LIMIT`` caps the result, so neither
        the SQL result set nor host memory materializes the whole corpus (a
        100-row corpus with bound 10 loads ~10 ids, not 100). Ordering is by the
        source summary's ``latest_at DESC`` so a newest-first backfill cannot
        invert the bounded retrieval window when older summaries are embedded
        later.

        Column filters are enforced BEFORE the bound (they are in ``WHERE``), so
        a filtered match inside the most-recent window is never dropped; only
        matches beyond the bounded window are out of scope, which is exactly
        what ``coverage='bounded'`` signals. The source-lineage filter is a
        recursive descendant walk, so it is applied to the already-bounded id
        set afterward (and fails closed when provenance is unverifiable).

        ``limit=_SCAN_ALL_ROWS`` enumerates every live candidate (the batched
        full-corpus scan bounds memory per BATCH, not per corpus).
        """
        if limit == 0:
            return []
        if conversation_ids is not None and not list(conversation_ids):
            return []
        columns = self._table_columns("summary_nodes")
        if not columns:
            return []
        if (since is not None or until is not None) and "latest_at" not in columns:
            return []
        # Per-row fallback matters, not just column existence: the DAG
        # migration adds ``latest_at`` without backfilling legacy rows, and a
        # bare ``latest_at DESC`` sorts those NULLs last — silently dropping
        # legacy summaries out of the bounded window on upgraded databases.
        recency_expr = (
            "COALESCE(sn.latest_at, sn.created_at)"
            if "latest_at" in columns
            else "sn.created_at"
        )
        where = ["m.identity_hash = ?", "m.archived = 0"]
        args: list[object] = [str(identity_hash)]
        if "suppressed_at" in columns:
            where.append("sn.suppressed_at IS NULL")
        if since is not None:
            where.append(f"{recency_expr} >= ?")
            args.append(float(since))
        if until is not None:
            where.append(f"{recency_expr} <= ?")
            args.append(float(until))
        if access_scope is not None:
            # The OWNER predicate, same as the FTS corpus. Applied in the WHERE
            # clause so it is enforced BEFORE the bound -- a filter applied after
            # a LIMIT would return another principal's rows whenever the bound
            # bit first, which is precisely when it matters.
            where.append("m.access_scope = ?")
            args.append(str(access_scope))
        args.append(int(limit))
        with self._optional_temp_id_table(conversation_ids) as conversation_table:
            conversation_join = (
                f"JOIN {conversation_table} c ON c.id = sn.session_id"
                if conversation_table is not None
                else ""
            )
            rows = self._conn.execute(
                f"""
                SELECT m.embedded_id
                FROM lcm_embedding_meta m
                JOIN summary_nodes sn ON sn.node_id = CAST(m.embedded_id AS INTEGER)
                {conversation_join}
                WHERE {' AND '.join(where)}
                ORDER BY {recency_expr} DESC, sn.node_id DESC
                LIMIT ?
                """,
                args,
            ).fetchall()
        bounded_ids = [str(row[0]) for row in rows]
        if source and bounded_ids:
            with self._temp_id_table(bounded_ids) as table:
                allowed = self._source_allowed_ids(table, source)
            bounded_ids = [
                embedded_id for embedded_id in bounded_ids if embedded_id in allowed
            ]
        return bounded_ids

    @staticmethod
    def _decode_stored_vec(blob: bytes, dim: int, dtype: str) -> tuple[float, ...] | None:
        """Decode a stored ``vec`` BLOB to floats per the profile dtype.

        float32 blobs are ``dim*4`` bytes; int8 blobs are ``dim`` bytes + a
        float32 scale (dequantized here). A blob whose length does not match the
        expected dtype layout is rejected (returns None) rather than
        misinterpreted — the same defensive guard the float path always had.
        """
        if dtype == _INT8_DTYPE:
            return _decode_int8_vector(blob, dim)
        if len(blob) != dim * 4:
            return None
        try:
            return struct.unpack(f"<{dim}f", blob)
        except struct.error:
            return None

    def _vector_rows_cursor(
        self,
        identity_hash: str,
        *,
        chunk: bool,
        candidate_ids: Sequence[str] | None = None,
    ) -> sqlite3.Cursor:
        """Stream live vectors, newest-first when candidates were enumerated."""
        candidate_order = candidate_ids is not None
        args: list[object] = [str(identity_hash)]
        if chunk:
            order_by = (
                "msg.timestamp DESC, v.chunk_id DESC"
                if candidate_order
                else "v.rowid"
            )
            return self._conn.execute(
                f"""
                SELECT v.rowid AS vector_rowid, v.chunk_id AS vector_id,
                       'chunk' AS vector_kind, v.vec AS vector_blob
                FROM lcm_chunk_vectors v
                JOIN lcm_chunk_meta m
                 ON m.chunk_id = v.chunk_id
                 AND m.identity_hash = v.identity_hash
                JOIN messages msg
                  ON msg.store_id = m.store_id
                WHERE v.identity_hash = ? AND m.archived = 0
                ORDER BY {order_by}
                """,
                args,
            )
        summary_columns = self._table_columns("summary_nodes")
        suppressed_clause = (
            " AND sn.suppressed_at IS NULL"
            if "suppressed_at" in summary_columns
            else ""
        )
        recency_expr = (
            "COALESCE(sn.latest_at, sn.created_at)"
            if "latest_at" in summary_columns
            else "sn.created_at"
        )
        order_by = (
            f"{recency_expr} DESC, sn.node_id DESC"
            if candidate_order
            else "v.rowid"
        )
        return self._conn.execute(
            f"""
            SELECT v.rowid AS vector_rowid, v.embedded_id AS vector_id,
                   m.embedded_kind AS vector_kind, v.vec AS vector_blob
            FROM lcm_embedding_vectors v
            JOIN lcm_embedding_meta m
              ON m.embedded_id = v.embedded_id
             AND m.identity_hash = v.identity_hash
            JOIN summary_nodes sn
              ON sn.node_id = CAST(m.embedded_id AS INTEGER)
            WHERE v.identity_hash = ? AND m.archived = 0{suppressed_clause}
            ORDER BY {order_by}
            """,
            args,
        )

    @contextmanager
    def _vector_rows_stream(
        self,
        identity_hash: str,
        *,
        chunk: bool,
        candidate_ids: Sequence[str] | None,
    ) -> Iterator[sqlite3.Cursor]:
        cursor = self._vector_rows_cursor(
            identity_hash,
            chunk=chunk,
            candidate_ids=candidate_ids,
        )
        try:
            yield cursor
        finally:
            cursor.close()

    @staticmethod
    def _vectorized_batch(
        numpy: Any,
        rows: Sequence[tuple[int, str, str, bytes]],
        dim: int,
        dtype: str,
    ) -> tuple[list[int], list[str], list[str], Any, Any]:
        """Decode one homogeneous BLOB batch with NumPy, not per-row unpack."""
        rowids = [row[0] for row in rows]
        ids = [row[1] for row in rows]
        kinds = [row[2] for row in rows]
        if dtype == _INT8_DTYPE:
            matrix = numpy.frombuffer(
                b"".join(row[3][:dim] for row in rows), dtype=numpy.int8
            ).reshape(len(rows), dim)
            scales = numpy.frombuffer(
                b"".join(row[3][dim:] for row in rows), dtype="<f4"
            )
            return rowids, ids, kinds, matrix, scales
        matrix = numpy.frombuffer(
            b"".join(row[3] for row in rows), dtype="<f4"
        ).reshape(len(rows), dim)
        return rowids, ids, kinds, matrix, None

    def _iter_vectorized_batches(
        self,
        numpy: Any,
        identity_hash: str,
        dim: int,
        candidate_ids: Sequence[str] | None,
        dtype: str,
        *,
        chunk: bool,
        batch_rows: int,
        deadline: float | None = None,
    ) -> Iterator[tuple[list[int], list[str], list[str], Any, Any]]:
        """Yield selected live vectors from one streaming cursor."""
        wanted = (
            None
            if candidate_ids is None
            else {str(value) for value in candidate_ids}
        )
        if wanted == set():
            return
        expected_bytes = dim + 4 if dtype == _INT8_DTYPE else dim * 4
        pending: list[tuple[int, str, str, bytes]] = []
        completed = 0
        with self._vector_rows_stream(
            identity_hash, chunk=chunk, candidate_ids=candidate_ids
        ) as cursor:
            for row in cursor:
                if _prescreen_deadline_expired(deadline, completed):
                    raise _PrescreenDeadlineExpired(completed)
                vector_id = str(row["vector_id"])
                if wanted is not None and vector_id not in wanted:
                    continue
                try:
                    blob = bytes(row["vector_blob"])
                except (TypeError, ValueError):
                    continue
                if len(blob) != expected_bytes:
                    continue
                pending.append(
                    (
                        int(row["vector_rowid"]),
                        vector_id,
                        str(row["vector_kind"]),
                        blob,
                    )
                )
                if len(pending) >= max(1, int(batch_rows)):
                    yield self._vectorized_batch(numpy, pending, dim, dtype)
                    completed += len(pending)
                    pending = []
            if pending:
                yield self._vectorized_batch(numpy, pending, dim, dtype)

    def _load_vector_matrix(
        self,
        numpy: Any,
        identity_hash: str,
        dim: int,
        candidate_ids: Sequence[str],
        dtype: str,
        *,
        chunk: bool,
    ) -> tuple[list[int], list[str], list[str], Any]:
        """Load one candidate set through the shared vectorized cursor."""
        batches = list(
            self._iter_vectorized_batches(
                numpy,
                identity_hash,
                dim,
                candidate_ids,
                dtype,
                chunk=chunk,
                batch_rows=max(1, len(candidate_ids)),
            )
        )
        if not batches:
            return [], [], [], numpy.empty((0, dim), dtype=numpy.float32)
        rowids, ids, kinds, matrix, scales = batches[0]
        if scales is not None:
            matrix = (
                matrix.astype(numpy.float32)
                * scales.astype(numpy.float32, copy=False)[:, None]
            )
        return rowids, ids, kinds, matrix

    def _load_python_vectors_for_ids(
        self,
        identity_hash: str,
        dim: int,
        candidate_ids: Sequence[str],
        dtype: str,
        *,
        chunk: bool,
    ) -> tuple[list[int], list[str], list[str], list[list[float]]]:
        """Dependency-free fallback over the same ordered live-row stream."""
        wanted = {str(value) for value in candidate_ids}
        rowids: list[int] = []
        ids: list[str] = []
        kinds: list[str] = []
        vectors: list[list[float]] = []
        if not wanted:
            return rowids, ids, kinds, vectors
        with self._vector_rows_stream(
            identity_hash, chunk=chunk, candidate_ids=candidate_ids
        ) as cursor:
            for row in cursor:
                vector_id = str(row["vector_id"])
                if vector_id not in wanted:
                    continue
                try:
                    vector = self._decode_stored_vec(
                        bytes(row["vector_blob"]), dim, dtype
                    )
                except (TypeError, ValueError):
                    continue
                if vector is None:
                    continue
                rowids.append(int(row["vector_rowid"]))
                ids.append(vector_id)
                kinds.append(str(row["vector_kind"]))
                vectors.append(list(vector))
        return rowids, ids, kinds, vectors

    def _load_vectors_for_ids(
        self,
        identity_hash: str,
        dim: int,
        embedded_ids: Sequence[str],
        dtype: str = _VECTOR_DTYPE,
    ) -> tuple[list[int], list[str], list[str], list[list[float]]]:
        """Pre-R1 loader for a bounded, already-filtered summary id set."""
        rowids: list[int] = []
        out_ids: list[str] = []
        kinds: list[str] = []
        vectors: list[list[float]] = []
        if not embedded_ids:
            return rowids, out_ids, kinds, vectors
        with self._temp_id_table(embedded_ids) as table:
            rows = self._conn.execute(
                f"""
                SELECT v.rowid, v.embedded_id, m.embedded_kind, v.vec
                FROM {table} t
                JOIN lcm_embedding_vectors v
                  ON v.embedded_id = t.id AND v.identity_hash = ?
                JOIN lcm_embedding_meta m
                  ON m.embedded_id = v.embedded_id
                 AND m.identity_hash = v.identity_hash
                WHERE m.archived = 0
                """,
                (identity_hash,),
            ).fetchall()
        for row in rows:
            try:
                vector = self._decode_stored_vec(bytes(row["vec"]), dim, dtype)
            except (TypeError, ValueError):
                continue
            if vector is None:
                continue
            rowids.append(int(row["rowid"]))
            out_ids.append(str(row["embedded_id"]))
            kinds.append(str(row["embedded_kind"]))
            vectors.append(list(vector))
        return rowids, out_ids, kinds, vectors

    @staticmethod
    def _rank_key(row: tuple[int, str, float, str]) -> tuple[float, int, str]:
        """Sort key for one ``(rowid, embedded_id, score, kind)`` candidate."""
        return (-float(row[2]), -int(row[0]), str(row[1]))

    @classmethod
    def _ranked(
        cls,
        rowids: Sequence[int],
        embedded_ids: Sequence[str],
        kinds: Sequence[str],
        scores: Sequence[float],
        limit: int,
    ) -> list[tuple[str, float, str]]:
        ranked = sorted(zip(rowids, embedded_ids, scores, kinds), key=cls._rank_key)
        return [
            (str(embedded_id), float(score), str(kind))
            for _, embedded_id, score, kind in ranked[:limit]
        ]

    def _release_matrix_caches(self) -> None:
        """Drop transient float matrices before a streamed scan allocates batches.

        ``cache=False`` alone keeps NEW batches out of the LRU but does nothing
        about what is already in it, and the retrieval-core pool keeps a store
        alive for the whole process. A probe over a two-batch scan measured
        cache_before=4 / cache_after=4: four warm matrices (~192MB of summary
        float32 at a 25k batch, before the separate chunk cache) coexisting with
        the streamed batch, which is not the one-batch bound this batching
        promises.

        Persistent binary and int8 resident matrices are intentionally retained:
        they have their own explicit cache/budget contracts and are exactly the
        warm state a multi-batch fallback must not destroy (#171 trap 1).

        Safe against a concurrent reader on a shared pooled store: clearing only
        drops the dict entries. A caller mid-scan holds its own reference to the
        tuple it was handed, so its matrix stays alive until it returns and is
        freed normally afterwards.
        """
        with self._cache_lock:
            self._matrix_cache.clear()
            self._chunk_matrix_cache.clear()

    @staticmethod
    def _score_vectorized_batch(
        numpy: Any, matrix: Any, scales: Any, query: Any
    ) -> Any:
        if scales is None:
            return matrix @ query
        # Preserve the pre-#171 score operation order for bit-identical output:
        # dequantize each int8 value to float32, then take the dot product.
        dequantized = (
            matrix.astype(numpy.float32)
            * scales.astype(numpy.float32, copy=False)[:, None]
        )
        return dequantized @ query

    @staticmethod
    def _quantize_float_matrix(numpy: Any, matrix: Any) -> tuple[Any, Any]:
        """Apply ``_encode_int8_vector`` semantics to float32 rows in bulk."""
        if matrix.shape[0] == 0:
            return (
                numpy.empty(matrix.shape, dtype=numpy.int8),
                numpy.empty(0, dtype=numpy.float32),
            )
        # _encode_int8_vector computes a per-vector scale in Python float before
        # packing it to float32. Promote first so the vectorized build preserves
        # those rounding semantics instead of taking a float32 max/division.
        wide = matrix.astype(numpy.float64, copy=False)
        max_abs = numpy.max(numpy.abs(wide), axis=1)
        scale64 = max_abs / float(_INT8_MAX)
        quantized = numpy.zeros(matrix.shape, dtype=numpy.int8)
        nonzero = scale64 > 0.0
        if numpy.any(nonzero):
            quantized[nonzero] = numpy.clip(
                numpy.rint(wide[nonzero] / scale64[nonzero, None]),
                -_INT8_MAX,
                _INT8_MAX,
            ).astype(numpy.int8)
        return quantized, scale64.astype(numpy.float32)

    @staticmethod
    def _use_streaming_scan(candidate_count: int, batch_rows: int) -> bool:
        """Use R1 once its measured corpus-size crossover applies."""
        del batch_rows
        return int(candidate_count) >= _FAST_SCAN_STREAMING_MIN_ROWS

    def _scan_ranked(
        self,
        *,
        candidate_ids: Sequence[str],
        batch_rows: int,
        budget_s: float,
        deadline: float | None,
        caller_deadline: float | None,
        limit: int,
        score_batch: Any,
    ) -> tuple[list[tuple[str, float, str]], int, bool, bool]:
        """Score every candidate in ``batch_rows`` chunks, keeping a running top-k.

        ``batch_rows`` bounds PEAK MEMORY (one batch of vectors resident at a
        time), not the corpus reached: the whole candidate set is scored, so a
        memory older than the batch size stays retrievable. That is the fix for
        the 25k recency window that made 86% of a 185k-vector corpus invisible
        to semantic recall (FINDING-F31 §2).

        ``budget_s`` (0 = no relative early stop, the default) and the caller's
        absolute operation ``deadline`` can cut the scan short; when either
        does, the caller degrades to ``coverage='bounded'`` and the existing
        disclosure names the ratio. ``deadline`` is the effective stop deadline,
        which may include the relative budget; ``caller_deadline`` is only the
        caller's absolute deadline.
        Returns ``(ranked top-k, candidates scored, stopped early,
        deadline expired)``. The final flag reports only caller-deadline expiry
        and is intentionally separate from ``stopped_early``: a relative budget
        or unreadable live vector can make the scan bounded without expiring the
        caller's absolute deadline.

        A MULTI-BATCH sweep streams past the matrix LRU (``cache=False``). The
        cache holds 4 entries, so a corpus needing more batches than that evicts
        each one before the next sweep reaches it — every repetition pays a full
        reload for zero hits, while still retaining 4 float32 matrices (153.6MB
        on the 185k-vector chunk arm) and breaking the one-batch memory bound
        this batching exists to provide. A single-batch scan — every lcm_grep
        call and any corpus under the batch size — still caches exactly as
        before, so the pooled-store warm path is unchanged.
        """
        best: list[tuple[int, str, float, str]] = []
        scanned = 0
        stopped_early = False
        deadline_expired = False
        cache_batches = len(candidate_ids) <= batch_rows
        if not cache_batches:
            # Keeping the new batches OUT of the cache is only half the bound:
            # matrices warmed by EARLIER calls stay resident for the pooled
            # store's whole lifetime. Release them before the first batch is
            # allocated, so peak really is one batch.
            self._release_matrix_caches()
        started = _monotonic()
        for start in range(0, len(candidate_ids), batch_rows):
            stop_expired = deadline is not None and _monotonic() >= deadline
            if stop_expired:
                stopped_early = True
                deadline_expired = _caller_deadline_expired(
                    caller_deadline,
                    stop_deadline=deadline,
                    stop_expired=stop_expired,
                )
                break
            batch = candidate_ids[start:start + batch_rows]
            rowids, embedded_ids, kinds, scores = score_batch(batch, cache_batches)
            scanned += len(rowids)
            best.extend(
                (int(rowid), str(embedded_id), float(score), str(kind))
                for rowid, embedded_id, score, kind in zip(
                    rowids, embedded_ids, scores, kinds
                )
            )
            if len(best) > limit:
                best.sort(key=self._rank_key)
                del best[limit:]
            exhausted = start + batch_rows >= len(candidate_ids)
            if not exhausted:
                budget_expired = (
                    budget_s > 0 and (_monotonic() - started) >= budget_s
                )
                stop_expired = (
                    deadline is not None and _monotonic() >= deadline
                )
                if budget_expired or stop_expired:
                    stopped_early = True
                    deadline_expired = _caller_deadline_expired(
                        caller_deadline,
                        stop_deadline=deadline,
                        stop_expired=stop_expired,
                    )
                    break
        if scanned < len(candidate_ids):
            stopped_early = True
        best.sort(key=self._rank_key)
        return (
            [
                (embedded_id, score, kind)
                for _, embedded_id, score, kind in best[:limit]
            ],
            scanned,
            stopped_early,
            deadline_expired,
        )

    def _scan_vectorized_ranked(
        self,
        *,
        numpy: Any,
        identity_hash: str,
        dim: int,
        dtype: str,
        candidate_ids: Sequence[str],
        chunk: bool,
        batch_rows: int,
        budget_s: float,
        deadline: float | None,
        caller_deadline: float | None,
        limit: int,
        query: Any,
    ) -> tuple[list[tuple[str, float, str]], int, bool, bool]:
        """Exact multi-batch R1 scan using one cursor and vectorized decode."""
        self._release_matrix_caches()
        best: list[tuple[int, str, float, str]] = []
        scanned = 0
        stopped_early = False
        deadline_expired = False
        started = _monotonic()
        batches = None
        try:
            batches = self._iter_vectorized_batches(
                numpy,
                identity_hash,
                dim,
                candidate_ids,
                dtype,
                chunk=chunk,
                batch_rows=batch_rows,
                deadline=deadline,
            )
            for rowids, ids, kinds, matrix, scales in batches:
                scores = self._score_vectorized_batch(
                    numpy, matrix, scales, query
                )
                scanned += len(ids)
                best.extend(
                    (int(rowid), str(vector_id), float(score), str(kind))
                    for rowid, vector_id, score, kind in zip(
                        rowids, ids, scores, kinds
                    )
                )
                if len(best) > limit:
                    best.sort(key=self._rank_key)
                    del best[limit:]
                budget_expired = (
                    budget_s > 0 and (_monotonic() - started) >= budget_s
                )
                stop_expired = _prescreen_deadline_expired(deadline, scanned)
                if budget_expired or stop_expired:
                    stopped_early = scanned < len(candidate_ids)
                    if stopped_early:
                        deadline_expired = _caller_deadline_expired(
                            caller_deadline,
                            stop_deadline=deadline,
                            stop_expired=stop_expired,
                        )
                        break
        except _PrescreenDeadlineExpired as exc:
            scanned = max(scanned, exc.scanned)
            stopped_early = scanned < len(candidate_ids)
            if stopped_early:
                deadline_expired = _caller_deadline_expired(
                    caller_deadline,
                    stop_deadline=deadline,
                    stop_expired=True,
                )
        finally:
            if batches is not None:
                batches.close()
        if scanned < len(candidate_ids):
            stopped_early = True
        best.sort(key=self._rank_key)
        return (
            [
                (vector_id, score, kind)
                for _, vector_id, score, kind in best[:limit]
            ],
            scanned,
            stopped_early,
            deadline_expired,
        )

    def _source_allowed_ids(self, table: str, source: str) -> set[str]:
        """Root candidate ids whose source subtree reaches a message with ``source``.

        Enforced at the store layer (before the top-k cap) by walking each
        candidate summary's source tree down to its raw messages — the same
        recursion the DAG uses — so an ineligible high-scoring vector cannot
        consume a slot ahead of an eligible lower one.
        """
        from .store import _UNKNOWN_SOURCE, _legacy_blank_source_clause, _normalize_source_value

        # A VectorStore-only worker DB may predate MessageStore's source repair
        # (``_ensure_source_column``) and lack ``messages.source`` entirely.
        # Provenance is then unverifiable, so the source filter must FAIL CLOSED:
        # return no allowed ids rather than treating "can't check" as "all
        # allowed". Failing open let a source-filtered query surface a legacy
        # summary whose source could not be confirmed; an empty allowed set
        # instead yields no false-positive hit (and lets the caller degrade to
        # full_text), which is the safe direction for an unverifiable filter.
        message_columns = {
            str(row[1])
            for row in self._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "source" not in message_columns:
            return set()

        normalized_source = _normalize_source_value(source)
        legacy_blank_clause = _legacy_blank_source_clause("m.source")
        rows = self._conn.execute(
            f"""
            WITH RECURSIVE walk(root_id, source_type, source_id) AS (
                SELECT t.id, sn.source_type, CAST(j.value AS INTEGER)
                FROM {table} t
                JOIN summary_nodes sn ON sn.node_id = CAST(t.id AS INTEGER)
                JOIN json_each(sn.source_ids) j

                UNION ALL

                SELECT w.root_id, child.source_type, CAST(j.value AS INTEGER)
                FROM walk w
                JOIN summary_nodes child
                  ON w.source_type = 'nodes' AND child.node_id = w.source_id
                JOIN json_each(child.source_ids) j
                LIMIT ?
            ), matched AS (
                SELECT DISTINCT w.root_id
                FROM walk w
                JOIN messages m
                  ON w.source_type = 'messages' AND m.store_id = w.source_id
                WHERE CASE
                        WHEN ? = ? THEN (m.source = ? OR {legacy_blank_clause})
                        ELSE m.source = ?
                      END
            ), work_count AS (
                SELECT COUNT(*) AS visited FROM walk
            )
            SELECT root_id, 0 AS overflow FROM matched
            UNION ALL
            SELECT NULL, 1 FROM work_count WHERE visited > ?
            """,
            (
                _SOURCE_LINEAGE_WORK_LIMIT + 1,
                normalized_source,
                _UNKNOWN_SOURCE,
                normalized_source,
                normalized_source,
                _SOURCE_LINEAGE_WORK_LIMIT,
            ),
        ).fetchall()
        if any(int(row[1]) for row in rows):
            raise _UnverifiableProvenance(
                "source lineage exceeded the bounded verification budget"
            )
        return {str(row[0]) for row in rows}

    def _filtered_candidate_indexes(
        self,
        embedded_ids: Sequence[str],
        *,
        since: float | None,
        until: float | None,
        conversation_ids: Sequence[str] | None,
        source: str | None,
    ) -> list[int]:
        if not embedded_ids:
            return []
        if conversation_ids is not None and not list(conversation_ids):
            return []
        unique_ids = list(dict.fromkeys(str(value) for value in embedded_ids))
        columns = self._table_columns("summary_nodes")
        if (since is not None or until is not None) and "latest_at" not in columns:
            return []
        recency_expr = "sn.latest_at"
        with self._temp_id_table(unique_ids) as table:
            where = ["1 = 1"]
            args: list[object] = []
            if "suppressed_at" in columns:
                where.append("sn.suppressed_at IS NULL")
            if since is not None:
                where.append(f"{recency_expr} >= ?")
                args.append(float(since))
            if until is not None:
                where.append(f"{recency_expr} <= ?")
                args.append(float(until))
            with self._optional_temp_id_table(conversation_ids) as conversation_table:
                conversation_join = (
                    f"JOIN {conversation_table} c ON c.id = sn.session_id"
                    if conversation_table is not None
                    else ""
                )
                rows = self._conn.execute(
                    f"""
                    SELECT t.id
                    FROM {table} t
                    JOIN summary_nodes sn ON sn.node_id = CAST(t.id AS INTEGER)
                    {conversation_join}
                    WHERE {' AND '.join(where)}
                    """,
                    args,
                ).fetchall()
            allowed = {str(row[0]) for row in rows}
            if source:
                allowed &= self._source_allowed_ids(table, source)
        return [
            index
            for index, embedded_id in enumerate(embedded_ids)
            if str(embedded_id) in allowed
        ]

    # -- Two-stage KNN (binary Hamming prescreen -> int8/float rescore) -----
    #
    # SPEC C1: when the active identity has a sign-bit prescreen column, KNN runs
    # over the WHOLE corpus (coverage='full') within the deadline: stage 1 ranks
    # every live vector by Hamming distance to the query's sign bits (packed 1
    # bit/dim, ~20MB for 260k x 1024bit) and keeps M = mult x k survivors; stage 2
    # rescores only those M by exact int8 cosine and returns top-k. Stage 1 is an
    # APPROXIMATE prescreen — recall@M is the guarantee, not exact top-k — so M is
    # deliberately several times k.

    def _has_binary(self, identity_hash: str, *, chunk: bool) -> bool:
        """Does the active identity carry a sign-bit prescreen (int8 identities)?"""
        table = "lcm_chunk_binary" if chunk else "lcm_embedding_binary"
        try:
            row = self._conn.execute(
                f"SELECT 1 FROM {table} WHERE identity_hash = ? LIMIT 1",
                (str(identity_hash),),
            ).fetchone()
        except sqlite3.OperationalError:
            return False
        return row is not None

    def _binary_fully_synced(self, identity_hash: str, *, chunk: bool) -> bool:
        """True iff EVERY stored vector for the identity has a sign-bit row.

        Defense-in-depth for FIX 1: the two-stage path INNER JOINs the binary
        table, so it is only correct when the binary corpus is a COMPLETE mirror
        of the vector corpus. A partial binary corpus — e.g.
        ``LCM_EMBEDDING_BINARY_PRESCREEN`` flipped on mid-life so sign-bits exist
        only for rows written after the flip — would silently exclude every
        pre-flip vector while reporting coverage='full'. When the counts disagree
        (or no binary rows exist) this returns False so ``knn`` stays on the exact
        scan with honest coverage. Both tables are written/deleted in lockstep for
        a genuine prescreen identity (int8 or float32-with-prescreen), so a
        fully-synced corpus reports equal counts.
        """
        binary_table = "lcm_chunk_binary" if chunk else "lcm_embedding_binary"
        vector_table = "lcm_chunk_vectors" if chunk else "lcm_embedding_vectors"
        try:
            binary_count = self._conn.execute(
                f"SELECT COUNT(*) FROM {binary_table} WHERE identity_hash = ?",
                (str(identity_hash),),
            ).fetchone()
            vector_count = self._conn.execute(
                f"SELECT COUNT(*) FROM {vector_table} WHERE identity_hash = ?",
                (str(identity_hash),),
            ).fetchone()
        except sqlite3.OperationalError:
            return False
        binary = int(binary_count[0]) if binary_count else 0
        vectors = int(vector_count[0]) if vector_count else 0
        return binary > 0 and binary == vectors

    def _load_chunk_binary_matrix(
        self,
        numpy: Any,
        identity_hash: str,
        *,
        since: float | None,
        until: float | None,
        conversation_ids: Sequence[str] | None,
        source: str | None,
    ) -> tuple[list[str], Any]:
        """Load the FULL live sign-bit corpus for a chunk identity as a uint8 matrix.

        The same message-column filters the bounded chunk path applies
        (recency/session/source) run in the WHERE clause, but WITHOUT a LIMIT —
        the whole point of the two-stage path is full-corpus reach. Returns the
        parallel chunk-id list and an ``(N, nbytes)`` uint8 matrix.
        """
        where = ["cm.identity_hash = ?", "cm.archived = 0"]
        args: list[object] = [str(identity_hash)]
        if since is not None:
            where.append("m.timestamp >= ?")
            args.append(float(since))
        if until is not None:
            where.append("m.timestamp <= ?")
            args.append(float(until))
        if source is not None:
            where.append("m.source = ?")
            args.append(str(source))
        with self._optional_temp_id_table(conversation_ids) as conversation_table:
            conversation_join = (
                f"JOIN {conversation_table} c ON c.id = m.session_id"
                if conversation_table is not None
                else ""
            )
            rows = self._conn.execute(
                f"""
                SELECT b.chunk_id, b.bits
                FROM lcm_chunk_binary b
                JOIN lcm_chunk_meta cm
                  ON cm.chunk_id = b.chunk_id AND cm.identity_hash = b.identity_hash
                JOIN messages m ON m.store_id = cm.store_id
                {conversation_join}
                WHERE {' AND '.join(where)}
                """,
                args,
            ).fetchall()
        return self._binary_rows_to_matrix(numpy, rows)

    def _load_embedding_binary_matrix(
        self,
        numpy: Any,
        identity_hash: str,
        *,
        since: float | None,
        until: float | None,
        conversation_ids: Sequence[str] | None,
    ) -> tuple[list[str], Any]:
        """Load the FULL live sign-bit corpus for a summary identity (no source).

        since/until/conversation filters (plain ``summary_nodes`` columns) run in
        the WHERE clause with no LIMIT. The ``source`` lineage filter is NOT
        handled here — its recursive descendant walk is budget-bounded, so a
        source-filtered summary query stays on the exact bounded path (the caller
        gates on ``source is None`` before calling this).
        """
        columns = self._table_columns("summary_nodes")
        recency_expr = (
            "COALESCE(sn.latest_at, sn.created_at)"
            if "latest_at" in columns
            else "sn.created_at"
        )
        where = ["m.identity_hash = ?", "m.archived = 0"]
        args: list[object] = [str(identity_hash)]
        if "suppressed_at" in columns:
            where.append("sn.suppressed_at IS NULL")
        if since is not None:
            where.append(f"{recency_expr} >= ?")
            args.append(float(since))
        if until is not None:
            where.append(f"{recency_expr} <= ?")
            args.append(float(until))
        with self._optional_temp_id_table(conversation_ids) as conversation_table:
            conversation_join = (
                f"JOIN {conversation_table} c ON c.id = sn.session_id"
                if conversation_table is not None
                else ""
            )
            rows = self._conn.execute(
                f"""
                SELECT b.embedded_id, b.bits
                FROM lcm_embedding_binary b
                JOIN lcm_embedding_meta m
                  ON m.embedded_id = b.embedded_id AND m.identity_hash = b.identity_hash
                JOIN summary_nodes sn ON sn.node_id = CAST(b.embedded_id AS INTEGER)
                {conversation_join}
                WHERE {' AND '.join(where)}
                """,
                args,
            ).fetchall()
        return self._binary_rows_to_matrix(numpy, rows)

    def _cached_binary_matrix(
        self, cache: "OrderedDict[tuple[str, int], tuple[list[str], Any]]",
        identity_hash: str, loader, *, cacheable: bool, deadline: float | None,
    ) -> tuple[list[str], Any]:
        """Load a full-corpus binary matrix, caching only the UNFILTERED case.

        A filtered load (recency/session/source) varies per call, so it bypasses
        the cache; the common unfiltered full-corpus recall reuses the loaded
        sign-bit matrix across calls (keyed on data_version for cross-process
        invalidation), which is what makes back-to-back two-stage recalls cheap.
        """
        if deadline is not None and _monotonic() >= deadline:
            raise _PrescreenDeadlineExpired()
        if not cacheable:
            loaded = loader()
            if deadline is not None and _monotonic() >= deadline:
                raise _PrescreenDeadlineExpired()
            return loaded
        with self._cache_lock:
            key = (identity_hash, self._data_version(identity_hash))
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                if deadline is not None and _monotonic() >= deadline:
                    raise _PrescreenDeadlineExpired()
                return cached
            loaded = loader()
            if deadline is not None and _monotonic() >= deadline:
                raise _PrescreenDeadlineExpired()
            cache[key] = loaded
            while len(cache) > self._MATRIX_CACHE_MAX_ENTRIES:
                cache.popitem(last=False)
            return loaded

    def _load_binary_with_deadline(self, loader, deadline: float | None):
        """Run a full binary load on an interruptible read-only connection."""
        if deadline is None:
            return loader(self)
        remaining = deadline - _monotonic()
        if remaining <= 0:
            raise _PrescreenDeadlineExpired()
        conn: sqlite3.Connection | None = None
        expired = [False]

        def interrupt_if_expired() -> int:
            if _monotonic() >= deadline:
                expired[0] = True
                return 1
            return 0

        db_path = str(self.db_path)
        query = db_path.partition("?")[2]
        in_memory = (
            db_path == ":memory:"
            or db_path.startswith("file::memory:")
            or any(part == "mode=memory" for part in query.split("&"))
        )
        if in_memory:
            with self._write_lock:
                try:
                    self._conn.set_progress_handler(interrupt_if_expired, 1000)
                    loaded = loader(self)
                    if _monotonic() >= deadline:
                        raise _PrescreenDeadlineExpired()
                    return loaded
                except sqlite3.OperationalError as exc:
                    if expired[0] or _monotonic() >= deadline:
                        raise _PrescreenDeadlineExpired() from exc
                    raise
                finally:
                    self._conn.set_progress_handler(None, 0)

        uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
        try:
            conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=max(0.001, remaining),
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.set_progress_handler(interrupt_if_expired, 1000)
            reader = copy.copy(self)
            reader._conn = conn
            reader._write_lock = threading.RLock()
            reader._cache_lock = threading.RLock()
            loaded = loader(reader)
            if _monotonic() >= deadline:
                raise _PrescreenDeadlineExpired()
            return loaded
        except sqlite3.OperationalError as exc:
            if expired[0] or _monotonic() >= deadline:
                raise _PrescreenDeadlineExpired() from exc
            raise
        finally:
            if conn is not None:
                conn.close()

    def _read_with_deadline(self, loader, deadline: float | None):
        """Run a candidate read through the shared interruptible read path."""
        return self._load_binary_with_deadline(loader, deadline)

    @staticmethod
    def _binary_rows_to_matrix(numpy: Any, rows: Sequence[sqlite3.Row]) -> tuple[list[str], Any]:
        ids: list[str] = []
        raw: list[bytes] = []
        nbytes = 0
        for row in rows:
            blob = bytes(row[1])
            if not blob:
                continue
            if nbytes == 0:
                nbytes = len(blob)
            elif len(blob) != nbytes:
                # A mixed-width row means a corrupt/foreign identity leaked in;
                # skip it rather than mis-shape the matrix.
                continue
            ids.append(str(row[0]))
            raw.append(blob)
        if not raw:
            return ids, numpy.empty((0, 0), dtype=numpy.uint8)
        matrix = numpy.frombuffer(b"".join(raw), dtype=numpy.uint8).reshape(
            len(raw), nbytes
        )
        return ids, matrix

    def _two_stage_rank(
        self,
        numpy: Any,
        identity_hash: str,
        dim: int,
        dtype: str,
        query: Sequence[float],
        k: int,
        binary_ids: Sequence[str],
        binary_matrix: Any,
        *,
        chunk: bool,
        deadline: float | None,
    ) -> list[tuple[str, float, str]]:
        """Stage-1 Hamming prescreen over ``binary_matrix`` then stage-2 rescore.

        Stage 1 packs the query sign bits, XORs against every corpus row, popcounts
        (byte lookup table), and keeps the M = mult x k lowest-Hamming survivors.
        Stage 2 loads the survivors' int8 vectors, dequantizes, and ranks by exact
        cosine (the query is unit-normalized, so the dot product is cosine).
        """
        n = binary_matrix.shape[0]
        if n == 0:
            return []
        query_bits = numpy.packbits(
            (numpy.asarray(query, dtype=numpy.float32) >= 0.0).astype(numpy.uint8)
        )
        # Guard against a query/corpus width mismatch (defensive; identities pin
        # the dim so this should not happen).
        width = min(binary_matrix.shape[1], int(query_bits.shape[0]))
        popcount = numpy.asarray(_POPCOUNT_TABLE, dtype=numpy.uint16)
        hamming = numpy.empty(n, dtype=numpy.uint32)
        prescreen_rows = min(max(1, self.bounded_scan_rows), 4096)
        for start in range(0, n, prescreen_rows):
            if _prescreen_deadline_expired(deadline, start):
                raise _PrescreenDeadlineExpired(start)
            end = min(n, start + prescreen_rows)
            xor = numpy.bitwise_xor(
                binary_matrix[start:end, :width], query_bits[:width]
            )
            hamming[start:end] = popcount[xor].sum(axis=1)
            if _prescreen_deadline_expired(deadline, end):
                raise _PrescreenDeadlineExpired(end)
        m = min(n, max(k, self.knn_prescreen_multiplier * k))
        if m < n:
            survivors = numpy.argpartition(hamming, m - 1)[:m]
        else:
            survivors = numpy.arange(n)
        survivor_ids = [str(binary_ids[int(index)]) for index in survivors]
        if deadline is not None and _monotonic() >= deadline:
            raise _PrescreenDeadlineExpired(n)
        rowids, out_ids, kinds, matrix = self._load_vector_matrix(
            numpy,
            identity_hash,
            dim,
            survivor_ids,
            dtype,
            chunk=chunk,
        )
        if deadline is not None and _monotonic() >= deadline:
            raise _PrescreenDeadlineExpired(n)
        if matrix.shape[0] == 0:
            return []
        scores = matrix @ numpy.asarray(query, dtype=numpy.float32)
        if deadline is not None and _monotonic() >= deadline:
            raise _PrescreenDeadlineExpired(n)
        return self._ranked(rowids, out_ids, kinds, scores, k)

    def knn(
        self,
        query_vec: Sequence[float],
        k: int = 50,
        model: str | None = None,
        since: float | None = None,
        until: float | None = None,
        conversation_ids: Sequence[str] | None = None,
        source: str | None = None,
        provider: str | None = None,
        full_scan: bool = False,
        scan_max_rows: int = 0,
        scan_budget_s: float = 0.0,
        deadline: float | None = None,
        access_scope: str | None = None,
    ) -> KNNResult:
        """Summary KNN with independent reach and scoring disclosure.

        ``coverage`` reports how much of the requested candidate set was
        reached. ``scoring`` is ``float32_exact`` on exact float paths and
        ``int8_quantized`` whenever int8 values are scored, including resident
        quantize-on-load scans whose reach remains ``coverage="full"``.
        """
        operation_started = _monotonic()
        k = int(k)
        if k <= 0:
            return KNNResult(coverage="none")
        summary_columns = self._table_columns("summary_nodes")
        if conversation_ids is not None and "session_id" not in summary_columns:
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        if (since is not None or until is not None) and "latest_at" not in summary_columns:
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        if source is not None and "source" not in self._table_columns("messages"):
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        # Resolve by the FULL configured identity (provider + model) when the
        # caller knows its provider, so a config switch A→B for the same model
        # name scores the query against B's vectors, not whichever provider is
        # active. An unbackfilled identity resolves to None → coverage "none" →
        # the caller degrades to full_text.
        profile = (
            self._resolve_profile(str(model), provider=provider)
            if model is not None
            else self._current_profile()
        )
        if profile is None:
            return KNNResult(coverage="none")
        identity = str(profile["identity_hash"])
        dim = int(profile["dim"])
        dtype = str(profile["dtype"])
        exact_scoring = (
            "int8_quantized" if dtype == _INT8_DTYPE else "float32_exact"
        )
        query = self._normalized(query_vec, expected_dim=dim)
        try:
            numpy = _load_numpy()
        except ImportError:
            numpy = None

        resident_eligible = (
            numpy is not None
            and full_scan
            and since is None
            and until is None
            and conversation_ids is None
            and source is None
            # An owner-scoped query takes the exact path, for the same reason it
            # skips the sign-bit prescreen below: the resident matrix is a whole-
            # identity snapshot, POOLED AND CACHED on (identity, data_version)
            # only, so it cannot express a per-row owner predicate and a scoped
            # query would rank across every principal's vectors.
            and access_scope is None
            and not self._scan_bounds_requested(scan_max_rows, scan_budget_s)
        )
        if resident_eligible:
            try:
                resident = self._resident_int8_matrix(
                    numpy,
                    identity,
                    dim,
                    dtype,
                    data_version=int(profile["data_version"]),
                    chunk=False,
                    deadline=deadline,
                )
                if resident is not None:
                    candidates = self._rank_resident_int8(
                        numpy,
                        resident,
                        numpy.asarray(query, dtype=numpy.float32),
                        k,
                        deadline=deadline,
                    )
                    return KNNResult(
                        candidates,
                        coverage="full",
                        scoring="int8_quantized",
                    )
            except _PrescreenDeadlineExpired as exc:
                return KNNResult(
                    coverage="bounded",
                    scoring="int8_quantized",
                    scanned=exc.scanned,
                )

        # Two-stage full-corpus path when this identity's sign-bit prescreen is a
        # COMPLETE mirror of its vectors. A partial binary corpus (FIX 1) stays on
        # the exact scan below so no vector is silently INNER-JOINed away. The
        # source filter stays on the exact bounded path (its lineage walk is
        # budget-bounded and does not scale to a full-corpus scan). A requested
        # hard bound also routes to the exact path, which is the only one that
        # can enforce and disclose it.
        if (
            numpy is not None
            and source is None
            # An owner-scoped query takes the exact path for the same reason a
            # source-filtered one does: the binary prescreen mirrors the whole
            # corpus and cannot express a per-row filter, so running it would
            # prescreen across every principal before any scoping applied.
            and access_scope is None
            and not (full_scan and dtype == _INT8_DTYPE)
            and not self._scan_bounds_requested(scan_max_rows, scan_budget_s)
            and self._binary_fully_synced(identity, chunk=False)
        ):
            unfiltered = since is None and until is None and conversation_ids is None
            try:
                binary_ids, binary_matrix = self._cached_binary_matrix(
                    self._binary_matrix_cache,
                    identity,
                    lambda: self._load_binary_with_deadline(
                        lambda reader: reader._load_embedding_binary_matrix(
                            numpy,
                            identity,
                            since=since,
                            until=until,
                            conversation_ids=conversation_ids,
                        ),
                        deadline,
                    ),
                    cacheable=unfiltered,
                    deadline=deadline,
                )
            except _PrescreenDeadlineExpired as exc:
                return KNNResult(
                    coverage="bounded",
                    scoring=exact_scoring,
                    scanned=exc.scanned,
                )
            if binary_matrix.shape[0] == 0:
                return KNNResult(coverage="none", scoring=exact_scoring)
            try:
                candidates = self._two_stage_rank(
                    numpy,
                    identity,
                    dim,
                    dtype,
                    query,
                    k,
                    binary_ids,
                    binary_matrix,
                    chunk=False,
                    deadline=deadline,
                )
            except _PrescreenDeadlineExpired as exc:
                return KNNResult(
                    coverage="bounded",
                    scoring=exact_scoring,
                    scanned=exc.scanned,
                    total=len(binary_ids),
                )
            # coverage='full_approx' (FIX 2): the whole corpus is REACHED, but
            # stage-1 Hamming keeps only M=mult*k survivors, so top-k is an
            # approximate (recall@M) result, not exact like the exact-scan 'full'.
            return KNNResult(
                candidates,
                coverage="full_approx",
                scoring=exact_scoring,
            )

        probe_limit, scan_limit = self._scan_limits(
            full_scan=full_scan, scan_max_rows=scan_max_rows
        )
        scan_deadline = deadline
        if scan_budget_s > 0:
            budget_deadline = operation_started + scan_budget_s
            scan_deadline = (
                budget_deadline
                if scan_deadline is None
                else min(scan_deadline, budget_deadline)
            )
        # Probe one past the scan limit through the indexed candidate query.
        # This determines full-vs-bounded coverage without COUNT(*) scanning the
        # entire identity on every request.
        try:
            probed_ids = self._read_with_deadline(
                lambda reader: reader._bounded_candidate_ids(
                    identity,
                    since=since,
                    until=until,
                    conversation_ids=conversation_ids,
                    source=source,
                    limit=probe_limit,
                    access_scope=access_scope,
                ),
                scan_deadline,
            )
        except _UnverifiableProvenance:
            return KNNResult(
                coverage="none",
                scoring=exact_scoring,
                reason="unverifiable_provenance",
            )
        except _PrescreenDeadlineExpired as exc:
            deadline_expired = _caller_deadline_expired(
                deadline,
                stop_deadline=scan_deadline,
                stop_expired=True,
            )
            return self._scanned_knn_result(
                coverage="bounded",
                scoring=exact_scoring,
                scanned_rows=exc.scanned,
                deadline_expired=deadline_expired,
                identity_hash=identity,
                chunk=False,
            )
        if not probed_ids:
            return KNNResult(coverage="none", scoring=exact_scoring)
        scan_ids = probed_ids if scan_limit is None else probed_ids[:scan_limit]
        candidate_coverage = (
            "bounded"
            if source is not None
            or (scan_limit is not None and len(probed_ids) > scan_limit)
            else "full"
        )

        if numpy is not None:
            query_array = numpy.asarray(query, dtype=numpy.float32)

            # Deadline-aware int8 decoding must preserve its per-batch scored-row
            # accounting. Small float32 scans keep the retained single-batch LRU
            # path so pooled recall calls still warm and reuse their matrix.
            if (
                deadline is not None and dtype == _INT8_DTYPE
            ) or self._use_streaming_scan(
                len(scan_ids), max(1, self.bounded_scan_rows)
            ):
                candidates, scanned_rows, stopped_early, deadline_expired = (
                    self._scan_vectorized_ranked(
                        numpy=numpy,
                        identity_hash=identity,
                        dim=dim,
                        dtype=dtype,
                        candidate_ids=scan_ids,
                        chunk=False,
                        batch_rows=max(1, self.bounded_scan_rows),
                        budget_s=scan_budget_s,
                        deadline=scan_deadline,
                        caller_deadline=deadline,
                        limit=k,
                        query=query_array,
                    )
                )
                coverage = "bounded" if stopped_early else candidate_coverage
                return self._scanned_knn_result(
                    candidates,
                    coverage=coverage,
                    scoring=exact_scoring,
                    scanned_rows=scanned_rows,
                    deadline_expired=deadline_expired,
                    identity_hash=identity,
                    chunk=False,
                    complete_total=(len(scan_ids) if scan_deadline is not None else None),
                )

            def score_batch(
                batch_ids: Sequence[str], cache: bool
            ) -> tuple[Any, Any, Any, Any]:
                rowids, embedded_ids, kinds, matrix = self._numpy_rows(
                    numpy,
                    identity,
                    dim,
                    batch_ids,
                    dtype,
                    cache=cache,
                )
                return rowids, embedded_ids, kinds, matrix @ query_array
        else:
            # The candidate enumeration itself runs at the SQL layer: the column
            # filters + ORDER BY source latest_at DESC run inside SQLite, and
            # only ONE batch of vectors is decoded into host memory at a time.
            # Filters live in the WHERE clause, so a filtered match is never
            # lost; only the source-lineage walk runs on the enumerated set.
            def score_batch(
                batch_ids: Sequence[str], _cache: bool
            ) -> tuple[Any, Any, Any, Any]:
                rowids, embedded_ids, kinds, vectors = self._load_vectors_for_ids(
                    identity,
                    dim,
                    batch_ids,
                    dtype,
                )
                return rowids, embedded_ids, kinds, [
                    sum(value * query_value for value, query_value in zip(vector, query))
                    for vector in vectors
                ]

        candidates, scanned_rows, stopped_early, deadline_expired = self._scan_ranked(
            candidate_ids=scan_ids,
            batch_rows=max(1, self.bounded_scan_rows),
            budget_s=scan_budget_s,
            deadline=scan_deadline,
            caller_deadline=deadline,
            limit=k,
            score_batch=score_batch,
        )
        coverage = "bounded" if stopped_early else candidate_coverage
        return self._scanned_knn_result(
            candidates,
            coverage=coverage,
            scoring=exact_scoring,
            scanned_rows=scanned_rows,
            deadline_expired=deadline_expired,
            identity_hash=identity,
            chunk=False,
            complete_total=(len(scan_ids) if scan_deadline is not None else None),
        )

    def _scan_limits(
        self, *, full_scan: bool, scan_max_rows: int
    ) -> tuple[int, int | None]:
        """Resolve ``(candidate probe limit, hard scan cap)`` for one KNN call.

        Default (``full_scan=False``) keeps the historical recency window:
        ``bounded_scan_rows`` candidates, probed one deeper so a larger corpus
        reports ``coverage='bounded'``. ``full_scan=True`` (the lcm_recall
        contract) enumerates the WHOLE corpus and uses ``bounded_scan_rows``
        only as the scan's batch size; ``scan_max_rows`` (0 = unlimited) is the
        escape hatch for a pathological corpus and is disclosed as bounded
        exactly like the recency window was.
        """
        if not full_scan:
            bound = max(0, self.bounded_scan_rows)
            return bound + 1, bound
        if scan_max_rows > 0:
            return scan_max_rows + 1, scan_max_rows
        return _SCAN_ALL_ROWS, None

    @staticmethod
    def _scan_bounds_requested(scan_max_rows: int, scan_budget_s: float) -> bool:
        """Whether the caller asked for a hard bound on this scan.

        The two-stage binary-prescreen path can honor neither: it is one matmul
        over the whole binary mirror, with no candidate window to cap and no
        batch boundary to stop at. When a bound is requested we therefore route
        to the exact batched scan, which enforces it and reports the bounded
        coverage. Both knobs default to 0, so an unbounded recall keeps the
        two-stage path and its ``full_approx`` disclosure unchanged.
        """
        return scan_max_rows > 0 or scan_budget_s > 0

    # -- Chunk corpus ------------------------------------------------------
    #
    # The chunk corpus mirrors the summary corpus exactly: identity-hashed
    # profiles (task='chunk') in the shared lcm_embedding_profile table, the
    # same bounded-candidate KNN + coverage contract, the same lease publication
    # CAS. It differs only in its source table (raw messages, keyed by store_id)
    # and its own meta/vectors tables. Chunk schema is materialized lazily on
    # first chunk-corpus use so a summary-only install never creates it.

    def ensure_chunk_schema(self) -> None:
        """Materialize (and verify) the opt-in chunk tables on first chunk use.

        Same discipline as ``_ensure_embedding_schema``: the ``chunk_vectors_v1``
        marker is not trusted on its own; the tables/indexes are re-ensured and
        VERIFIED, and only a fully-materialized schema keeps the marker.
        """
        if self._chunk_schema_ready:
            return
        with self._write_lock:
            ensure_embedding_tables(self._conn)
            ensure_chunk_tables(self._conn)
            errors = verify_chunk_schema(self._conn)
            if errors:
                ensure_chunk_tables(self._conn)
                errors = verify_chunk_schema(self._conn)
                if errors:
                    raise sqlite3.OperationalError(
                        "chunk schema incompatible after ensure: " + "; ".join(errors)
                    )
            # Read-before-write: skip the needless marker re-write + commit when
            # already stamped and verified clean (sprint-opt-1).
            if not self._migration_step_present("chunk_vectors_v1"):
                mark_migration_step_complete(self._conn, "chunk_vectors_v1")
                self._conn.commit()
        self._chunk_schema_ready = True

    def _write_chunk_row(
        self,
        chunk_id: str,
        *,
        store_id: int,
        chunk_index: int,
        char_start: int,
        char_end: int,
        token_estimate: int,
        vec: Sequence[float],
        identity: EmbeddingIdentity,
        profile: sqlite3.Row,
    ) -> None:
        identity_hash = identity.identity_hash
        normalized, packed, sign_bits = self._encode_stored_vector(vec, profile)
        embedded_at = self._now()
        source_message = self._conn.execute(
            "SELECT access_scope FROM messages WHERE store_id = ?",
            (int(store_id),),
        ).fetchone()
        access_scope = source_message[0] if source_message is not None else None
        self._conn.execute(
            "DELETE FROM lcm_chunk_vectors WHERE chunk_id = ? AND identity_hash = ?",
            (chunk_id, identity_hash),
        )
        self._conn.execute(
            "DELETE FROM lcm_chunk_meta WHERE chunk_id = ? AND identity_hash = ?",
            (chunk_id, identity_hash),
        )
        self._conn.execute(
            "DELETE FROM lcm_chunk_binary WHERE chunk_id = ? AND identity_hash = ?",
            (chunk_id, identity_hash),
        )
        self._conn.execute(
            "INSERT INTO lcm_chunk_vectors(chunk_id, identity_hash, vec, access_scope) VALUES(?, ?, ?, ?)",
            (chunk_id, identity_hash, packed, access_scope),
        )
        if sign_bits is not None:
            self._conn.execute(
                "INSERT INTO lcm_chunk_binary(chunk_id, identity_hash, bits, access_scope) "
                "VALUES(?, ?, ?, ?)",
                (chunk_id, identity_hash, sign_bits, access_scope),
            )
        self._conn.execute(
            """
            INSERT INTO lcm_chunk_meta(
                chunk_id, identity_hash, store_id, chunk_index, char_start,
                char_end, token_estimate, embedded_at, archived, access_scope
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                chunk_id,
                identity_hash,
                int(store_id),
                int(chunk_index),
                int(char_start),
                int(char_end),
                int(token_estimate),
                embedded_at,
                access_scope,
            ),
        )
        self._bump_data_version(identity_hash)

    def record_chunk_embedding(
        self,
        chunk_id: str,
        model: str,
        vec: Sequence[float],
        *,
        store_id: int,
        chunk_index: int,
        char_start: int,
        char_end: int,
        token_estimate: int,
        identity: EmbeddingIdentity,
    ) -> None:
        self.ensure_chunk_schema()
        with self._write_transaction():
            profile = self._validate_embedding_write(identity, model=model)
            self._write_chunk_row(
                str(chunk_id),
                store_id=store_id,
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
                token_estimate=token_estimate,
                vec=vec,
                identity=identity,
                profile=profile,
            )

    def publish_chunk_embedding_under_lease(
        self,
        chunk_id: str,
        model: str,
        vec: Sequence[float],
        *,
        store_id: int,
        chunk_index: int,
        char_start: int,
        char_end: int,
        token_estimate: int,
        identity: EmbeddingIdentity,
        claim_key: str,
        lease_id: str,
        generation: int,
        request_id: str,
    ) -> EmbeddingPublishOutcome:
        """Atomically revalidate ownership and publish one accepted chunk vector."""
        chunk_id = str(chunk_id)
        return self._publish_under_lease(
            chunk_id,
            model=model,
            identity=identity,
            claim_key=claim_key,
            lease_id=lease_id,
            generation=generation,
            request_id=request_id,
            write_row=lambda profile: self._write_chunk_row(
                chunk_id,
                store_id=store_id,
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
                token_estimate=token_estimate,
                vec=vec,
                identity=identity,
                profile=profile,
            ),
        )

    def publish_chunk_embedding_batch_under_lease(
        self,
        rows: Sequence[dict],
        *,
        model: str,
        identity: EmbeddingIdentity,
        claim_key: str,
        lease_id: str,
        generation: int,
    ) -> list[BatchPublishResult]:
        """Publish an accepted network batch of chunk vectors in ONE transaction.

        Chunk analogue of :meth:`publish_embedding_batch_under_lease`. Each row
        dict carries ``chunk_id``, ``vec``, ``request_id`` and the chunk-meta
        fields (``store_id``/``chunk_index``/``char_start``/``char_end``/
        ``token_estimate``); each publishes under its own savepoint via the
        single-row ``publish_chunk_embedding_under_lease``.
        """
        return self._run_publish_batch(
            [
                (
                    str(row["chunk_id"]),
                    lambda row=row: self.publish_chunk_embedding_under_lease(
                        row["chunk_id"],
                        model,
                        row["vec"],
                        store_id=row["store_id"],
                        chunk_index=row["chunk_index"],
                        char_start=row["char_start"],
                        char_end=row["char_end"],
                        token_estimate=row["token_estimate"],
                        identity=identity,
                        claim_key=claim_key,
                        lease_id=lease_id,
                        generation=generation,
                        request_id=row["request_id"],
                    ),
                )
                for row in rows
            ]
        )

    def archive_chunks_for_messages(self, store_ids: Sequence[int | str]) -> int:
        """Soft-archive chunks whose source messages were purged/GC'd.

        Mirrors ``purge_embeddings_for_nodes`` but ARCHIVES (archived=1) rather
        than deletes, matching the meta.archived contract KNN filters on: an
        archived chunk is dropped from ranking but its row survives for bounded
        recovery. No-op (0) when the chunk schema was never materialized.
        """
        if not self._chunk_tables_exist():
            return 0
        unique_ids = list(dict.fromkeys(int(store_id) for store_id in store_ids))
        if not unique_ids:
            return 0
        with self._write_transaction():
            with self._temp_id_table([str(value) for value in unique_ids]) as table:
                cur = self._conn.execute(
                    f"UPDATE lcm_chunk_meta SET archived = 1 "
                    f"WHERE archived = 0 AND store_id IN "
                    f"(SELECT CAST(id AS INTEGER) FROM {table})"
                )
            if cur.rowcount:
                self._bump_all_data_versions()
        return int(cur.rowcount or 0)

    @staticmethod
    def archive_chunks_for_messages_on_connection(
        conn: sqlite3.Connection, store_ids: Sequence[int | str]
    ) -> int:
        """Archive one already-bounded message batch in the caller's transaction."""
        unique_ids = list(dict.fromkeys(int(store_id) for store_id in store_ids))
        if not unique_ids:
            return 0
        if len(unique_ids) > 256:
            raise ValueError("chunk archive batch exceeds 256 rows")
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ("
                "'lcm_chunk_meta','lcm_embedding_profile')"
            ).fetchall()
        }
        if not {"lcm_chunk_meta", "lcm_embedding_profile"} <= tables:
            return 0
        placeholders = ",".join("?" for _ in unique_ids)
        cur = conn.execute(
            f"UPDATE lcm_chunk_meta SET archived = 1 "
            f"WHERE archived = 0 AND store_id IN ({placeholders})",
            unique_ids,
        )
        if cur.rowcount:
            conn.execute(
                "UPDATE lcm_embedding_profile SET data_version = data_version + 1"
            )
        return int(cur.rowcount or 0)

    def _chunk_tables_exist(self) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_chunk_meta'"
        ).fetchone() is not None

    def _bounded_chunk_candidate_ids(
        self,
        identity_hash: str,
        *,
        since: float | None,
        until: float | None,
        conversation_ids: Sequence[str] | None,
        source: str | None,
        limit: int,
        access_scope: str | None = None,
    ) -> list[str]:
        """Enumerate at most ``limit`` live chunk ids, most-recent-message first.

        The candidate enumeration is bounded at the SQL layer exactly like the
        summary path: recency (message timestamp), conversation (message
        session_id), and source (message source) filters live in the WHERE
        clause and a hard LIMIT caps the result. Unlike the summary corpus a
        chunk maps DIRECTLY to one raw message, so the source filter is a plain
        indexed column match (no recursive lineage walk) and needs no
        fail-open/closed budget.

        ``limit=_SCAN_ALL_ROWS`` enumerates every live candidate (the batched
        full-corpus scan bounds memory per BATCH, not per corpus).
        """
        if limit == 0:
            return []
        if conversation_ids is not None and not list(conversation_ids):
            return []
        message_columns = self._table_columns("messages")
        if not message_columns:
            return []
        where = ["cm.identity_hash = ?", "cm.archived = 0"]
        args: list[object] = [str(identity_hash)]
        if since is not None:
            where.append("m.timestamp >= ?")
            args.append(float(since))
        if until is not None:
            where.append("m.timestamp <= ?")
            args.append(float(until))
        if source is not None:
            where.append("m.source = ?")
            args.append(str(source))
        if access_scope is not None:
            # Owner predicate for the CHUNK corpus, mirroring the summary path.
            # In the WHERE clause so it is enforced before the bound: a filter
            # applied after LIMIT returns another principal's rows exactly when
            # the bound bites, which is when it matters most.
            where.append("cm.access_scope = ?")
            args.append(str(access_scope))
        args.append(int(limit))
        with self._optional_temp_id_table(conversation_ids) as conversation_table:
            conversation_join = (
                f"JOIN {conversation_table} c ON c.id = m.session_id"
                if conversation_table is not None
                else ""
            )
            rows = self._conn.execute(
                f"""
                SELECT cm.chunk_id
                FROM lcm_chunk_meta cm
                JOIN messages m ON m.store_id = cm.store_id
                {conversation_join}
                WHERE {' AND '.join(where)}
                ORDER BY m.timestamp DESC, cm.chunk_id DESC
                LIMIT ?
                """,
                args,
            ).fetchall()
        return [str(row[0]) for row in rows]

    def _load_chunk_vectors_for_ids(
        self,
        identity_hash: str,
        dim: int,
        chunk_ids: Sequence[str],
        dtype: str = _VECTOR_DTYPE,
    ) -> tuple[list[int], list[str], list[str], list[list[float]]]:
        """Pre-R1 loader for a bounded, already-filtered chunk id set."""
        rowids: list[int] = []
        out_ids: list[str] = []
        kinds: list[str] = []
        vectors: list[list[float]] = []
        if not chunk_ids:
            return rowids, out_ids, kinds, vectors
        with self._temp_id_table(chunk_ids) as table:
            rows = self._conn.execute(
                f"""
                SELECT v.rowid, v.chunk_id, v.vec
                FROM {table} t
                JOIN lcm_chunk_vectors v
                  ON v.chunk_id = t.id AND v.identity_hash = ?
                JOIN lcm_chunk_meta m
                  ON m.chunk_id = v.chunk_id
                 AND m.identity_hash = v.identity_hash
                WHERE m.archived = 0
                """,
                (identity_hash,),
            ).fetchall()
        for row in rows:
            try:
                vector = self._decode_stored_vec(bytes(row["vec"]), dim, dtype)
            except (TypeError, ValueError):
                continue
            if vector is None:
                continue
            rowids.append(int(row["rowid"]))
            out_ids.append(str(row["chunk_id"]))
            kinds.append("chunk")
            vectors.append(list(vector))
        return rowids, out_ids, kinds, vectors

    def _numpy_chunk_rows(
        self,
        numpy: Any,
        identity_hash: str,
        dim: int,
        chunk_ids: Sequence[str],
        dtype: str = _VECTOR_DTYPE,
        cache: bool = True,
    ) -> tuple[list[int], list[str], list[str], Any]:
        if not cache:
            return self._load_chunk_matrix(numpy, identity_hash, dim, chunk_ids, dtype)
        with self._cache_lock:
            data_version = self._data_version(identity_hash)
            key = (identity_hash, data_version, tuple(str(value) for value in chunk_ids))
            cached = self._chunk_matrix_cache.get(key)
            if cached is not None:
                self._chunk_matrix_cache.move_to_end(key)  # mark most-recently used
                return cached
            loaded = self._load_chunk_matrix(numpy, identity_hash, dim, chunk_ids, dtype)
            self._chunk_matrix_cache[key] = loaded
            while len(self._chunk_matrix_cache) > self._MATRIX_CACHE_MAX_ENTRIES:
                self._chunk_matrix_cache.popitem(last=False)  # evict oldest
            return loaded

    def _load_chunk_matrix(
        self,
        numpy: Any,
        identity_hash: str,
        dim: int,
        chunk_ids: Sequence[str],
        dtype: str,
    ) -> tuple[list[int], list[str], list[str], Any]:
        """Decode one chunk candidate set through the retained pre-R1 loader."""
        rowids, loaded_ids, kinds, raw_vectors = self._load_chunk_vectors_for_ids(
            identity_hash, dim, chunk_ids, dtype
        )
        matrix = (
            numpy.asarray(raw_vectors, dtype=numpy.float32)
            if raw_vectors
            else numpy.empty((0, dim), dtype=numpy.float32)
        )
        return rowids, loaded_ids, kinds, matrix

    def knn_chunks(
        self,
        query_vec: Sequence[float],
        k: int = 50,
        model: str | None = None,
        since: float | None = None,
        until: float | None = None,
        conversation_ids: Sequence[str] | None = None,
        source: str | None = None,
        provider: str | None = None,
        full_scan: bool = False,
        scan_max_rows: int = 0,
        scan_budget_s: float = 0.0,
        deadline: float | None = None,
        access_scope: str | None = None,
    ) -> KNNResult:
        """Chunk KNN with the summary reach/scoring contract.

        Coverage is full|bounded|none exactly as for summaries: ``none`` when
        the corpus/identity is unbackfilled or a requested filter is
        unverifiable (missing message column), ``bounded`` when the scan was
        cut short by a hard cap, latency budget, or operation deadline, and
        ``full`` when the requested candidate set was scanned completely.
        Scoring is disclosed separately as ``float32_exact`` or
        ``int8_quantized``.
        """
        operation_started = _monotonic()
        k = int(k)
        if k <= 0:
            return KNNResult(coverage="none")
        if not self._chunk_tables_exist():
            return KNNResult(coverage="none")
        message_columns = self._table_columns("messages")
        if conversation_ids is not None and "session_id" not in message_columns:
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        if (since is not None or until is not None) and "timestamp" not in message_columns:
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        if source is not None and "source" not in message_columns:
            return KNNResult(coverage="none", reason="unverifiable_provenance")
        profile = (
            self._resolve_chunk_profile(str(model), provider=provider)
            if model is not None
            else self._current_chunk_profile()
        )
        if profile is None:
            return KNNResult(coverage="none")
        identity = str(profile["identity_hash"])
        dim = int(profile["dim"])
        dtype = str(profile["dtype"])
        exact_scoring = (
            "int8_quantized" if dtype == _INT8_DTYPE else "float32_exact"
        )
        query = self._normalized(query_vec, expected_dim=dim)
        try:
            numpy = _load_numpy()
        except ImportError:
            numpy = None

        resident_eligible = (
            numpy is not None
            and full_scan
            and since is None
            and until is None
            and conversation_ids is None
            and source is None
            # An owner-scoped query takes the exact path, for the same reason it
            # skips the sign-bit prescreen below: the resident matrix is a whole-
            # identity snapshot, POOLED AND CACHED on (identity, data_version)
            # only, so it cannot express a per-row owner predicate and a scoped
            # query would rank across every principal's vectors.
            and access_scope is None
            and not self._scan_bounds_requested(scan_max_rows, scan_budget_s)
        )
        if resident_eligible:
            try:
                resident = self._resident_int8_matrix(
                    numpy,
                    identity,
                    dim,
                    dtype,
                    data_version=int(profile["data_version"]),
                    chunk=True,
                    deadline=deadline,
                )
                if resident is not None:
                    candidates = self._rank_resident_int8(
                        numpy,
                        resident,
                        numpy.asarray(query, dtype=numpy.float32),
                        k,
                        deadline=deadline,
                    )
                    return KNNResult(
                        candidates,
                        coverage="full",
                        scoring="int8_quantized",
                    )
            except _PrescreenDeadlineExpired as exc:
                return KNNResult(
                    coverage="bounded",
                    scoring="int8_quantized",
                    scanned=exc.scanned,
                )

        # Two-stage full-corpus path when this chunk identity's sign-bit prescreen
        # is a COMPLETE mirror of its vectors (FIX 1 — a partial binary corpus
        # would be silently truncated by the INNER JOIN). Chunk filters are plain
        # indexed message columns (timestamp/session_id/source), so they all apply
        # in the binary load over the whole (filtered) corpus. A requested hard
        # bound routes to the exact path, which is the only one that can enforce
        # and disclose it.
        if (
            numpy is not None
            # An owner-scoped query takes the exact path, exactly as the summary
            # arm above already required. The binary prescreen mirrors the WHOLE
            # corpus and cannot express a per-row filter, so running it here
            # prescreened across every principal before any scoping applied.
            # This guard was present on the summary twin and absent here, so a
            # scoped CHUNK query silently ranked against other principals'
            # vectors -- `knn_chunks` accepts `access_scope` and this path
            # ignored it.
            and access_scope is None
            and not (full_scan and dtype == _INT8_DTYPE)
            and not self._scan_bounds_requested(scan_max_rows, scan_budget_s)
            and self._binary_fully_synced(identity, chunk=True)
        ):
            unfiltered = (
                since is None and until is None
                and conversation_ids is None and source is None
            )
            try:
                binary_ids, binary_matrix = self._cached_binary_matrix(
                    self._chunk_binary_matrix_cache,
                    identity,
                    lambda: self._load_binary_with_deadline(
                        lambda reader: reader._load_chunk_binary_matrix(
                            numpy,
                            identity,
                            since=since,
                            until=until,
                            conversation_ids=conversation_ids,
                            source=source,
                        ),
                        deadline,
                    ),
                    cacheable=unfiltered,
                    deadline=deadline,
                )
            except _PrescreenDeadlineExpired as exc:
                return KNNResult(
                    coverage="bounded",
                    scoring=exact_scoring,
                    scanned=exc.scanned,
                )
            if binary_matrix.shape[0] == 0:
                return KNNResult(coverage="none", scoring=exact_scoring)
            try:
                candidates = self._two_stage_rank(
                    numpy,
                    identity,
                    dim,
                    dtype,
                    query,
                    k,
                    binary_ids,
                    binary_matrix,
                    chunk=True,
                    deadline=deadline,
                )
            except _PrescreenDeadlineExpired as exc:
                return KNNResult(
                    coverage="bounded",
                    scoring=exact_scoring,
                    scanned=exc.scanned,
                    total=len(binary_ids),
                )
            # coverage='full_approx' (FIX 2): whole corpus reached, but stage-1
            # keeps only M=mult*k survivors -> approximate top-k, not exact.
            return KNNResult(
                candidates,
                coverage="full_approx",
                scoring=exact_scoring,
            )

        probe_limit, scan_limit = self._scan_limits(
            full_scan=full_scan, scan_max_rows=scan_max_rows
        )
        scan_deadline = deadline
        if scan_budget_s > 0:
            budget_deadline = operation_started + scan_budget_s
            scan_deadline = (
                budget_deadline
                if scan_deadline is None
                else min(scan_deadline, budget_deadline)
            )
        try:
            probed_ids = self._read_with_deadline(
                lambda reader: reader._bounded_chunk_candidate_ids(
                    identity,
                    since=since,
                    until=until,
                    conversation_ids=conversation_ids,
                    source=source,
                    limit=probe_limit,
                    access_scope=access_scope,
                ),
                scan_deadline,
            )
        except _PrescreenDeadlineExpired as exc:
            deadline_expired = _caller_deadline_expired(
                deadline,
                stop_deadline=scan_deadline,
                stop_expired=True,
            )
            return self._scanned_knn_result(
                coverage="bounded",
                scoring=exact_scoring,
                scanned_rows=exc.scanned,
                deadline_expired=deadline_expired,
                identity_hash=identity,
                chunk=True,
            )
        if not probed_ids:
            return KNNResult(coverage="none", scoring=exact_scoring)
        scan_ids = probed_ids if scan_limit is None else probed_ids[:scan_limit]
        candidate_coverage = (
            "bounded"
            if scan_limit is not None and len(probed_ids) > scan_limit
            else "full"
        )

        if numpy is not None:
            query_array = numpy.asarray(query, dtype=numpy.float32)

            # Match the summary arm: int8 deadlines use the scored-row-aware
            # streaming decoder; small float32 calls retain their pooled LRU.
            if (
                deadline is not None and dtype == _INT8_DTYPE
            ) or self._use_streaming_scan(
                len(scan_ids), max(1, self.bounded_scan_rows)
            ):
                candidates, scanned_rows, stopped_early, deadline_expired = (
                    self._scan_vectorized_ranked(
                        numpy=numpy,
                        identity_hash=identity,
                        dim=dim,
                        dtype=dtype,
                        candidate_ids=scan_ids,
                        chunk=True,
                        batch_rows=max(1, self.bounded_scan_rows),
                        budget_s=scan_budget_s,
                        deadline=scan_deadline,
                        caller_deadline=deadline,
                        limit=k,
                        query=query_array,
                    )
                )
                coverage = "bounded" if stopped_early else candidate_coverage
                return self._scanned_knn_result(
                    candidates,
                    coverage=coverage,
                    scoring=exact_scoring,
                    scanned_rows=scanned_rows,
                    deadline_expired=deadline_expired,
                    identity_hash=identity,
                    chunk=True,
                    complete_total=(len(scan_ids) if scan_deadline is not None else None),
                )

            def score_batch(
                batch_ids: Sequence[str], cache: bool
            ) -> tuple[Any, Any, Any, Any]:
                rowids, chunk_ids, kinds, matrix = self._numpy_chunk_rows(
                    numpy, identity, dim, batch_ids, dtype, cache=cache
                )
                return rowids, chunk_ids, kinds, matrix @ query_array
        else:
            def score_batch(
                batch_ids: Sequence[str], _cache: bool
            ) -> tuple[Any, Any, Any, Any]:
                rowids, chunk_ids, kinds, vectors = self._load_chunk_vectors_for_ids(
                    identity, dim, batch_ids, dtype
                )
                return rowids, chunk_ids, kinds, [
                    sum(value * query_value for value, query_value in zip(vector, query))
                    for vector in vectors
                ]

        candidates, scanned_rows, stopped_early, deadline_expired = self._scan_ranked(
            candidate_ids=scan_ids,
            batch_rows=max(1, self.bounded_scan_rows),
            budget_s=scan_budget_s,
            deadline=scan_deadline,
            caller_deadline=deadline,
            limit=k,
            score_batch=score_batch,
        )
        coverage = "bounded" if stopped_early else candidate_coverage
        return self._scanned_knn_result(
            candidates,
            coverage=coverage,
            scoring=exact_scoring,
            scanned_rows=scanned_rows,
            deadline_expired=deadline_expired,
            identity_hash=identity,
            chunk=True,
            complete_total=(len(scan_ids) if scan_deadline is not None else None),
        )

    def _current_chunk_profile(self) -> sqlite3.Row | None:
        """The active profile registered under task='chunk' (most recent)."""
        return self._conn.execute(
            """
            SELECT identity_hash, model_name, provider, revision, dim, dtype,
                   byteorder, task, registered_at, active, archived_at, data_version
            FROM lcm_embedding_profile
            WHERE active = 1 AND archived_at IS NULL AND task = ?
            ORDER BY registered_at DESC, identity_hash DESC
            LIMIT 1
            """,
            (_CHUNK_TASK,),
        ).fetchone()

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass
            conn.close()
            self._conn = None
        with self._cache_lock:
            self._matrix_cache.clear()
            self._chunk_matrix_cache.clear()
            self._binary_matrix_cache.clear()
            self._chunk_binary_matrix_cache.clear()
        with _RESIDENT_MATRIX_REGISTRY_LOCK:
            if getattr(self, "_resident_registry_is_memory", False):
                _drop_resident_registry_db_locked(
                    self._resident_registry_db_path,
                    self._resident_registry_incarnation,
                )
                for build_key in list(_RESIDENT_MATRIX_BUILD_LOCKS):
                    if (
                        build_key[0] == self._resident_registry_db_path
                        and build_key[1:3]
                        == self._resident_registry_incarnation
                    ):
                        _RESIDENT_MATRIX_BUILD_LOCKS.pop(build_key, None)
            self._resident_matrix_cache.clear()
            self._resident_ineligible_cache.clear()
            _RESIDENT_MATRIX_INSTANCES.discard(self)

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass
