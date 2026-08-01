"""Shared retrieval plumbing for LCM search tools.

This module factors the retrieval/fusion core out of ``tools.py`` so that
``lcm_grep`` (and the forthcoming ``lcm_recall``) call one engine instead of
duplicating ranking logic. Everything here is a pure move from ``tools.py`` —
callers keep their existing contracts, guards, and error strings; only the
plumbing relocated.

Guards (role/time/conversation/scope/content_scope degrade matrix) intentionally
stay in ``tools.py``: they are contract, not plumbing. So do the bounded worker /
deadline machinery (``_run_within_deadline`` and its semaphores) and the
provider-resolution / query-embedding steps, because their module-level names are
monkeypatched through the ``tools`` namespace and the semaphore default binding is
lexical to that module.
"""

from __future__ import annotations

import copy
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import LCMEngine


# -- Pooled VectorStore instances -------------------------------------------
# run_knn / run_chunk_knn used to build a fresh VectorStore per call and close()
# it in a finally, which cleared both matrix caches every time -- the "cache"
# never survived the single call that built it, so back-to-back identical recalls
# re-paid the full candidate-load + matmul (F2-matrix-cache-never-persists). We
# now keep a small LRU pool of long-lived stores keyed by (db_path, scan_rows).
# The per-identity/data_version cache keys already invalidate on any committed
# write (the durable data_version counter is bumped inside every vector write),
# so a pooled store observes cross-process writes and never serves stale vectors.
_POOL_MAX_PATHS = 2
_pool_lock = threading.Lock()
# (db_path, bounded_scan_rows) -> {"store": VectorStore, "lock": RLock}
_vector_store_pool: "OrderedDict[tuple[str, int], dict[str, Any]]" = OrderedDict()


def _reset_vector_store_pool() -> None:
    """Close and drop every pooled VectorStore (test hygiene / shutdown)."""
    with _pool_lock:
        while _vector_store_pool:
            _, entry = _vector_store_pool.popitem()
            try:
                entry["store"].close()
            except Exception:  # pragma: no cover - defensive
                pass


def _acquire_vector_store(
    engine: "LCMEngine", *, vector_store_cls: Any, scan_rows: int | None
) -> tuple[Any, Any, bool]:
    """Return ``(store, per_store_lock_or_None, is_transient)``.

    Only the genuine pooling-capable store (``_supports_pooling``) is pooled so
    its matrix caches survive across calls; injected test doubles are constructed
    transiently and the caller closes them. The pool is an LRU bounded to
    ``_POOL_MAX_PATHS`` (db_path, scan_rows) keys and an evicted store is closed.
    The returned lock MUST be held while querying: a pooled sqlite connection is
    shared across callers and is not safe for concurrent use.
    """
    resolved_scan = int(scan_rows) if scan_rows is not None else -1
    key = (str(engine._store.db_path), resolved_scan)
    with _pool_lock:
        entry = _vector_store_pool.get(key)
        if entry is not None:
            _vector_store_pool.move_to_end(key)
            return entry["store"], entry["lock"], False
        store = vector_store_cls(
            engine._store.db_path, config=engine._config, bounded_scan_rows=scan_rows
        )
        if not getattr(vector_store_cls, "_supports_pooling", False):
            return store, None, True  # transient: caller closes it
        entry = {"store": store, "lock": threading.RLock()}
        _vector_store_pool[key] = entry
        while len(_vector_store_pool) > _POOL_MAX_PATHS:
            _, evicted = _vector_store_pool.popitem(last=False)
            with evicted["lock"]:  # wait out any in-flight query before closing
                try:
                    evicted["store"].close()
                except Exception:  # pragma: no cover - defensive
                    pass
        return store, entry["lock"], False


def _run_pooled_knn(
    engine: "LCMEngine",
    *,
    vector_store_cls: Any,
    scan_rows: int | None,
    deadline: float,
    query: Any,
) -> Any:
    """Run ``query(store)`` on a pooled/transient store under its deadline guard.

    The per-call progress handler is installed for this deadline and cleared in a
    finally so a pooled connection never carries a stale/expired deadline into the
    next caller.
    """
    store, store_lock, transient = _acquire_vector_store(
        engine, vector_store_cls=vector_store_cls, scan_rows=scan_rows
    )
    try:
        with (store_lock if store_lock is not None else nullcontext()):
            vector_conn = getattr(store, "_conn", None)
            if vector_conn is not None:
                vector_conn.set_progress_handler(
                    lambda: 1 if time.monotonic() >= deadline else 0, 1000
                )
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError("semantic vector search deadline exhausted")
                try:
                    return query(store)
                except sqlite3.OperationalError as exc:
                    if "interrupt" in str(exc).lower() or time.monotonic() >= deadline:
                        raise TimeoutError(
                            "semantic vector search deadline exhausted"
                        ) from exc
                    raise
            finally:
                if vector_conn is not None:
                    vector_conn.set_progress_handler(None, 1000)
    finally:
        if transient:
            store.close()


def _lcm_grep_confidence(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.5:
        return "medium"
    if score >= 0.35:
        return "low"
    return "noise"


def _lcm_grep_deadline_error(mode: str, stage: str) -> dict[str, Any]:
    return {
        "error": "lcm_grep request deadline exceeded",
        "mode": mode,
        "timeout": True,
        "timeout_stage": stage,
    }


def _shape_message_hit(
    hit: Dict[str, Any],
    *,
    current_session_id: str | None,
    has_current_session: bool,
) -> dict[str, Any]:
    """Shape a raw MessageStore hit into an lcm_grep result row."""
    timestamp_value = hit.get("timestamp", 0) or 0
    return {
        "type": "message",
        "depth": "raw",
        "store_id": hit["store_id"],
        "session_id": hit["session_id"],
        "source": hit.get("source") or "",
        "conversation_id": hit.get("conversation_id") or "",
        "role": hit["role"],
        "timestamp": timestamp_value,
        "snippet": hit.get("snippet", hit.get("content", "")[:200]),
        "from_current_session": has_current_session
        and hit["session_id"] == current_session_id,
        "_sort_ts": timestamp_value,
        "_sort_rank": hit.get("search_rank"),
        "_sort_directness": hit.get("_directness_score") or 0.0,
    }


def _shape_summary_hit(node: Any) -> dict[str, Any]:
    """Shape a SummaryDAG node hit into an lcm_grep result row."""
    return {
        "type": "summary",
        "depth": f"d{node.depth}",
        "node_id": node.node_id,
        "session_id": node.session_id,
        "snippet": node.summary[:300],
        "token_count": node.token_count,
        "expand_hint": node.expand_hint,
        "earliest_at": node.earliest_at,
        "latest_at": node.latest_at,
        "from_current_session": True,
        "_sort_ts": node.latest_at or node.created_at,
        "_sort_rank": node.search_rank,
        "_sort_directness": node.search_directness or 0.0,
    }


def _resolve_semantic_conversation_scope(
    engine: "LCMEngine",
    *,
    search_session_id: str | None,
    conversation_id: str | None,
) -> list[str] | None:
    """Resolve the conversation filter to the session_ids KNN should allow.

    Summaries are keyed by session_id, so a message-level ``conversation_id`` is
    enforced by resolving it to the sessions that carry it (intersected with the
    active scope). Returns ``None`` for "no session constraint", or a possibly
    empty list when a conversation matches no sessions (which then degrades).
    """
    if not conversation_id:
        return [search_session_id] if search_session_id is not None else None
    try:
        rows = engine._store.connection.execute(
            "SELECT DISTINCT session_id FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        conv_sessions = {str(row[0]) for row in rows if row and row[0] is not None}
    except Exception:  # pragma: no cover - defensive; degrade on any store error
        conv_sessions = set()
    if search_session_id is not None:
        return sorted(conv_sessions & {search_session_id})
    return sorted(conv_sessions)


def run_knn(
    engine: "LCMEngine",
    *,
    query_vector: list[float],
    provider: Any,
    knn_limit: int,
    deadline: float,
    since: float | None,
    until: float | None,
    conversation_ids: list[str] | None,
    source: str | None,
    vector_store_cls: Any,
    scan_rows: int | None = None,
    full_scan: bool = False,
) -> Any:
    """Run the vector KNN query inside the operation's absolute deadline.

    ``vector_store_cls`` is injected so callers (and their tests) keep resolving
    the VectorStore binding through the ``tools`` module namespace. ``scan_rows``
    overrides the candidate-scan bound when set (``None`` keeps the configured
    ``embedding_bounded_scan_rows`` — the lcm_grep contract is unchanged); a
    cross-conversation caller passes a larger bound so "all time" is real.
    """
    if time.monotonic() >= deadline:
        raise TimeoutError("semantic vector search deadline exhausted")
    return _run_pooled_knn(
        engine,
        vector_store_cls=vector_store_cls,
        scan_rows=scan_rows,
        deadline=deadline,
        query=lambda store: store.knn(
            query_vector,
            k=knn_limit,
            model=provider.model_id,
            provider=provider.provider_id,
            since=since,
            until=until,
            conversation_ids=conversation_ids,
            source=source,
            full_scan=full_scan,
            deadline=deadline,
        ),
    )


def run_chunk_knn(
    engine: "LCMEngine",
    *,
    query_vector: list[float],
    provider: Any,
    knn_limit: int,
    deadline: float,
    since: float | None,
    until: float | None,
    conversation_ids: list[str] | None,
    source: str | None,
    vector_store_cls: Any,
    scan_rows: int | None = None,
    full_scan: bool = False,
) -> Any:
    """Run the chunk-corpus KNN query inside the operation's absolute deadline.

    Mirrors ``run_knn`` for the second (chunk) corpus: the same injected
    ``vector_store_cls`` binding, ``scan_rows`` candidate-bound override, and
    progress-handler deadline guard, calling ``knn_chunks`` instead of ``knn``.
    Returns the store's coverage contract (full|bounded|none) so the caller
    degrades identically to the summary arm.
    """
    if time.monotonic() >= deadline:
        raise TimeoutError("chunk vector search deadline exhausted")
    return _run_pooled_knn(
        engine,
        vector_store_cls=vector_store_cls,
        scan_rows=scan_rows,
        deadline=deadline,
        query=lambda store: store.knn_chunks(
            query_vector,
            k=knn_limit,
            model=provider.model_id,
            provider=provider.provider_id,
            since=since,
            until=until,
            conversation_ids=conversation_ids,
            source=source,
            full_scan=full_scan,
            deadline=deadline,
        ),
    )


def hydrate_chunk_hits(
    engine: "LCMEngine",
    *,
    ranked_rows: list[Any],
    knn_limit: int,
    deadline: float,
    snippet_chars: int,
) -> list[tuple[dict[str, Any], float]]:
    """Resolve ranked chunk ids to message-excerpt hits on a read-only connection.

    A chunk id is ``store_id:chunk_index``; ``lcm_chunk_meta`` carries the span
    (char_start/char_end) and the raw ``messages`` row supplies session/time and
    the verbatim excerpt. Each hit maps 1:1 to
    ``lcm_expand(store_id=..., content_offset=char_start)`` and is keyed by
    ``store_id`` so RRF fuses it against an FTS raw hit for the same message.
    """
    conn: sqlite3.Connection | None = None

    def require_remaining(stage: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"chunk result hydration deadline exhausted before {stage}"
            )
        return remaining

    try:
        require_remaining("database path resolution")
        db_path = Path(engine._store.db_path).resolve()
        uri = f"{db_path.as_uri()}?mode=ro"
        require_remaining("database connection")
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=max(0.001, require_remaining("database connection")),
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(
            lambda: 1 if time.monotonic() >= deadline else 0,
            1000,
        )
        # Rank-ordered chunk ids (bounded to knn_limit), then ONE batched JOIN
        # instead of a SELECT per hit (F4-chunk-hydrate-n-plus-1).
        ordered_ids: list[str] = []
        scores: dict[str, float] = {}
        for chunk_id, score, kind in ranked_rows:
            if kind != "chunk":
                continue
            cid = str(chunk_id)
            if cid in scores:
                continue
            ordered_ids.append(cid)
            scores[cid] = float(score)
            if len(ordered_ids) >= knn_limit:
                break
        if not ordered_ids:
            return []
        rows_by_id: dict[str, sqlite3.Row] = {}
        # Chunk in bounded batches so the IN(...) placeholder list stays well
        # under SQLite's variable limit even for a large knn_limit.
        batch_size = 500
        for start in range(0, len(ordered_ids), batch_size):
            require_remaining("chunk lookup")
            batch = ordered_ids[start:start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            for row in conn.execute(
                f"""
                SELECT cm.chunk_id, cm.store_id, cm.chunk_index, cm.char_start,
                       cm.char_end, m.session_id, m.source, m.role, m.timestamp,
                       m.content
                FROM lcm_chunk_meta cm
                JOIN messages m ON m.store_id = cm.store_id
                WHERE cm.chunk_id IN ({placeholders}) AND cm.archived = 0
                """,
                batch,
            ):
                rows_by_id.setdefault(str(row["chunk_id"]), row)
        hydrated: list[tuple[dict[str, Any], float]] = []
        for cid in ordered_ids:  # preserve KNN rank order
            row = rows_by_id.get(cid)
            if row is None:
                continue
            content = str(row["content"] or "")
            char_start = int(row["char_start"])
            char_end = int(row["char_end"])
            excerpt = content[char_start:char_end]
            hit = {
                "kind": "message_excerpt",
                "store_id": int(row["store_id"]),
                "session_id": row["session_id"],
                "source": row["source"] or "",
                "role": row["role"],
                "timestamp": row["timestamp"] or 0,
                "chunk_span": {
                    "chunk_index": int(row["chunk_index"]),
                    "char_start": char_start,
                    "char_end": char_end,
                },
                "content_offset": char_start,
                "snippet": excerpt[:snippet_chars],
            }
            hydrated.append((hit, scores[cid]))
        return hydrated
    finally:
        if conn is not None:
            conn.close()


def hydrate_semantic_nodes(
    engine: "LCMEngine",
    *,
    ranked_rows: list[Any],
    knn_limit: int,
    deadline: float,
) -> list[tuple[Any, float]]:
    """Hydrate ranked vector hits into summary nodes on a read-only connection."""
    conn: sqlite3.Connection | None = None

    def require_remaining(stage: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"semantic result hydration deadline exhausted before {stage}"
            )
        return remaining

    try:
        require_remaining("database path resolution")
        db_path = Path(engine._store.db_path).resolve()
        require_remaining("database path resolution")
        uri = f"{db_path.as_uri()}?mode=ro"
        require_remaining("database URI construction")
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=max(0.001, require_remaining("database connection")),
        )
        require_remaining("database connection")
        conn.row_factory = sqlite3.Row
        require_remaining("connection setup")
        conn.execute("PRAGMA query_only=ON")
        require_remaining("connection setup")
        conn.set_progress_handler(
            lambda: 1 if time.monotonic() >= deadline else 0,
            1000,
        )
        require_remaining("DAG setup")
        read_dag = copy.copy(engine._dag)
        read_dag._conn = conn
        read_dag._db_lock = threading.RLock()
        require_remaining("DAG setup")
        hydrated: list[tuple[Any, float]] = []
        for embedded_id, score, kind in ranked_rows:
            require_remaining("node lookup")
            if kind != "summary":
                continue
            try:
                node_id = int(embedded_id)
            except (TypeError, ValueError):
                continue
            require_remaining("node lookup")
            node = read_dag.get_node(node_id)
            require_remaining("node lookup")
            if node is not None:
                hydrated.append((node, float(score)))
            if len(hydrated) >= knn_limit:
                break
        return hydrated
    finally:
        if conn is not None:
            conn.close()


def _hit_identity(hit: dict[str, Any]) -> tuple[str, Any]:
    if hit.get("node_id") is not None:
        return ("node", hit.get("node_id"))
    return ("message", hit.get("store_id"))


def rrf_fuse(
    arms: list[list[dict[str, Any]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Reciprocal-rank fusion over ranked hit arms, keyed by hit identity.

    Each entry accumulates ``weight_arm / (k + rank)`` across the arms it appears
    in. Returns entries ordered by descending RRF score with the source hit under
    ``hit``, the fused score under ``rrf_score``, and each arm's 1-based rank in
    ``ranks`` (keyed by arm index). Callers own arm-specific metadata (which arm
    is FTS vs semantic, confidence, snippet provenance).

    ``weights`` optionally scales each arm's per-rank contribution (positional,
    aligned to ``arms``). It defaults to ``1.0`` for every arm, which is
    byte-identical to unweighted RRF -- a weak arm can then be down-weighted so a
    3-arm hybrid is never dragged below its best arm (measured on LongMemEval:
    naive equal-weight fusion cost 21 R@5 points versus pure vectors because the
    weak FTS arm got equal say). A ``weights`` shorter than ``arms`` (or with a
    missing/non-finite entry) falls back to ``1.0`` for the unspecified arms. A
    negative entry is clamped to ``0.0`` (the arm drops out) rather than allowed
    to invert rank-monotonicity.

    A single identity that appears more than once within the SAME arm (e.g. a
    message chunked into several pieces, each a separate chunk-arm hit) is
    collapsed to its best (first, since arms are best-first ordered) rank and
    contributes exactly one ``weight_arm / (k + rank)`` term for that arm --
    otherwise a multi-chunk message double-counts and out-scores a genuine
    higher-rank match (RRF-1).
    """

    def _arm_weight(arm_index: int) -> float:
        if weights is None or arm_index >= len(weights):
            return 1.0
        raw = weights[arm_index]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 1.0
        if value != value or value in (float("inf"), float("-inf")):
            return 1.0
        if value < 0.0:
            # A negative weight would invert rank-monotonicity (a rank-1 hit
            # scoring below a rank-2 hit). Clamp to 0.0 so the arm cleanly drops
            # out of the fusion rather than corrupting the ordering.
            return 0.0
        return value

    fused: dict[tuple[str, Any], dict[str, Any]] = {}
    for arm_index, arm in enumerate(arms):
        weight = _arm_weight(arm_index)
        for rank, hit in enumerate(arm, start=1):
            key = _hit_identity(hit)
            entry = fused.setdefault(
                key, {"hit": dict(hit), "rrf_score": 0.0, "ranks": {}}
            )
            if arm_index in entry["ranks"]:
                # Already scored this identity for this arm at a better rank.
                continue
            entry["ranks"][arm_index] = rank
            entry["rrf_score"] += weight / (k + rank)
    return sorted(
        fused.values(),
        key=lambda entry: (
            -float(entry["rrf_score"]),
            *(int(entry["ranks"].get(i, 10**9)) for i in range(len(arms))),
            _hit_identity(entry["hit"]),
        ),
    )
