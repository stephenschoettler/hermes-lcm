"""Slash-style /lcm command helpers for Hermes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
import dataclasses
import json
import math
import os
import re
import sqlite3
import time
from typing import Any
import uuid

from .db_bootstrap import (
    check_external_content_fts_integrity,
    external_content_fts_needs_repair,
    inspect_lcm_schema_health,
    join_background_integrity_scans,
    load_integrity_failed,
    remediate_interim_schema_stamp,
    repair_external_content_fts,
)
from .diagnostics import (
    _has_lifecycle_fragmentation,
    _state_db_path_for_engine,
    doctor_guidance_for_checks,
)
from .ingest_protection import (
    externalized_payload_stats,
    scan_externalized_payload_integrity,
    scan_sqlite_payload_risks,
    sensitive_pattern_status,
)
from .dag import SummaryDAG, build_nodes_fts_spec
from .presets import (
    explicit_operator_overrides,
    get_preset,
    invalid_operator_overrides,
    preset_confidence_reasons,
    preset_env_diff,
    preset_match_confidence,
    shipped_presets,
    suggest_preset_for_engine,
    unsupported_runtime_fields_text,
)
from .maintenance import backup_database, rotate_backup_database
from . import rollup_builder
from .rollup_store import RollupStore
from .session_patterns import build_session_match_keys, matches_session_pattern
from .store import build_message_fts_spec
from .chunking import (
    VALID_CONTENT_POLICIES,
    chunk_message,
    group_by_store_id,
    normalize_content_policy,
)
from .embedding_provider import (
    EmbeddedDocumentBatch,
    EmbeddingProviderError,
    FastembedProvider,
    ProviderPreDispatchError,
    VoyageError,
    _VOYAGE_CONTEXT_DOCUMENT_TOKEN_BUDGET,
    _VOYAGE_CONTEXT_MAX_CHUNK_TOKENS,
    _VOYAGE_CONTEXT_MAX_REQUEST_CHUNKS,
    _VOYAGE_CONTEXT_REQUEST_TOKEN_BUDGET,
    _VOYAGE_MAX_BATCH_ITEMS,
    _is_voyage_context_model,
    _plan_contextualized_requests,
    default_chunk_model,
    fastembed_download_size_note,
    resolve_provider,
)
from .tokens import count_tokens
from .vector_store import EmbeddingIdentity, EmbeddingPublishOutcome, VectorStore


_EMBEDDING_BACKFILL_CLAIM_KEY = "lcm_embedding_backfill_claim"
_EMBEDDING_BACKFILL_CLAIM_TTL_S = 10 * 60
_EMBEDDING_BACKFILL_BATCH_SIZE = 32


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _embedding_backfill_lease_ttl_s() -> float:
    return _env_float("LCM_EMBEDDING_BACKFILL_LEASE_TTL_S", float(_EMBEDDING_BACKFILL_CLAIM_TTL_S)) or float(
        _EMBEDDING_BACKFILL_CLAIM_TTL_S
    )


def _embedding_backfill_heartbeat_s() -> float:
    # Refresh cadence: renew the lease at most this often during a long run so a
    # live owner keeps a lease a second worker would otherwise steal as expired.
    return _env_float("LCM_EMBEDDING_BACKFILL_HEARTBEAT_S", 60.0)


def _embedding_backfill_budget_s() -> float:
    # Operation-wide wall-clock budget (0 = unlimited). When exceeded the run
    # stops between batches and reports partial rather than running unbounded.
    return _env_float("LCM_EMBEDDING_BACKFILL_BUDGET_S", 0.0)


def _ensure_inflight_table(conn: sqlite3.Connection) -> None:
    expected_columns = (
        ("embedded_id", "TEXT", 0, None, 1),
        ("identity_hash", "TEXT", 0, None, 2),
        ("lease_id", "TEXT", 0, None, 0),
        ("generation", "INTEGER", 0, None, 0),
        ("claimed_at", "REAL", 0, None, 0),
        ("state", "TEXT", 1, "'claimed'", 0),
        ("request_id", "TEXT", 0, None, 0),
        ("updated_at", "REAL", 0, None, 0),
        ("last_error", "TEXT", 0, None, 0),
    )
    expected_checks = ("statein('claimed','dispatched','uncertain')",)

    def normalize_check_expression(expression: str) -> str:
        # SQL keywords/identifiers are case-insensitive, but quoted literal
        # bytes are not. Preserve literal case and whitespace so e.g.
        # 'CLAIMED' or 'claimed ' cannot collide with the canonical CHECK.
        normalized: list[str] = []
        quote: str | None = None
        cursor = 0
        while cursor < len(expression):
            char = expression[cursor]
            if quote is not None:
                normalized.append(char)
                if quote == "]":
                    if char == "]":
                        quote = None
                elif char == quote:
                    if cursor + 1 < len(expression) and expression[cursor + 1] == quote:
                        cursor += 1
                        normalized.append(expression[cursor])
                    else:
                        quote = None
            elif char in {"'", '"', "`"}:
                quote = char
                normalized.append(char)
            elif char == "[":
                quote = "]"
                normalized.append(char)
            elif not char.isspace():
                normalized.append(char.lower())
            cursor += 1
        return "".join(normalized)

    def check_expressions(sql: str) -> tuple[str, ...]:
        """Return every normalized CHECK body, including duplicates and order."""
        lowered = sql.lower()
        expressions: list[str] = []
        offset = 0
        while True:
            match = re.search(r"\bcheck\s*\(", lowered[offset:])
            if match is None:
                return tuple(expressions)
            body_start = offset + match.end()
            cursor = body_start
            depth = 1
            quote: str | None = None
            while cursor < len(lowered) and depth:
                char = lowered[cursor]
                if quote is not None:
                    if quote == "]":
                        if char == "]":
                            quote = None
                    elif char == quote:
                        # SQLite escapes quotes by doubling them.
                        if cursor + 1 < len(lowered) and lowered[cursor + 1] == quote:
                            cursor += 1
                        else:
                            quote = None
                elif char in {"'", '"', "`"}:
                    quote = char
                elif char == "[":
                    quote = "]"
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                cursor += 1
            if depth:
                return ("<malformed>",)
            expressions.append(
                normalize_check_expression(sql[body_start:cursor - 1])
            )
            offset = cursor

    def create_table() -> None:
        conn.execute(
            """
            CREATE TABLE lcm_embedding_backfill_inflight (
                embedded_id TEXT,
                identity_hash TEXT,
                lease_id TEXT,
                generation INTEGER,
                claimed_at REAL,
                state TEXT NOT NULL DEFAULT 'claimed'
                    CHECK(state IN ('claimed', 'dispatched', 'uncertain')),
                request_id TEXT,
                updated_at REAL,
                last_error TEXT,
                PRIMARY KEY(embedded_id, identity_hash)
            )
            """
        )

    def table_info() -> tuple[tuple[object, ...], ...]:
        return tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), row[4], int(row[5]))
            for row in conn.execute(
                "PRAGMA table_info(lcm_embedding_backfill_inflight)"
            ).fetchall()
        )

    started_transaction = not conn.in_transaction
    try:
        if started_transaction:
            conn.execute("BEGIN IMMEDIATE")
        # Inspect only after obtaining the DDL write lock. Two first-run
        # backfills may arrive together; a pre-lock snapshot lets both observe
        # a missing table and makes the loser execute a stale CREATE TABLE.
        exists = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='lcm_embedding_backfill_inflight'"
        ).fetchone()
        if exists is None:
            create_table()
        else:
            actual = table_info()
            actual_names = {str(column[0]) for column in actual}
            expected_by_name = {column[0]: column for column in expected_columns}
            required_legacy = {column[0] for column in expected_columns[:5]}
            if not required_legacy <= actual_names or not actual_names <= set(expected_by_name):
                raise RuntimeError(
                    "incompatible lcm_embedding_backfill_inflight columns"
                )
            # Types and the composite PK are never safe to reinterpret. The
            # newer lifecycle columns are additive and can be rebuilt into the
            # exact checked/defaulted shape below.
            for column in actual:
                expected = expected_by_name[str(column[0])]
                if str(column[1]) != expected[1] or int(column[4]) != expected[4]:
                    raise RuntimeError(
                        "incompatible lcm_embedding_backfill_inflight column "
                        f"{column[0]}"
                    )
            table_sql = str(exists[0] or "")
            exact = (
                actual == expected_columns
                and check_expressions(table_sql) == expected_checks
            )
            if not exact:
                old_table = "lcm_embedding_backfill_inflight_legacy"
                if conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name=?", (old_table,)
                ).fetchone() is not None:
                    raise RuntimeError(
                        "cannot repair lcm_embedding_backfill_inflight: "
                        "legacy staging table already exists"
                    )
                conn.execute(
                    "ALTER TABLE lcm_embedding_backfill_inflight "
                    f"RENAME TO {old_table}"
                )
                create_table()
                state_expr = (
                    "CASE WHEN state IN ('claimed','dispatched','uncertain') "
                    "THEN state ELSE 'uncertain' END"
                    if "state" in actual_names
                    else "'uncertain'"
                )
                request_expr = "request_id" if "request_id" in actual_names else "NULL"
                updated_expr = (
                    "COALESCE(updated_at, claimed_at)"
                    if "updated_at" in actual_names
                    else "claimed_at"
                )
                error_expr = "last_error" if "last_error" in actual_names else "NULL"
                conn.execute(
                    "INSERT INTO lcm_embedding_backfill_inflight("
                    "embedded_id, identity_hash, lease_id, generation, claimed_at, "
                    "state, request_id, updated_at, last_error) "
                    "SELECT embedded_id, identity_hash, lease_id, generation, claimed_at, "
                    f"{state_expr}, {request_expr}, {updated_expr}, {error_expr} "
                    f"FROM {old_table}"
                )
                conn.execute(f"DROP TABLE {old_table}")

        if table_info() != expected_columns:
            raise RuntimeError(
                "lcm_embedding_backfill_inflight schema verification failed"
            )
        verified_sql = str(conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='lcm_embedding_backfill_inflight'"
        ).fetchone()[0] or "")
        if check_expressions(verified_sql) != expected_checks:
            raise RuntimeError(
                "lcm_embedding_backfill_inflight CHECK constraints are not exact"
            )

        expected_indexes = {
            "idx_lcm_embedding_inflight_identity_state": (
                "identity_hash", "state", "embedded_id"
            ),
            "idx_lcm_embedding_inflight_maintenance": (
                "identity_hash", "state", "updated_at", "embedded_id"
            ),
        }

        def index_shape(name: str) -> tuple[tuple[object, ...], ...]:
            return tuple(
                (
                    None if row[2] is None else str(row[2]),
                    int(row[3]),
                    str(row[4]).upper(),
                    int(row[5]),
                )
                for row in conn.execute(f"PRAGMA index_xinfo({name})").fetchall()
            )

        for name, columns in expected_indexes.items():
            expected_shape = tuple(
                (column, 0, "BINARY", 1) for column in columns
            ) + ((None, 0, "BINARY", 0),)
            index = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
                (name,),
            ).fetchone()
            needs_create = index is None
            if index is not None:
                index_flags = next(
                    (
                        (int(row[2]), int(row[4]))
                        for row in conn.execute(
                            "PRAGMA index_list(lcm_embedding_backfill_inflight)"
                        ).fetchall()
                        if str(row[1]) == name
                    ),
                    None,
                )
                if index_shape(name) != expected_shape or index_flags != (0, 0):
                    conn.execute(f"DROP INDEX {name}")
                    needs_create = True
            if needs_create:
                conn.execute(
                    f"CREATE INDEX {name} "
                    f"ON lcm_embedding_backfill_inflight({', '.join(columns)})"
                )
            if index_shape(name) != expected_shape:
                raise RuntimeError(f"in-flight index verification failed: {name}")
        if started_transaction:
            conn.commit()
    except Exception:
        if started_transaction:
            conn.rollback()
        raise
_VOYAGE_MAX_BATCH_TOKENS = 80_000
_VOYAGE_MAX_DOCUMENT_TOKENS = 27_000
_VOYAGE_USD_PER_MILLION_TOKENS = {
    "voyage-4-large": 0.12,
    "voyage-4": 0.06,
    "voyage-4-lite": 0.02,
    "voyage-context-3": 0.18,
    "voyage-3-lite": 0.02,
    "voyage-3.5-lite": 0.02,
    "voyage-3": 0.06,
    "voyage-3.5": 0.06,
    "voyage-3-large": 0.18,
    "voyage-code-3": 0.18,
}


_ROLLUPS_OUTPUT_CHAR_LIMIT = 20_000


class _RollupBuildOutcome(str, Enum):
    READY = "ready"
    DEFERRED = "deferred"
    NO_SOURCE = "no-source"
    SUPERSEDED = "superseded"
    FAILED = "failed"
    QUEUED = "queued"


@dataclass(frozen=True)
class _RollupRebuildResult:
    period_kind: str
    period_start: str
    outcome: _RollupBuildOutcome
    attempted: bool
    detail: str | None = None


def _bounded_rollups_text(text: str) -> str:
    """Apply one final bound to every ``/lcm rollups`` serialization."""
    if len(text) <= _ROLLUPS_OUTPUT_CHAR_LIMIT:
        return text
    marker = (
        "\ntruncated: true"
        f"\nchar_limit: {_ROLLUPS_OUTPUT_CHAR_LIMIT}"
        "\ntruncation_reason: response_char_limit"
    )
    return text[: _ROLLUPS_OUTPUT_CHAR_LIMIT - len(marker)] + marker

def _fmt_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _fmt_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024
        unit += 1
    precision = 0 if value >= 100 else 1 if value >= 10 else 2
    return f"{value:.{precision}f} {units[unit]}"


def _help_text(error: str | None = None) -> str:
    lines = []
    if error:
        lines.append(error)
        lines.append("")
    lines.extend([
        "LCM command help",
        "- /lcm or /lcm status: show current LCM runtime/session status",
        "- /lcm doctor: run read-only LCM health checks",
        "- /lcm doctor clean: best-effort scan of obvious junk/noise session candidates without deleting anything",
        "- /lcm doctor clean apply: backup-first cleanup for safe pattern-matched candidates only",
        "- /lcm doctor clean lifecycle: read-only scan for lifecycle rows with zero messages/nodes",
        "- /lcm doctor clean lifecycle apply: backup-first cleanup of empty lifecycle rows only",
        "- /lcm doctor repair: read-only scan for SQLite/FTS index repair needs",
        "- /lcm doctor repair apply: backup-first repair/rebuild of message and summary FTS indexes",
        "- /lcm doctor repair schema-stamp: read-only scan for an interim-build schema_version stamp ahead of the actual v5 shape",
        "- /lcm doctor repair schema-stamp apply: backup-first reset of an interim schema_version stamp back to the supported version",
        "- /lcm doctor source: read-only scan for legacy blank-source rows",
        "- /lcm doctor source apply: backup-first normalization of legacy blank-source rows to unknown",
        "- /lcm doctor retention: read-only retention analysis for stored session footprint and age",
        "- /lcm backup: create a timestamped SQLite backup before any future cleanup workflow",
        "- /lcm rotate: preview a tail-preserving in-place compact of the active session (read-only)",
        "- /lcm rotate apply: backup-first rotate that advances the lifecycle frontier past pre-tail raw messages",
        "- /lcm rollups: show temporal-rollup status for the current session",
        "- /lcm rollups rebuild <day|week|month|all> [date]: synchronously rebuild a bounded UTC target set",
        "- /lcm preset show [name]: inspect shipped preset metadata and benchmark provenance",
        "- /lcm preset suggest: preview the best shipped preset for the current engine state",
        "- /lcm preset apply <name> --dry-run: preview env-var changes without mutating live config",
        "- /lcm embed warmup: download/probe the configured embedding model and register its dimension",
        "- /lcm embed backfill [--apply] [--limit N]: preview or populate missing leaf-summary embeddings",
        "- /lcm help: show this help",
    ])
    return "\n".join(lines)


def _status_text(engine) -> str:
    status = engine.get_status()
    db_path = Path(engine._store.db_path)
    db_exists = db_path.exists()
    db_size = db_path.stat().st_size if db_exists else 0
    session_bound = bool(engine.current_session_id)
    source_stats = status.get("source_lineage") or {}
    runtime_identity = status.get("runtime_identity") or {}
    source_stats = {
        "messages_total": int(source_stats.get("messages_total", 0) or 0),
        "attributed_messages": int(source_stats.get("attributed_messages", 0) or 0),
        "normalized_unknown_messages": int(source_stats.get("normalized_unknown_messages", 0) or 0),
        "legacy_blank_source_messages": int(source_stats.get("legacy_blank_source_messages", 0) or 0),
        "effective_unknown_messages": int(source_stats.get("effective_unknown_messages", 0) or 0),
        **({"error": source_stats.get("error")} if source_stats.get("error") else {}),
    }
    protection = status.get("ingest_protection") or sensitive_pattern_status(engine._config)
    config_sources = status.get("config_sources") or {}
    config_source_warnings = status.get("config_source_warnings") or []
    ignored_config_yaml_lcm_keys = status.get("ignored_config_yaml_lcm_keys") or []

    uninitialized = "(uninitialized)"
    unknown = "(unknown)"
    model = (engine.model or unknown) if session_bound else uninitialized
    provider = (engine.provider or unknown) if session_bound else uninitialized
    context_length_source = (
        (getattr(engine, "_context_length_source", "") or unknown)
        if session_bound
        else uninitialized
    )

    lines = [
        "LCM status",
        f"engine: {status.get('engine', engine.name)}",
        f"plugin_name: {runtime_identity.get('plugin_name', '(unknown)')}",
        f"plugin_version: {runtime_identity.get('plugin_version', '(unknown)')}",
        f"plugin_path: {runtime_identity.get('plugin_path', '(unknown)')}",
        f"module_path: {runtime_identity.get('module_path', '(unknown)')}",
        f"plugin_git_commit: {runtime_identity.get('plugin_git_commit') or '(unavailable)'}",
        f"plugin_git_branch: {runtime_identity.get('plugin_git_branch') or '(unavailable)'}",
        f"plugin_git_dirty: {runtime_identity.get('plugin_git_dirty') if runtime_identity.get('plugin_git_dirty') is not None else '(unavailable)'}",
        f"hermes_home: {runtime_identity.get('hermes_home', '') or '(unset)'}",
        f"session_id: {engine.current_session_id or '(unbound)'}",
        f"session_platform: {engine.current_session_platform or ('(unbound)' if not session_bound else '(unknown)')}",
        f"model: {model}",
        f"provider: {provider}",
        f"database_path: {db_path}",
        f"database_path_source: {runtime_identity.get('database_path_source', '(unknown)')}",
        f"database_exists: {_fmt_bool(db_exists)}",
        f"database_size: {_fmt_size(db_size) if db_exists else 'missing'}",
        f"compression_count: {engine.compression_count}",
        f"total_compactions: {status.get('total_compactions', 0)}",
        f"total_compactions_scope: {status.get('total_compactions_scope', 'current_conversation')}",
        f"last_compression_status: {status.get('last_compression_status', 'idle')}",
        f"last_compression_noop_reason: {status.get('last_compression_noop_reason', '') or '(none)'}",
        f"context_length: {engine.context_length if session_bound else '(uninitialized)'}",
        f"raw_context_length: {status.get('raw_context_length', 0) if session_bound else '(uninitialized)'}",
        f"effective_context_length_cap: {status.get('effective_context_length_cap') or '(none)'}",
        f"effective_context_length_reason: {status.get('effective_context_length_reason') or '(none)'}",
        f"context_length_source: {context_length_source}",
        f"configured_context_threshold: {status.get('configured_context_threshold', engine._config.context_threshold)}",
        f"context_threshold: {status.get('context_threshold', engine._config.context_threshold)}",
        f"context_threshold_source: {status.get('context_threshold_source', config_sources.get('context_threshold', 'manual_or_default'))}",
        f"context_threshold_autoraised: {status.get('context_threshold_autoraised') or '(none)'}",
        f"threshold_tokens: {engine.threshold_tokens if session_bound else '(uninitialized)'}",
        f"cache_metrics_available: {_fmt_bool(status.get('cache_metrics_available'))}",
        f"last_input_tokens: {status.get('last_input_tokens', 0)}",
        f"last_output_tokens: {status.get('last_output_tokens', 0)}",
        f"last_cache_read_tokens: {status.get('last_cache_read_tokens', 0)}",
        f"last_cache_write_tokens: {status.get('last_cache_write_tokens', 0)}",
        f"last_reasoning_tokens: {status.get('last_reasoning_tokens', 0)}",
        f"cache_read_ratio: {float(status.get('cache_read_ratio', 0.0) or 0.0) * 100:.1f}%",
        f"sensitive_patterns_enabled: {_fmt_bool(protection.get('enabled'))}",
        f"sensitive_patterns: {', '.join(protection.get('patterns') or []) or '(none)'}",
        f"sensitive_patterns_source: {protection.get('source', 'default')}",
        # Filter classification for current_session_id (the foreground view).
        # When a side channel is in flight, get_status() reports the bound
        # session's flags; we read the engine properties instead so this row
        # stays consistent with the session_id row above.
        f"session_ignored: {_fmt_bool(engine.current_session_ignored)}",
        f"session_stateless: {_fmt_bool(engine.current_session_stateless)}",
        f"side_channel_active: {_fmt_bool(engine.side_channel_active)}",
        f"conversation_id: {runtime_identity.get('conversation_id', '') or '(unbound)'}",
        f"lifecycle_current_session_id: {runtime_identity.get('lifecycle_current_session_id', '') or '(none)'}",
        f"lifecycle_last_finalized_session_id: {runtime_identity.get('lifecycle_last_finalized_session_id', '') or '(none)'}",
        f"source_messages_total: {source_stats['messages_total']}",
        f"source_attributed_messages: {source_stats['attributed_messages']}",
        f"source_unknown_messages: {source_stats['normalized_unknown_messages']}",
        f"source_legacy_blank_messages: {source_stats['legacy_blank_source_messages']}",
        f"source_effective_unknown_messages: {source_stats['effective_unknown_messages']}",
    ]

    last_rotate_at = status.get("last_rotate_at")
    if last_rotate_at:
        lines.append(
            f"last_rotate_at: "
            f"{datetime.fromtimestamp(float(last_rotate_at), tz=timezone.utc).isoformat(timespec='seconds')}"
        )
        rotate_backup_size = int(status.get("rotate_backup_size", 0) or 0)
        if rotate_backup_size:
            lines.append(f"rotate_backup_size: {_fmt_size(rotate_backup_size)}")
    else:
        lines.append("last_rotate_at: (never)")
    if status.get("rotate_backup_path"):
        lines.append(f"rotate_backup_path: {status['rotate_backup_path']}")

    if session_bound:
        lines.extend([
            f"store_messages: {status.get('store_messages', 0)}",
            f"dag_nodes: {status.get('dag_nodes', 0)}",
        ])
    else:
        lines.append(
            "note: no active Hermes session has initialized LCM in this process yet — after a fresh restart, send one normal message first if you want live per-session runtime details"
        )

    if "ignore_session_patterns_source" in status:
        lines.append(
            f"ignore_session_patterns_source: {status.get('ignore_session_patterns_source')}"
        )
    if "stateless_session_patterns_source" in status:
        lines.append(
            f"stateless_session_patterns_source: {status.get('stateless_session_patterns_source')}"
        )
    if config_source_warnings:
        lines.append("config_source_warnings: " + "; ".join(config_source_warnings))
    if ignored_config_yaml_lcm_keys:
        lines.append(
            "ignored_config_yaml_lcm_keys: "
            + ", ".join(f"lcm.{key}" for key in ignored_config_yaml_lcm_keys)
        )
    if source_stats.get("error"):
        lines.append(f"source_lineage_error: {source_stats['error']}")
    return "\n".join(lines)


def _scan_clean_candidates(engine) -> dict[str, Any]:
    try:
        rows = engine._store.scan_session_cleanup_stats()
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "error": str(exc),
            "candidates": [],
            "ignored_count": 0,
            "stateless_count": 0,
            "protected_count": 0,
        }

    candidates = []
    ignored_count = 0
    stateless_count = 0
    protected_count = 0

    for session_id, message_count, token_total, node_count in rows:
        keys = build_session_match_keys(session_id)
        matched_classes = []
        if matches_session_pattern(keys, engine._compiled_ignore_session_patterns):
            matched_classes.append("ignored-pattern")
            ignored_count += 1
        elif matches_session_pattern(keys, engine._compiled_stateless_session_patterns):
            matched_classes.append("stateless-pattern")
            stateless_count += 1
        if not matched_classes:
            continue
        # Protect the actively-bound session from cleanup, not the foreground
        # view. While a cron tick has rebound the engine, _session_id points
        # at the cron session and the engine is actively writing through it
        # via lifecycle hooks; deleting that data mid-flight would corrupt
        # the cleanup pass. current_session_id (foreground) is the wrong
        # field here.
        if session_id == getattr(engine, "_session_id", ""):
            protected_count += 1
            continue
        candidates.append(
            {
                "session_id": session_id,
                "classes": matched_classes,
                "message_count": int(message_count),
                "node_count": int(node_count),
                "token_total": int(token_total),
            }
        )

    return {
        "error": None,
        "candidates": candidates,
        "ignored_count": ignored_count,
        "stateless_count": stateless_count,
        "protected_count": protected_count,
    }


def _scan_retention_candidates(engine) -> dict[str, Any]:
    now = datetime.now().timestamp()
    # SQL is scoped to the foreground session so /lcm doctor retention
    # reports the operator's real conversation rather than whatever side
    # channel (cron tick, debug probe) currently owns engine._session_id.
    # The "protected" flag below still keys off engine._session_id (the
    # actively-bound row) because that is the row receiving live writes
    # from the concurrent run.
    session_id = engine.current_session_id
    if not session_id:
        return {
            "error": None,
            "sessions": [],
            "sessions_analyzed": 0,
            "stale_sessions_30d": 0,
            "stale_sessions_90d": 0,
            "retained_tokens_30d": 0,
            "retained_tokens_90d": 0,
            "protected_count": 0,
        }
    try:
        rows = engine._store.scan_session_retention_stats(session_id)
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "error": str(exc),
            "sessions": [],
            "sessions_analyzed": 0,
            "stale_sessions_30d": 0,
            "stale_sessions_90d": 0,
            "retained_tokens_30d": 0,
            "retained_tokens_90d": 0,
            "protected_count": 0,
        }

    sessions = []
    protected_count = 0
    stale_sessions_30d = 0
    stale_sessions_90d = 0
    retained_tokens_30d = 0
    retained_tokens_90d = 0

    for row in rows:
        (
            session_id,
            message_count,
            token_total,
            node_count,
            node_token_total,
            first_message_at,
            last_message_at,
            first_node_at,
            last_node_at,
        ) = row
        timestamps = [
            ts for ts in (first_message_at, last_message_at, first_node_at, last_node_at)
            if ts is not None
        ]
        if not timestamps:
            continue
        first_activity_at = min(float(ts) for ts in (first_message_at, first_node_at) if ts is not None)
        last_activity_at = max(float(ts) for ts in (last_message_at, last_node_at) if ts is not None)
        age_days = max(0.0, (now - last_activity_at) / 86400.0)
        # Bound (not foreground): protect the live session from retention
        # bookkeeping while the engine may still be writing to it.
        protected = session_id == getattr(engine, "_session_id", "")
        total_footprint_tokens = int(token_total) + int(node_token_total)
        if protected:
            protected_count += 1
        if age_days >= 30.0:
            stale_sessions_30d += 1
            retained_tokens_30d += total_footprint_tokens
        if age_days >= 90.0:
            stale_sessions_90d += 1
            retained_tokens_90d += total_footprint_tokens
        sessions.append(
            {
                "session_id": session_id,
                "protected": protected,
                "message_count": int(message_count),
                "node_count": int(node_count),
                "token_total": total_footprint_tokens,
                "raw_token_total": int(token_total),
                "summary_token_total": int(node_token_total),
                "first_activity_at": float(first_activity_at),
                "last_activity_at": float(last_activity_at),
                "age_days": age_days,
            }
        )

    sessions.sort(
        key=lambda item: (
            1 if item["protected"] else 0,
            0 if item["age_days"] >= 30.0 else 1,
            -item["token_total"],
            -item["node_count"],
            -item["message_count"],
            item["last_activity_at"],
            item["session_id"],
        )
    )

    return {
        "error": None,
        "sessions": sessions,
        "sessions_analyzed": len(sessions),
        "stale_sessions_30d": stale_sessions_30d,
        "stale_sessions_90d": stale_sessions_90d,
        "retained_tokens_30d": retained_tokens_30d,
        "retained_tokens_90d": retained_tokens_90d,
        "protected_count": protected_count,
    }


def _rotate_text(engine) -> str:
    preview = engine.rotate_active_session(apply=False)
    if not preview.get("ok"):
        reason = preview.get("reason", "unknown")
        lines = [
            "LCM rotate",
            "status: refused",
            f"reason: {reason}",
        ]
        session_id = preview.get("session_id")
        if session_id:
            lines.append(f"session_id: {session_id}")
        lines.append("note: read-only preview — no changes were made")
        return "\n".join(lines)

    backup_path = engine.rotate_backup_path()
    lines = [
        "LCM rotate",
        f"status: {'noop' if preview.get('noop') else 'preview'}",
        f"session_id: {preview['session_id']}",
        f"conversation_id: {preview['conversation_id']}",
        f"total_message_count: {preview['total_message_count']}",
        f"fresh_tail_count: {preview['fresh_tail_count']}",
        f"fresh_tail_max_tokens: {preview['fresh_tail_max_tokens']}",
        f"effective_fresh_tail_count: {preview['effective_fresh_tail_count']}",
        f"effective_fresh_tail_tokens: {preview['effective_fresh_tail_tokens']}",
        f"pre_tail_message_count: {preview.get('pre_tail_message_count', 0)}",
        f"current_frontier_store_id: {preview['current_frontier_store_id']}",
        f"new_frontier_store_id: {preview['new_frontier_store_id']}",
        f"rotate_backup_path: {backup_path}",
    ]
    if preview.get("noop"):
        lines.append(f"reason: {preview.get('reason', 'no_change')}")
        lines.append("note: read-only preview — rotate apply would be a no-op for this session")
    else:
        lines.append("note: read-only preview — use `/lcm rotate apply` to advance the frontier (backup-first)")
        lines.append("note: pre-tail raw messages remain in the store and recoverable via lcm_load_session")
    return "\n".join(lines)


def _rotate_apply_text(engine) -> str:
    # Pre-flight refusal AND noop check before touching disk. This avoids
    # both writing a backup for a session that would refuse and overwriting
    # the previous known-good rolling backup when the apply would be a no-op
    # (e.g., idempotent rerun on an already-rotated session).
    pre = engine.rotate_active_session(apply=False)
    if not pre.get("ok"):
        reason = pre.get("reason", "unknown")
        lines = [
            "LCM rotate apply",
            "status: refused",
            f"reason: {reason}",
        ]
        session_id = pre.get("session_id")
        if session_id:
            lines.append(f"session_id: {session_id}")
        lines.append("note: rotate apply refused; no backup was created and no lifecycle state was changed")
        return "\n".join(lines)

    if pre.get("noop"):
        # Surface the same shape as a successful apply but with status:noop so
        # operators get the standard fields without a fresh backup write
        # destroying the previous known-good snapshot.
        lines = [
            "LCM rotate apply",
            "status: noop",
            f"session_id: {pre['session_id']}",
            f"conversation_id: {pre['conversation_id']}",
            f"total_message_count: {pre['total_message_count']}",
            f"fresh_tail_count: {pre['fresh_tail_count']}",
            f"fresh_tail_max_tokens: {pre['fresh_tail_max_tokens']}",
            f"effective_fresh_tail_count: {pre['effective_fresh_tail_count']}",
            f"effective_fresh_tail_tokens: {pre['effective_fresh_tail_tokens']}",
            f"pre_tail_message_count: {pre.get('pre_tail_message_count', 0)}",
            f"previous_frontier_store_id: {pre['current_frontier_store_id']}",
            f"new_frontier_store_id: {pre['new_frontier_store_id']}",
            f"reason: {pre.get('reason', 'no_change')}",
            "note: rotate is a no-op; rolling backup was not written so the previous rotate-latest snapshot is preserved",
        ]
        return "\n".join(lines)

    backup = rotate_backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM rotate apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: backup failed: {backup['error']}",
            "note: rotate apply aborted before any lifecycle mutation",
        ])

    result = engine.rotate_active_session(apply=True)
    if not result.get("ok"):
        return "\n".join([
            "LCM rotate apply",
            "status: refused",
            f"reason: {result.get('reason', 'unknown')}",
            f"rotate_backup_path: {backup['backup_path']}",
            f"rotate_backup_size: {_fmt_size(int(backup['backup_size']))}",
            "note: backup was created before rotate refused; lifecycle state unchanged",
        ])

    is_noop = bool(result.get("noop"))
    lines = [
        "LCM rotate apply",
        f"status: {'noop' if is_noop else 'ok'}",
        f"session_id: {result['session_id']}",
        f"conversation_id: {result['conversation_id']}",
        f"rotate_backup_path: {backup['backup_path']}",
        f"rotate_backup_size: {_fmt_size(int(backup['backup_size']))}",
        f"total_message_count: {result['total_message_count']}",
        f"fresh_tail_count: {result['fresh_tail_count']}",
        f"fresh_tail_max_tokens: {result['fresh_tail_max_tokens']}",
        f"effective_fresh_tail_count: {result['effective_fresh_tail_count']}",
        f"effective_fresh_tail_tokens: {result['effective_fresh_tail_tokens']}",
        f"pre_tail_message_count: {result.get('pre_tail_message_count', 0)}",
        f"previous_frontier_store_id: {result['current_frontier_store_id']}",
        f"new_frontier_store_id: {result.get('applied_frontier_store_id', result['new_frontier_store_id'])}",
    ]
    if is_noop:
        lines.append(f"reason: {result.get('reason', 'no_change')}")
        lines.append("note: lifecycle state already at or ahead of the target frontier")
    else:
        lines.append("note: pre-tail raw messages remain in the store and recoverable via lcm_load_session")
        lines.append("note: rolling backup overwrites the previous rotate-latest slot")
    return "\n".join(lines)


def _scan_fts_repair(engine) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    specs = {
        "messages_fts": build_message_fts_spec(),
        "nodes_fts": build_nodes_fts_spec(),
    }
    conn = engine._store.connection
    for label, spec in specs.items():
        try:
            structural_needs_repair = external_content_fts_needs_repair(conn, spec)
            integrity_check = check_external_content_fts_integrity(conn, spec)
            integrity_status = str(integrity_check.get("status") or "fail")
            needs_repair = structural_needs_repair or integrity_status == "fail"
            content_count = int(conn.execute(
                f"SELECT COUNT(*) FROM {spec.content_table}"
            ).fetchone()[0])
            try:
                fts_count = int(conn.execute(f"SELECT COUNT(*) FROM {spec.table_name}").fetchone()[0])
            except sqlite3.Error:
                fts_count = None
            checks[label] = {
                "ok": not needs_repair,
                "needs_repair": needs_repair,
                "content_rows": content_count,
                "fts_rows": fts_count,
                "integrity_status": integrity_status,
                "integrity_detail": integrity_check.get("detail"),
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - defensive
            checks[label] = {
                "ok": False,
                "needs_repair": True,
                "content_rows": None,
                "fts_rows": None,
                "integrity_status": "error",
                "integrity_detail": str(exc),
                "error": str(exc),
            }
    return {
        "checks": checks,
        "needs_repair": any(item["needs_repair"] for item in checks.values()),
    }


def _doctor_repair_text(engine) -> str:
    scan = _scan_fts_repair(engine)
    lines = [
        "LCM doctor repair",
        f"status: {'repair-needed' if scan['needs_repair'] else 'ok'}",
    ]
    for label, item in scan["checks"].items():
        state = "repair-needed" if item["needs_repair"] else "ok"
        lines.append(f"{label}: {state}")
        if item["error"]:
            lines.append(f"{label}_error: {item['error']}")
        else:
            lines.append(f"{label}_content_rows: {item['content_rows']}")
            lines.append(f"{label}_fts_rows: {item['fts_rows']}")
            lines.append(f"{label}_integrity_status: {item['integrity_status']}")
    lines.append("note: read-only scan only — no FTS tables were repaired")
    if scan["needs_repair"]:
        lines.append("note: use `/lcm doctor repair apply` to create a backup and repair FTS indexes")
    return "\n".join(lines)


def _doctor_repair_apply_text(engine) -> str:
    backup = backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM doctor repair apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: backup failed: {backup['error']}",
            "note: repair apply aborted before any FTS tables were repaired",
        ])

    # Join any in-flight background integrity scan first: otherwise a scan still
    # mid-flight can error out (or re-check) after the repair commits and re-write
    # a fresh fts_integrity_failed marker, reproducing F1's stuck false-positive
    # via a race (F3).
    join_background_integrity_scans()

    conn = engine._store.connection
    try:
        messages_result = repair_external_content_fts(conn, build_message_fts_spec())
        nodes_result = repair_external_content_fts(conn, build_nodes_fts_spec())
    except sqlite3.Error as exc:
        return "\n".join([
            "LCM doctor repair apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"backup_path: {backup['backup_path']}",
            f"backup_size: {_fmt_size(int(backup['backup_size']))}",
            f"error: FTS repair failed: {exc}",
            "note: backup was created before repair apply",
        ])

    return "\n".join([
        "LCM doctor repair apply",
        "status: ok",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
        f"messages_fts_rebuilt: {_fmt_bool(messages_result['rebuilt'])}",
        f"messages_fts_triggers_recreated: {_fmt_bool(messages_result['triggers_recreated'])}",
        f"messages_fts_degraded: {_fmt_bool(messages_result['degraded'])}",
        f"nodes_fts_rebuilt: {_fmt_bool(nodes_result['rebuilt'])}",
        f"nodes_fts_triggers_recreated: {_fmt_bool(nodes_result['triggers_recreated'])}",
        f"nodes_fts_degraded: {_fmt_bool(nodes_result['degraded'])}",
        "note: backup created before repair apply",
    ])


def _classify_schema_stamp(db_path: Path) -> dict[str, Any]:
    """Open the DB independently (read-only) and classify a too-new stamp.

    Doctor must work even when the engine store refused to open the stamped DB,
    so this uses a fresh read-only connection keyed on the path rather than the
    engine's own connection.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return remediate_interim_schema_stamp(conn, apply=False)
    finally:
        conn.close()


def _schema_stamp_note(plan: dict[str, Any]) -> str:
    status = plan["status"]
    if status == "noop":
        return f"note: schema_version is already v{plan['target_version']}; nothing to repair"
    if status == "refused":
        return (
            "note: schema is genuinely newer than this build — refusing to "
            "downgrade; restore a pre-upgrade backup (.db/-wal/-shm) instead"
        )
    return (
        "note: use `/lcm doctor repair schema-stamp apply` to create a backup "
        f"and reset the stamp to v{plan['target_version']}"
    )


def _schema_stamp_drop_lines(plan: dict[str, Any], *, applied: bool) -> list[str]:
    """Per-early-feature-table lines: what was (or would be) dropped + why."""
    verb = "dropped" if applied else "would_drop"
    lines: list[str] = []
    for family in plan.get("drop_plan") or []:
        hint = family["rebuild_hint"]
        for table in family["tables"]:
            lines.append(f"{verb}: {table} ({hint})")
    return lines


def _doctor_repair_schema_stamp_text(engine) -> str:
    db_path = Path(engine._store.db_path)
    lines = ["LCM doctor repair schema-stamp"]
    if not db_path.exists():
        return "\n".join([
            *lines,
            "status: error",
            f"database_path: {db_path}",
            "error: database file does not exist",
        ])
    try:
        plan = _classify_schema_stamp(db_path)
    except sqlite3.Error as exc:  # pragma: no cover - defensive
        return "\n".join([
            *lines,
            "status: error",
            f"database_path: {db_path}",
            f"error: schema-stamp scan failed: {exc}",
        ])
    status = "repair-needed" if plan["status"] == "dry-run" else plan["status"]
    return "\n".join([
        *lines,
        f"status: {status}",
        f"database_path: {db_path}",
        f"stored_schema_version: {plan['current_version']}",
        f"target_schema_version: {plan['target_version']}",
        f"classification: {plan['classification'] or 'none'}",
        *_schema_stamp_drop_lines(plan, applied=False),
        "note: read-only scan only — no schema changes were made",
        _schema_stamp_note(plan),
    ])


def _doctor_repair_schema_stamp_apply_text(engine) -> str:
    db_path = Path(engine._store.db_path)
    lines = ["LCM doctor repair schema-stamp apply"]
    if not db_path.exists():
        return "\n".join([
            *lines,
            "status: error",
            f"database_path: {db_path}",
            "error: database file does not exist",
        ])

    # Classify first so a genuinely-newer or already-current DB never triggers a
    # pointless backup or any mutation.
    try:
        plan = _classify_schema_stamp(db_path)
    except sqlite3.Error as exc:  # pragma: no cover - defensive
        return "\n".join([
            *lines,
            "status: error",
            f"database_path: {db_path}",
            f"error: schema-stamp scan failed: {exc}",
        ])
    if plan["status"] != "dry-run":
        return "\n".join([
            *lines,
            f"status: {plan['status']}",
            f"database_path: {db_path}",
            f"stored_schema_version: {plan['current_version']}",
            f"target_schema_version: {plan['target_version']}",
            f"classification: {plan['classification'] or 'none'}",
            _schema_stamp_note(plan),
        ])

    backup = backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            *lines,
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: backup failed: {backup['error']}",
            "note: schema-stamp apply aborted before any schema change",
        ])

    conn = sqlite3.connect(str(db_path))
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    except sqlite3.Error as exc:
        return "\n".join([
            *lines,
            "status: error",
            f"database_path: {backup['db_path']}",
            f"backup_path: {backup['backup_path']}",
            f"backup_size: {_fmt_size(int(backup['backup_size']))}",
            f"error: schema-stamp reset failed: {exc}",
            "note: backup was created before schema-stamp apply",
        ])
    finally:
        conn.close()

    dropped = result.get("dropped_tables") or []
    return "\n".join([
        *lines,
        f"status: {result['status']}",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
        f"stored_schema_version: {result['current_version']}",
        f"schema_version_reset_to: {result['target_version']}",
        f"classification: {result['classification']}",
        f"applied: {_fmt_bool(bool(result['applied']))}",
        f"dropped_feature_tables: {len(dropped)}",
        *_schema_stamp_drop_lines(result, applied=True),
        "note: backup created before schema-stamp apply",
        *(
            ["note: dropped tables are derived caches — rebuild them with the "
             "commands above; no lossless message/summary data was affected"]
            if dropped else []
        ),
    ])


def _doctor_source_text(engine) -> str:
    try:
        plan = engine._store.get_source_normalization_plan()
    except Exception as exc:  # pragma: no cover - defensive
        return "\n".join([
            "LCM doctor source",
            "status: error",
            f"error: source-lineage scan failed: {exc}",
            "note: read-only scan only — no source rows were updated",
        ])

    stats = plan["stats_before"]
    would_update = int(plan["would_update_messages"])
    lines = [
        "LCM doctor source",
        f"status: {'normalization-needed' if would_update else 'ok'}",
        f"messages_total: {stats['messages_total']}",
        f"attributed_messages: {stats['attributed_messages']}",
        f"unknown_messages: {stats['normalized_unknown_messages']}",
        f"legacy_blank_messages: {stats['legacy_blank_source_messages']}",
        f"effective_unknown_messages: {stats['effective_unknown_messages']}",
        f"target_source: {plan['target_source']}",
        f"would_update_messages: {would_update}",
        f"affected_sessions: {plan['affected_sessions']}",
        "note: read-only scan only — no source rows were updated",
    ]
    if would_update:
        lines.append(
            "note: use `/lcm doctor source apply` to create a backup and normalize legacy blank-source rows"
        )
    else:
        lines.append("note: no legacy blank-source rows need normalization")
    return "\n".join(lines)


def _doctor_source_apply_text(engine) -> str:
    try:
        plan = engine._store.get_source_normalization_plan()
    except Exception as exc:  # pragma: no cover - defensive
        return "\n".join([
            "LCM doctor source apply",
            "status: error",
            f"error: source-lineage scan failed: {exc}",
            "note: source normalization apply aborted before any rows were updated",
        ])

    if int(plan["would_update_messages"]) == 0:
        stats = plan["stats_before"]
        return "\n".join([
            "LCM doctor source apply",
            "status: ok",
            f"target_source: {plan['target_source']}",
            "updated_messages: 0",
            f"legacy_blank_before: {stats['legacy_blank_source_messages']}",
            f"legacy_blank_after: {stats['legacy_blank_source_messages']}",
            "note: no legacy blank-source rows needed normalization",
        ])

    backup = backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM doctor source apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: backup failed: {backup['error']}",
            "note: source normalization apply aborted before any rows were updated",
        ])

    try:
        result = engine._store.normalize_legacy_blank_sources()
    except sqlite3.Error as exc:
        return "\n".join([
            "LCM doctor source apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"backup_path: {backup['backup_path']}",
            f"backup_size: {_fmt_size(int(backup['backup_size']))}",
            f"error: source normalization failed: {exc}",
            "note: backup was created before source normalization apply",
        ])

    before = result["stats_before"]
    after = result["stats_after"]
    return "\n".join([
        "LCM doctor source apply",
        "status: ok",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
        f"target_source: {result['target_source']}",
        f"updated_messages: {result['updated_messages']}",
        f"legacy_blank_before: {before['legacy_blank_source_messages']}",
        f"legacy_blank_after: {after['legacy_blank_source_messages']}",
        f"unknown_before: {before['normalized_unknown_messages']}",
        f"unknown_after: {after['normalized_unknown_messages']}",
        "note: backup created before source normalization apply",
    ])


def _doctor_text(engine) -> str:
    db_path = Path(engine._store.db_path)
    runtime_identity = engine.get_runtime_identity()
    store_conn = engine._store.connection
    dag_conn = engine._dag.connection

    issues: list[str] = []
    recommended_actions: list[str] = []
    schema_health = inspect_lcm_schema_health(store_conn, database_path=str(db_path))
    schema_missing_raw = schema_health.get("missing_tables")
    schema_missing_tables = [str(name) for name in schema_missing_raw] if isinstance(schema_missing_raw, list) else []
    schema_existing_raw = schema_health.get("existing_tables")
    schema_existing_tables = [str(name) for name in schema_existing_raw] if isinstance(schema_existing_raw, list) else []
    schema_core_status = "error" if schema_health.get("error") else "missing" if schema_missing_tables else "ok"
    if schema_missing_tables or schema_health.get("error"):
        issues.append("schema_core_tables")

    def _safe_count(conn, query: str, issue_key: str) -> int | str:
        try:
            return int(conn.execute(query).fetchone()[0])
        except Exception as exc:  # pragma: no cover - defensive
            issues.append(issue_key)
            return f"error: {exc}"

    try:
        integrity_row = store_conn.execute("PRAGMA integrity_check").fetchone()
        integrity = str(integrity_row[0]) if integrity_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        integrity = f"error: {exc}"
        issues.append("sqlite_integrity")

    def _fts_text_status(result: dict[str, Any]) -> str:
        status = str(result.get("status") or "fail")
        return "ok" if status == "pass" else status

    try:
        store_fts_count = int(store_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
        store_fts_integrity = check_external_content_fts_integrity(store_conn, build_message_fts_spec())
        store_fts = _fts_text_status(store_fts_integrity)
        if store_fts == "fail":
            issues.append("messages_fts")
        elif store_fts == "unchecked":
            recommended_actions.append("rerun `/lcm doctor` with read-write SQLite access if a deep messages FTS check is needed")
    except Exception as exc:  # pragma: no cover - defensive
        store_fts_count = f"error: {exc}"
        store_fts = f"error: {exc}"
        store_fts_integrity = {"status": "fail", "detail": str(exc)}
        issues.append("messages_fts")

    try:
        node_fts_count = int(dag_conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0])
        node_fts_integrity = check_external_content_fts_integrity(dag_conn, build_nodes_fts_spec())
        node_fts = _fts_text_status(node_fts_integrity)
        if node_fts == "fail":
            issues.append("nodes_fts")
        elif node_fts == "unchecked":
            recommended_actions.append("rerun `/lcm doctor` with read-write SQLite access if a deep nodes FTS check is needed")
    except Exception as exc:  # pragma: no cover - defensive
        node_fts_count = f"error: {exc}"
        node_fts = f"error: {exc}"
        node_fts_integrity = {"status": "fail", "detail": str(exc)}
        issues.append("nodes_fts")

    # A prior non-blocking background integrity scan (issue #6) records a
    # persisted ``fts_integrity_failed:<table>`` flag when it finds corruption
    # without rebuilding. Surface it even when this doctor run's live deep check
    # could not confirm it (e.g. read-only access), pointing at the explicit
    # repair path.
    try:
        store_fts_failed_flag = load_integrity_failed(store_conn, build_message_fts_spec())
    except Exception:  # pragma: no cover - defensive
        store_fts_failed_flag = None
    try:
        node_fts_failed_flag = load_integrity_failed(dag_conn, build_nodes_fts_spec())
    except Exception:  # pragma: no cover - defensive
        node_fts_failed_flag = None
    if store_fts_failed_flag and "messages_fts" not in issues:
        issues.append("messages_fts")
    if node_fts_failed_flag and "nodes_fts" not in issues:
        issues.append("nodes_fts")

    total_messages = _safe_count(store_conn, "SELECT COUNT(*) FROM messages", "messages_total")
    total_message_sessions = _safe_count(
        store_conn,
        "SELECT COUNT(DISTINCT session_id) FROM messages",
        "message_sessions_total",
    )
    total_nodes = _safe_count(dag_conn, "SELECT COUNT(*) FROM summary_nodes", "summary_nodes_total")
    total_node_sessions = _safe_count(
        dag_conn,
        "SELECT COUNT(DISTINCT session_id) FROM summary_nodes",
        "summary_node_sessions_total",
    )

    db_exists = db_path.exists()
    db_size = db_path.stat().st_size if db_exists else 0
    wal_path = Path(str(db_path) + "-wal")
    wal_size = wal_path.stat().st_size if wal_path.exists() else 0
    try:
        journal_row = store_conn.execute("PRAGMA journal_mode").fetchone()
        journal_mode = str(journal_row[0]) if journal_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        journal_mode = f"error: {exc}"
        issues.append("sqlite_journal_mode")
    try:
        quick_row = store_conn.execute("PRAGMA quick_check").fetchone()
        quick_check = str(quick_row[0]) if quick_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        quick_check = f"error: {exc}"
        issues.append("sqlite_quick_check")
    payload_storage_error = ""
    try:
        payload_risks = scan_sqlite_payload_risks(store_conn)
        externalized_stats = externalized_payload_stats(engine._config, hermes_home=engine._hermes_home)
        externalized_integrity = scan_externalized_payload_integrity(
            store_conn,
            engine._config,
            hermes_home=engine._hermes_home,
        )
    except Exception as exc:  # pragma: no cover - defensive
        payload_storage_error = str(exc)
        payload_risks = {
            "largest_content_rows": [],
            "largest_tool_calls_rows": [],
            "suspicious_data_uri_content_rows": [],
            "suspicious_data_uri_tool_calls_rows": [],
            "suspicious_base64_like_rows": [],
            "quarantined_assistant_rows": [],
            "suspicious_repetitive_assistant_rows": [],
            "heartbeat_noise_rows": [],
        }
        externalized_stats = {
            "externalized_payload_count": 0,
            "externalized_payload_bytes": 0,
            "externalized_payload_chars": 0,
            "externalized_payload_dir": "",
            "latest_externalized_payload_path": "",
            "latest_externalized_payload_mtime": 0,
        }
        externalized_integrity = {
            "externalized_payload_refs_total": 0,
            "externalized_payload_refs_existing": 0,
            "externalized_payload_refs_missing": 0,
            "externalized_payload_files_unreferenced": 0,
            "missing_externalized_payload_refs": [],
            "unreferenced_externalized_payload_files": [],
        }
        issues.append("payload_storage")
    clean_scan = _scan_clean_candidates(engine)

    debt_rows = []
    lifecycle_conn = getattr(getattr(engine, "_lifecycle", None), "connection", None)
    if lifecycle_conn is not None:
        try:
            debt_rows = lifecycle_conn.execute(
                """
                SELECT conversation_id, debt_kind, debt_size_estimate
                FROM lcm_lifecycle_state
                WHERE debt_kind IS NOT NULL AND debt_size_estimate > 0
                ORDER BY updated_at DESC
                """
            ).fetchall()
        except Exception as exc:  # pragma: no cover - defensive
            issues.append("lifecycle_state")
            debt_rows = [(f"error: {exc}", "error", 0)]

    observations: list[str] = []
    missing_externalized_refs = int(externalized_integrity.get("externalized_payload_refs_missing", 0) or 0)
    suspicious_payload_rows = sum(
        len(payload_risks.get(key) or [])
        for key in (
            "suspicious_data_uri_content_rows",
            "suspicious_data_uri_tool_calls_rows",
            "suspicious_base64_like_rows",
            "suspicious_repetitive_assistant_rows",
        )
    )

    if schema_health.get("error"):
        observations.append(f"schema_core_tables: error: {schema_health['error']}")
        recommended_actions.append(
            "verify SQLite can read sqlite_master for the database inspected by Hermes"
        )
    elif schema_missing_tables:
        observations.append(
            "schema_core_tables: missing " + ", ".join(schema_missing_tables)
        )
        recommended_actions.append(
            "verify HERMES_HOME/LCM_DATABASE_PATH point at the database inspected by Hermes"
        )
    else:
        observations.append("schema_core_tables: ok")

    if debt_rows:
        first = debt_rows[0]
        observations.append(
            f"maintenance_debt: {len(debt_rows)} conversation(s) currently carry deferred maintenance debt; first={first[0]} kind={first[1]} size={first[2]}"
        )
        recommended_actions.append(
            "let normal compaction turns reduce maintenance debt before attempting broader cleanup"
        )

    if clean_scan["error"]:
        observations.append(f"cleanup_candidates: scan error: {clean_scan['error']}")
    elif clean_scan["candidates"]:
        observations.append(
            f"cleanup_candidates: {len(clean_scan['candidates'])} pattern-matched junk/noise session candidate(s) detected"
        )
        recommended_actions.append("inspect candidate sessions with `/lcm doctor clean`")
        recommended_actions.append("create a safety snapshot first with `/lcm backup`")
    else:
        observations.append("cleanup_candidates: none")

    if missing_externalized_refs:
        issues.append("payload_storage")
        observations.append(
            f"payload_storage: {missing_externalized_refs} externalized payload ref(s) point to missing JSON files"
        )
        recommended_actions.append(
            "inspect missing externalized payload refs and restore from backups if needed"
        )
    if suspicious_payload_rows:
        observations.append(
            f"payload_storage: {suspicious_payload_rows} suspicious inline/base64 payload row(s) need review"
        )
        recommended_actions.append(
            "inspect suspicious payload rows before cleanup; restore payload files from backup before deleting or rewriting anything"
        )
    if payload_storage_error:
        observations.append(f"payload_storage_error: {payload_storage_error}")
        recommended_actions.append("inspect payload storage diagnostics before cleanup or deletion")

    try:
        source_stats = engine._store.get_source_stats()
    except Exception as exc:  # pragma: no cover - defensive
        issues.append("source_lineage")
        source_stats = {
            "messages_total": 0,
            "attributed_messages": 0,
            "normalized_unknown_messages": 0,
            "legacy_blank_source_messages": 0,
            "effective_unknown_messages": 0,
            "error": str(exc),
        }
    observations.append(
        "source_lineage: "
        f"attributed={source_stats['attributed_messages']} "
        f"unknown={source_stats['normalized_unknown_messages']} "
        f"legacy_blank={source_stats['legacy_blank_source_messages']} "
        f"effective_unknown={source_stats['effective_unknown_messages']}"
    )
    if source_stats.get("error"):
        observations.append(f"source_lineage_error: {source_stats['error']}")
    if source_stats["legacy_blank_source_messages"]:
        observations.append(
            "legacy blank-source rows are normalized as `source=unknown` for back-compat filters"
        )

    try:
        lifecycle_stats = engine._lifecycle.get_fragmentation_stats(
            state_db_path=_state_db_path_for_engine(engine)
        )
    except Exception as exc:  # pragma: no cover - defensive
        issues.append("lifecycle_fragmentation")
        lifecycle_stats = {"error": str(exc)}
    else:
        observations.append(
            "lifecycle_fragmentation: "
            f"lifecycle_rows={lifecycle_stats['lifecycle_rows']} "
            f"empty_lifecycle_rows={lifecycle_stats.get('empty_lifecycle_rows', 0)} "
            f"message_sessions={lifecycle_stats['distinct_message_sessions']} "
            f"node_sessions={lifecycle_stats['distinct_node_sessions']} "
            f"current_missing_in_lcm_any={lifecycle_stats['lifecycle_current_missing_in_lcm_any']} "
            f"last_finalized_missing_in_lcm_any={lifecycle_stats['lifecycle_last_finalized_missing_in_lcm_any']} "
            f"current_missing_in_state={lifecycle_stats['lifecycle_current_missing_in_state']} "
            f"last_finalized_missing_in_state={lifecycle_stats['lifecycle_last_finalized_missing_in_state']} "
            f"message_sessions_missing_in_state={lifecycle_stats['lcm_message_sessions_missing_in_state']} "
            f"node_sessions_missing_in_state={lifecycle_stats['lcm_node_sessions_missing_in_state']} "
            f"message_sessions_without_lifecycle_current={lifecycle_stats['message_sessions_without_lifecycle_current']} "
            f"message_sessions_without_lifecycle_reference={lifecycle_stats['message_sessions_without_lifecycle_reference']} "
            f"node_sessions_without_lifecycle_reference={lifecycle_stats['node_sessions_without_lifecycle_reference']} "
            f"state_sessions_missing_in_lcm_any={lifecycle_stats['state_sessions_missing_in_lcm_any']}"
        )
        if lifecycle_stats.get("state_db_error"):
            observations.append(f"lifecycle_fragmentation_state_db_error: {lifecycle_stats['state_db_error']}")
        classification = lifecycle_stats.get("classification") or {}
        categories = classification.get("categories") or []
        if classification:
            observations.append(
                "lifecycle_fragmentation_classification: "
                f"{classification.get('status', 'unknown')}; {len(categories)} categories need review"
            )
            for category in categories:
                sample = ",".join(category.get("sample_session_ids") or []) or "(none)"
                observations.append(
                    "lifecycle_category "
                    f"{category.get('name')}: count={category.get('count', 0)} sample={sample}"
                )
        if _has_lifecycle_fragmentation(lifecycle_stats):
            recommended_actions.append(
                "inspect lifecycle fragmentation before any cleanup/repair behavior mutates state"
            )
            recommended_actions.append(
                "treat this as read-only evidence; do not infer every mismatch is harmful"
            )

    if clean_scan.get("protected_count"):
        observations.append(
            f"protected_sessions: skipped {clean_scan['protected_count']} currently bound session(s) from cleanup candidates"
        )

    protection = sensitive_pattern_status(engine._config)
    if protection["enabled"] and protection["active_patterns"]:
        observations.append(
            "sensitive_pattern_handling: enabled; matching raw secret values are replaced before SQLite, FTS, summaries, active replay, and externalized payloads"
        )
    elif protection["enabled"]:
        observations.append("sensitive_pattern_handling: enabled but no active known patterns are configured")
        recommended_actions.append("set LCM_SENSITIVE_PATTERNS to one or more known names, or disable sensitive handling")
    else:
        observations.append("sensitive_pattern_handling: disabled")
    if protection["unknown_patterns"]:
        issues.append("sensitive_pattern_config")
        recommended_actions.append(
            "remove unknown LCM_SENSITIVE_PATTERNS entries or replace them with supported names"
        )

    if store_fts_failed_flag:
        observations.append(
            "messages_fts_integrity: a background integrity scan flagged corruption "
            f"(detail: {store_fts_failed_flag['detail'] or 'unknown'})"
        )
        recommended_actions.append(
            "run `/lcm doctor repair`, then `/lcm backup` and `/lcm doctor repair apply` to rebuild messages_fts"
        )
    if node_fts_failed_flag:
        observations.append(
            "nodes_fts_integrity: a background integrity scan flagged corruption "
            f"(detail: {node_fts_failed_flag['detail'] or 'unknown'})"
        )
        recommended_actions.append(
            "run `/lcm doctor repair`, then `/lcm backup` and `/lcm doctor repair apply` to rebuild nodes_fts"
        )

    triage_checks: list[dict[str, Any]] = []
    if integrity != "ok":
        triage_checks.append({"check": "database_integrity", "status": "fail", "detail": integrity})
    if schema_health.get("error") or schema_missing_tables:
        triage_checks.append({"check": "schema_core_tables", "status": "fail", "detail": schema_health})
    if store_fts != "ok":
        triage_checks.append({
            "check": "messages_fts_integrity",
            "status": "warn" if store_fts == "unchecked" else "fail",
            "detail": store_fts_integrity,
        })
    if node_fts != "ok":
        triage_checks.append({
            "check": "nodes_fts_integrity",
            "status": "warn" if node_fts == "unchecked" else "fail",
            "detail": node_fts_integrity,
        })
    if store_fts_failed_flag and store_fts != "fail":
        triage_checks.append({
            "check": "messages_fts_integrity",
            "status": "fail",
            "detail": {"status": "fail", "background_flag": store_fts_failed_flag},
        })
    if node_fts_failed_flag and node_fts != "fail":
        triage_checks.append({
            "check": "nodes_fts_integrity",
            "status": "fail",
            "detail": {"status": "fail", "background_flag": node_fts_failed_flag},
        })
    if clean_scan["candidates"]:
        triage_checks.append({"check": "cleanup_candidates", "status": "warn", "detail": clean_scan})
    if payload_storage_error or missing_externalized_refs or any(payload_risks.get(key) for key in (
        "suspicious_data_uri_content_rows",
        "suspicious_data_uri_tool_calls_rows",
        "suspicious_base64_like_rows",
        "suspicious_repetitive_assistant_rows",
        "heartbeat_noise_rows",
    )):
        detail = {**payload_risks, **externalized_integrity}
        if payload_storage_error:
            detail["error"] = payload_storage_error
        triage_checks.append({
            "check": "payload_storage",
            "status": "fail" if payload_storage_error else "warn",
            "detail": detail,
        })
    if (protection["enabled"] and not protection["active_patterns"]) or protection["unknown_patterns"]:
        triage_checks.append({"check": "sensitive_pattern_handling", "status": "warn", "detail": protection})
    if source_stats.get("error"):
        triage_checks.append({"check": "source_lineage_hygiene", "status": "fail", "detail": source_stats})
    if lifecycle_stats.get("error") or _has_lifecycle_fragmentation(lifecycle_stats):
        lifecycle_status = "fail" if lifecycle_stats.get("error") else "warn"
        triage_checks.append({"check": "lifecycle_fragmentation", "status": lifecycle_status, "detail": lifecycle_stats})
    triage_guidance = doctor_guidance_for_checks(triage_checks)

    doctor_status = "issues-found" if integrity != "ok" or issues else (
        "action-recommended" if recommended_actions else "ok"
    )
    lines = [
        "LCM doctor",
        f"status: {doctor_status}",
        f"plugin_name: {runtime_identity.get('plugin_name', '(unknown)')}",
        f"plugin_version: {runtime_identity.get('plugin_version', '(unknown)')}",
        f"plugin_path: {runtime_identity.get('plugin_path', '(unknown)')}",
        f"module_path: {runtime_identity.get('module_path', '(unknown)')}",
        f"plugin_git_commit: {runtime_identity.get('plugin_git_commit') or '(unavailable)'}",
        f"plugin_git_branch: {runtime_identity.get('plugin_git_branch') or '(unavailable)'}",
        f"plugin_git_dirty: {runtime_identity.get('plugin_git_dirty') if runtime_identity.get('plugin_git_dirty') is not None else '(unavailable)'}",
        f"database_path: {db_path}",
        f"database_exists: {_fmt_bool(db_exists)}",
        f"database_size: {_fmt_size(db_size) if db_exists else 'missing'}",
        f"wal_size: {_fmt_size(wal_size)}",
        f"schema_core_tables: {schema_core_status}",
        f"schema_missing_tables: {', '.join(schema_missing_tables) or '(none)'}",
        f"schema_existing_tables: {', '.join(schema_existing_tables) or '(none)'}",
        f"journal_mode: {journal_mode}",
        f"quick_check: {quick_check}",
        f"sqlite_integrity: {integrity}",
        f"messages_total: {total_messages}",
        f"message_sessions_total: {total_message_sessions}",
        f"summary_nodes_total: {total_nodes}",
        f"summary_node_sessions_total: {total_node_sessions}",
        f"messages_fts: {store_fts}",
        f"messages_fts_rows: {store_fts_count}",
        f"nodes_fts: {node_fts}",
        f"nodes_fts_rows: {node_fts_count}",
        f"largest_content_rows: {payload_risks['largest_content_rows']}",
        f"largest_tool_calls_rows: {payload_risks['largest_tool_calls_rows']}",
        f"suspicious_data_uri_content_rows: {payload_risks['suspicious_data_uri_content_rows']}",
        f"suspicious_data_uri_tool_calls_rows: {payload_risks['suspicious_data_uri_tool_calls_rows']}",
        f"suspicious_base64_like_rows: {payload_risks['suspicious_base64_like_rows']}",
        f"quarantined_assistant_rows: {payload_risks['quarantined_assistant_rows']}",
        f"suspicious_repetitive_assistant_rows: {payload_risks['suspicious_repetitive_assistant_rows']}",
        f"heartbeat_noise_rows: {payload_risks['heartbeat_noise_rows']}",
        f"sensitive_patterns_enabled: {_fmt_bool(protection.get('enabled'))}",
        f"sensitive_patterns: {', '.join(protection.get('patterns') or []) or '(none)'}",
        f"sensitive_patterns_source: {protection.get('source', 'default')}",
        f"sensitive_patterns_unknown: {', '.join(protection.get('unknown_patterns') or []) or '(none)'}",
        f"externalized_payload_dir: {externalized_stats['externalized_payload_dir']}",
        f"externalized_payload_count: {externalized_stats['externalized_payload_count']}",
        f"externalized_payload_bytes: {externalized_stats['externalized_payload_bytes']}",
        f"externalized_payload_chars: {externalized_stats['externalized_payload_chars']}",
        f"latest_externalized_payload_path: {externalized_stats['latest_externalized_payload_path'] or '(none)'}",
        f"externalized_payload_refs_total: {externalized_integrity['externalized_payload_refs_total']}",
        f"externalized_payload_refs_existing: {externalized_integrity['externalized_payload_refs_existing']}",
        f"externalized_payload_refs_missing: {externalized_integrity['externalized_payload_refs_missing']}",
        f"externalized_payload_files_unreferenced: {externalized_integrity['externalized_payload_files_unreferenced']}",
        f"missing_externalized_payload_refs: {externalized_integrity['missing_externalized_payload_refs']}",
        f"unreferenced_externalized_payload_files: {externalized_integrity['unreferenced_externalized_payload_files']}",
    ]
    if issues:
        lines.append(f"issues: {', '.join(issues)}")
    else:
        lines.append("issues: none")
    lines.append("observations:")
    for item in observations:
        lines.append(f"- {item}")
    lines.append("recommended_actions:")
    if recommended_actions:
        for item in recommended_actions:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("triage_guidance:")
    if triage_guidance:
        for item in triage_guidance:
            warning_suffix = " warning-only" if item.get("warning_only") else ""
            lines.append(
                "- "
                f"{item['check']}: {item['action']}{warning_suffix} — "
                f"{item['operator_action']}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def _doctor_clean_text(engine) -> str:
    scan = _scan_clean_candidates(engine)
    if scan["error"]:
        return "\n".join([
            "LCM doctor clean",
            "status: error",
            f"error: {scan['error']}",
            "note: read-only scan only — no rows were deleted",
        ])

    candidates = scan["candidates"]
    lines = [
        "LCM doctor clean",
        f"status: {'candidates-found' if candidates else 'ok'}",
        f"candidate_sessions: {len(candidates)}",
        f"ignored_pattern_matches: {scan['ignored_count']}",
        f"stateless_pattern_matches: {scan['stateless_count']}",
    ]
    if scan["protected_count"]:
        lines.append(f"protected_sessions_skipped: {scan['protected_count']}")

    if not candidates:
        lines.append("result: no obvious junk/noise session candidates detected")
        return "\n".join(lines)

    lines.append("candidates:")
    for item in candidates[:20]:
        classes = ", ".join(item["classes"])
        lines.append(
            "- "
            f"{item['session_id']} | class={classes} | messages={item['message_count']} | "
            f"nodes={item['node_count']} | tokens={item['token_total']}"
        )
    if len(candidates) > 20:
        lines.append(f"... {len(candidates) - 20} more candidate session(s) omitted")
    lines.append("note: best-effort stored-session scan only — platform-only matches may not be reconstructable from the SQLite state")
    lines.append("note: read-only scan only — no rows were deleted")
    lines.append("note: use `/lcm doctor clean apply` only after a backup-first review of these safe candidates")
    return "\n".join(lines)


def _doctor_retention_text(engine) -> str:
    scan = _scan_retention_candidates(engine)
    if scan["error"]:
        return "\n".join([
            "LCM doctor retention",
            "status: error",
            f"error: {scan['error']}",
            "note: read-only analysis only — no rows were deleted",
        ])

    sessions = scan["sessions"]
    lines = [
        "LCM doctor retention",
        f"status: {'analysis-ready' if sessions else 'ok'}",
        f"sessions_analyzed: {scan['sessions_analyzed']}",
        f"stale_sessions_30d: {scan['stale_sessions_30d']}",
        f"stale_sessions_90d: {scan['stale_sessions_90d']}",
        f"retained_tokens_30d: {scan['retained_tokens_30d']}",
        f"retained_tokens_90d: {scan['retained_tokens_90d']}",
    ]
    if scan["protected_count"]:
        lines.append(f"protected_sessions: {scan['protected_count']}")

    if not sessions:
        lines.append("result: no stored sessions found for retention analysis")
        lines.append("note: read-only analysis only — no rows were deleted")
        return "\n".join(lines)

    lines.append("retention_candidates:")
    for item in sessions[:20]:
        lines.append(
            "- "
            f"{item['session_id']} | protected={'yes' if item['protected'] else 'no'} | "
            f"messages={item['message_count']} | nodes={item['node_count']} | "
            f"tokens={item['token_total']} | age_days={item['age_days']:.1f}"
        )
    if len(sessions) > 20:
        lines.append(f"... {len(sessions) - 20} more session(s) omitted")
    lines.append("note: retention analysis is scoped to the active session only")
    lines.append("note: stale sessions are listed before fresh ones; within each bucket, candidates are sorted by footprint (tokens/nodes/messages), with protected current-session entries listed after non-protected ones")
    lines.append("note: read-only analysis only — no rows were deleted")
    lines.append("note: if you prune later, create a safety snapshot first with `/lcm backup`")
    return "\n".join(lines)


def _delete_clean_candidates_atomically(engine, session_ids: set[str]) -> dict[str, int]:
    """Delete cleanup candidates in one SQLite transaction.

    All LCM tables live in the same SQLite database, but the store, DAG, and
    lifecycle helpers use separate connections and commit internally. Cleanup
    apply is destructive, so do the coordinated deletes on one connection to
    avoid half-cleaned state if a later table delete fails.
    """
    conn = engine._store.connection
    # Protect the actively-bound session id, not current_session_id. While a
    # cron tick has rebound the engine, _session_id is the row the engine is
    # actively writing to via lifecycle hooks; deleting it during cleanup
    # would race with that ingest.
    protected_session_ids = {getattr(engine, "_session_id", "")}
    protected_session_ids = {str(s) for s in protected_session_ids if s}
    session_ids = {str(s) for s in session_ids if s and str(s) not in protected_session_ids}
    if not session_ids:
        return {
            "messages_deleted": 0,
            "nodes_deleted": 0,
            "lifecycle_deleted": 0,
            "lifecycle_skipped": 0,
        }

    try:
        conn.execute("BEGIN IMMEDIATE")
        SummaryDAG.stage_delete_session_scope(conn, session_ids)
        scope_table = SummaryDAG.DELETE_SESSION_SCOPE_TABLE
        # Capture the store_ids about to be deleted so their raw-history chunks
        # can be archived in this same transaction (chunks map to messages by
        # store_id; a deleted message's chunks must drop from ranking).
        deleted_store_ids = [
            int(row[0])
            for row in conn.execute(
                f"SELECT store_id FROM messages WHERE EXISTS ("
                f"SELECT 1 FROM {scope_table} AS scope "
                "WHERE scope.session_id = messages.session_id)"
            ).fetchall()
        ]
        msg_cur = conn.execute(
            f"DELETE FROM messages WHERE EXISTS ("
            f"SELECT 1 FROM {scope_table} AS scope "
            "WHERE scope.session_id = messages.session_id)"
        )
        archive_chunks = getattr(engine, "_archive_chunks_for_messages", None)
        if callable(archive_chunks) and deleted_store_ids:
            archive_chunks(deleted_store_ids, connection=conn)
        nodes_deleted = 0
        purge = getattr(engine, "_purge_embeddings_for_nodes", None)
        while True:
            deleted_ids = SummaryDAG.delete_node_batch(
                conn,
                (),
                staged_scope=True,
            )
            if not deleted_ids:
                break
            nodes_deleted += len(deleted_ids)
            if callable(purge):
                purge(deleted_ids, connection=conn)

        lifecycle_scope = "temp_lcm_delete_lifecycle_scope"
        conn.execute(
            f"CREATE TEMP TABLE IF NOT EXISTS {lifecycle_scope}("
            "conversation_id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        conn.execute(f"DELETE FROM {lifecycle_scope}")
        conn.execute(
            f"INSERT OR IGNORE INTO {lifecycle_scope}(conversation_id) "
            f"SELECT state.conversation_id FROM {scope_table} AS scope "
            "JOIN lcm_lifecycle_state AS state "
            "INDEXED BY idx_lcm_lifecycle_current_session "
            "ON state.current_session_id = scope.session_id"
        )
        conn.execute(
            f"INSERT OR IGNORE INTO {lifecycle_scope}(conversation_id) "
            f"SELECT state.conversation_id FROM {scope_table} AS scope "
            "JOIN lcm_lifecycle_state AS state "
            "INDEXED BY idx_lcm_lifecycle_last_finalized_session "
            "ON state.last_finalized_session_id = scope.session_id"
        )

        protected = next(iter(protected_session_ids), "")
        deletable_where = f"""
            scoped.conversation_id = state.conversation_id
            AND (state.current_session_id IS NULL OR state.current_session_id = ''
                 OR EXISTS (SELECT 1 FROM {scope_table} AS current_scope
                            WHERE current_scope.session_id = state.current_session_id))
            AND (state.last_finalized_session_id IS NULL
                 OR state.last_finalized_session_id = ''
                 OR EXISTS (SELECT 1 FROM {scope_table} AS finalized_scope
                            WHERE finalized_scope.session_id = state.last_finalized_session_id))
            AND (? = '' OR COALESCE(state.current_session_id, '') != ?)
            AND (? = '' OR COALESCE(state.last_finalized_session_id, '') != ?)
        """
        scoped_count = int(
            conn.execute(f"SELECT COUNT(*) FROM {lifecycle_scope}").fetchone()[0]
        )
        lifecycle_deleted = 0
        while True:
            rows = conn.execute(
                f"SELECT state.conversation_id FROM lcm_lifecycle_state AS state "
                f"JOIN {lifecycle_scope} AS scoped ON {deletable_where} "
                "ORDER BY state.conversation_id LIMIT 256",
                (protected, protected, protected, protected),
            ).fetchall()
            if not rows:
                break
            conversation_ids = [str(row[0]) for row in rows]
            placeholders = ",".join("?" for _ in conversation_ids)
            cur = conn.execute(
                f"DELETE FROM lcm_lifecycle_state "
                f"WHERE conversation_id IN ({placeholders})",
                conversation_ids,
            )
            lifecycle_deleted += cur.rowcount if cur.rowcount is not None else 0
        lifecycle_skipped = scoped_count - lifecycle_deleted
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "messages_deleted": msg_cur.rowcount if msg_cur.rowcount is not None else 0,
        "nodes_deleted": nodes_deleted,
        "lifecycle_deleted": lifecycle_deleted,
        "lifecycle_skipped": lifecycle_skipped,
    }


def _doctor_clean_apply_text(engine) -> str:
    if not getattr(getattr(engine, "_config", None), "doctor_clean_apply_enabled", False):
        return "\n".join([
            "LCM doctor clean apply",
            "status: denied",
            "error: destructive cleanup is disabled by default",
            "note: set LCM_DOCTOR_CLEAN_APPLY_ENABLED=true only in trusted operator environments",
            "note: no rows were deleted",
        ])

    scan = _scan_clean_candidates(engine)
    if scan["error"]:
        return "\n".join([
            "LCM doctor clean apply",
            "status: error",
            f"error: {scan['error']}",
            "note: cleanup apply aborted before any rows were deleted",
        ])

    candidates = scan["candidates"]
    if not candidates:
        return "\n".join([
            "LCM doctor clean apply",
            "status: ok",
            "candidate_sessions: 0",
            "result: no safe cleanup candidates detected",
            "note: nothing was deleted",
        ])

    backup = backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM doctor clean apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: backup failed: {backup['error']}",
            "note: cleanup apply aborted before any rows were deleted",
        ])

    session_ids = {item["session_id"] for item in candidates}
    try:
        deleted = _delete_clean_candidates_atomically(engine, session_ids)
    except sqlite3.Error as exc:
        return "\n".join([
            "LCM doctor clean apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"backup_path: {backup['backup_path']}",
            f"backup_size: {_fmt_size(int(backup['backup_size']))}",
            f"error: cleanup apply failed: {exc}",
            "note: cleanup apply rolled back; restore from the backup if you need to inspect pre-apply state",
        ])

    return "\n".join([
        "LCM doctor clean apply",
        "status: ok",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
        f"candidate_sessions: {len(candidates)}",
        f"messages_deleted: {deleted['messages_deleted']}",
        f"nodes_deleted: {deleted['nodes_deleted']}",
        f"lifecycle_rows_deleted: {deleted['lifecycle_deleted']}",
        f"lifecycle_rows_skipped: {deleted['lifecycle_skipped']}",
        "note: backup created before cleanup apply",
    ])


def _doctor_clean_lifecycle_text(engine) -> str:
    count = engine._lifecycle.row_count()
    protected = {str(getattr(engine, "_session_id", "") or "")}
    protected = {s for s in protected if s}

    conn = engine._lifecycle.connection
    sessions_with_data: set[str] = set()
    for row in conn.execute("SELECT DISTINCT session_id FROM messages").fetchall():
        sessions_with_data.add(str(row[0]))
    for row in conn.execute("SELECT DISTINCT session_id FROM summary_nodes").fetchall():
        sessions_with_data.add(str(row[0]))

    empty_current = 0
    empty_finalized = 0
    empty_protected = 0
    rows = conn.execute("SELECT * FROM lcm_lifecycle_state").fetchall()
    for row in rows:
        cur = str(row["current_session_id"] or "")
        fin = str(row["last_finalized_session_id"] or "")
        if ((cur and cur in sessions_with_data)
                or (fin and fin in sessions_with_data)):
            continue
        refs = {r for r in (cur, fin) if r}
        if refs & protected:
            empty_protected += 1
            continue
        if cur and not fin:
            empty_current += 1
        else:
            empty_finalized += 1

    total_empty = empty_current + empty_finalized
    if total_empty == 0:
        return "\n".join([
            "LCM doctor clean lifecycle",
            "status: ok",
            f"lifecycle_rows: {count}",
            "empty_rows: 0",
            "note: no empty lifecycle rows to prune",
        ])

    return "\n".join([
        "LCM doctor clean lifecycle",
        "status: candidates-found",
        f"lifecycle_rows: {count}",
        f"empty_rows: {total_empty}",
        f"  empty_current: {empty_current}",
        f"  empty_finalized: {empty_finalized}",
        f"  empty_protected: {empty_protected}",
        "note: read-only scan — no rows were deleted",
        "note: empty rows reference sessions with zero messages and zero nodes",
        "note: use `/lcm doctor clean lifecycle apply` to delete empty rows",
    ])


def _doctor_clean_lifecycle_apply_text(engine) -> str:
    if not getattr(getattr(engine, "_config", None), "doctor_clean_apply_enabled", False):
        return "\n".join([
            "LCM doctor clean lifecycle apply",
            "status: denied",
            "error: destructive cleanup is disabled by default",
            "note: set LCM_DOCTOR_CLEAN_APPLY_ENABLED=true only in trusted operator environments",
            "note: no rows were deleted",
        ])

    backup = backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM doctor clean lifecycle apply",
            "status: error",
            "error: failed to create backup before destructive cleanup",
            f"database_path: {backup['db_path']}",
            f"backup_error: {backup['error']}",
            "note: no rows were deleted",
        ])

    before = engine._lifecycle.row_count()
    protected = {str(getattr(engine, "_session_id", "") or "")}
    protected = {s for s in protected if s}

    try:
        deleted = engine._lifecycle.prune_empty_sessions(
            protected_session_ids=protected,
        )
    except Exception as exc:
        return "\n".join([
            "LCM doctor clean lifecycle apply",
            "status: error",
            "error: failed to prune empty sessions",
            f"backup_path: {backup['backup_path']}",
            f"prune_error: {exc}",
            "note: no rows were deleted",
        ])

    after = engine._lifecycle.row_count()
    return "\n".join([
        "LCM doctor clean lifecycle apply",
        "status: ok",
        f"lifecycle_rows_before: {before}",
        f"lifecycle_rows_deleted: {deleted}",
        f"lifecycle_rows_remaining: {after}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size_bytes: {backup['backup_size']}",
        "note: only empty lifecycle rows were deleted — messages and nodes untouched",
    ])


def _backup_text(engine) -> str:
    backup = backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM backup",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: {backup['error']}",
        ])

    return "\n".join([
        "LCM backup",
        "status: ok",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
        "note: backup created before any future cleanup/apply workflow",
    ])


def _unknown_preset_text(name: str) -> str:
    available = ", ".join(preset.name for preset in shipped_presets()) or "(none)"
    return "\n".join([
        "LCM preset",
        "status: error",
        f"error: unknown preset {name}",
        f"available_presets: {available}",
    ])


def _preset_show_text(tokens: list[str], engine) -> str:
    if len(tokens) > 1:
        return _help_text("`/lcm preset show` accepts at most one preset name.")
    preset = get_preset(tokens[0] if tokens else None)
    if preset is None:
        return _unknown_preset_text(tokens[0])
    provenance = dict(preset.provenance)
    metric_summary = dict(provenance.get("metric_summary") or {})
    fixture_suite = ", ".join(str(item) for item in provenance.get("fixture_suite") or []) or "(unknown)"
    applies_to = ", ".join(preset.applies_to) if preset.applies_to else "(unspecified)"
    lines = [
        "LCM preset show",
        f"preset: {preset.name}",
        f"family: {preset.family}",
        f"description: {preset.description}",
        f"policy_version: {preset.policy_version}",
        f"policy_path: {preset.policy_path}",
        f"benchmark_version: {provenance.get('benchmark_version', '(unknown)')}",
        f"fixture_suite: {fixture_suite}",
        f"score: {metric_summary.get('score', '(unknown)')}",
        f"baseline_score: {metric_summary.get('baseline_score', '(unknown)')}",
        f"retrieval_canary_recall: {metric_summary.get('retrieval_canary_recall', '(unknown)')}",
        f"applies_to: {applies_to}",
        "runtime_env:",
    ]
    for item in preset_env_diff(preset, engine._config):
        lines.append(f"- {item}")
    lines.extend([
        f"unsupported_runtime_fields: {unsupported_runtime_fields_text(preset)}",
        "operator_config_precedence: explicit preset-managed LCM_* overrides win",
        "runtime_mutation: no",
        f"notes: {preset.notes}",
    ])
    return "\n".join(lines)


def _preset_suggest_text(engine) -> str:
    preset, reason = suggest_preset_for_engine(engine)
    lines = ["LCM preset suggest"]
    if preset is None:
        lines.extend([
            "suggested_preset: (none)",
            f"reason: {reason}",
            "note: run deterministic benchmarks before promoting a runtime preset",
            "note: suggestion only; no live config was changed",
        ])
        return "\n".join(lines)

    explicit = explicit_operator_overrides()
    invalid = invalid_operator_overrides()
    invalid_text = ", ".join(
        f"{env_var}={os.environ.get(env_var, '')}" for env_var in sorted(invalid.values())
    ) if invalid else "(none)"
    lines.extend([
        f"suggested_preset: {preset.name}",
        f"reason: {reason}",
        f"match_confidence: {preset_match_confidence(engine, preset)}",
        f"policy_version: {preset.policy_version}",
        f"benchmark_version: {preset.provenance.get('benchmark_version', '(unknown)')}",
        "explicit_overrides: " + (", ".join(sorted(explicit.values())) if explicit else "(none)"),
        f"invalid_overrides: {invalid_text}",
        "confidence_reasons:",
    ])
    for item in preset_confidence_reasons(engine, preset, reason):
        lines.append(f"- {item}")
    lines.extend([
        "preview:",
    ])
    for item in preset_env_diff(
        preset,
        engine._config,
        runtime_context_threshold=getattr(engine, "context_threshold", None),
        runtime_context_threshold_source=getattr(engine, "_context_threshold_source", ""),
    ):
        lines.append(f"- {item}")
    lines.extend([
        f"unsupported_runtime_fields: {unsupported_runtime_fields_text(preset)}",
        "note: suggestion only; no live config was changed",
    ])
    return "\n".join(lines)


def _preset_apply_text(tokens: list[str], engine) -> str:
    if not tokens:
        return _help_text("`/lcm preset apply` requires a preset name and `--dry-run`.")
    dry_run = "--dry-run" in tokens
    selected = [token for token in tokens if token != "--dry-run"]
    if len(selected) != 1:
        return _help_text("`/lcm preset apply` accepts exactly one preset name and optional `--dry-run`.")
    preset_name = selected[0]
    preset = get_preset(preset_name)
    if preset is None:
        return _unknown_preset_text(preset_name)
    if not dry_run:
        return "\n".join([
            "LCM preset apply",
            "status: denied",
            "error: preset apply is preview-only for now; pass --dry-run",
            "note: no live config was changed",
        ])

    lines = [
        "LCM preset apply",
        "status: dry-run",
        f"preset: {preset.name}",
        "would_set:",
    ]
    for item in preset_env_diff(
        preset,
        engine._config,
        runtime_context_threshold=getattr(engine, "context_threshold", None),
        runtime_context_threshold_source=getattr(engine, "_context_threshold_source", ""),
    ):
        lines.append(f"- {item}")
    lines.extend([
        f"unsupported_runtime_fields: {unsupported_runtime_fields_text(preset)}",
        "operator_config_precedence: explicit preset-managed LCM_* overrides win",
        "note: no live config was changed",
    ])
    return "\n".join(lines)


def _preset_text(tokens: list[str], engine) -> str:
    if not tokens:
        return _help_text("`/lcm preset` requires `show`, `suggest`, or `apply`.")
    subcommand = tokens[0].lower()
    rest = tokens[1:]
    if subcommand == "show":
        return _preset_show_text(rest, engine)
    if subcommand == "suggest":
        if rest:
            return _help_text("`/lcm preset suggest` does not accept extra arguments.")
        return _preset_suggest_text(engine)
    if subcommand == "apply":
        return _preset_apply_text(rest, engine)
    return _help_text("`/lcm preset` supports `show`, `suggest`, and `apply`.")


def _rollups_status_text(engine) -> str:
    # Import lazily to avoid making the slash-command module part of the tool
    # module's import path while still guaranteeing both surfaces use one shape.
    from .tools import _temporal_rollups_status

    status = _temporal_rollups_status(engine)
    lines = [
        "LCM temporal rollups",
        f"enabled: {_fmt_bool(status['enabled'])}",
        f"scope: {status['scope'] or '(unbound)'}",
        "period | ready | stale | building | failed",
    ]
    for kind in ("day", "week", "month"):
        counts = status["counts"][kind]
        lines.append(
            f"{kind} | {counts['ready']} | {counts['stale']} | "
            f"{counts['building']} | {counts['failed']}"
        )
    oldest_age = status["oldest_stale_age_seconds"]
    lines.append(
        "oldest_stale_age_seconds: "
        + (str(oldest_age) if oldest_age is not None else "(none)")
    )
    for kind in ("day", "week", "month"):
        cursor = status["last_build_cursors"][kind]
        built_at = status["last_built_at"][kind]
        lines.append(f"last_build_cursor_{kind}: {cursor or '(none)'}")
        lines.append(f"last_built_at_{kind}: {built_at or '(never)'}")
    lines.append(f"last_error: {status['last_error'] or '(none)'}")
    if status.get("query_error"):
        lines.append(f"query_error: {status['query_error']}")
    if status.get("truncated_fields"):
        lines.append("truncated: true")
        lines.append("truncated_fields: " + ", ".join(status["truncated_fields"]))
    if not status["enabled"]:
        lines.append("note: temporal rollups are disabled; set LCM_TEMPORAL_ROLLUPS_ENABLED=true and restart Hermes to enable them")
    return "\n".join(lines)


def _rollup_period_targets(kind: str, target_date: date) -> list[tuple[str, date]]:
    week_start = target_date.fromordinal(target_date.toordinal() - target_date.weekday())
    month_start = target_date.replace(day=1)
    starts = {
        "day": target_date,
        "week": week_start,
        "month": month_start,
    }
    kinds = ("day", "week", "month") if kind == "all" else (kind,)
    return [(period_kind, starts[period_kind]) for period_kind in kinds]


def _classify_rollup_build_outcome(
    result: dict[str, object] | None,
    row: dict[str, object] | None,
) -> tuple[_RollupBuildOutcome, str | None]:
    """Map a completed builder call to an honest typed operator outcome."""
    if result is not None and str(result.get("status") or "") == "ready":
        return _RollupBuildOutcome.READY, None
    if row is None:
        return _RollupBuildOutcome.NO_SOURCE, None

    status = str(row.get("status") or "missing")
    error = str(row.get("error") or "").strip() or None
    if status == "ready":
        return _RollupBuildOutcome.READY, None
    if status == "failed":
        return _RollupBuildOutcome.FAILED, error
    if status == "stale" and error and error.startswith("incomplete:"):
        return _RollupBuildOutcome.DEFERRED, error
    # A builder that returns with its claimed row stale/building/missing a
    # success result no longer owns a publishable terminal transition.
    return _RollupBuildOutcome.SUPERSEDED, status


def _rollups_rebuild_text(tokens: list[str], engine) -> str:
    if not engine._config.temporal_rollups_enabled:
        return "\n".join([
            "LCM temporal rollup rebuild",
            "status: disabled",
            "error: temporal rollups are disabled",
            "note: set LCM_TEMPORAL_ROLLUPS_ENABLED=true and restart Hermes before rebuilding",
        ])
    if not engine.current_session_id:
        return "\n".join([
            "LCM temporal rollup rebuild",
            "status: refused",
            "error: no active session",
        ])
    if not tokens or len(tokens) > 2:
        return _help_text("`/lcm rollups rebuild` requires <day|week|month|all> and accepts one optional YYYY-MM-DD date.")

    kind = tokens[0].lower()
    if kind not in {"day", "week", "month", "all"}:
        return _help_text("`/lcm rollups rebuild` period must be one of: day, week, month, all.")
    if len(tokens) == 2:
        try:
            target_date = date.fromisoformat(tokens[1])
        except ValueError:
            return _help_text("`/lcm rollups rebuild` date must be a valid YYYY-MM-DD UTC date.")
    else:
        target_date = datetime.now(timezone.utc).date()

    scope = engine.current_session_id
    targets = _rollup_period_targets(kind, target_date)
    limit = max(0, int(engine._config.rollup_builds_per_pass))
    store = RollupStore(engine._dag.db_path)
    outcomes: list[_RollupRebuildResult] = []
    try:
        if store.connection is None:  # pragma: no cover - RollupStore initialization contract
            raise RuntimeError("temporal rollup store is unavailable")
        # Durably seed a stale row for EVERY requested target BEFORE applying the
        # per-pass build budget, so targets beyond the budget remain durably
        # 'stale' (not absent) and are built by later maintenance (maintainer #391
        # blocker). The whole multi-target seed is ONE transaction so a mid-batch
        # failure cannot leave some targets seeded and others absent — e.g. a
        # missing month with no row (maintainer #391 D2). upsert_stale_many leaves
        # a currently-'building' row untouched.
        store.upsert_stale_many(
            [
                (period_kind, period_start.isoformat(), scope)
                for period_kind, period_start in targets
            ]
        )

        builders = {
            "day": rollup_builder.build_day,
            "week": rollup_builder.build_week,
            "month": rollup_builder.build_month,
        }
        for index, (period_kind, period_start) in enumerate(targets):
            period_key = period_start.isoformat()
            if index >= limit:
                row = store.get_rollup(period_kind, period_key, scope)
                status = str(row["status"]) if row else "missing"
                outcomes.append(
                    _RollupRebuildResult(
                        period_kind,
                        period_key,
                        _RollupBuildOutcome.QUEUED,
                        attempted=False,
                        detail=status,
                    )
                )
                continue
            result = builders[period_kind](
                store,
                engine._dag,
                engine._config,
                scope,
                period_start,
                circuit_breaker=engine._summary_circuit_breaker,
                spend_guard=engine._summary_spend_guard,
            )
            row = store.get_rollup(period_kind, period_key, scope)
            outcome, detail = _classify_rollup_build_outcome(result, row)
            outcomes.append(
                _RollupRebuildResult(
                    period_kind,
                    period_key,
                    outcome,
                    attempted=True,
                    detail=detail,
                )
            )
    except Exception as exc:  # pragma: no cover - defensive operator surface
        return "\n".join([
            "LCM temporal rollup rebuild",
            "status: error",
            f"error: {type(exc).__name__}: {exc}",
        ])
    finally:
        store.close()

    # ``complete`` is reserved for attempted targets that all reached ready.
    # Explicitly bounded, unattempted queued debt may coexist with complete.
    attempted_incomplete = any(
        outcome.attempted and outcome.outcome is not _RollupBuildOutcome.READY
        for outcome in outcomes
    )
    top_status = "partial" if attempted_incomplete else "complete"
    lines = [
        "LCM temporal rollup rebuild",
        f"status: {top_status}",
        f"scope: {scope}",
        f"requested: {kind}",
        f"date_utc: {target_date.isoformat()}",
        f"build_limit: {limit}",
        "outcomes:",
    ]
    for outcome in outcomes:
        if not outcome.attempted:
            rendered = f"{outcome.detail or 'missing'} (bounded; not attempted)"
        else:
            rendered = outcome.outcome.value
            if outcome.detail:
                rendered += f" ({outcome.detail})"
        lines.append(f"- {outcome.period_kind} {outcome.period_start}: {rendered}")
    return "\n".join(lines)


def _rollups_text(tokens: list[str], engine) -> str:
    if not tokens:
        result = _rollups_status_text(engine)
    elif tokens[0].lower() == "rebuild":
        result = _rollups_rebuild_text(tokens[1:], engine)
    else:
        result = _help_text("`/lcm rollups` accepts only `rebuild <day|week|month|all> [date]`.")
    return _bounded_rollups_text(result)
def _resolve_storage_dtype(config, override: str | None = None) -> str:
    """Resolve the vector storage dtype: an explicit --dtype flag, else config.

    SPEC C1: float32 (default) keeps vectors byte-identical; int8 stores
    quantized vectors + a sign-bit prescreen and unlocks two-stage full-corpus
    KNN. Any unrecognized value degrades to float32.
    """
    value = str(
        override
        if override is not None
        else getattr(config, "embedding_storage_dtype", "float32")
    ).strip().lower()
    return "int8" if value == "int8" else "float32"


def _resolve_store_dim(config, provider_dim: int, override: int | None = None) -> int:
    """Resolve the Matryoshka store dim: --store-dim flag or config, capped to provider dim.

    0 (default) or any value >= the provider dim means "store the full dim".
    """
    raw = override if override is not None else getattr(config, "embedding_store_dim", 0)
    try:
        store_dim = int(raw or 0)
    except (TypeError, ValueError):
        store_dim = 0
    if store_dim <= 0 or store_dim >= int(provider_dim):
        return int(provider_dim)
    return store_dim


def _embedding_warmup_text(engine) -> str:
    """Warm and dimension-lock both summary and chunk vector profiles."""
    try:
        if not bool(getattr(engine._config, "embeddings_enabled", False)):
            return (
                "LCM embedding warmup\n"
                "status: disabled\n"
                "error: embeddings are disabled; set LCM_EMBEDDINGS_ENABLED=true"
            )
        provider = resolve_provider(engine._config)
        if provider is None:
            return (
                "LCM embedding warmup\n"
                "status: error\n"
                "error: embedding provider is not configured; set "
                "LCM_EMBEDDING_PROVIDER and LCM_EMBEDDING_MODEL"
            )

        if provider.provider_id == FastembedProvider.provider_id:
            vector = provider.warmup()
            progress = (
                f"download: ready ({fastembed_download_size_note(provider.model_id)})"
            )
            cost_note = "local model; no per-call API charge"
        else:
            vector = provider.embed_query("warmup")
            progress = "probe: complete"
            cost_note = (
                "API usage may incur provider charges"
                if provider.provider_id == "voyage"
                else "local Ollama model; no per-call API charge"
            )

        dim = len(vector)
        if dim < 1:
            raise ValueError("provider returned an empty warmup embedding")
        storage_dtype = _resolve_storage_dtype(engine._config)
        store_dim = _resolve_store_dim(engine._config, dim)
        chunk_model = default_chunk_model(provider.provider_id, provider.model_id)
        chunk_provider = provider
        chunk_dim = dim
        chunk_progress = "shared summary probe"
        if chunk_model != provider.model_id:
            chunk_config = dataclasses.replace(
                engine._config,
                embedding_provider=provider.provider_id,
                embedding_model=chunk_model,
            )
            chunk_provider = resolve_provider(chunk_config)
            if chunk_provider is None:
                raise ValueError("chunk embedding provider is not configured")
            chunk_vector = chunk_provider.embed_query("warmup")
            chunk_dim = len(chunk_vector)
            if chunk_dim < 1:
                raise ValueError("chunk provider returned an empty warmup embedding")
            chunk_progress = "probe: complete"
        chunk_store_dim = _resolve_store_dim(engine._config, chunk_dim)
        store = VectorStore(engine._store.db_path, config=engine._config)
        try:
            store.register_profile(
                provider.model_id,
                provider.provider_id,
                store_dim,
                dtype=storage_dtype,
            )
            store.register_profile(
                chunk_provider.model_id,
                chunk_provider.provider_id,
                chunk_store_dim,
                dtype=storage_dtype,
                task="chunk",
            )
        finally:
            store.close()
        # Semantic search caches provider instances by configured provider and
        # model. Replace any pre-warmup instance (which may have an open
        # breaker) with the provider that just completed warmup successfully.
        engine._lcm_embedding_provider_cache = (
            (
                str(getattr(engine._config, "embedding_provider", "") or "")
                .strip()
                .lower(),
                str(getattr(engine._config, "embedding_model", "") or "").strip(),
            ),
            provider,
        )
        return "\n".join([
            "LCM embedding warmup",
            "status: ready",
            progress,
            f"provider: {provider.provider_id}",
            f"model: {provider.model_id}",
            f"dim: {store_dim}",
            f"chunk_model: {chunk_provider.model_id}",
            f"chunk_dim: {chunk_store_dim}",
            f"chunk_probe: {chunk_progress}",
            f"dtype: {storage_dtype}",
            f"cost_note: {cost_note}",
        ])
    except Exception as exc:
        return "\n".join([
            "LCM embedding warmup",
            "status: error",
            f"error: {exc}",
        ])


def _embedding_read_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open the existing database without allowing a dry run to create it."""
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _embedding_current_profile(conn: sqlite3.Connection) -> sqlite3.Row | None:
    try:
        return conn.execute(
            """
            SELECT identity_hash, model_name, provider, revision, dim, dtype,
                   byteorder, task, registered_at
            FROM lcm_embedding_profile
            WHERE active = 1 AND archived_at IS NULL AND task = 'summary'
            ORDER BY registered_at DESC, identity_hash DESC
            LIMIT 1
            """
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return None
        raise


def _embedding_pending_rows(
    conn: sqlite3.Connection,
    identity_hash: str,
    limit: int,
) -> tuple[int, list[sqlite3.Row]]:
    # Discovery excludes both recorded rows and every durable in-flight state.
    # A dispatched/uncertain row may already have been accepted remotely, so it
    # must never be silently resent; only explicit --retry-uncertain operator
    # authorization can return it to discovery.
    inflight_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='lcm_embedding_backfill_inflight'"
    ).fetchone() is not None
    inflight_clause = (
        """
        AND NOT EXISTS (
            SELECT 1 FROM lcm_embedding_backfill_inflight AS f
            WHERE f.embedded_id = CAST(n.node_id AS TEXT)
              AND f.identity_hash = ?
        )
        """
        if inflight_exists
        else ""
    )
    where = """
        n.depth = 0
        AND NOT EXISTS (
            SELECT 1
            FROM lcm_embedding_meta AS m
            WHERE m.embedded_id = CAST(n.node_id AS TEXT)
              AND m.embedded_kind = 'summary'
              AND m.identity_hash = ?
        )
    """ + inflight_clause
    args: tuple[object, ...] = (
        (identity_hash, identity_hash) if inflight_exists else (identity_hash,)
    )
    total = int(conn.execute(
        f"SELECT COUNT(*) FROM summary_nodes AS n WHERE {where}",
        args,
    ).fetchone()[0])
    rows = conn.execute(
        f"""
        SELECT n.node_id, n.summary
        FROM summary_nodes AS n
        WHERE {where}
        ORDER BY COALESCE(n.latest_at, n.created_at) DESC, n.node_id DESC
        LIMIT ?
        """,
        (*args, limit),
    ).fetchall()
    return total, rows


def _embedding_authorized_uncertain_rows(
    conn: sqlite3.Connection,
    identity_hash: str,
    limit: int,
) -> tuple[int, list[sqlite3.Row]]:
    """Select exact uncertain rows without consuming their durable markers."""
    where = """
        f.identity_hash = ?
        AND f.state = 'uncertain'
        AND n.depth = 0
        AND NOT EXISTS (
            SELECT 1
            FROM lcm_embedding_meta AS m
            WHERE m.embedded_id = f.embedded_id
              AND m.embedded_kind = 'summary'
              AND m.identity_hash = f.identity_hash
        )
    """
    total = int(
        conn.execute(
            "SELECT COUNT(*) "
            "FROM lcm_embedding_backfill_inflight AS f "
            "JOIN summary_nodes AS n ON n.node_id = CAST(f.embedded_id AS INTEGER) "
            f"WHERE {where}",
            (identity_hash,),
        ).fetchone()[0]
    )
    rows = conn.execute(
        "SELECT n.node_id, n.summary "
        "FROM lcm_embedding_backfill_inflight AS f "
        "JOIN summary_nodes AS n ON n.node_id = CAST(f.embedded_id AS INTEGER) "
        f"WHERE {where} "
        "ORDER BY f.updated_at, f.embedded_id LIMIT ?",
        (identity_hash, max(0, int(limit))),
    ).fetchall()
    return total, rows


def _embedding_batch_estimate(provider: str, token_counts: list[int]) -> int:
    if not token_counts:
        return 0
    if provider.lower() != "voyage":
        return int(math.ceil(len(token_counts) / _EMBEDDING_BACKFILL_BATCH_SIZE))
    estimated = 0
    for offset in range(0, len(token_counts), _EMBEDDING_BACKFILL_BATCH_SIZE):
        batch_tokens = 0
        for tokens in token_counts[offset:offset + _EMBEDDING_BACKFILL_BATCH_SIZE]:
            if tokens > _VOYAGE_MAX_DOCUMENT_TOKENS:
                continue
            if batch_tokens and batch_tokens + tokens > _VOYAGE_MAX_BATCH_TOKENS:
                estimated += 1
                batch_tokens = 0
            batch_tokens += tokens
        if batch_tokens:
            estimated += 1
    return estimated


def _chunk_context_estimates(
    documents: list[tuple[str, str, int]]
) -> tuple[int, int, int]:
    """Estimate tokens/billable-tokens/requests for the contextualized chunk path.

    The dry-run estimate must match what the grouped apply path actually sends
    (FIX 4). Apply slices the selected documents into ``_EMBEDDING_BACKFILL_BATCH_SIZE``
    batches, groups each batch's chunks per source message
    (``group_by_store_id``), drops any single chunk over the per-chunk context cap
    (``_VOYAGE_CONTEXT_MAX_CHUNK_TOKENS`` = 32K, NOT the flat 27K per-document cap),
    and packs the surviving per-message documents into requests via
    ``_plan_contextualized_requests``. Mirror that flow exactly so the request
    count and billable tokens preview equal apply's, rather than the pre-C2 flat
    per-document packing (``_embedding_batch_estimate``).
    """
    total_tokens = sum(int(document[2]) for document in documents)
    billable_tokens = 0
    total_requests = 0
    for offset in range(0, len(documents), _EMBEDDING_BACKFILL_BATCH_SIZE):
        batch = documents[offset:offset + _EMBEDDING_BACKFILL_BATCH_SIZE]
        store_ids = [str(item[0]).split(":", 1)[0] for item in batch]
        per_document_tokens: list[list[int]] = []
        for group_indexes in group_by_store_id(store_ids):
            kept: list[int] = []
            for index in group_indexes:
                tokens = int(batch[index][2])
                # A single chunk above the per-chunk cap is non-embeddable and is
                # skipped by the provider before grouping -> not sent, not billed.
                if tokens > _VOYAGE_CONTEXT_MAX_CHUNK_TOKENS:
                    continue
                kept.append(tokens)
                billable_tokens += tokens
            if kept:
                per_document_tokens.append(kept)
        if per_document_tokens:
            total_requests += len(
                _plan_contextualized_requests(
                    per_document_tokens,
                    doc_token_budget=_VOYAGE_CONTEXT_DOCUMENT_TOKEN_BUDGET,
                    request_token_budget=_VOYAGE_CONTEXT_REQUEST_TOKEN_BUDGET,
                    request_chunk_budget=_VOYAGE_CONTEXT_MAX_REQUEST_CHUNKS,
                    max_inputs=_VOYAGE_MAX_BATCH_ITEMS,
                )
            )
    return total_tokens, billable_tokens, total_requests


def _embedding_estimated_cost(provider: str, model: str, tokens: int) -> float:
    if provider.lower() != "voyage":
        return 0.0
    rate = _VOYAGE_USD_PER_MILLION_TOKENS.get(model, 0.18)
    return (max(0, tokens) / 1_000_000.0) * rate


def _embedding_lease_heartbeat(raw: str) -> float:
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            # Tolerate the pre-heartbeat claim shape ({owner, claimed_at}).
            return float(
                payload.get("heartbeat_at", payload.get("claimed_at", 0.0)) or 0.0
            )
        return float(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0


class _BackfillLeaseLost(RuntimeError):
    """The caller no longer owns the exact backfill lease generation."""


class _LocalPublishError(RuntimeError):
    """The batch-publish CALL itself failed locally (e.g. SQLITE_BUSY on BEGIN
    IMMEDIATE, or a commit I/O error), as opposed to a provider/network error.

    Carries the operator-facing ``local_error:`` reason so the outer handler
    routes the quarantine reason text correctly and counts the rows, instead of
    mislabeling a local storage/lock failure as a provider error and omitting the
    unpublished rows from the run's ``failed`` count.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _BackfillLease:
    """A renewable owner-CAS lease for the one-shot embedding backfill worker.

    Only a *truly-expired* lease is stealable: while an owner heartbeats within
    the TTL, a second worker's acquire is refused. Renewal is a compare-and-swap
    on the owner id, so if this lease was stolen (its TTL lapsed and another
    worker took over), ``renew`` returns False and the run aborts instead of
    writing under a lease it no longer holds.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        generation: int,
        *,
        ttl_s: float,
        heartbeat_s: float,
        now: float,
    ) -> None:
        self.conn = conn
        self.lease_id = lease_id
        self.generation = generation
        self.ttl_s = ttl_s
        self.heartbeat_s = heartbeat_s
        self._last_heartbeat = now

    def _value(self, heartbeat_at: float) -> str:
        return json.dumps(
            {
                "owner": self.lease_id,
                "generation": self.generation,
                "heartbeat_at": heartbeat_at,
            },
            sort_keys=True,
        )

    def renew(self, *, now: float | None = None, force: bool = False) -> bool:
        now = time.time() if now is None else float(now)
        if not force and (now - self._last_heartbeat) < self.heartbeat_s:
            return True
        cur = self.conn.execute(
            """
            UPDATE metadata
            SET value = ?
            WHERE key = ?
              AND json_extract(value, '$.owner') = ?
              AND CAST(json_extract(value, '$.generation') AS INTEGER) = ?
            """,
            (
                self._value(now),
                _EMBEDDING_BACKFILL_CLAIM_KEY,
                self.lease_id,
                self.generation,
            ),
        )
        if not cur.rowcount:
            return False
        self._last_heartbeat = now
        return True

    def release(self) -> None:
        self.conn.execute(
            """
            DELETE FROM metadata
            WHERE key = ?
              AND json_extract(value, '$.owner') = ?
              AND CAST(json_extract(value, '$.generation') AS INTEGER) = ?
            """,
            (_EMBEDDING_BACKFILL_CLAIM_KEY, self.lease_id, self.generation),
        )


def _acquire_embedding_backfill_lease(
    conn: sqlite3.Connection,
    *,
    ttl_s: float,
    heartbeat_s: float,
    now: float | None = None,
) -> _BackfillLease | None:
    """Atomically acquire the lease, stealing only a truly-expired one."""
    now = time.time() if now is None else float(now)
    lease_id = uuid.uuid4().hex
    generation = 1
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (_EMBEDDING_BACKFILL_CLAIM_KEY,),
        ).fetchone()
        if row is not None:
            heartbeat_at = _embedding_lease_heartbeat(str(row[0]))
            if (now - heartbeat_at) < ttl_s:
                conn.rollback()
                return None
            try:
                prior = json.loads(str(row[0]))
                generation = int(prior.get("generation", 0) or 0) + 1
            except (TypeError, ValueError, json.JSONDecodeError):
                generation = 1
        lease = _BackfillLease(
            conn, lease_id, generation, ttl_s=ttl_s, heartbeat_s=heartbeat_s, now=now
        )
        conn.execute(
            """
            INSERT INTO metadata(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (_EMBEDDING_BACKFILL_CLAIM_KEY, lease._value(now)),
        )
        conn.commit()
        return lease
    except Exception:
        conn.rollback()
        raise


def _prepare_inflight_for_lease(
    conn: sqlite3.Connection,
    identity_hash: str,
    lease: "_BackfillLease",
) -> None:
    """Recover predecessor rows atomically under the exact current lease CAS."""
    # ``claimed`` is written before dispatch. If an older lease died in that
    # state, no provider call had begun and retry is safe. ``dispatched`` is the
    # opposite: remote acceptance is unknowable after a crash, so convert it to
    # the durable operator state instead of auto-clearing it.
    maintenance_limit = 256
    try:
        conn.execute("BEGIN IMMEDIATE")
        owner = conn.execute(
            "SELECT 1 FROM metadata WHERE key=? "
            "AND json_extract(value, '$.owner')=? "
            "AND CAST(json_extract(value, '$.generation') AS INTEGER)=?",
            (
                _EMBEDDING_BACKFILL_CLAIM_KEY,
                lease.lease_id,
                lease.generation,
            ),
        ).fetchone()
        if owner is None:
            raise _BackfillLeaseLost("embedding backfill lease lost before maintenance")

        # Freeze the exact predecessor ownership tuples before changing any
        # row. BEGIN IMMEDIATE prevents a real successor from appearing while
        # this transaction is open; the tuple predicates additionally ensure
        # a replaced row can never be mistaken for the staged predecessor.
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS "
            "lcm_embedding_backfill_predecessor_batch("
            "source_rowid INTEGER PRIMARY KEY, embedded_id TEXT, "
            "identity_hash TEXT, lease_id TEXT, generation INTEGER, "
            "state TEXT, request_id TEXT, sort_updated_at REAL)"
        )
        conn.execute(
            "DELETE FROM lcm_embedding_backfill_predecessor_batch"
        )
        state_limits = [
            ("claimed", maintenance_limit),
            ("dispatched", maintenance_limit),
        ]
        for state, state_limit in state_limits:
            # idx_lcm_embedding_inflight_maintenance serves the complete
            # identity/state/order prefix, so each predecessor class is both
            # SQL-limited and index-bounded rather than corpus-sorted.
            conn.execute(
                "INSERT INTO lcm_embedding_backfill_predecessor_batch("
                "source_rowid, embedded_id, identity_hash, lease_id, generation, "
                "state, request_id, sort_updated_at) "
                "SELECT rowid, embedded_id, identity_hash, lease_id, generation, "
                "state, request_id, updated_at "
                "FROM lcm_embedding_backfill_inflight "
                "WHERE identity_hash = ? AND state = ? "
                "AND (lease_id IS NOT ? OR generation IS NOT ?) "
                "ORDER BY updated_at, embedded_id LIMIT ?",
                (
                    identity_hash,
                    state,
                    lease.lease_id,
                    lease.generation,
                    state_limit,
                ),
            )
        conn.execute(
            "DELETE FROM lcm_embedding_backfill_inflight "
            "WHERE rowid IN (SELECT source_rowid "
            "FROM lcm_embedding_backfill_predecessor_batch "
            "WHERE state = 'claimed') AND state = 'claimed' AND EXISTS ("
            "SELECT 1 FROM lcm_embedding_backfill_predecessor_batch staged "
            "WHERE staged.source_rowid = lcm_embedding_backfill_inflight.rowid "
            "AND staged.embedded_id IS lcm_embedding_backfill_inflight.embedded_id "
            "AND staged.identity_hash IS lcm_embedding_backfill_inflight.identity_hash "
            "AND staged.lease_id IS lcm_embedding_backfill_inflight.lease_id "
            "AND staged.generation IS lcm_embedding_backfill_inflight.generation "
            "AND staged.request_id IS lcm_embedding_backfill_inflight.request_id "
            "AND staged.state = 'claimed')"
        )
        conn.execute(
            "UPDATE lcm_embedding_backfill_inflight "
            "SET state = 'uncertain', updated_at = ?, "
            "last_error = COALESCE(last_error, 'prior lease ended after dispatch') "
            "WHERE rowid IN (SELECT source_rowid "
            "FROM lcm_embedding_backfill_predecessor_batch "
            "WHERE state = 'dispatched') AND state = 'dispatched' AND EXISTS ("
            "SELECT 1 FROM lcm_embedding_backfill_predecessor_batch staged "
            "WHERE staged.source_rowid = lcm_embedding_backfill_inflight.rowid "
            "AND staged.embedded_id IS lcm_embedding_backfill_inflight.embedded_id "
            "AND staged.identity_hash IS lcm_embedding_backfill_inflight.identity_hash "
            "AND staged.lease_id IS lcm_embedding_backfill_inflight.lease_id "
            "AND staged.generation IS lcm_embedding_backfill_inflight.generation "
            "AND staged.request_id IS lcm_embedding_backfill_inflight.request_id "
            "AND staged.state = 'dispatched')",
            (time.time(),),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _mark_inflight(
    conn: sqlite3.Connection,
    identity_hash: str,
    lease: "_BackfillLease",
    embedded_ids: list[str],
    *,
    authorized_uncertain_ids: set[str] | frozenset[str] = frozenset(),
    now: float | None = None,
) -> None:
    """Reserve the exact selected rows without consuming retry authorization."""
    now = time.time() if now is None else float(now)
    authorized = {str(embedded_id) for embedded_id in authorized_uncertain_ids}
    try:
        conn.execute("BEGIN IMMEDIATE")
        owner = conn.execute(
            "SELECT 1 FROM metadata WHERE key=? "
            "AND json_extract(value, '$.owner')=? "
            "AND CAST(json_extract(value, '$.generation') AS INTEGER)=?",
            (
                _EMBEDDING_BACKFILL_CLAIM_KEY,
                lease.lease_id,
                lease.generation,
            ),
        ).fetchone()
        if owner is None:
            raise _BackfillLeaseLost("embedding backfill lease lost before reservation")

        for raw_embedded_id in embedded_ids:
            embedded_id = str(raw_embedded_id)
            if embedded_id in authorized:
                # Keep the durable state uncertain until the exact provider
                # request begins. A crash, budget expiry, or lease loss before
                # dispatch therefore cannot turn an authorized retry into an
                # ordinary automatically-discoverable row.
                cur = conn.execute(
                    "UPDATE lcm_embedding_backfill_inflight "
                    "SET lease_id=?, generation=?, claimed_at=?, request_id=NULL, "
                    "updated_at=? WHERE embedded_id=? AND identity_hash=? "
                    "AND state='uncertain'",
                    (
                        lease.lease_id,
                        lease.generation,
                        now,
                        now,
                        embedded_id,
                        identity_hash,
                    ),
                )
                if int(cur.rowcount or 0) != 1:
                    raise RuntimeError(
                        "authorized uncertain embedding changed before reservation"
                    )
                continue

            cur = conn.execute(
                "INSERT OR IGNORE INTO lcm_embedding_backfill_inflight("
                "embedded_id, identity_hash, lease_id, generation, claimed_at, "
                "state, request_id, updated_at, last_error) "
                "VALUES(?, ?, ?, ?, ?, 'claimed', NULL, ?, NULL)",
                (
                    embedded_id,
                    identity_hash,
                    lease.lease_id,
                    lease.generation,
                    now,
                    now,
                ),
            )
            if int(cur.rowcount or 0) != 1:
                raise RuntimeError("embedding candidate changed before reservation")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _mark_dispatched(
    conn: sqlite3.Connection,
    identity_hash: str,
    lease: "_BackfillLease",
    embedded_ids: list[str],
    request_id: str,
    *,
    authorized_uncertain_ids: set[str] | frozenset[str] = frozenset(),
) -> int:
    if not embedded_ids:
        return 0
    authorized = {str(embedded_id) for embedded_id in authorized_uncertain_ids}
    try:
        conn.execute("BEGIN IMMEDIATE")
        owner = conn.execute(
            "SELECT 1 FROM metadata WHERE key=? "
            "AND json_extract(value, '$.owner')=? "
            "AND CAST(json_extract(value, '$.generation') AS INTEGER)=?",
            (
                _EMBEDDING_BACKFILL_CLAIM_KEY,
                lease.lease_id,
                lease.generation,
            ),
        ).fetchone()
        if owner is None:
            conn.rollback()
            return 0
        now = time.time()
        for raw_embedded_id in embedded_ids:
            embedded_id = str(raw_embedded_id)
            prior_state = "uncertain" if embedded_id in authorized else "claimed"
            cur = conn.execute(
                "UPDATE lcm_embedding_backfill_inflight "
                "SET state='dispatched', request_id=?, updated_at=? "
                "WHERE embedded_id=? AND identity_hash=? AND lease_id=? "
                "AND generation=? AND state=? AND request_id IS NULL",
                (
                    request_id,
                    now,
                    embedded_id,
                    identity_hash,
                    lease.lease_id,
                    lease.generation,
                    prior_state,
                ),
            )
            if int(cur.rowcount or 0) != 1:
                conn.rollback()
                return 0
        conn.commit()
        return len(embedded_ids)
    except Exception:
        conn.rollback()
        raise


def _owned_inflight_transition(
    conn: sqlite3.Connection,
    identity_hash: str,
    lease: "_BackfillLease",
    request_id: str | None,
    *,
    embedded_id: str | None = None,
    error: str | None = None,
    retryable: bool = False,
    authorized_uncertain_ids: set[str] | frozenset[str] = frozenset(),
) -> bool:
    """Mutate only the caller-owned exact request under the lease CAS."""
    authorized = {str(item) for item in authorized_uncertain_ids}
    try:
        conn.execute("BEGIN IMMEDIATE")
        owner = conn.execute(
            "SELECT 1 FROM metadata WHERE key=? "
            "AND json_extract(value, '$.owner')=? "
            "AND CAST(json_extract(value, '$.generation') AS INTEGER)=?",
            (
                _EMBEDDING_BACKFILL_CLAIM_KEY,
                lease.lease_id,
                lease.generation,
            ),
        ).fetchone()
        if owner is None:
            conn.rollback()
            return False
        if embedded_id is not None:
            embedded_id = str(embedded_id)
            if embedded_id in authorized and request_id is None:
                # Reservation deliberately leaves authorized rows uncertain;
                # no provider dispatch occurred, so there is nothing to undo.
                pass
            elif embedded_id in authorized:
                conn.execute(
                    "UPDATE lcm_embedding_backfill_inflight "
                    "SET state='uncertain', request_id=NULL, updated_at=?, "
                    "last_error=? WHERE embedded_id=? AND identity_hash=? "
                    "AND lease_id=? AND generation=? AND request_id=? "
                    "AND state='dispatched'",
                    (
                        time.time(),
                        str(error or "authorized retry was not published"),
                        embedded_id,
                        identity_hash,
                        lease.lease_id,
                        lease.generation,
                        request_id,
                    ),
                )
            elif request_id is None:
                conn.execute(
                    "DELETE FROM lcm_embedding_backfill_inflight "
                    "WHERE embedded_id=? AND identity_hash=? AND lease_id=? "
                    "AND generation=? AND state='claimed'",
                    (
                        embedded_id,
                        identity_hash,
                        lease.lease_id,
                        lease.generation,
                    ),
                )
            else:
                conn.execute(
                    "DELETE FROM lcm_embedding_backfill_inflight "
                    "WHERE embedded_id=? AND identity_hash=? AND lease_id=? "
                    "AND generation=? AND request_id=? AND state='dispatched'",
                    (
                        embedded_id,
                        identity_hash,
                        lease.lease_id,
                        lease.generation,
                        request_id,
                    ),
                )
        elif retryable:
            for authorized_id in authorized:
                conn.execute(
                    "UPDATE lcm_embedding_backfill_inflight "
                    "SET state='uncertain', request_id=NULL, updated_at=?, "
                    "last_error=? WHERE embedded_id=? AND identity_hash=? "
                    "AND lease_id=? AND generation=? AND request_id=? "
                    "AND state='dispatched'",
                    (
                        time.time(),
                        str(error or "authorized retry was definitively rejected"),
                        authorized_id,
                        identity_hash,
                        lease.lease_id,
                        lease.generation,
                        request_id,
                    ),
                )
            if authorized:
                placeholders = ",".join("?" for _ in authorized)
                conn.execute(
                    "DELETE FROM lcm_embedding_backfill_inflight "
                    "WHERE identity_hash=? AND lease_id=? AND generation=? "
                    f"AND request_id=? AND state='dispatched' "
                    f"AND embedded_id NOT IN ({placeholders})",
                    (
                        identity_hash,
                        lease.lease_id,
                        lease.generation,
                        request_id,
                        *sorted(authorized),
                    ),
                )
            else:
                conn.execute(
                    "DELETE FROM lcm_embedding_backfill_inflight "
                    "WHERE identity_hash=? AND lease_id=? AND generation=? "
                    "AND request_id=? AND state='dispatched'",
                    (
                        identity_hash,
                        lease.lease_id,
                        lease.generation,
                        request_id,
                    ),
                )
        else:
            conn.execute(
                "UPDATE lcm_embedding_backfill_inflight "
                "SET state='uncertain', updated_at=?, last_error=? "
                "WHERE identity_hash=? AND lease_id=? AND generation=? "
                "AND request_id=? AND state='dispatched'",
                (
                    time.time(),
                    str(error or "remote acceptance could not be verified"),
                    identity_hash,
                    lease.lease_id,
                    lease.generation,
                    request_id,
                ),
            )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def _embedding_backfill_options(
    tokens: list[str],
) -> tuple[bool, int, bool, str, str, bool, str | None] | str:
    apply = False
    limit = 200
    retry_uncertain = False
    corpus = "summary"
    policy = ""
    confirm_raw_text = False
    expected_dtype: str | None = None
    seen_corpus = False
    seen_policy = False
    seen_dtype = False
    index = 0
    while index < len(tokens):
        token = tokens[index].lower()
        if token == "--apply" and not apply:
            apply = True
            index += 1
            continue
        if token == "--dtype" and index + 1 < len(tokens) and not seen_dtype:
            expected_dtype = tokens[index + 1].lower()
            if expected_dtype not in {"float32", "int8"}:
                return "`--dtype` must be one of `float32` or `int8`."
            seen_dtype = True
            index += 2
            continue
        if token == "--retry-uncertain" and not retry_uncertain:
            retry_uncertain = True
            index += 1
            continue
        if token == "--confirm-raw-text" and not confirm_raw_text:
            confirm_raw_text = True
            index += 1
            continue
        if token == "--limit" and index + 1 < len(tokens):
            try:
                limit = int(tokens[index + 1])
            except ValueError:
                return "`--limit` must be a positive integer."
            if limit < 1:
                return "`--limit` must be a positive integer."
            index += 2
            continue
        if token == "--corpus" and index + 1 < len(tokens) and not seen_corpus:
            corpus = tokens[index + 1].lower()
            if corpus not in {"summary", "chunks", "both"}:
                return "`--corpus` must be one of `summary`, `chunks`, or `both`."
            seen_corpus = True
            index += 2
            continue
        if token == "--policy" and index + 1 < len(tokens) and not seen_policy:
            policy = tokens[index + 1].lower()
            if policy not in VALID_CONTENT_POLICIES:
                return "`--policy` must be one of `conversational`, `heads`, or `full`."
            seen_policy = True
            index += 2
            continue
        return (
            "`/lcm embed backfill` accepts only `--apply`, `--retry-uncertain`, "
            "`--confirm-raw-text`, `--limit N`, `--corpus summary|chunks|both`, "
            "`--policy conversational|heads|full`, and `--dtype float32|int8`."
        )
    if retry_uncertain and not apply:
        return "`--retry-uncertain` requires `--apply` because it may incur charges."
    if policy and corpus == "summary":
        return "`--policy` only applies to the chunk corpus; add `--corpus chunks`."
    if confirm_raw_text and corpus == "summary":
        return (
            "`--confirm-raw-text` only applies to the chunk corpus; add "
            "`--corpus chunks` or `--corpus both`."
        )
    return apply, limit, retry_uncertain, corpus, policy, confirm_raw_text, expected_dtype


def _embedding_backfill_report(
    *,
    mode: str,
    status: str,
    provider: str,
    model: str,
    pending: int,
    selected: int,
    estimated_tokens: int,
    estimated_cost_tokens: int,
    estimated_batches: int,
    embedded: int,
    skipped: list[tuple[str, str]],
    failed: list[tuple[str, str]],
    remaining: int,
    duration: float,
    consumed_tokens: int,
    error: str | None = None,
    in_flight: int = 0,
    uncertain: int = 0,
    stop_reason: str | None = None,
    corpus: str | None = None,
    policy: str | None = None,
    include_next_hint: bool = True,
) -> str:
    header = "LCM embedding backfill" if corpus is None else f"LCM {corpus} backfill"
    lines = [header, f"mode: {mode}"]
    if corpus is not None:
        lines.append(f"corpus: {corpus}")
    if policy is not None:
        lines.append(f"policy: {policy}")
    lines += [
        f"status: {status}",
        f"provider: {provider}",
        f"model: {model}",
        f"pending: {pending}",
        f"selected: {selected}",
        f"estimated_tokens: {estimated_tokens}",
        f"estimated_batches: {estimated_batches}",
        f"estimated_cost_usd: ${_embedding_estimated_cost(provider, model, estimated_cost_tokens):.6f}",
        f"embedded: {embedded}",
        f"skipped_overcap: {len(skipped)}",
        f"failed: {len(failed)}",
        f"in_flight: {in_flight}",
        f"uncertain_remote_acceptance: {uncertain}",
        f"remaining: {remaining}",
        f"duration_seconds: {duration:.3f}",
        f"tokens_consumed: {consumed_tokens}",
    ]
    if stop_reason:
        lines.append(f"stop_reason: {stop_reason}")
    if error:
        lines.append(f"error: {error}")
    if uncertain:
        lines.append(
            "warning: remote acceptance is uncertain; rows are excluded from "
            "automatic retry to prevent duplicate billing"
        )
        lines.append(
            "next: inspect provider records, then use `--apply --retry-uncertain` "
            "only if re-embedding is explicitly authorized"
        )
    lines.extend(
        f"skipped_detail: node_id={node_id} reason={reason}"
        for node_id, reason in skipped
    )
    lines.extend(
        f"failed_detail: node_id={node_id} reason={reason}"
        for node_id, reason in failed
    )
    if mode == "dry-run":
        lines.append("note: preview only; no provider calls or database writes were made")
        # In a `--corpus both` preview the caller emits ONE combined next-hint for
        # the whole run, so the per-corpus reports suppress their own (F6).
        if include_next_hint:
            apply_hint = (
                "/lcm embed backfill --apply"
                if corpus is None
                else f"/lcm embed backfill --corpus {corpus} --apply"
            )
            lines.append(f"next: run `{apply_hint}` to populate embeddings")
    return "\n".join(lines)


def _provider_document_batches(provider, texts: list[str], *, before_dispatch):
    """Yield accepted provider requests, adapting legacy deterministic fakes."""
    method = getattr(provider, "embed_document_batches", None)
    if callable(method):
        yield from method(texts, before_dispatch=before_dispatch)
        return
    before_dispatch(tuple(range(len(texts))))
    vectors = provider.embed_documents(texts)
    skipped = {
        int(index)
        for index in getattr(provider, "last_skipped_documents", [])
        if 0 <= int(index) < len(texts)
    }
    indexes = tuple(index for index in range(len(texts)) if index not in skipped)
    yield EmbeddedDocumentBatch(
        indexes=indexes,
        vectors=tuple(tuple(vector) for vector in vectors),
    )


def _provider_chunk_document_batches(
    provider, batch, chunk_meta, *, before_dispatch
):
    """Yield accepted provider requests for a chunk backfill batch.

    A contextualized provider (voyage-context-*) groups each message's chunks
    into one inputs inner-list so the context model embeds them together; every
    accepted batch still carries per-chunk flat indexes + vectors, so the
    downstream publish/lease/inflight path stays per-chunk and unchanged.
    Non-context / local providers keep the flat per-chunk document path.
    """
    if getattr(provider, "supports_contextualized_grouping", False):
        store_ids = [chunk_meta[item[0]][0] for item in batch]
        groups = [
            [(index, batch[index][1]) for index in group_indexes]
            for group_indexes in group_by_store_id(store_ids)
        ]
        yield from provider.embed_chunk_group_batches(
            groups, before_dispatch=before_dispatch
        )
        return
    yield from _provider_document_batches(
        provider, [item[1] for item in batch], before_dispatch=before_dispatch
    )


def _embedding_backfill_text(tokens: list[str], engine) -> str:
    parsed = _embedding_backfill_options(tokens)
    if isinstance(parsed, str):
        return _help_text(parsed)
    apply, limit, retry_uncertain, corpus, policy, confirm_raw_text, expected_dtype = parsed
    if corpus == "chunks":
        return _chunk_backfill_text(
            engine, apply=apply, limit=limit,
            retry_uncertain=retry_uncertain, policy=policy,
            confirm_raw_text=confirm_raw_text, expected_dtype=expected_dtype,
        )
    if corpus == "both":
        summary_report = _embedding_backfill_summary_text(
            engine, apply=apply, limit=limit, retry_uncertain=retry_uncertain,
            include_next_hint=False, expected_dtype=expected_dtype,
        )
        chunk_report = _chunk_backfill_text(
            engine, apply=apply, limit=limit,
            retry_uncertain=retry_uncertain, policy=policy,
            confirm_raw_text=confirm_raw_text,
            include_next_hint=False, expected_dtype=expected_dtype,
        )
        combined = summary_report + "\n\n" + chunk_report
        if not apply:
            # One coherent next-hint matching the actual `--corpus both` invocation,
            # instead of the two contradictory per-corpus hints (F6).
            combined += (
                "\n\nnext: run `/lcm embed backfill --corpus both --apply` to "
                "populate both corpora (add `--confirm-raw-text` to authorize the "
                "chunk corpus on a cloud provider)"
            )
        return combined
    return _embedding_backfill_summary_text(
        engine, apply=apply, limit=limit, retry_uncertain=retry_uncertain,
        expected_dtype=expected_dtype,
    )


def _embedding_backfill_summary_text(
    engine, *, apply: bool, limit: int, retry_uncertain: bool,
    include_next_hint: bool = True, expected_dtype: str | None = None,
) -> str:
    mode = "apply" if apply else "dry-run"
    started = time.monotonic()

    if not bool(getattr(engine._config, "embeddings_enabled", False)):
        return "\n".join([
            "LCM embedding backfill",
            f"mode: {mode}",
            "status: refused",
            "error: embeddings are disabled; set LCM_EMBEDDINGS_ENABLED=true, then run `/lcm embed warmup`",
        ])

    db_path = engine._store.db_path
    # Resolve the active profile (read-only) — needed by both modes. The
    # *pending* set for apply is (re-)discovered AFTER the lease is claimed.
    try:
        read_conn = _embedding_read_connection(db_path)
    except sqlite3.Error as exc:
        return "\n".join([
            "LCM embedding backfill",
            f"mode: {mode}",
            "status: refused",
            f"error: embedding database is unavailable ({exc}); run `/lcm embed warmup` first",
        ])
    try:
        profile = _embedding_current_profile(read_conn)
        if profile is None:
            return "\n".join([
                "LCM embedding backfill",
                f"mode: {mode}",
                "status: refused",
                "error: no current embedding profile is registered; run `/lcm embed warmup` first",
            ])
        identity = str(profile["identity_hash"])
        model = str(profile["model_name"])
        provider_name = str(profile["provider"])
        profile_dtype = str(profile["dtype"] or "float32")
        if expected_dtype is not None and expected_dtype != profile_dtype:
            return "\n".join([
                "LCM embedding backfill",
                f"mode: {mode}",
                "status: refused",
                f"error: --dtype {expected_dtype} does not match the registered "
                f"summary profile dtype ({profile_dtype}); re-run `/lcm embed warmup` "
                f"with LCM_EMBEDDING_STORAGE_DTYPE={expected_dtype} to register that identity",
            ])
        if not apply:
            pending, rows = _embedding_pending_rows(read_conn, identity, limit)
    except sqlite3.Error as exc:
        return "\n".join([
            "LCM embedding backfill",
            f"mode: {mode}",
            "status: error",
            f"error: could not discover pending summaries ({exc})",
        ])
    finally:
        read_conn.close()

    def _estimates(documents: list[tuple[str, str, int]]) -> tuple[int, int, int]:
        est_tokens = sum(document[2] for document in documents)
        est_cost_tokens = sum(
            document[2]
            for document in documents
            if provider_name.lower() != "voyage"
            or document[2] <= _VOYAGE_MAX_DOCUMENT_TOKENS
        )
        est_batches = _embedding_batch_estimate(
            provider_name, [document[2] for document in documents]
        )
        return est_tokens, est_cost_tokens, est_batches

    if not apply:
        documents = [
            (str(row["node_id"]), str(row["summary"]), count_tokens(row["summary"]))
            for row in rows
        ]
        estimated_tokens, estimated_cost_tokens, estimated_batches = _estimates(documents)
        return _embedding_backfill_report(
            mode=mode,
            status="dry-run",
            provider=provider_name,
            model=model,
            pending=pending,
            selected=len(documents),
            estimated_tokens=estimated_tokens,
            estimated_cost_tokens=estimated_cost_tokens,
            estimated_batches=estimated_batches,
            embedded=0,
            skipped=[],
            failed=[],
            remaining=pending,
            duration=time.monotonic() - started,
            consumed_tokens=0,
            include_next_hint=include_next_hint,
        )

    ttl_s = _embedding_backfill_lease_ttl_s()
    heartbeat_s = _embedding_backfill_heartbeat_s()
    budget_s = _embedding_backfill_budget_s()

    store: VectorStore | None = None
    lease: _BackfillLease | None = None
    documents: list[tuple[str, str, int]] = []
    pending = 0
    embedded = 0
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    consumed_tokens = 0
    error: str | None = None
    stop_reason: str | None = None
    lease_lost = False
    identity_superseded = False
    budget_exhausted = False
    try:
        store = VectorStore(db_path, config=engine._config)
        conn = store.connection
        _ensure_inflight_table(conn)
        # Claim BEFORE discovery: acquire the lease, then re-query pending rows
        # so the batch reflects the state at claim time, not a stale pre-claim
        # snapshot another writer may have since drained.
        lease = _acquire_embedding_backfill_lease(
            conn, ttl_s=ttl_s, heartbeat_s=heartbeat_s
        )
        if lease is None:
            return "\n".join([
                "LCM embedding backfill",
                "mode: apply",
                "status: refused",
                "error: another embedding backfill holds the lease; retry after it exits or after the lease TTL expires",
            ])
        _prepare_inflight_for_lease(conn, identity, lease)
        captured_identity = store.capture_identity(model, provider=provider_name)
        if captured_identity.identity_hash != identity:
            raise ValueError(
                "active embedding identity changed before backfill dispatch; "
                "run `/lcm embed warmup` and retry"
            )
        # A risky retry invocation is an exact, exclusive authorization for
        # the bounded uncertain rows selected here. Ordinary pending work is
        # deliberately left for a normal invocation, so a newer ordinary row
        # can never displace an older authorized row under --limit.
        if retry_uncertain:
            pending, rows = _embedding_authorized_uncertain_rows(
                conn, identity, limit
            )
            authorized_uncertain_ids = {
                str(row["node_id"]) for row in rows
            }
        else:
            pending, rows = _embedding_pending_rows(conn, identity, limit)
            authorized_uncertain_ids = set()
        documents = [
            (str(row["node_id"]), str(row["summary"]), count_tokens(row["summary"]))
            for row in rows
        ]

        # Bulk backfill bypasses the interactive per-minute spend guard (it has
        # its own op budget + lease); otherwise a large --apply run trips the
        # 60/min guard mid-way and stalls.
        provider = resolve_provider(engine._config, for_backfill=True)
        if provider is None:
            error = "embedding provider is not configured; run `/lcm embed warmup`"
        elif (
            provider.model_id != model
            or str(provider.provider_id).lower() != provider_name.lower()
        ):
            error = "configured provider does not match the current profile; run `/lcm embed warmup`"
        else:
            for offset in range(0, len(documents), _EMBEDDING_BACKFILL_BATCH_SIZE):
                # Renew the heartbeat lease; if it was stolen (TTL lapsed and a
                # second owner took over), stop rather than write under a lease
                # we no longer hold.
                if not lease.renew():
                    lease_lost = True
                    stop_reason = "lease_lost"
                    break
                if budget_s > 0 and (time.monotonic() - started) > budget_s:
                    budget_exhausted = True
                    stop_reason = "op_budget_exhausted"
                    break
                batch = documents[offset:offset + _EMBEDDING_BACKFILL_BATCH_SIZE]
                # Claim the bounded outer batch. The provider invokes the
                # callback immediately before EACH real sub-request, and only
                # those exact indexes become dispatched under that request's
                # distinct durable id.
                batch_authorized_ids = {
                    item[0]
                    for item in batch
                    if item[0] in authorized_uncertain_ids
                }
                _mark_inflight(
                    conn,
                    identity,
                    lease,
                    [item[0] for item in batch],
                    authorized_uncertain_ids=batch_authorized_ids,
                )
                request_by_index: dict[int, str] = {}
                indexes_by_request: dict[str, set[int]] = {}
                transitioned_requests: set[str] = set()
                latest_request_id: str | None = None

                def before_dispatch(indexes: tuple[int, ...]) -> None:
                    nonlocal latest_request_id
                    normalized = tuple(int(index) for index in indexes)
                    if (
                        not normalized
                        or len(set(normalized)) != len(normalized)
                        or any(index < 0 or index >= len(batch) for index in normalized)
                        or any(index in request_by_index for index in normalized)
                    ):
                        raise EmbeddingProviderError(
                            "provider dispatched invalid or duplicate document indexes"
                        )
                    request_id = uuid.uuid4().hex
                    ids = [batch[index][0] for index in normalized]
                    if _mark_dispatched(
                        conn,
                        identity,
                        lease,
                        ids,
                        request_id,
                        authorized_uncertain_ids=batch_authorized_ids,
                    ) != len(ids):
                        raise EmbeddingProviderError(
                            "embedding lease lost before provider dispatch"
                        )
                    indexes_by_request[request_id] = set(normalized)
                    request_by_index.update(
                        (index, request_id) for index in normalized
                    )
                    latest_request_id = request_id

                try:
                    accepted_indexes: set[int] = set()
                    for accepted_batch in _provider_document_batches(
                        provider,
                        [item[1] for item in batch],
                        before_dispatch=before_dispatch,
                    ):
                        if len(accepted_batch.indexes) != len(accepted_batch.vectors):
                            raise EmbeddingProviderError(
                                "provider returned a different number of indexes and embeddings"
                            )
                        publish_rows = []
                        publish_items = []
                        for batch_index, vector in zip(
                            accepted_batch.indexes, accepted_batch.vectors
                        ):
                            index = int(batch_index)
                            if index < 0 or index >= len(batch) or index in accepted_indexes:
                                raise EmbeddingProviderError(
                                    "provider returned an invalid accepted document index"
                                )
                            accepted_indexes.add(index)
                            item = batch[index]
                            request_id = request_by_index.get(index)
                            if request_id is None:
                                raise EmbeddingProviderError(
                                    "provider returned an embedding before durable dispatch"
                                )
                            publish_rows.append(
                                (item[0], "summary", model, vector, request_id)
                            )
                            publish_items.append((item, request_id))
                        # One BEGIN IMMEDIATE per accepted network batch (one fsync)
                        # instead of one per row; each row still runs the full CAS
                        # under its own savepoint, so a mid-batch failure quarantines
                        # only its request while the committed siblings survive.
                        try:
                            published_results = store.publish_embedding_batch_under_lease(
                                publish_rows,
                                identity=captured_identity,
                                claim_key=_EMBEDDING_BACKFILL_CLAIM_KEY,
                                lease_id=lease.lease_id,
                                generation=lease.generation,
                            )
                        except Exception as publish_exc:
                            # The batch publish CALL itself failed locally (e.g.
                            # SQLITE_BUSY on BEGIN IMMEDIATE, or a commit I/O
                            # error) -- NOT a provider error. The generic handler
                            # below would mislabel it provider_error and omit
                            # these rows from `failed` (accepted_indexes already
                            # holds them, so failed_indexes is empty). Record
                            # every accepted-batch row as a local_error failure
                            # here; the rows still go uncertain via the inflight
                            # fallback in the outer handler, now carrying the
                            # correct local_error reason text.
                            local_reason = f"local_error:{publish_exc}"
                            failed.extend(
                                (item[0], local_reason) for item, _rid in publish_items
                            )
                            raise _LocalPublishError(local_reason) from publish_exc
                        for result, (item, request_id) in zip(
                            published_results,
                            publish_items,
                        ):
                            if result.error is not None:
                                exc = result.error
                                failed.append((item[0], f"record_error:{exc}"))
                                if not _owned_inflight_transition(
                                    conn,
                                    identity,
                                    lease,
                                    request_id,
                                    error=f"accepted remotely; local publish failed: {exc}",
                                    authorized_uncertain_ids=batch_authorized_ids,
                                ):
                                    lease_lost = True
                                    stop_reason = "lease_lost"
                                else:
                                    transitioned_requests.add(request_id)
                                    raise EmbeddingProviderError(
                                        "accepted remotely; local publication failed"
                                    ) from exc
                                break
                            publish_outcome = result.outcome
                            if publish_outcome is EmbeddingPublishOutcome.OWNERSHIP_LOST:
                                lease_lost = True
                                stop_reason = "lease_lost"
                                break
                            if (
                                publish_outcome
                                is EmbeddingPublishOutcome.IDENTITY_SUPERSEDED
                            ):
                                identity_superseded = True
                                stop_reason = "identity_superseded"
                                failed.append(
                                    (
                                        item[0],
                                        "identity_superseded_after_remote_acceptance",
                                    )
                                )
                                transitioned_requests.add(request_id)
                                break
                            embedded += 1
                            consumed_tokens += item[2]
                        if lease_lost or identity_superseded:
                            break
                    if lease_lost or identity_superseded:
                        break
                    skipped_indexes = {
                        int(index)
                        for index in getattr(provider, "last_skipped_documents", [])
                        if 0 <= int(index) < len(batch)
                    }
                    for index in sorted(skipped_indexes):
                        skipped.append((batch[index][0], "provider_document_token_cap"))
                        request_id = request_by_index.get(index)
                        if not _owned_inflight_transition(
                            conn,
                            identity,
                            lease,
                            request_id,
                            embedded_id=batch[index][0],
                            error="provider document token cap",
                            authorized_uncertain_ids=batch_authorized_ids,
                        ):
                            lease_lost = True
                            stop_reason = "lease_lost"
                            break
                    unresolved = set(range(len(batch))) - accepted_indexes - skipped_indexes
                    if unresolved:
                        raise EmbeddingProviderError(
                            "provider did not resolve every dispatched document"
                        )
                except Exception as exc:
                    if isinstance(exc, _LocalPublishError):
                        # Local storage/lock failure of the batch publish call:
                        # the accepted rows were already appended to `failed` with
                        # this reason; do not relabel it a provider error.
                        reason = exc.reason
                    elif isinstance(exc, VoyageError):
                        reason = f"provider_{exc.kind}:{exc}"
                    else:
                        reason = f"provider_error:{exc}"
                    definitive_rejection = (
                        isinstance(exc, ProviderPreDispatchError)
                        or (
                            isinstance(exc, VoyageError)
                            and exc.kind in {"auth", "bad_request", "rate_limit"}
                        )
                    )
                    if latest_request_id is not None:
                        failed_indexes = indexes_by_request[latest_request_id] - accepted_indexes
                        failed.extend((batch[index][0], reason) for index in failed_indexes)
                        if (
                            not lease_lost
                            and latest_request_id not in transitioned_requests
                            and not _owned_inflight_transition(
                                conn,
                                identity,
                                lease,
                                latest_request_id,
                                error=reason,
                                retryable=definitive_rejection,
                                authorized_uncertain_ids=batch_authorized_ids,
                            )
                        ):
                            lease_lost = True
                            stop_reason = "lease_lost"
                    # Exact rows not yet handed to a provider remain provably
                    # unsent; clear their claims so ordinary discovery can
                    # retry them without operator billing authorization.
                    if not lease_lost:
                        for index, item in enumerate(batch):
                            if index in request_by_index:
                                continue
                            if not _owned_inflight_transition(
                                conn,
                                identity,
                                lease,
                                None,
                                embedded_id=item[0],
                                authorized_uncertain_ids=batch_authorized_ids,
                            ):
                                lease_lost = True
                                stop_reason = "lease_lost"
                                break
                    if isinstance(exc, VoyageError) and exc.kind == "auth":
                        error = f"provider authentication failed; {exc}"
                        break
    except _BackfillLeaseLost:
        lease_lost = True
        stop_reason = "lease_lost"
    except Exception as exc:
        error = str(exc)
    finally:
        if store is not None:
            if lease is not None and not lease_lost and not identity_superseded:
                try:
                    lease.release()
                except Exception as exc:
                    error = (
                        f"{error}; lease release failed: {exc}"
                        if error
                        else f"lease release failed: {exc}"
                    )
            store.close()

    remaining, in_flight_count, uncertain_count = _embedding_backfill_remaining(
        db_path, identity, pending, embedded
    )
    selected_embeddable = len(documents) - len(skipped)
    status = _embedding_backfill_status(
        error=error,
        lease_lost=lease_lost,
        budget_exhausted=budget_exhausted,
        embedded=embedded,
        selected_embeddable=selected_embeddable,
        failed=failed,
        uncertain=uncertain_count,
        skipped=len(skipped),
    )
    estimated_tokens, estimated_cost_tokens, estimated_batches = _estimates(documents)
    return _embedding_backfill_report(
        mode=mode,
        status=status,
        provider=provider_name,
        model=model,
        pending=pending,
        selected=len(documents),
        estimated_tokens=estimated_tokens,
        estimated_cost_tokens=estimated_cost_tokens,
        estimated_batches=estimated_batches,
        embedded=embedded,
        skipped=skipped,
        failed=failed,
        remaining=remaining,
        duration=time.monotonic() - started,
        consumed_tokens=consumed_tokens,
        error=error,
        in_flight=in_flight_count,
        uncertain=uncertain_count,
        stop_reason=stop_reason,
    )


def _embedding_backfill_remaining(
    db_path, identity_hash: str, pending: int, embedded: int
) -> tuple[int, int, int]:
    """Re-read pending + in_flight counts after an apply run for a truthful report."""
    try:
        check_conn = _embedding_read_connection(db_path)
        try:
            remaining, _ = _embedding_pending_rows(check_conn, identity_hash, 1)
            try:
                in_flight = int(check_conn.execute(
                    "SELECT COUNT(*) FROM lcm_embedding_backfill_inflight WHERE identity_hash = ?",
                    (identity_hash,),
                ).fetchone()[0])
                uncertain = int(check_conn.execute(
                    "SELECT COUNT(*) FROM lcm_embedding_backfill_inflight "
                    "WHERE identity_hash = ? AND state IN ('dispatched', 'uncertain')",
                    (identity_hash,),
                ).fetchone()[0])
            except sqlite3.OperationalError:
                in_flight = 0
                uncertain = 0
        finally:
            check_conn.close()
    except sqlite3.Error:
        remaining = max(0, pending - embedded)
        in_flight = 0
        uncertain = 0
    return remaining, in_flight, uncertain


# -- Chunk corpus backfill -------------------------------------------------
#
# The chunk corpus reuses the shipped lease/inflight/uncertain MACHINERY
# unchanged (``_ensure_inflight_table``, ``_acquire_embedding_backfill_lease``,
# ``_prepare_inflight_for_lease``, ``_mark_inflight``, ``_mark_dispatched``,
# ``_owned_inflight_transition``, ``_provider_document_batches``) — those
# functions key on ``(embedded_id, identity_hash)`` and are corpus-agnostic, so
# a chunk's ``store_id:chunk_index`` id and its distinct task='chunk' identity
# share the one inflight table with summaries without collision. Only discovery
# (policy-chunked messages vs summary_nodes) and the publish call differ.


def _chunk_current_profile(conn: sqlite3.Connection) -> sqlite3.Row | None:
    try:
        return conn.execute(
            """
            SELECT identity_hash, model_name, provider, revision, dim, dtype,
                   byteorder, task, registered_at
            FROM lcm_embedding_profile
            WHERE active = 1 AND archived_at IS NULL AND task = 'chunk'
            ORDER BY registered_at DESC, identity_hash DESC
            LIMIT 1
            """
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return None
        raise


def _chunk_embedded_ids(conn: sqlite3.Connection, identity_hash: str | None) -> set[str]:
    if not identity_hash:
        return set()
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_chunk_meta'"
    ).fetchone()
    if exists is None:
        return set()
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT chunk_id FROM lcm_chunk_meta WHERE identity_hash = ?",
            (identity_hash,),
        ).fetchall()
    }


def _chunk_inflight_ids(conn: sqlite3.Connection, identity_hash: str | None) -> set[str]:
    if not identity_hash:
        return set()
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='lcm_embedding_backfill_inflight'"
    ).fetchone()
    if exists is None:
        return set()
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT embedded_id FROM lcm_embedding_backfill_inflight "
            "WHERE identity_hash = ?",
            (identity_hash,),
        ).fetchall()
    }


def _chunk_pending_rows(
    conn: sqlite3.Connection,
    identity_hash: str | None,
    policy: str,
    limit: int,
) -> tuple[int, list[tuple[str, str, int]], dict[str, tuple[int, int, int, int]]]:
    """Discover policy-chunked messages whose chunks are not yet embedded.

    Streams messages most-recent-first (so ``--limit`` selects recent chunks,
    mirroring the summary path's ``latest_at DESC``), chunks each under the
    active policy, and excludes chunk ids already embedded under the identity or
    held in-flight. Returns the total pending count, up to ``limit`` documents
    as ``(chunk_id, text, tokens)``, and a metadata map for the selected chunks.
    """
    already = _chunk_embedded_ids(conn, identity_hash)
    inflight = _chunk_inflight_ids(conn, identity_hash)
    total = 0
    documents: list[tuple[str, str, int]] = []
    meta: dict[str, tuple[int, int, int, int]] = {}
    for row in conn.execute(
        "SELECT store_id, role, content FROM messages ORDER BY store_id DESC"
    ):
        for chunk in chunk_message(row[0], row[1], row[2], policy=policy):
            chunk_id = chunk.chunk_id
            if chunk_id in already or chunk_id in inflight:
                continue
            # Write-seam guard: a degenerate/zero span is non-embeddable and would
            # persist an empty snippet + bogus lcm_expand offset (F3).
            if chunk.char_end <= chunk.char_start:
                continue
            total += 1
            if len(documents) < limit:
                documents.append((chunk_id, chunk.text, chunk.token_estimate))
                meta[chunk_id] = (
                    chunk.store_id,
                    chunk.chunk_index,
                    chunk.char_start,
                    chunk.char_end,
                )
    return total, documents, meta


def _rebuild_chunk_document(
    conn: sqlite3.Connection, chunk_id: str, policy: str
) -> tuple[str, int, int, int] | None:
    """Reconstruct one chunk's text/tokens/span by re-chunking its source message.

    Returns ``(text, token_estimate, char_start, char_end)`` for the chunk whose
    index matches ``chunk_id`` (``store_id:chunk_index``), or None if the message
    is gone or no longer produces that chunk under ``policy`` (content changed).
    The real char span is carried so ``--retry-uncertain`` recovery persists a
    correct verbatim span instead of a ``(0, 0)`` placeholder (F3).
    """
    try:
        store_id_str, index_str = chunk_id.split(":", 1)
        store_id = int(store_id_str)
        chunk_index = int(index_str)
    except (ValueError, AttributeError):
        return None
    row = conn.execute(
        "SELECT role, content FROM messages WHERE store_id = ?", (store_id,)
    ).fetchone()
    if row is None:
        return None
    for chunk in chunk_message(store_id, row[0], row[1], policy=policy):
        if chunk.chunk_index == chunk_index:
            return chunk.text, chunk.token_estimate, chunk.char_start, chunk.char_end
    return None


def _chunk_authorized_uncertain_rows(
    conn: sqlite3.Connection,
    identity_hash: str,
    policy: str,
    limit: int,
) -> tuple[int, list[tuple[str, str, int]], dict[str, tuple[int, int, int, int]]]:
    """Select uncertain chunk rows without consuming their durable markers."""
    rows = conn.execute(
        """
        SELECT f.embedded_id
        FROM lcm_embedding_backfill_inflight AS f
        WHERE f.identity_hash = ?
          AND f.state = 'uncertain'
          AND NOT EXISTS (
              SELECT 1 FROM lcm_chunk_meta AS m
              WHERE m.chunk_id = f.embedded_id
                AND m.identity_hash = f.identity_hash
          )
        ORDER BY f.updated_at, f.embedded_id
        LIMIT ?
        """,
        (identity_hash, max(0, int(limit))),
    ).fetchall()
    documents: list[tuple[str, str, int]] = []
    meta: dict[str, tuple[int, int, int, int]] = {}
    for row in rows:
        chunk_id = str(row[0])
        rebuilt = _rebuild_chunk_document(conn, chunk_id, policy)
        if rebuilt is None:
            continue
        text, tokens, char_start, char_end = rebuilt
        # Write-seam guard: never persist a degenerate/zero span — it yields an
        # empty verbatim snippet and a bogus lcm_expand offset (F3).
        if char_end <= char_start:
            continue
        store_id_str, index_str = chunk_id.split(":", 1)
        documents.append((chunk_id, text, tokens))
        meta[chunk_id] = (int(store_id_str), int(index_str), char_start, char_end)
    # Rows are SELECTed by (updated_at, embedded_id) to pick WHICH uncertain
    # chunks fall inside the retry budget, but that interleaves store_ids.
    # ``group_by_store_id`` only merges ADJACENT equal store_ids, so an
    # interleaved order collapses every retry chunk into a singleton group,
    # defeating C2 cross-chunk contextualization on the retry path. Stably
    # re-sort the selected documents by (store_id, chunk_index) so a message's
    # chunks are contiguous and group into one contextualization document —
    # matching the discovery path's natural store_id-ordered emission (FIX 3).
    documents.sort(key=lambda item: meta[item[0]][:2])
    return len(documents), documents, meta


def _chunk_backfill_remaining(
    db_path, identity_hash: str | None, policy: str, pending: int, embedded: int
) -> tuple[int, int, int]:
    try:
        check_conn = _embedding_read_connection(db_path)
        try:
            remaining, _, _ = _chunk_pending_rows(check_conn, identity_hash, policy, 1)
            if identity_hash:
                try:
                    in_flight = int(check_conn.execute(
                        "SELECT COUNT(*) FROM lcm_embedding_backfill_inflight "
                        "WHERE identity_hash = ?",
                        (identity_hash,),
                    ).fetchone()[0])
                    uncertain = int(check_conn.execute(
                        "SELECT COUNT(*) FROM lcm_embedding_backfill_inflight "
                        "WHERE identity_hash = ? AND state IN ('dispatched', 'uncertain')",
                        (identity_hash,),
                    ).fetchone()[0])
                except sqlite3.OperationalError:
                    in_flight = 0
                    uncertain = 0
            else:
                in_flight = 0
                uncertain = 0
        finally:
            check_conn.close()
    except sqlite3.Error:
        remaining = max(0, pending - embedded)
        in_flight = 0
        uncertain = 0
    return remaining, in_flight, uncertain


# Providers that embed on this machine and never transmit text off-box. The raw-
# text consent gate on the chunk corpus is waived for these (F1).
_LOCAL_EMBEDDING_PROVIDERS = frozenset({"fastembed", "ollama"})


def _is_local_embedding_provider(provider_name: str) -> bool:
    return str(provider_name or "").strip().lower() in _LOCAL_EMBEDDING_PROVIDERS


def _chunk_backfill_text(
    engine, *, apply: bool, limit: int, retry_uncertain: bool, policy: str,
    confirm_raw_text: bool = False, include_next_hint: bool = True,
    expected_dtype: str | None = None,
) -> str:
    policy = normalize_content_policy(policy or getattr(
        engine._config, "embedding_content_policy", "conversational"
    ))
    mode = "apply" if apply else "dry-run"
    started = time.monotonic()

    def _refused(message: str) -> str:
        return "\n".join([
            "LCM chunk backfill",
            f"mode: {mode}",
            "corpus: chunks",
            f"policy: {policy}",
            "status: refused",
            f"error: {message}",
        ])

    if not bool(getattr(engine._config, "embeddings_enabled", False)):
        return _refused(
            "embeddings are disabled; set LCM_EMBEDDINGS_ENABLED=true, then run "
            "`/lcm embed warmup`"
        )

    db_path = engine._store.db_path
    configured_provider = str(
        getattr(engine._config, "embedding_provider", "") or ""
    ).strip().lower()
    configured_model = str(getattr(engine._config, "embedding_model", "") or "").strip()

    try:
        read_conn = _embedding_read_connection(db_path)
    except sqlite3.Error as exc:
        return _refused(f"embedding database is unavailable ({exc})")
    try:
        profile = _chunk_current_profile(read_conn)
        if profile is not None:
            identity = str(profile["identity_hash"])
            model = str(profile["model_name"])
            provider_name = str(profile["provider"])
            profile_dim = int(profile["dim"])
            profile_dtype = str(profile["dtype"] or "float32")
        else:
            # No chunk profile yet: a dry-run can still estimate pending/tokens/
            # cost using the default chunk model for the configured provider.
            identity = None
            provider_name = configured_provider
            model = default_chunk_model(configured_provider, configured_model)
            profile_dim = 0
            profile_dtype = "float32"
        if (
            expected_dtype is not None
            and profile is not None
            and expected_dtype != profile_dtype
        ):
            return _refused(
                f"--dtype {expected_dtype} does not match the registered chunk "
                f"profile dtype ({profile_dtype}); re-run `/lcm embed warmup` with "
                f"LCM_EMBEDDING_STORAGE_DTYPE={expected_dtype} to register that identity"
            )
        if not apply:
            pending, rows, _ = _chunk_pending_rows(read_conn, identity, policy, limit)
    except sqlite3.Error as exc:
        return _refused(f"could not discover pending chunks ({exc})")
    finally:
        read_conn.close()

    def _estimates(documents: list[tuple[str, str, int]]) -> tuple[int, int, int]:
        # The contextualized (voyage-context) chunk apply path groups a message's
        # chunks into one document, skips a chunk only above the 32K per-chunk
        # context cap, and plans requests by the context budgets. Model that
        # exactly so the cost/consent preview matches apply (FIX 4). Plain-voyage
        # and local providers keep the flat per-document estimate.
        if (
            provider_name.strip().lower() in {"voyage", "voyageai"}
            and _is_voyage_context_model(model)
        ):
            return _chunk_context_estimates(documents)
        est_tokens = sum(document[2] for document in documents)
        est_cost_tokens = sum(
            document[2]
            for document in documents
            if provider_name.lower() != "voyage"
            or document[2] <= _VOYAGE_MAX_DOCUMENT_TOKENS
        )
        est_batches = _embedding_batch_estimate(
            provider_name, [document[2] for document in documents]
        )
        return est_tokens, est_cost_tokens, est_batches

    if not apply:
        estimated_tokens, estimated_cost_tokens, estimated_batches = _estimates(rows)
        return _embedding_backfill_report(
            mode=mode,
            status="dry-run",
            provider=provider_name,
            model=model,
            pending=pending,
            selected=len(rows),
            estimated_tokens=estimated_tokens,
            estimated_cost_tokens=estimated_cost_tokens,
            estimated_batches=estimated_batches,
            embedded=0,
            skipped=[],
            failed=[],
            remaining=pending,
            duration=time.monotonic() - started,
            consumed_tokens=0,
            corpus="chunks",
            policy=policy,
            include_next_hint=include_next_hint,
        )

    # -- apply --
    if profile is None:
        return _refused(
            "no chunk embedding profile is registered; register one before "
            "`--corpus chunks --apply`"
        )

    # Consent gate: unlike the summary corpus (which sends only generated
    # summaries), the chunk corpus sends RAW, VERBATIM message text — including
    # tool-result and error/traceback content, exactly the content most likely to
    # carry secrets — to the embedding provider. Require an explicit
    # acknowledgment before sending that to a CLOUD provider; local providers
    # (fastembed/ollama) keep the text on this machine and are exempt (F1).
    if not _is_local_embedding_provider(provider_name) and not confirm_raw_text:
        return _refused(
            f"the chunk corpus sends RAW, VERBATIM message text — including "
            f"tool-result and error/traceback content that the summary corpus "
            f"never exposes — to the '{provider_name}' cloud embedding provider. "
            f"Re-run with `--confirm-raw-text` to acknowledge this, or switch to a "
            f"local provider (fastembed/ollama). Note: LCM_SENSITIVE_PATTERNS_ENABLED "
            f"redaction runs at INGEST, so text already stored is not "
            f"retro-redacted before being sent."
        )

    ttl_s = _embedding_backfill_lease_ttl_s()
    heartbeat_s = _embedding_backfill_heartbeat_s()
    budget_s = _embedding_backfill_budget_s()

    store: VectorStore | None = None
    lease: _BackfillLease | None = None
    documents: list[tuple[str, str, int]] = []
    chunk_meta: dict[str, tuple[int, int, int, int]] = {}
    pending = 0
    embedded = 0
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    consumed_tokens = 0
    error: str | None = None
    stop_reason: str | None = None
    lease_lost = False
    identity_superseded = False
    budget_exhausted = False
    try:
        store = VectorStore(db_path, config=engine._config)
        store.ensure_chunk_schema()
        conn = store.connection
        _ensure_inflight_table(conn)
        lease = _acquire_embedding_backfill_lease(
            conn, ttl_s=ttl_s, heartbeat_s=heartbeat_s
        )
        if lease is None:
            store.close()
            return _refused(
                "another embedding backfill holds the lease; retry after it exits "
                "or after the lease TTL expires"
            )
        _prepare_inflight_for_lease(conn, identity, lease)
        captured_identity = EmbeddingIdentity.canonical(
            provider_name,
            model,
            str(profile["revision"] or ""),
            profile_dim,
            profile_dtype,
            str(profile["byteorder"] or "little"),
            "chunk",
        )
        if captured_identity.identity_hash != identity:
            raise ValueError(
                "active chunk embedding identity changed before backfill dispatch"
            )
        if retry_uncertain:
            pending, rows, chunk_meta = _chunk_authorized_uncertain_rows(
                conn, identity, policy, limit
            )
            authorized_uncertain_ids = {row[0] for row in rows}
        else:
            pending, rows, chunk_meta = _chunk_pending_rows(
                conn, identity, policy, limit
            )
            authorized_uncertain_ids = set()
        documents = rows

        chunk_provider_config = dataclasses.replace(
            engine._config, embedding_model=model
        )
        provider = resolve_provider(chunk_provider_config, for_backfill=True)
        if provider is None:
            error = "embedding provider is not configured; run `/lcm embed warmup`"
        elif (
            provider.model_id != model
            or str(provider.provider_id).lower() != provider_name.lower()
        ):
            error = "configured provider does not match the chunk profile; run `/lcm embed warmup`"
        else:
            for offset in range(0, len(documents), _EMBEDDING_BACKFILL_BATCH_SIZE):
                if not lease.renew():
                    lease_lost = True
                    stop_reason = "lease_lost"
                    break
                if budget_s > 0 and (time.monotonic() - started) > budget_s:
                    budget_exhausted = True
                    stop_reason = "op_budget_exhausted"
                    break
                batch = documents[offset:offset + _EMBEDDING_BACKFILL_BATCH_SIZE]
                batch_authorized_ids = {
                    item[0] for item in batch if item[0] in authorized_uncertain_ids
                }
                _mark_inflight(
                    conn, identity, lease, [item[0] for item in batch],
                    authorized_uncertain_ids=batch_authorized_ids,
                )
                request_by_index: dict[int, str] = {}
                indexes_by_request: dict[str, set[int]] = {}
                transitioned_requests: set[str] = set()
                latest_request_id: str | None = None

                def before_dispatch(indexes: tuple[int, ...]) -> None:
                    nonlocal latest_request_id
                    normalized = tuple(int(index) for index in indexes)
                    if (
                        not normalized
                        or len(set(normalized)) != len(normalized)
                        or any(index < 0 or index >= len(batch) for index in normalized)
                        or any(index in request_by_index for index in normalized)
                    ):
                        raise EmbeddingProviderError(
                            "provider dispatched invalid or duplicate document indexes"
                        )
                    request_id = uuid.uuid4().hex
                    ids = [batch[index][0] for index in normalized]
                    if _mark_dispatched(
                        conn, identity, lease, ids, request_id,
                        authorized_uncertain_ids=batch_authorized_ids,
                    ) != len(ids):
                        raise EmbeddingProviderError(
                            "embedding lease lost before provider dispatch"
                        )
                    indexes_by_request[request_id] = set(normalized)
                    request_by_index.update(
                        (index, request_id) for index in normalized
                    )
                    latest_request_id = request_id

                try:
                    accepted_indexes: set[int] = set()
                    for accepted_batch in _provider_chunk_document_batches(
                        provider, batch, chunk_meta,
                        before_dispatch=before_dispatch,
                    ):
                        if len(accepted_batch.indexes) != len(accepted_batch.vectors):
                            raise EmbeddingProviderError(
                                "provider returned a different number of indexes and embeddings"
                            )
                        publish_rows = []
                        publish_items = []
                        for batch_index, vector in zip(
                            accepted_batch.indexes, accepted_batch.vectors
                        ):
                            index = int(batch_index)
                            if index < 0 or index >= len(batch) or index in accepted_indexes:
                                raise EmbeddingProviderError(
                                    "provider returned an invalid accepted document index"
                                )
                            accepted_indexes.add(index)
                            item = batch[index]
                            request_id = request_by_index.get(index)
                            if request_id is None:
                                raise EmbeddingProviderError(
                                    "provider returned an embedding before durable dispatch"
                                )
                            store_id, chunk_index, char_start, char_end = chunk_meta[item[0]]
                            publish_rows.append({
                                "chunk_id": item[0], "vec": vector,
                                "store_id": store_id, "chunk_index": chunk_index,
                                "char_start": char_start, "char_end": char_end,
                                "token_estimate": item[2], "request_id": request_id,
                            })
                            publish_items.append((item, request_id))
                        # One BEGIN IMMEDIATE per accepted network batch (one fsync)
                        # instead of one per row; each row still runs the full CAS
                        # under its own savepoint (see the summary path).
                        try:
                            published_results = store.publish_chunk_embedding_batch_under_lease(
                                publish_rows,
                                model=model,
                                identity=captured_identity,
                                claim_key=_EMBEDDING_BACKFILL_CLAIM_KEY,
                                lease_id=lease.lease_id,
                                generation=lease.generation,
                            )
                        except Exception as publish_exc:
                            # Local storage/lock failure of the batch publish call
                            # itself (see the summary path) -- record every
                            # accepted-batch row as a local_error failure rather
                            # than letting the generic handler mislabel it a
                            # provider error and drop the count.
                            local_reason = f"local_error:{publish_exc}"
                            failed.extend(
                                (item[0], local_reason) for item, _rid in publish_items
                            )
                            raise _LocalPublishError(local_reason) from publish_exc
                        for result, (item, request_id) in zip(
                            published_results,
                            publish_items,
                        ):
                            if result.error is not None:
                                exc = result.error
                                failed.append((item[0], f"record_error:{exc}"))
                                if not _owned_inflight_transition(
                                    conn, identity, lease, request_id,
                                    error=f"accepted remotely; local publish failed: {exc}",
                                    authorized_uncertain_ids=batch_authorized_ids,
                                ):
                                    lease_lost = True
                                    stop_reason = "lease_lost"
                                else:
                                    transitioned_requests.add(request_id)
                                    raise EmbeddingProviderError(
                                        "accepted remotely; local publication failed"
                                    ) from exc
                                break
                            publish_outcome = result.outcome
                            if publish_outcome is EmbeddingPublishOutcome.OWNERSHIP_LOST:
                                lease_lost = True
                                stop_reason = "lease_lost"
                                break
                            if publish_outcome is EmbeddingPublishOutcome.IDENTITY_SUPERSEDED:
                                identity_superseded = True
                                stop_reason = "identity_superseded"
                                failed.append(
                                    (item[0], "identity_superseded_after_remote_acceptance")
                                )
                                transitioned_requests.add(request_id)
                                break
                            embedded += 1
                            consumed_tokens += item[2]
                        if lease_lost or identity_superseded:
                            break
                    if lease_lost or identity_superseded:
                        break
                    skipped_indexes = {
                        int(index)
                        for index in getattr(provider, "last_skipped_documents", [])
                        if 0 <= int(index) < len(batch)
                    }
                    for index in sorted(skipped_indexes):
                        skipped.append((batch[index][0], "provider_document_token_cap"))
                        request_id = request_by_index.get(index)
                        if not _owned_inflight_transition(
                            conn, identity, lease, request_id,
                            embedded_id=batch[index][0],
                            error="provider document token cap",
                            authorized_uncertain_ids=batch_authorized_ids,
                        ):
                            lease_lost = True
                            stop_reason = "lease_lost"
                            break
                    unresolved = set(range(len(batch))) - accepted_indexes - skipped_indexes
                    if unresolved:
                        raise EmbeddingProviderError(
                            "provider did not resolve every dispatched document"
                        )
                except Exception as exc:
                    if isinstance(exc, _LocalPublishError):
                        # Local storage/lock failure of the batch publish call:
                        # rows already appended to `failed` with this reason.
                        reason = exc.reason
                    elif isinstance(exc, VoyageError):
                        reason = f"provider_{exc.kind}:{exc}"
                    else:
                        reason = f"provider_error:{exc}"
                    definitive_rejection = (
                        isinstance(exc, ProviderPreDispatchError)
                        or (
                            isinstance(exc, VoyageError)
                            and exc.kind in {"auth", "bad_request", "rate_limit"}
                        )
                    )
                    if latest_request_id is not None:
                        failed_indexes = indexes_by_request[latest_request_id] - accepted_indexes
                        failed.extend((batch[index][0], reason) for index in failed_indexes)
                        if (
                            not lease_lost
                            and latest_request_id not in transitioned_requests
                            and not _owned_inflight_transition(
                                conn, identity, lease, latest_request_id,
                                error=reason, retryable=definitive_rejection,
                                authorized_uncertain_ids=batch_authorized_ids,
                            )
                        ):
                            lease_lost = True
                            stop_reason = "lease_lost"
                    if not lease_lost:
                        for index, item in enumerate(batch):
                            if index in request_by_index:
                                continue
                            if not _owned_inflight_transition(
                                conn, identity, lease, None,
                                embedded_id=item[0],
                                authorized_uncertain_ids=batch_authorized_ids,
                            ):
                                lease_lost = True
                                stop_reason = "lease_lost"
                                break
                    if isinstance(exc, VoyageError) and exc.kind == "auth":
                        error = f"provider authentication failed; {exc}"
                        break
    except _BackfillLeaseLost:
        lease_lost = True
        stop_reason = "lease_lost"
    except Exception as exc:
        error = str(exc)
    finally:
        if store is not None:
            if lease is not None and not lease_lost and not identity_superseded:
                try:
                    lease.release()
                except Exception as exc:
                    error = (
                        f"{error}; lease release failed: {exc}"
                        if error else f"lease release failed: {exc}"
                    )
            store.close()

    remaining, in_flight_count, uncertain_count = _chunk_backfill_remaining(
        db_path, identity, policy, pending, embedded
    )
    selected_embeddable = len(documents) - len(skipped)
    status = _embedding_backfill_status(
        error=error, lease_lost=lease_lost, budget_exhausted=budget_exhausted,
        embedded=embedded, selected_embeddable=selected_embeddable,
        failed=failed, uncertain=uncertain_count, skipped=len(skipped),
    )
    estimated_tokens, estimated_cost_tokens, estimated_batches = _estimates(documents)
    return _embedding_backfill_report(
        mode=mode, status=status, provider=provider_name, model=model,
        pending=pending, selected=len(documents),
        estimated_tokens=estimated_tokens, estimated_cost_tokens=estimated_cost_tokens,
        estimated_batches=estimated_batches, embedded=embedded, skipped=skipped,
        failed=failed, remaining=remaining, duration=time.monotonic() - started,
        consumed_tokens=consumed_tokens, error=error, in_flight=in_flight_count,
        uncertain=uncertain_count, stop_reason=stop_reason,
        corpus="chunks", policy=policy,
    )


def _embedding_backfill_status(
    *,
    error: str | None,
    lease_lost: bool,
    budget_exhausted: bool,
    embedded: int,
    selected_embeddable: int,
    failed: list[tuple[str, str]],
    uncertain: int = 0,
    skipped: int = 0,
) -> str:
    """Report the truthful terminal status — never a premature ``complete``."""
    if error:
        return "error"
    if lease_lost or budget_exhausted:
        return "partial"
    if uncertain or skipped:
        return "partial"
    if failed:
        return "failed" if embedded == 0 else "partial"
    if embedded >= selected_embeddable:
        return "complete"
    return "partial"


def handle_lcm_command(raw_args: str | None, engine) -> str:
    tokens = [part.strip() for part in (raw_args or "").strip().split() if part.strip()]
    if not tokens:
        return _status_text(engine)

    head = tokens[0].lower()
    rest = tokens[1:]

    if head == "status":
        if rest:
            return _help_text("`/lcm status` does not accept extra arguments.")
        return _status_text(engine)

    if head == "doctor":
        if not rest:
            return _doctor_text(engine)
        if len(rest) == 1 and rest[0].lower() == "clean":
            return _doctor_clean_text(engine)
        if len(rest) == 1 and rest[0].lower() == "repair":
            return _doctor_repair_text(engine)
        if len(rest) == 1 and rest[0].lower() == "source":
            return _doctor_source_text(engine)
        if len(rest) == 1 and rest[0].lower() == "retention":
            return _doctor_retention_text(engine)
        if len(rest) == 2 and rest[0].lower() == "clean" and rest[1].lower() == "apply":
            return _doctor_clean_apply_text(engine)
        if len(rest) == 2 and rest[0].lower() == "clean" and rest[1].lower() == "lifecycle":
            return _doctor_clean_lifecycle_text(engine)
        if len(rest) == 3 and rest[0].lower() == "clean" and rest[1].lower() == "lifecycle" and rest[2].lower() == "apply":
            return _doctor_clean_lifecycle_apply_text(engine)
        if len(rest) == 2 and rest[0].lower() == "repair" and rest[1].lower() == "apply":
            return _doctor_repair_apply_text(engine)
        if len(rest) == 2 and rest[0].lower() == "repair" and rest[1].lower() == "schema-stamp":
            return _doctor_repair_schema_stamp_text(engine)
        if (
            len(rest) == 3
            and rest[0].lower() == "repair"
            and rest[1].lower() == "schema-stamp"
            and rest[2].lower() == "apply"
        ):
            return _doctor_repair_schema_stamp_apply_text(engine)
        if len(rest) == 2 and rest[0].lower() == "source" and rest[1].lower() == "apply":
            return _doctor_source_apply_text(engine)
        return _help_text("`/lcm doctor` currently supports `clean`, `clean apply`, `clean lifecycle`, `clean lifecycle apply`, `repair`, `repair apply`, `repair schema-stamp`, `repair schema-stamp apply`, `source`, `source apply`, and `retention` as extra subcommands.")

    if head == "backup":
        if rest:
            return _help_text("`/lcm backup` does not accept extra arguments.")
        return _backup_text(engine)

    if head == "rotate":
        if not rest:
            return _rotate_text(engine)
        if len(rest) == 1 and rest[0].lower() == "apply":
            return _rotate_apply_text(engine)
        return _help_text("`/lcm rotate` accepts an optional `apply` subcommand.")

    if head == "rollups":
        return _rollups_text(rest, engine)

    if head == "preset":
        return _preset_text(rest, engine)

    if head == "embed":
        if len(rest) == 1 and rest[0].lower() == "warmup":
            return _embedding_warmup_text(engine)
        if rest and rest[0].lower() == "backfill":
            return _embedding_backfill_text(rest[1:], engine)
        return _help_text("`/lcm embed` requires the `warmup` or `backfill` subcommand.")

    if head == "help":
        return _help_text()

    return _help_text(f"Unknown subcommand: {tokens[0]}")
