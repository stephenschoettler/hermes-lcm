"""Shared read-only diagnostic helpers for LCM tools and commands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DOCTOR_ACTION_SAFE_IGNORE = "safe/ignore"
DOCTOR_ACTION_INSPECT = "inspect"
DOCTOR_ACTION_BACKUP_FIRST_CLEANUP = "backup-first cleanup"


def _enforce_state_db_containment(path: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    env_base = os.environ.get("LCM_HERMES_BASE_DIR")
    if env_base:
        allowed_base = Path(env_base).expanduser().resolve()
        try:
            resolved.relative_to(allowed_base)
        except ValueError:
            raise ValueError(
                f"{description} resolves to {resolved} which is not within allowed base {allowed_base}"
            )
    return resolved


def state_db_path_for_engine(engine: Any) -> Path:
    """Return the Hermes state database path for an LCM engine.

    The path is read-only diagnostic input. When ``LCM_HERMES_BASE_DIR`` is
    configured, enforce the same containment guard for all diagnostic surfaces.
    """
    hermes_home = getattr(engine, "_hermes_home", "") or ""
    if hermes_home:
        return _enforce_state_db_containment(
            Path(hermes_home) / "state.db",
            description=f"hermes_home {hermes_home}",
        )
    db_path = Path(getattr(engine._store, "db_path", Path.home() / ".hermes" / "lcm.db"))
    return _enforce_state_db_containment(
        db_path.parent / "state.db",
        description=f"state database fallback from LCM database {db_path}",
    )


def has_lifecycle_fragmentation(stats: dict[str, Any]) -> bool:
    """Return whether lifecycle diagnostics should be treated as warning evidence.

    Retained-history drift is intentionally read-only diagnostic context. Keep the
    doctor warning for concrete operator action (empty lifecycle rows that the
    explicit backup-first cleanup path can prune) or diagnostic unreadability, but
    do not make overall health unhealthy solely because historical LCM/state
    indexes no longer agree.
    """
    empty_lifecycle_rows = int(stats.get("empty_lifecycle_rows", 0) or 0)
    return empty_lifecycle_rows > 0 or (
        bool(stats.get("state_db_checked")) and bool(stats.get("state_db_error"))
    )


def doctor_guidance_for_check(check: dict[str, Any]) -> dict[str, Any] | None:
    """Return operator triage guidance for one lcm_doctor check.

    Guidance is deliberately conservative: most warning classes are inspect-only
    evidence, and any mutation path is framed as preview/backup/apply rather than
    implied automatic cleanup.
    """
    status = str(check.get("status") or "")
    if status not in {"warn", "fail"}:
        return None

    name = str(check.get("check") or "unknown")
    detail = check.get("detail")
    action = DOCTOR_ACTION_INSPECT
    command = "inspect the reported detail and confirm the active HERMES_HOME/LCM_DATABASE_PATH"
    warning_only = False
    rationale = "operator review required before changing persisted LCM state"

    if name == "database_integrity":
        command = "stop and inspect the SQLite database path; restore from backup if integrity_check is not ok"
    elif name == "schema_core_tables":
        command = "verify HERMES_HOME/LCM_DATABASE_PATH points at the intended LCM database before repair or restore"
    elif name in {"messages_fts_integrity", "nodes_fts_integrity", "fts_index_sync"}:
        if status == "warn" and isinstance(detail, dict) and detail.get("status") == "unchecked":
            action = DOCTOR_ACTION_INSPECT
            command = "rerun `/lcm doctor` with read-write SQLite access if a deep FTS integrity result is needed"
            warning_only = True
            rationale = "the deep FTS check could not run, but this is not evidence that the index is corrupt"
        else:
            action = DOCTOR_ACTION_BACKUP_FIRST_CLEANUP
            command = "run `/lcm doctor repair` first; if it still recommends repair, run `/lcm backup` before `/lcm doctor repair apply`"
            rationale = "FTS repair is rebuildable, but it still mutates SQLite indexes"
    elif name == "sqlite_storage":
        command = "inspect journal/quick_check output and database/WAL size; restore from backup if SQLite reports corruption"
    elif name == "payload_storage":
        missing_refs = 0
        heartbeat_rows = 0
        suspicious_rows = 0
        if isinstance(detail, dict):
            missing_refs = int(detail.get("externalized_payload_refs_missing", 0) or 0)
            heartbeat_rows = len(detail.get("heartbeat_noise_rows") or [])
            suspicious_rows = sum(
                len(detail.get(key) or [])
                for key in (
                    "suspicious_data_uri_content_rows",
                    "suspicious_data_uri_tool_calls_rows",
                    "suspicious_base64_like_rows",
                    "suspicious_repetitive_assistant_rows",
                )
            )
        if status == "warn" and heartbeat_rows and not missing_refs and not suspicious_rows:
            action = DOCTOR_ACTION_SAFE_IGNORE
            command = "safe to ignore unless heartbeat/progress noise is crowding useful recall; consider message/session filters for future rows"
            rationale = "heartbeat rows are read-only noise diagnostics, not corruption"
        else:
            command = "inspect payload rows/refs; restore missing externalized payload files from backup before deleting or rewriting anything"
            if status == "warn":
                warning_only = True
                rationale = "payload warnings may represent preserved user/tool data"
            else:
                rationale = "payload diagnostic failures mean doctor could not read storage risk state reliably"
    elif name == "sensitive_pattern_handling":
        command = "inspect LCM_SENSITIVE_PATTERNS settings; remove unknown names or configure supported catalog entries"
    elif name == "orphaned_dag_nodes":
        command = "inspect affected DAG/source IDs; do not auto-delete summaries without confirming recall impact"
        if status == "warn":
            warning_only = True
        else:
            rationale = "DAG diagnostic failures mean doctor could not read summary/source state reliably"
    elif name == "summary_quality":
        command = "inspect worst_nodes and retrieval behavior; treat as summary quality evidence, not cleanup input"
        if status == "warn":
            warning_only = True
        else:
            rationale = "summary-quality diagnostic failures mean doctor could not read DAG quality state reliably"
    elif name == "config_validation":
        command = "inspect LCM_* environment/config values and adjust only intentional operator overrides"
    elif name == "source_lineage_hygiene" and status == "warn":
        action = DOCTOR_ACTION_SAFE_IGNORE
        command = "safe to ignore legacy blank-source observations; use `/lcm doctor source` only when you intentionally want backup-first normalization"
        rationale = "legacy blank sources are normalized to unknown for compatibility"
    elif name == "source_lineage_hygiene":
        command = "inspect source-lineage diagnostics and SQLite read errors before running any source normalization workflow"
        rationale = "source-lineage failures indicate the doctor could not read attribution state reliably"
    elif name == "lifecycle_fragmentation":
        command = "inspect lifecycle categories; only use explicit backup-first lifecycle cleanup for empty lifecycle rows"
        if status == "warn":
            warning_only = True
            rationale = "not every lifecycle/state mismatch is harmful or safe to mutate"
        else:
            rationale = "lifecycle diagnostic failures mean doctor could not read session lifecycle state reliably"
    elif name == "context_pressure":
        action = DOCTOR_ACTION_SAFE_IGNORE
        command = "safe to ignore if compaction proceeds normally; inspect lcm_status only if pressure stays high or compaction loops"
        warning_only = True
        rationale = "context pressure is an operating state, not persisted-state corruption"
    elif name == "cleanup_candidates":
        action = DOCTOR_ACTION_BACKUP_FIRST_CLEANUP
        command = "run `/lcm doctor clean` first; if candidates are expected junk/noise, run `/lcm backup` before `/lcm doctor clean apply`"
        rationale = "candidate cleanup deletes rows and must stay preview-and-backup gated"

    return {
        "check": name,
        "status": status,
        "action": action,
        "operator_action": command,
        "warning_only": warning_only,
        "rationale": rationale,
    }


def doctor_guidance_for_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return actionable guidance for warning/failing lcm_doctor checks."""
    guidance = []
    for check in checks:
        item = doctor_guidance_for_check(check)
        if item is not None:
            guidance.append(item)
    return guidance


# Backward-compatible private aliases for existing command/tool internals and tests.
_state_db_path_for_engine = state_db_path_for_engine
_has_lifecycle_fragmentation = has_lifecycle_fragmentation
