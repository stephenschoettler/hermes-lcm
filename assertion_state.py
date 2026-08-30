"""Typed, conflict-preserving state reads over source-valid V4 assertions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

from .assertion_store import AssertionStore


_INACTIVE_RELATION_STATUS = {
    "cancels": "cancelled",
    "fulfills": "fulfilled",
    "reverses": "reversed",
    "supersedes": "superseded",
}

_RECOMMENDATION_ACCEPT_RELATIONS = frozenset({"confirms"})
_RECOMMENDATION_DECLINE_RELATIONS = frozenset({"cancels", "contradicts", "reverses"})


@dataclass(frozen=True)
class AssertionStateResult:
    subject_key: str
    predicate_key: str | None
    as_of: float | None
    assertions: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...]
    active_assertion_ids: tuple[str, ...] | None
    conflict_assertion_ids: tuple[str, ...]
    assertions_truncated: bool
    relations_truncated: bool


def _object_identity(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "object": row.get("object_value"),
            "polarity": row.get("polarity"),
            "value_text": row.get("value_text"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _attribution(row: dict[str, Any]) -> str:
    speaker = str(row.get("speaker_role") or "")
    subject = str(row.get("subject_key") or "")
    if speaker in {"user", "assistant"} and subject == f"{speaker}:self":
        return "first_person"
    addressee = {"user": "assistant:self", "assistant": "user:self"}.get(speaker)
    if addressee and subject == addressee:
        return "addressee"
    return "third_party"


def _semantic_state(
    row: dict[str, Any],
    lifecycle_status: set[str],
    recommendation_dispositions: set[str],
) -> str:
    kind = str(row.get("kind") or "")
    if kind == "recommendation":
        if len(recommendation_dispositions) > 1:
            return "conflicting_disposition"
        if recommendation_dispositions:
            return next(iter(recommendation_dispositions))
        if lifecycle_status:
            return sorted(lifecycle_status)[0]
        return "unanswered"
    if kind == "commitment":
        if "fulfilled" in lifecycle_status:
            return "fulfilled"
        if "cancelled" in lifecycle_status:
            return "cancelled"
        if lifecycle_status:
            return sorted(lifecycle_status)[0]
        return "pending"
    if lifecycle_status:
        return sorted(lifecycle_status)[0]
    return "current"


def query_assertion_state(
    store: AssertionStore,
    *,
    subject_key: str,
    predicate_key: str | None = None,
    kinds: Iterable[str] | None = None,
    scope_key: str | None = None,
    speaker_role: str | None = None,
    as_of: float | None = None,
    access_scope: str | None = None,
    limit: int = 100,
) -> AssertionStateResult:
    """Return explicit lifecycle state without recency-based conflict erasure.

    ``access_scope`` is the per-principal owner predicate, forwarded to both
    queries. Absent -- every default-off and non-Teams caller -- the behaviour
    is unchanged. It has to reach BOTH: assertions carry their source quote and
    relations carry the quotes of both endpoints.
    """

    normalized_subject = str(subject_key or "").strip()
    if not normalized_subject:
        raise ValueError("subject_key is required for a bounded state query")
    rows = store.query_assertions(
        subject_key=normalized_subject,
        predicate_key=predicate_key,
        kinds=kinds,
        scope_key=scope_key,
        speaker_role=speaker_role,
        as_of=as_of,
        access_scope=access_scope,
        limit=limit,
    )
    ids = [str(row["assertion_id"]) for row in rows]
    raw_relations = (
        store.query_relations(
            assertion_ids=ids, as_of=as_of, access_scope=access_scope, limit=500
        )
        if ids
        else []
    )
    relations_truncated = len(raw_relations) == 500
    id_set = set(ids)
    # ``query_relations(assertion_ids=...)`` already guarantees at least one
    # endpoint is selected. Keep the other endpoint visible so a kind-filtered
    # recommendation or commitment still carries its explicit disposition.
    relations = raw_relations

    lifecycle: dict[str, set[str]] = {assertion_id: set() for assertion_id in ids}
    rows_by_id = {str(row["assertion_id"]): row for row in rows}
    recommendation_dispositions: dict[str, set[str]] = {
        assertion_id: set()
        for assertion_id, row in rows_by_id.items()
        if str(row.get("kind") or "") == "recommendation"
    }
    explicit_conflicts: set[str] = set()
    for relation in relations:
        relation_type = str(relation["relation_type"])
        from_id = str(relation["from_assertion_id"])
        to_id = str(relation["to_assertion_id"])
        if relation_type in _INACTIVE_RELATION_STATUS and to_id in lifecycle:
            lifecycle[to_id].add(_INACTIVE_RELATION_STATUS[relation_type])
        if relation_type == "contradicts":
            explicit_conflicts.update({from_id, to_id} & id_set)
        if to_id in recommendation_dispositions:
            if relation_type in _RECOMMENDATION_ACCEPT_RELATIONS:
                recommendation_dispositions[to_id].add("accepted")
            if relation_type in _RECOMMENDATION_DECLINE_RELATIONS:
                recommendation_dispositions[to_id].add("declined")

    active_ids = {
        assertion_id
        for assertion_id, statuses in lifecycle.items()
        if not statuses
    }
    unresolved_conflicts = set(explicit_conflicts) & active_ids
    groups: dict[tuple[str, str, str, str], dict[str, list[str]]] = {}
    for row in rows:
        assertion_id = str(row["assertion_id"])
        if assertion_id not in active_ids:
            continue
        key = (
            str(row["subject_key"]),
            str(row["predicate_key"]),
            str(row["scope_key"]),
            str(row["kind"]),
        )
        groups.setdefault(key, {}).setdefault(_object_identity(row), []).append(assertion_id)
    for variants in groups.values():
        if len(variants) > 1:
            for assertion_ids in variants.values():
                unresolved_conflicts.update(assertion_ids)

    annotated: list[dict[str, Any]] = []
    for row in rows:
        assertion_id = str(row["assertion_id"])
        item = dict(row)
        item["active"] = None if relations_truncated else assertion_id in active_ids
        item["lifecycle_status"] = (
            None
            if relations_truncated
            else tuple(sorted(lifecycle[assertion_id]))
        )
        item["unresolved_conflict"] = assertion_id in unresolved_conflicts
        item["attribution"] = _attribution(item)
        item["semantic_state"] = (
            "unknown"
            if relations_truncated
            else _semantic_state(
                item,
                lifecycle[assertion_id],
                recommendation_dispositions.get(assertion_id, set()),
            )
        )
        annotated.append(item)
    return AssertionStateResult(
        subject_key=normalized_subject,
        predicate_key=predicate_key,
        as_of=as_of,
        assertions=tuple(annotated),
        relations=tuple(relations),
        active_assertion_ids=(
            None
            if relations_truncated
            else tuple(assertion_id for assertion_id in ids if assertion_id in active_ids)
        ),
        conflict_assertion_ids=tuple(
            assertion_id for assertion_id in ids if assertion_id in unresolved_conflicts
        ),
        assertions_truncated=len(rows) == int(limit),
        relations_truncated=relations_truncated,
    )
