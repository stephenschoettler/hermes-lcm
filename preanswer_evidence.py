"""Default-off, bounded product-owned evidence augmentation before an LLM turn.

This module is deliberately host-neutral.  The Hermes ``pre_llm_call`` hook and
the benchmark bridge both call this implementation.  It never returns prose as
truth: only exact cited evidence or an immutable computation trace may be
rendered into an ephemeral context block.  Every failure returns no context.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import time
from typing import Any, Callable, Mapping, Sequence

from .access_policy import policy_access_context, policy_for_engine
from .evidence_pack import build_evidence_pack, normalize_question_date
from .reasoning import EvidencePlan, compile_evidence_plan


PREANSWER_EVIDENCE_VERSION = "preanswer-evidence-v1"
_EXACT_REF_RE = re.compile(r"^lcm:(?P<store_id>[1-9]\d*):(?P<start>\d+)-(?P<end>\d+)$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_NUMBER_RE = re.compile(r"(?<![\w.])-?(?:\d+(?:,\d{3})*|\d*\.\d+)(?!\w)")
_STOP_WORDS = frozenset(
    {
        "a",
        "ago",
        "all",
        "and",
        "are",
        "between",
        "combined",
        "currently",
        "did",
        "do",
        "does",
        "five",
        "for",
        "four",
        "happened",
        "how",
        "i",
        "in",
        "is",
        "latest",
        "many",
        "more",
        "most",
        "my",
        "now",
        "number",
        "of",
        "one",
        "recent",
        "seven",
        "six",
        "the",
        "these",
        "this",
        "three",
        "total",
        "two",
        "value",
        "values",
        "what",
        "when",
        "where",
        "which",
        "who",
        "year",
    }
)


@dataclass(frozen=True)
class PreAnswerBudgets:
    max_retrieval_calls: int = 1
    max_novel_refs: int = 4
    max_input_refs: int = 25
    max_context_chars: int = 8_000

    def public_dict(self) -> dict[str, int]:
        return {
            "max_retrieval_calls": self.max_retrieval_calls,
            "max_novel_refs": self.max_novel_refs,
            "max_input_refs": self.max_input_refs,
            "max_context_chars": self.max_context_chars,
        }


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return min(maximum, max(minimum, parsed))


def _budgets(value: Any) -> PreAnswerBudgets:
    raw = value if isinstance(value, Mapping) else {}
    return PreAnswerBudgets(
        max_retrieval_calls=_bounded_int(
            raw.get("max_retrieval_calls"), default=1, minimum=0, maximum=1
        ),
        max_novel_refs=_bounded_int(
            raw.get("max_novel_refs"), default=4, minimum=1, maximum=8
        ),
        max_input_refs=_bounded_int(
            raw.get("max_input_refs"), default=25, minimum=1, maximum=50
        ),
        max_context_chars=_bounded_int(
            raw.get("max_context_chars"), default=8_000, minimum=256, maximum=16_000
        ),
    )


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base_result(
    *,
    status: str,
    reason_code: str,
    budgets: PreAnswerBudgets,
    started: float,
    retrieval: Mapping[str, Any] | None = None,
    decision: Mapping[str, Any] | None = None,
    baseline_refs: Sequence[str] = (),
) -> dict[str, Any]:
    refs = list(baseline_refs)
    return {
        "version": PREANSWER_EVIDENCE_VERSION,
        "status": status,
        "reason_code": reason_code,
        "context": None,
        "decision": dict(decision or {}),
        "baseline": {
            "exact_ref_count": len(refs),
            "exact_refs_sha256": _sha256_json(refs),
        },
        "retrieval": dict(
            retrieval
            or {
                "calls": 0,
                "status": "not_called",
                "novel_exact_refs": [],
                "usage": {},
            }
        ),
        "novel_exact_refs": [],
        "evidence": [],
        "computation": None,
        "computation_sha256": None,
        "budgets": budgets.public_dict(),
        "trace": {
            "version": PREANSWER_EVIDENCE_VERSION,
            "context_sha256": None,
            "truncated": False,
        },
        "metrics": {
            "latency_ms": round((time.perf_counter() - started) * 1_000.0, 3),
            "context_chars": 0,
        },
        "provenance": {
            "storage": "same_lcm_db",
            "provider_neutral_analysis": True,
            "final_prose_cached": False,
        },
    }


def _normalize_refs(
    raw_refs: Sequence[Any], *, limit: int
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    exact_refs: list[str] = []
    for raw in raw_refs[:limit]:
        if isinstance(raw, str):
            candidate = {"exact_ref": raw.strip()}
        elif isinstance(raw, Mapping):
            candidate = dict(raw)
            candidate["exact_ref"] = str(candidate.get("exact_ref") or "").strip()
        else:
            continue
        exact_ref = candidate["exact_ref"]
        if not _EXACT_REF_RE.fullmatch(exact_ref) or exact_ref in exact_refs:
            continue
        exact_refs.append(exact_ref)
        candidates.append(candidate)
    return candidates, exact_refs


def authorize_supplied_baseline_refs(engine: Any, refs: Any) -> tuple:
    """Authorize caller-supplied exact refs before any content is read.

    The generated baseline goes through ``handle_tool_call("lcm_recall", ...)``,
    which is gated. Caller-supplied refs skipped that entirely: each names a
    store_id whose exact span is read by :func:`_source_window` and injected
    into model context, so a ref to another principal's known store_id was
    disclosed verbatim without the policy ever seeing it.

    Unauthorized refs are dropped rather than raised on, so a partly-authorized
    payload still answers from the part the caller may see.
    """
    policy = policy_for_engine(engine)
    access_context = policy_access_context(engine)
    authorized: list = []
    for ref in refs:
        # Callers pass either a bare "lcm:<store_id>:<start>-<end>" string or a
        # mapping carrying it under "exact_ref" alongside its quote.
        exact_ref = ref.get("exact_ref") if isinstance(ref, Mapping) else ref
        match = _EXACT_REF_RE.fullmatch(str(exact_ref or ""))
        if match is None:
            # Not an exact store span, so not a disclosure this gate governs.
            # Passed through untouched: dropping it would silently empty the
            # payload for callers whose refs were never store references.
            authorized.append(ref)
            continue
        expected_scope = {
            "kind": "preanswer_baseline_ref",
            "store_id": int(match.group("store_id")),
            "exact_ref": str(exact_ref),
        }
        decision = policy.authorize_operation(access_context, "read", expected_scope)
        policy.audit_decision(
            access_context, "read", decision.denial_reason, decision.public()
        )
        if decision.allowed:
            # The ORIGINAL item, not the extracted ref -- downstream consumers
            # read the quote alongside it.
            authorized.append(ref)
    return tuple(authorized)


def _source_window(engine: Any, exact_ref: str) -> tuple[str, int, int] | None:
    match = _EXACT_REF_RE.fullmatch(exact_ref)
    if match is None:
        return None
    store_id = int(match.group("store_id"))
    start = int(match.group("start"))
    end = int(match.group("end"))
    row = engine._store.get(store_id)
    if row is None:
        return None
    content = str(row.get("content") or "")
    if start < 0 or end <= start or end > len(content):
        return None
    return content[start:end], start, end


def _unit_for_quote(quote: str, match: re.Match[str]) -> str | None:
    before = quote[max(0, match.start() - 16) : match.start()]
    after = quote[match.end() : match.end() + 20]
    if re.search(r"\$\s*$", before) or re.match(
        r"\s*(?:usd\b|dollars?\b)", after, re.IGNORECASE
    ):
        return "usd"
    unit_match = re.match(
        r"\s*(pages?|hours?|hrs?|minutes?|mins?|days?|weeks?|months?|points?|items?|events?)\b",
        after,
        re.IGNORECASE,
    )
    if unit_match is None:
        return None
    unit = unit_match.group(1).casefold()
    aliases = {
        "pages": "page",
        "hours": "hour",
        "hrs": "hour",
        "minutes": "minute",
        "mins": "minute",
        "days": "day",
        "weeks": "week",
        "months": "month",
        "points": "point",
        "items": "item",
        "events": "event",
    }
    return aliases.get(unit, unit)


def _numeric_facet(quote: str) -> tuple[float | int, str] | None:
    grounded: list[tuple[float | int, str]] = []
    for match in _NUMBER_RE.finditer(quote):
        raw = match.group(0).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        unit = _unit_for_quote(quote, match)
        if unit is None:
            continue
        grounded.append((int(value) if value.is_integer() else value, unit))
    return grounded[0] if len(grounded) == 1 else None


def _canonical_event_key(quote: str) -> str | None:
    tokens = [token.casefold() for token in _WORD_RE.findall(quote)]
    if not tokens:
        return None
    return " ".join(tokens)[:300]


def _automatic_candidate(
    raw: Mapping[str, Any], *, engine: Any, plan: EvidencePlan
) -> dict[str, Any] | None:
    exact_ref = str(raw.get("exact_ref") or "")
    window = _source_window(engine, exact_ref)
    if window is None:
        return None
    quote, _, _ = window
    candidate = {"exact_ref": exact_ref, "quote": quote}

    # Preserve caller facets only when present; the evidence pack revalidates
    # every value against the exact source span.
    for key in ("value", "unit", "key", "label"):
        if raw.get(key) is not None:
            candidate[key] = raw[key]

    if plan.operation in {"sum", "difference"} and "value" not in candidate:
        numeric = _numeric_facet(quote)
        if numeric is not None:
            candidate["value"], candidate["unit"] = numeric
    if plan.operation == "count_distinct" and "key" not in candidate:
        key = _canonical_event_key(quote)
        if key:
            candidate["key"] = key
    if plan.operation in {"date_filter", "order"} and not any(
        candidate.get(key) for key in ("value", "key", "label")
    ):
        # The full exact quote is a grounded human-readable event label.  It is
        # bounded by the evidence-pack quote cap and cannot invent content.
        candidate["label"] = quote[:300]
    return candidate


def _prepare_candidates(
    refs: Sequence[Mapping[str, Any]], *, engine: Any, plan: EvidencePlan
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for raw in refs:
        candidate = _automatic_candidate(raw, engine=engine, plan=plan)
        if candidate is not None:
            prepared.append(candidate)
    return prepared


def _query_terms(question: str, prepared: Sequence[Mapping[str, Any]]) -> list[str]:
    question_terms = [
        token.casefold()
        for token in _WORD_RE.findall(question)
        if len(token) >= 3 and token.casefold() not in _STOP_WORDS
    ]
    evidence_terms = {
        token.casefold()
        for candidate in prepared
        for token in _WORD_RE.findall(str(candidate.get("quote") or ""))
    }
    missing = [term for term in question_terms if term not in evidence_terms]
    return list(dict.fromkeys(missing or question_terms))[:3]


def _missing_requirement(
    question: str, plan: EvidencePlan, prepared: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    count = len(prepared)
    if plan.exact_operands is not None and count >= plan.exact_operands:
        return None
    terms = _query_terms(question, prepared)
    if plan.operation == "latest_fact":
        kind = "latest_state_update"
        if not terms:
            terms = ["current"]
    elif plan.operation in {"sum", "difference", "date_interval"}:
        kind = "missing_operand"
    elif plan.operation == "count_distinct":
        kind = "missing_distinct_event"
    elif plan.operation == "order":
        kind = "missing_order_event"
    elif plan.operation == "date_filter":
        kind = "missing_date_window_event"
    else:
        return None
    if not terms:
        return None
    return {
        "kind": kind,
        "operation": plan.operation,
        "query": " ".join(terms),
        "minimum_operands": plan.minimum_operands,
        "current_operands": count,
    }


def _pack(
    *,
    question: str,
    question_date: Any,
    candidates: Sequence[Mapping[str, Any]],
    engine: Any,
    budgets: PreAnswerBudgets,
) -> dict[str, Any]:
    raw = build_evidence_pack(
        {
            "question": question,
            "question_date": question_date,
            "baseline_refs": list(candidates),
            "budgets": {
                "max_refs": budgets.max_input_refs,
                "max_novel_refs": budgets.max_novel_refs,
                "max_retrieval_calls": 0,
            },
        },
        engine=engine,
    )
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else {"status": "fallback"}


def _render_computation(computation: Mapping[str, Any]) -> str:
    citations = computation.get("citations")
    refs = [str(ref) for ref in citations] if isinstance(citations, list) else []
    result = str(computation.get("result") or "").strip()
    lines = [
        "<lcm-preanswer-evidence>",
        "Validated deterministic result from exact stored evidence:",
        f"- result: {result}",
    ]
    if refs:
        lines.append(f"- exact refs: {', '.join(refs)}")
    lines.extend(
        [
            "Use this canonical result unchanged in the answer. Do not alter its value or unit.",
            "</lcm-preanswer-evidence>",
        ]
    )
    return "\n".join(lines)


def _render_evidence(evidence: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "<lcm-preanswer-evidence>",
        "Novel exact evidence from stored conversation history:",
    ]
    for item in evidence:
        exact_ref = str(item.get("exact_ref") or "")
        quote = " ".join(str(item.get("quote") or "").split())
        lines.append(f"- [{exact_ref}] {quote}")
    lines.extend(
        [
            "This is bounded evidence, not a claim that an open-world list is complete.",
            "</lcm-preanswer-evidence>",
        ]
    )
    return "\n".join(lines)


def _finalize_success(
    result: dict[str, Any],
    *,
    status: str,
    reason_code: str,
    context: str,
    evidence: Sequence[Mapping[str, Any]],
    computation: Mapping[str, Any] | None,
    novel_refs: Sequence[str],
    started: float,
) -> dict[str, Any]:
    if len(context) > int(result["budgets"]["max_context_chars"]):
        result["reason_code"] = "context_budget_exhausted"
        result["trace"]["truncated"] = True
        result["metrics"]["latency_ms"] = round(
            (time.perf_counter() - started) * 1_000.0, 3
        )
        return result
    result.update(
        {
            "status": status,
            "reason_code": reason_code,
            "context": context,
            "novel_exact_refs": list(novel_refs),
            "evidence": list(evidence),
            "computation": dict(computation) if computation is not None else None,
            "computation_sha256": (
                _sha256_json(computation) if computation is not None else None
            ),
        }
    )
    result["trace"]["context_sha256"] = hashlib.sha256(
        context.encode("utf-8")
    ).hexdigest()
    result["metrics"].update(
        {
            "latency_ms": round((time.perf_counter() - started) * 1_000.0, 3),
            "context_chars": len(context),
        }
    )
    return result


def build_preanswer_evidence(
    question: Any,
    *,
    engine: Any,
    baseline_refs: Sequence[Any] = (),
    question_date: Any = None,
    retrieve: Callable[[dict[str, Any]], Any] | None = None,
    enabled: bool = False,
    context_engine_enabled: bool = True,
    budgets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one bounded ephemeral context addition, or no augmentation.

    Runtime inputs are restricted to the question, its explicit date anchor,
    bounded exact refs, and product retrieval.  The function is fail-open for
    the host: it converts every error into ``context=None``.
    """

    started = time.perf_counter()
    limits = _budgets(budgets)
    if not enabled:
        return _base_result(
            status="no_augmentation",
            reason_code="feature_disabled",
            budgets=limits,
            started=started,
        )
    if not context_engine_enabled:
        return _base_result(
            status="no_augmentation",
            reason_code="context_engine_toolset_disabled",
            budgets=limits,
            started=started,
        )
    text = str(question or "").strip()
    if not text:
        return _base_result(
            status="no_augmentation",
            reason_code="question_required",
            budgets=limits,
            started=started,
        )
    normalized_date, date_error = normalize_question_date(question_date)
    if date_error:
        return _base_result(
            status="no_augmentation",
            reason_code=date_error,
            budgets=limits,
            started=started,
        )
    canonical_date = normalized_date.day.isoformat() if normalized_date else None
    plan_decision = compile_evidence_plan(text, canonical_date)
    if plan_decision.status != "planned" or plan_decision.plan is None:
        return _base_result(
            status="no_augmentation",
            reason_code=(
                "unsupported_or_ambiguous_operation"
                if plan_decision.status == "fallback"
                else "no_supported_operation"
            ),
            budgets=limits,
            started=started,
            decision={"status": plan_decision.status, "reason": plan_decision.reason},
        )
    plan = plan_decision.plan

    normalized_refs, exact_refs = _normalize_refs(
        baseline_refs, limit=limits.max_input_refs
    )
    prepared = _prepare_candidates(normalized_refs, engine=engine, plan=plan)
    decision: dict[str, Any] = {
        "status": "planned",
        "operation": plan.operation,
        "missing_requirement": None,
    }
    result = _base_result(
        status="no_augmentation",
        reason_code="baseline_not_actionable",
        budgets=limits,
        started=started,
        decision=decision,
        baseline_refs=exact_refs,
    )

    # Open enumeration cannot be repaired by one search.  Latest-state is the
    # one exception: a named update check can add fresher evidence without
    # claiming that an open list is complete.
    if (
        plan.requires_complete_evidence
        and plan.exact_operands is None
        and plan.operation != "latest_fact"
    ):
        result["reason_code"] = "open_cardinality"
        return result

    baseline_pack = (
        _pack(
            question=text,
            question_date=canonical_date,
            candidates=prepared,
            engine=engine,
            budgets=limits,
        )
        if prepared
        else None
    )
    baseline_computation = (
        baseline_pack.get("computation") if isinstance(baseline_pack, Mapping) else None
    )
    if isinstance(baseline_computation, Mapping):
        context = _render_computation(baseline_computation)
        return _finalize_success(
            result,
            status="computed",
            reason_code="validated_baseline_computation",
            context=context,
            evidence=baseline_pack.get("evidence", []),
            computation=baseline_computation,
            novel_refs=(),
            started=started,
        )

    missing = _missing_requirement(text, plan, prepared)
    result["decision"]["missing_requirement"] = missing
    if missing is None:
        result["reason_code"] = "baseline_validation_failed"
        return result
    if limits.max_retrieval_calls == 0:
        result["reason_code"] = "retrieval_budget_exhausted"
        result["retrieval"]["status"] = "budget_exhausted"
        return result
    if retrieve is None:
        result["reason_code"] = "retrieval_unavailable"
        result["retrieval"]["status"] = "unavailable"
        return result

    retrieval_args = {
        "query": missing["query"],
        "include": "verbatim",
        "detail": "answer_ready",
        "limit": limits.max_novel_refs,
        "scope_bias": 0.0,
        "seen_refs": exact_refs,
        "include_occurrence_time": True,
    }
    retrieval_started = time.perf_counter()
    try:
        raw = retrieve(retrieval_args)
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, Mapping):
            raise ValueError("retrieval result is not an object")
    except Exception as exc:  # noqa: BLE001 - host path must fail open
        result["reason_code"] = "retrieval_error"
        result["retrieval"] = {
            "calls": 1,
            "status": "error",
            "novel_exact_refs": [],
            "usage": {},
            "reason": str(exc)[:300],
            "latency_ms": round((time.perf_counter() - retrieval_started) * 1_000.0, 3),
        }
        return result
    if payload.get("timeout"):
        result["reason_code"] = "retrieval_timeout"
        result["retrieval"] = {
            "calls": 1,
            "status": "timeout",
            "novel_exact_refs": [],
            "usage": {},
            "latency_ms": round((time.perf_counter() - retrieval_started) * 1_000.0, 3),
        }
        return result
    if payload.get("error"):
        result["reason_code"] = "retrieval_error"
        result["retrieval"] = {
            "calls": 1,
            "status": "error",
            "novel_exact_refs": [],
            "usage": {},
            "reason": str(payload.get("error"))[:300],
            "latency_ms": round((time.perf_counter() - retrieval_started) * 1_000.0, 3),
        }
        return result

    hits = payload.get("hits")
    hit_list = hits if isinstance(hits, list) else []
    if not hit_list:
        result["reason_code"] = "no_hit"
        result["retrieval"] = {
            "calls": 1,
            "status": "no_hit",
            "novel_exact_refs": [],
            "usage": {},
            "latency_ms": round((time.perf_counter() - retrieval_started) * 1_000.0, 3),
        }
        return result
    novel_candidates: list[dict[str, Any]] = []
    novel_refs: list[str] = []
    seen = set(exact_refs)
    for hit in hit_list:
        if not isinstance(hit, Mapping):
            continue
        exact_ref = str(hit.get("exact_ref") or "").strip()
        if (
            not _EXACT_REF_RE.fullmatch(exact_ref)
            or exact_ref in seen
            or exact_ref in novel_refs
        ):
            continue
        novel_refs.append(exact_ref)
        novel_candidates.append(dict(hit))
        if len(novel_refs) >= limits.max_novel_refs:
            break
    metrics = payload.get("metrics")
    metric_map = metrics if isinstance(metrics, Mapping) else {}
    usage = {
        key: metric_map[key]
        for key in (
            "embedding_query_calls",
            "embedding_query_tokens",
            "embedding_query_tokens_complete",
            "embedding_queries",
        )
        if key in metric_map
    }
    result["retrieval"] = {
        "calls": 1,
        "status": "novel" if novel_refs else "no_novel",
        "query": missing["query"],
        "novel_exact_refs": novel_refs,
        "usage": usage,
        "latency_ms": round((time.perf_counter() - retrieval_started) * 1_000.0, 3),
    }
    if not novel_refs:
        result["reason_code"] = "no_novel_exact_ref"
        return result

    prepared_novel = _prepare_candidates(novel_candidates, engine=engine, plan=plan)
    if not prepared_novel:
        result["reason_code"] = "novel_exact_ref_not_hydratable"
        return result
    combined = [*prepared, *prepared_novel]
    final_pack = _pack(
        question=text,
        question_date=canonical_date,
        candidates=combined,
        engine=engine,
        budgets=limits,
    )
    computation = final_pack.get("computation")
    if isinstance(computation, Mapping):
        context = _render_computation(computation)
        return _finalize_success(
            result,
            status="computed",
            reason_code="validated_delta_computation",
            context=context,
            evidence=final_pack.get("evidence", []),
            computation=computation,
            novel_refs=novel_refs,
            started=started,
        )

    if plan.operation == "latest_fact":
        selection = final_pack.get("selection")
        selected_refs = (
            selection.get("exact_refs") if isinstance(selection, Mapping) else None
        )
        if isinstance(selected_refs, list) and len(selected_refs) == 1:
            selected_ref = str(selected_refs[0])
            if selected_ref in novel_refs:
                evidence = [
                    item
                    for item in final_pack.get("evidence", [])
                    if isinstance(item, Mapping)
                    and item.get("exact_ref") == selected_ref
                ]
                if evidence:
                    context = _render_evidence(evidence)
                    return _finalize_success(
                        result,
                        status="augmented",
                        reason_code="novel_latest_state_evidence",
                        context=context,
                        evidence=evidence,
                        computation=None,
                        novel_refs=[selected_ref],
                        started=started,
                    )

    result["reason_code"] = "delta_validation_failed"
    return result
