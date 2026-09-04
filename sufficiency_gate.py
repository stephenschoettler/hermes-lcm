"""Provider-free sufficiency policy at the preanswer finalize seam.

The gate is the single place where a finished preanswer-evidence result is
classified into the sufficiency vocabulary and mapped to one answer-contract
policy action:

- ``answer_sufficient`` / ``computation_sufficient`` / ``finite_coverage``
  → ``answer`` (the augmentation speaks for itself).
- ``partial`` → ``answer_with_disclosure``: a bounded disclosure of what is
  missing, rendered exclusively from fields already stored on the result.
- ``unknown`` / ``conflicted`` → ``annotate``: disclose rather than stay
  silent; unmarked absence is never presented as fact.

The gate never calls a provider, never mutates evidence or computation, and
never fabricates content.  For states where the gate itself makes no claim
(feature disabled, unsupported question, invalid host input) it adds nothing:
absence of a sufficiency section is itself the honest state.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping

SUFFICIENCY_GATE_VERSION = "sufficiency-gate-v1"

_ANSWER_STATES = frozenset(
    {"answer_sufficient", "computation_sufficient", "finite_coverage"}
)
_ANNOTATE_STATES = frozenset({"unknown", "conflicted"})
_KNOWN_STATES = _ANSWER_STATES | _ANNOTATE_STATES | {"partial"}

# Outcomes where the gate itself makes no claim at all: the ordinary answer
# path ran untouched, and adding a sufficiency section would fabricate a
# judgment the pipeline never made.
_NO_CLAIM_REASON_CODES = frozenset(
    {
        # gate/feature boundaries
        "feature_disabled",
        "context_engine_toolset_disabled",
        "question_required",
        "question_date_invalid",
        "unsupported_or_ambiguous_operation",
        "no_supported_operation",
        # answer-contract planning refusals (no evidence work was performed)
        "not_applicable",
        "question_too_long",
        "unsupported_operation",
        "question_as_of_invalid",
        "question_as_of_required",
        "insufficient_anchors",
        "generic_question_low_confidence",
        "ordinary_or_low_confidence",
        # hook-route refusals (no evidence work was performed)
        "ordinary_baseline",
        "disabled",
        "selector_unavailable",
    }
)

# Evidence work happened but did not close: the answer may proceed with an
# explicit disclosure of what is missing and why.
_PARTIAL_REASON_CODES = frozenset(
    {
        "baseline_not_actionable",
        "baseline_validation_failed",
        "open_cardinality",
        "retrieval_budget_exhausted",
        "retrieval_unavailable",
        "retrieval_error",
        "retrieval_timeout",
        "novel_exact_ref_not_hydratable",
        "delta_validation_failed",
        "context_budget_exhausted",
        "novel_ref_budget_exhausted",
    }
)

# Nothing usable was found at all: annotate instead of answering or refusing.
_UNKNOWN_REASON_CODES = frozenset(
    {
        "no_hit",
        "no_novel_exact_ref",
    }
)

# Two grounded candidates disagree: not unknown, but an unresolved conflict.
_CONFLICTED_REASON_CODES = frozenset({"state_conflicted"})


def _policy_for_state(state: str) -> tuple[str, str]:
    if state in _ANSWER_STATES:
        return state, "answer"
    if state == "partial":
        return state, "answer_with_disclosure"
    return state, "annotate"


def _refine_unknown_state(state: str, result: Mapping[str, Any]) -> str:
    """Split the compiler's blanket ``unknown`` into its honest sub-states.

    The requirements compiler only delivers ``answer_sufficient`` /
    ``computation_sufficient`` and otherwise leaves the base ``unknown``
    state.  The reason code says which kind of non-sufficiency happened;
    the gate surfaces that instead of flattening every miss to unknown.
    """
    reason_code = str(result.get("reason_code") or "")
    if reason_code in _CONFLICTED_REASON_CODES or reason_code.startswith(
        "ambiguous_"
    ):
        return "conflicted"
    evidence = result.get("evidence")
    if isinstance(evidence, list) and evidence:
        # Grounded evidence exists but the contract did not close: partial,
        # not unknown.  Absence of a delivered context is not absence of work.
        return "partial"
    return state


def _classify_legacy_result(result: Mapping[str, Any]) -> tuple[str, str] | None:
    """Map legacy (status, reason_code) pairs onto the state vocabulary.

    Unknown future reason codes make no claim: the gate stays silent rather
    than guessing a state the pipeline did not define.
    """
    status = str(result.get("status") or "")
    reason_code = str(result.get("reason_code") or "")
    if reason_code in _NO_CLAIM_REASON_CODES:
        return None
    if reason_code == "context_budget_exhausted":
        return ("partial", "answer_with_disclosure")
    state = result.get("state")
    if isinstance(state, str) and state in _KNOWN_STATES:
        # Only the compiler's blanket ``unknown`` needs refinement; delivered
        # states are the pipeline's own verdict and are never overridden.
        refined = _refine_unknown_state(state, result) if state == "unknown" else state
        return _policy_for_state(refined)
    if status == "computed":
        return ("computation_sufficient", "answer")
    if status == "augmented":
        return ("answer_sufficient", "answer")
    if status == "no_augmentation":
        if reason_code in _PARTIAL_REASON_CODES:
            return ("partial", "answer_with_disclosure")
        if reason_code in _UNKNOWN_REASON_CODES:
            return ("unknown", "annotate")
    return None


def _missing_line(missing: Any) -> str | None:
    if not isinstance(missing, Mapping):
        return None
    kind = str(missing.get("kind") or "").strip()
    query = str(missing.get("query") or "").strip()
    current = missing.get("current_operands")
    minimum = missing.get("minimum_operands")
    parts = [f"missing: {kind}"]
    if query:
        parts.append(f'query "{query}"')
    if current is not None and minimum is not None:
        parts.append(f"operands {current}/{minimum}")
    return " ".join(part for part in parts if part)


def render_disclosure(
    result: dict[str, Any], state: str, policy_action: str
) -> str:
    """Render the bounded disclosure from fields already stored on the result.

    Every line value is copied from the result; the template is the only
    authored text.  The block is an annotation about evidence state, never
    evidence itself.
    """
    decision = result.get("decision")
    decision_map = decision if isinstance(decision, dict) else {}
    retrieval = result.get("retrieval")
    retrieval_map = retrieval if isinstance(retrieval, dict) else {}
    baseline = result.get("baseline")
    baseline_map = baseline if isinstance(baseline, dict) else {}
    trace = result.get("trace")
    trace_map = trace if isinstance(trace, dict) else {}

    lines = [
        "<lcm-sufficiency-disclosure>",
        f"gate: {SUFFICIENCY_GATE_VERSION}",
        f"state: {state}",
        f"policy_action: {policy_action}",
    ]
    reason_code = str(result.get("reason_code") or "")
    if reason_code:
        lines.append(f"reason_code: {reason_code}")
    operation = str(decision_map.get("operation") or "").strip()
    if operation:
        lines.append(f"operation: {operation}")
    missing_line = _missing_line(decision_map.get("missing_requirement"))
    if missing_line:
        lines.append(missing_line)
    retrieval_status = str(retrieval_map.get("status") or "").strip()
    retrieval_calls = retrieval_map.get("calls")
    if retrieval_status and retrieval_status != "not_called":
        lines.append(f"retrieval: {retrieval_status} (calls: {retrieval_calls})")
    if trace_map.get("truncated"):
        lines.append("context_truncated: true")
    if baseline_map:
        lines.append(f"baseline exact refs: {baseline_map.get('exact_ref_count', 0)}")
    lines.extend(
        [
            "stored evidence did not establish sufficiency; disclose rather than imply sufficiency",
            "this block is an annotation, not evidence",
            "</lcm-sufficiency-disclosure>",
        ]
    )
    return "\n".join(lines)


def classify_preanswer_result(
    result: dict[str, Any],
) -> tuple[str, str] | None:
    """Return ``(state, policy_action)`` for a finished result, or ``None``.

    ``None`` means the gate makes no claim: the result stays unmarked and the
    ordinary answer path is untouched.
    """
    return _classify_legacy_result(result)


def apply_sufficiency_gate(
    result: dict[str, Any],
    *,
    enabled: bool = False,
    started: float | None = None,
) -> dict[str, Any]:
    """Attach the sufficiency verdict to a finished preanswer result.

    With ``enabled=False`` the result is returned untouched.  With the gate
    enabled, non-sufficient states with an empty context get their disclosure
    rendered as the ephemeral context block; the trace digest and context
    metrics are kept consistent with that block.  Existing evidence,
    computation, and status fields are never modified.
    """
    if not enabled:
        return result
    gate_started = started if started is not None else time.perf_counter()
    verdict = classify_preanswer_result(result)
    if verdict is None:
        return result
    state, policy_action = verdict
    disclosure: str | None = None
    if policy_action != "answer":
        disclosure = render_disclosure(result, state, policy_action)
    result["sufficiency"] = {
        "version": SUFFICIENCY_GATE_VERSION,
        "state": state,
        "policy_action": policy_action,
        "disclosure_context": disclosure,
    }
    if disclosure is not None and not isinstance(result.get("context"), str):
        result["context"] = disclosure
        result["trace"]["context_sha256"] = hashlib.sha256(
            disclosure.encode("utf-8")
        ).hexdigest()
        result["metrics"]["context_chars"] = len(disclosure)
    result["metrics"]["sufficiency_gate_latency_ms"] = round(
        (time.perf_counter() - gate_started) * 1_000.0, 3
    )
    return result


def sufficiency_digest(result: dict[str, Any]) -> str | None:
    """Stable digest of the sufficiency section, or ``None`` when unmarked."""
    section = result.get("sufficiency")
    if not isinstance(section, Mapping):
        return None
    return hashlib.sha256(
        json.dumps(section, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()