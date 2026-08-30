"""Third automated-review round.

Each test names the defect it pins rather than the function it calls, because
several of these passed a green suite before the fix: the narrowing bug was
invisible while nothing consulted operation_allowlist, and the recall arm's
hard-coded corpus was invisible while no policy narrowed it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_lcm import tools as tools_module
from hermes_lcm.access_context import Decision, DenialReason
from hermes_lcm.access_context.model import AccessContextV1
from hermes_lcm.access_context.validation import validate
from hermes_lcm.access_policy import FailClosedPolicy
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _context(**overrides) -> AccessContextV1:
    fields = dict(
        authenticated_transport="host-session",
        context_id="ctx-human",
        request_id="req-1",
        source_kind="human",
        deployment_id="dep-1",
        tenant_id="tenant-1",
        principal_id="principal-a",
        profile_id="profile-a",
        profile_incarnation="incarnation-1",
        session_id="session-a",
        session_owner_principal_id="principal-a",
        conversation_id="conversation-a",
        conversation_lane="lane-a",
        read_policy_ref="policy-a",
        lease_id="lease-a",
        issued_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        expires_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    fields.update(overrides)
    return AccessContextV1.from_host(**fields)


def test_repeated_narrowing_drops_the_superseded_operation() -> None:
    """A second narrowing must not leave the first one's wider token behind."""
    parent = _context(grants=frozenset({"read", "write"}))
    read_write = parent.narrow(operations={"read", "write"})
    assert read_write.operation_allowlist == frozenset({"read", "write"})

    read_only = read_write.narrow(operations={"read"})
    # The bug: child_narrowing started as a copy of the parent's tokens, so
    # operation:write survived and this nominally read-only context still
    # passed a write check.
    assert read_only.operation_allowlist == frozenset({"read"})
    assert "operation:write" not in read_only.narrowing


def test_repeated_narrowing_drops_the_superseded_collection() -> None:
    parent = _context(
        narrowing=frozenset({"collection:A", "collection:B"}),
        default_write_collection_id="A",
    )
    child = parent.narrow(collections={"A"})
    assert child.collection_allowlist == frozenset({"A"})
    assert "collection:B" not in child.narrowing


def test_a_single_revoked_context_id_is_not_shredded_into_characters() -> None:
    """A bare str satisfies Iterable[str]; frozenset() would split it."""
    context = _context()
    as_string = validate(context, now=NOW, revoked_context_ids="ctx-human")
    as_set = validate(context, now=NOW, revoked_context_ids={"ctx-human"})

    assert as_string.allowed is False
    assert as_string.denial_reason is DenialReason.CONTEXT_REVOKED
    # The two spellings must be indistinguishable. Before the fix the bare
    # string matched nothing and the revoked context was ALLOWED.
    assert as_string.allowed == as_set.allowed
    assert as_string.denial_reason == as_set.denial_reason


def test_a_context_used_before_issuance_is_rejected() -> None:
    context = _context()
    before = validate(context, now=datetime(2024, 1, 1, tzinfo=timezone.utc))
    during = validate(context, now=NOW)

    assert during.allowed is True
    assert before.allowed is False
    assert before.denial_reason is DenialReason.CONTEXT_EXPIRED


def test_fail_closed_resolver_returns_a_scope_mapping_not_a_sequence() -> None:
    """Callers read this with .get; the protocol declares a TargetScope."""
    resolved = FailClosedPolicy().resolve_authorized_targets(None, "read", {"limit": 1})
    assert hasattr(resolved, "get")
    assert not resolved


class _NarrowingPolicy:
    """Authorizes, then resolves a corpus narrower than the one requested."""

    def __init__(self, resolved: dict) -> None:
        self.resolved = resolved

    def authorize_operation(self, _context, _operation, _expected_scope) -> Decision:
        return Decision.allow()

    def resolve_authorized_targets(self, _context, _operation, _narrowing) -> dict:
        return self.resolved

    def audit_decision(self, *_args, **_kwargs) -> None:
        return None


@pytest.mark.parametrize(
    "resolved, expected_scope_value",
    [
        # The policy authorizes nothing in the corpus dimension: the hard-coded
        # "all" must be REMOVED, degrading to the tool's narrowest default.
        ({}, "current"),
        # The policy names one session but no scope: naming the implied scope
        # beats erroring with "session_id is only valid with scope=session".
        ({"session_id": "session-b"}, "session"),
    ],
)
def test_recall_arm_never_keeps_the_hardcoded_all_session_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resolved: dict,
    expected_scope_value: str,
) -> None:
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))
    captured: dict = {}

    def capture(fts_args, **_kwargs):
        captured.update(fts_args)
        return {"results": []}

    monkeypatch.setattr(
        tools_module, "policy_for_engine", lambda _e: _NarrowingPolicy(resolved)
    )
    monkeypatch.setattr(tools_module, "policy_access_context", lambda _e: None)
    monkeypatch.setattr(
        tools_module, "_lcm_grep_full_text_with_deadline", capture
    )
    tools_module._lcm_recall_fts_arm(
        engine, "anything", candidate_limit=5, deadline=float("inf")
    )

    assert captured.get("session_scope", "current") == expected_scope_value
    assert captured.get("session_scope") != "all"


class _DenyStoreReads:
    def authorize_operation(self, _context, _operation, expected_scope) -> Decision:
        if expected_scope.get("kind") == "preanswer_baseline_ref":
            return Decision.deny(DenialReason.SCOPE_FORBIDDEN)
        return Decision.allow()

    def audit_decision(self, *_args, **_kwargs) -> None:
        return None


def _baseline_refs(monkeypatch: pytest.MonkeyPatch, policy, refs):
    # The gate lives in preanswer_evidence, not in __init__.py: tests/conftest.py
    # registers the package WITHOUT executing __init__.py, so anything defined
    # at that level is unreachable from a test.
    from hermes_lcm import preanswer_evidence

    monkeypatch.setattr(preanswer_evidence, "policy_for_engine", lambda _e: policy)
    monkeypatch.setattr(preanswer_evidence, "policy_access_context", lambda _e: None)
    return preanswer_evidence.authorize_supplied_baseline_refs(object(), refs)


def test_supplied_baseline_refs_are_authorized_in_both_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers pass bare strings OR mappings carrying exact_ref plus a quote."""
    mapping_ref = {"exact_ref": "lcm:12:0-4", "quote": "text"}
    string_ref = "lcm:99:0-4"

    allowed = _baseline_refs(monkeypatch, _NarrowingPolicy({}), [mapping_ref, string_ref])
    # The ORIGINAL items survive -- downstream reads the quote alongside the ref.
    assert list(allowed) == [mapping_ref, string_ref]

    denied = _baseline_refs(monkeypatch, _DenyStoreReads(), [mapping_ref, string_ref])
    assert denied == ()


def test_supplied_refs_that_are_not_store_spans_pass_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only exact store spans are a disclosure this gate governs.

    Dropping everything else silently emptied the payload for callers whose
    refs were never store references, which changed unrelated downstream
    branching rather than closing a hole.
    """
    opaque = [{"quote": "no exact_ref here"}, "not-a-reference"]
    assert list(_baseline_refs(monkeypatch, _DenyStoreReads(), opaque)) == opaque


def test_compression_rollover_presents_the_source_session_to_the_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The source session is caller-supplied and may belong to someone else."""
    from hermes_lcm import engine as engine_module

    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))
    seen: list[dict] = []

    class _Recorder:
        def authorize_operation(self, _context, _operation, expected_scope):
            seen.append(dict(expected_scope))
            return Decision.deny(DenialReason.SCOPE_FORBIDDEN)

        def audit_decision(self, *_args, **_kwargs) -> None:
            return None

    monkeypatch.setattr(engine_module, "policy_for_engine", lambda _e: _Recorder())
    monkeypatch.setattr(engine_module, "policy_access_context", lambda _e: None)

    with pytest.raises(engine_module.AuthorizationRequiredError):
        engine.on_session_start(
            "session-destination",
            boundary_reason="compression",
            old_session_id="session-of-another-principal",
        )

    assert seen, "the policy was never consulted"
    # Before the fix the scope named only the destination, so a policy could
    # not refuse a rollover that reads and reassigns another principal's DAG.
    assert seen[0].get("source_session_id") == "session-of-another-principal"
