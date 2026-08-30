"""Owner resolution must not depend on the SHAPE a target arrives in.

An adversarial audit reproduced this against real customer data: the identical
row, the identical acting principal, and two different target shapes gave
opposite decisions.

    owner resolution, scalar store_id (lcm_expand shape) : ('carus',)
    owner resolution, baseline_refs LIST (evidence shape): ()

`_stored_access_scopes_for_targets` skipped any dict/list/tuple/set value, and
`baseline_refs` -- the declared target of `lcm_evidence_pack` and
`lcm_compile_evidence` -- is ALWAYS a list. So no `target_access_scopes` was
attached, TeamsPolicy had nothing to decide on, the gate allowed, and the pack
came back carrying another principal's message content verbatim.

The gate was present and correct at the declared site. It simply could not see
who owned what it was being asked about, which is the same failure as #218 in a
different disguise: the policy is handed a scope it has no rule for and reads
that as "yes".
"""

from __future__ import annotations

import pytest

from hermes_lcm.engine import LCMEngine


@pytest.mark.parametrize(
    "value, expected",
    [
        ("lcm:76:0-200", 76),
        ({"exact_ref": "lcm:76:0-200"}, 76),
        # Both shapes reach the authorization boundary: callers pass either the
        # bare string or a mapping carrying it under `exact_ref`.
        ({"exact_ref": "lcm:1:0-10", "note": "ignored"}, 1),
        ("lcm:bogus", None),
        ("", None),
        (None, None),
        ({"exact_ref": ""}, None),
        ({}, None),
        # A leading zero is not a valid store id in the canonical regex.
        ("lcm:0:0-1", None),
    ],
)
def test_the_store_id_inside_a_reference_is_recovered(value, expected) -> None:
    assert LCMEngine._store_id_from_exact_ref(value) == expected


def test_list_valued_targets_are_declared_for_owner_resolution() -> None:
    """The gap was structural: no lookup existed for a list-shaped target."""
    keys = {key for key, _table, _column in LCMEngine._TARGET_REF_LOOKUPS}
    assert "baseline_refs" in keys, (
        "baseline_refs must resolve owners; it is the declared target of "
        "lcm_evidence_pack and lcm_compile_evidence and is always a list"
    )


def test_every_declared_tool_target_can_be_resolved() -> None:
    """Structural, and the check that generalizes this defect.

    Any target a tool binding declares must be resolvable by one of the two
    lookup tables, or the policy is handed a name it cannot attach an owner to
    and allows. A new binding whose target is neither a known scalar id nor a
    known reference list fails here rather than shipping permissive.
    """
    from hermes_lcm.engine import LCM_TOOL_TARGET_BINDINGS

    resolvable = {key for key, _t, _c in LCMEngine._TARGET_OWNER_LOOKUPS}
    resolvable |= {key for key, _t, _c in LCMEngine._TARGET_REF_LOOKUPS}
    resolvable |= {key for key, _t, _c in LCMEngine._TARGET_ID_LIST_LOOKUPS}

    # Targets that name a scope or carry caller content rather than addressing a
    # row. These cannot bear an owner stamp and must be decided by a different
    # rule; each is listed with that rule, so adding a new one is a deliberate
    # act rather than an omission that silently allows.
    known_unresolvable = {
        "session_id",          # decided by the session-owner comparison
        "source_session_id",   # same, via the compression rollover path
        "partition_key",       # same, via rollup scheduling
        "conversation_id",     # travels with the session it belongs to
        "session_scope",       # a selector ("current"/"session"/"all"), not a row
        "source",              # a corpus selector, narrowed by resolve_authorized_targets
        # lcm_query_state addresses assertions by subject, and the assertion
        # tables carry no access_scope column at all -- so there is nothing to
        # resolve here yet. Tracked separately; this entry records that the
        # allow is currently UNJUSTIFIED rather than reasoned.
        "subject_key",
        "scope_key",
        # Caller-supplied payloads, not references to stored rows.
        "proposal",
        "operands",
        "identity",
        "tool_args",
        "computation",
    }

    declared: set[str] = set()
    for binding in LCM_TOOL_TARGET_BINDINGS.values():
        declared.update(binding.get("args", ()))

    unclassified = declared - resolvable - known_unresolvable
    assert not unclassified, (
        f"tool bindings declare targets nothing can resolve an owner for: "
        f"{sorted(unclassified)}. Add a lookup, or list it in "
        f"known_unresolvable with the rule that decides it."
    )
