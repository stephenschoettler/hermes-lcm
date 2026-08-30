"""The corpus queries carry an OWNER predicate.

The first TeamsPolicy narrowed the corpus by setting `source` to the context's
`default_write_collection_id`. On the synthetic smoke that worked, because the
fixture seeds rows with the same constant it puts in the context. On real
customer data the two are unrelated -- stored `source` values look like
``openclaw-lcm:agent:acorn:<uuid>`` -- so the filter matched nothing and recall
returned an EMPTY corpus to every principal.

That is isolation by breaking retrieval, and only a positive control catches it:
a leak probe is perfectly happy with zero results.

The fix is to narrow by ``access_scope`` -- the property rows actually carry,
and the same value the write path stamps, so reads and writes agree by
construction.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from hermes_lcm.access_context.model import AccessContextV1
from hermes_lcm.access_policy import TeamsPolicy


def _context(principal: str = "acorn") -> AccessContextV1:
    now = datetime.now(timezone.utc)
    return AccessContextV1.from_host(
        authenticated_transport="test",
        context_id="ctx",
        request_id="req",
        source_kind="human",
        deployment_id="dep",
        tenant_id="tenant",
        principal_id=principal,
        profile_id=principal,
        profile_incarnation="inc",
        session_id="session-a",
        session_owner_principal_id=principal,
        conversation_id="conv",
        conversation_lane="lane",
        default_write_collection_id="collection-acorn",
        read_policy_ref="policy",
        lease_id="lease",
        issued_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(hours=1),
    )


def test_the_corpus_is_narrowed_by_the_owner_stamp() -> None:
    resolved = TeamsPolicy(_context()).resolve_authorized_targets(
        None, "read", {"session_scope": "all", "source": None}
    )
    assert resolved["access_scope"] == "acorn"


def test_narrowing_does_not_ride_on_the_collection_id() -> None:
    """The defect this replaces, pinned so a regression is visible.

    `source` is a property of the STORED ROW; the context's collection id is a
    Teams concept. Overwriting one with the other filters real corpora down to
    nothing, which reads as perfect isolation and is total breakage.
    """
    resolved = TeamsPolicy(_context()).resolve_authorized_targets(
        None, "read", {"session_scope": "all", "source": "openclaw-lcm:agent:acorn:uuid"}
    )

    assert resolved.get("source") != "collection-acorn"
    assert resolved["access_scope"] == "acorn"


def test_the_predicate_survives_the_omission_rule() -> None:
    """The recall arm REMOVES keys the policy omits; the owner key is added.

    Those two rules run over the same mapping, so the owner predicate has to be
    applied after the removal loop or it would be stripped as "not authorized".
    """
    from hermes_lcm import tools as tools_module
    import inspect

    source = inspect.getsource(tools_module._lcm_recall_fts_arm)
    removal = source.index('fts_args.pop(key, None)')
    addition = source.index('fts_args["access_scope"]')
    assert addition > removal, (
        "the owner predicate must be applied AFTER the omission-removal loop, "
        "or the loop strips it"
    )


def test_a_context_without_a_principal_narrows_nothing() -> None:
    """No principal, no claim -- do not invent a predicate that matches nothing."""
    resolved = TeamsPolicy(None).resolve_authorized_targets(
        None, "read", {"session_scope": "all"}
    )
    assert "access_scope" not in resolved


def test_store_search_accepts_the_predicate_and_defaults_to_none() -> None:
    """Default-off callers must produce a byte-identical query."""
    import inspect

    from hermes_lcm.store import MessageStore

    signature = inspect.signature(MessageStore.search)
    assert "access_scope" in signature.parameters
    assert signature.parameters["access_scope"].default is None


def test_the_like_fallback_carries_the_predicate_too() -> None:
    """An FTS error must not silently WIDEN the corpus back to everyone."""
    import inspect

    from hermes_lcm.store import MessageStore

    signature = inspect.signature(MessageStore._search_like)
    assert "access_scope" in signature.parameters

    source = inspect.getsource(MessageStore.search)
    fallback = source.index("_search_like(")
    assert "access_scope=access_scope" in source[fallback:], (
        "the LIKE fallback must receive the owner predicate"
    )


def test_the_vector_paths_enforce_before_the_bound() -> None:
    """A filter applied after LIMIT leaks exactly when the bound bites."""
    import inspect

    from hermes_lcm.vector_store import VectorStore

    for helper in (
        VectorStore._bounded_candidate_ids,
        VectorStore._bounded_chunk_candidate_ids,
    ):
        source = inspect.getsource(helper)
        predicate = source.index("access_scope = ?")
        limit_arg = source.index("args.append(int(limit))")
        assert predicate < limit_arg, (
            f"{helper.__name__}: owner predicate must be in the WHERE clause, "
            f"applied before the LIMIT bound"
        )
