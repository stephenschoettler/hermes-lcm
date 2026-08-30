from __future__ import annotations

from pathlib import Path

import pytest

from hermes_lcm.access_context.fixtures import FIXTURE_KINDS, fixture_paths, load_fixture

def test_fixture_corpus_is_non_empty_in_every_kind() -> None:
    minimums = {"positive": 5, "negative": 16, "delegation": 10, "revocation": 4, "derivation": 8}
    for kind in FIXTURE_KINDS:
        paths = fixture_paths(kind)
        assert len(paths) >= minimums[kind], kind


@pytest.mark.parametrize("path", fixture_paths())
def test_each_discovered_fixture_has_an_envelope(path: Path) -> None:
    payload = load_fixture(path)
    assert payload["contract_revision"]
    assert payload["description"].count("\n") == 0
    assert "context" in payload
    if path.parent.name in {"negative", "delegation", "revocation", "derivation"}:
        assert "expected" in payload


def test_fixture_kinds_are_closed_and_shared_root_is_present() -> None:
    assert FIXTURE_KINDS == {"positive", "negative", "delegation", "revocation", "derivation"}
    assert fixture_paths()[0].parents[1].name == "access_context_v1"


def test_derived_scope_vectors_reject_widening_for_each_derived_kind() -> None:
    paths = fixture_paths("derivation")
    assert len(paths) >= 8
    seen: dict[str, set[bool]] = {}
    from access_context import AccessContextV1, is_subset_of

    for path in paths:
        payload = load_fixture(path)
        source = AccessContextV1.from_payload(payload["context"])
        derived = AccessContextV1.from_payload(payload["derived"])
        kind = payload["derived_kind"]
        expected_subset = bool(payload["expected"]["subset"])
        assert is_subset_of(derived, source) is expected_subset, path
        seen.setdefault(kind, set()).add(expected_subset)

    assert set(seen) == {"summary", "chunk", "vector_embedding", "rollup"}
    assert all(outcomes == {True, False} for outcomes in seen.values())
