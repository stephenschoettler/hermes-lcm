"""Lone UTF-16 surrogates must not abort LCM compaction hashing."""

from __future__ import annotations

import hashlib

from hermes_lcm.message_content import normalize_content_value, repair_unicode, utf8_bytes
from hermes_lcm.placeholder_ledger import PlaceholderLedgerMixin
from hermes_lcm.reconcile import ReconcileMixin


class _Fingerprint(PlaceholderLedgerMixin, ReconcileMixin):
    pass


def test_repair_keeps_well_formed_text_and_replaces_lone_surrogate():
    assert repair_unicode("plain cafe") == "plain cafe"
    assert repair_unicode("fire \U0001f525") == "fire \U0001f525"
    repaired = repair_unicode("tool \ud83d tail")
    repaired.encode("utf-8")
    assert repaired == "tool \ufffd tail"
    assert utf8_bytes("tool \ud83d tail") == "tool \ufffd tail".encode("utf-8")
    assert repair_unicode("\ud83d\ude00") == "\U0001f600"


def test_normalize_content_repairs_string_payloads():
    assert normalize_content_value("ok") == "ok"
    assert normalize_content_value("half \ud83d emoji") == "half \ufffd emoji"
    assert normalize_content_value("pair \ud83d\ude00") == "pair \U0001f600"


def test_dependent_reply_fingerprint_accepts_lone_surrogate():
    engine = _Fingerprint()
    text = "x" * 700 + "\ud83d"
    first = engine._ignored_dependent_reply_content_fingerprint(
        {"role": "tool", "tool_call_id": "call_1"},
        text,
    )
    second = engine._ignored_dependent_reply_content_fingerprint(
        {"role": "tool", "tool_call_id": "call_1"},
        text,
    )
    assert first == second
    assert len(first) == 16


def test_plain_dependent_reply_fingerprint_matches_strict_utf8():
    engine = _Fingerprint()
    text = "duplicate dependent reply"
    identity = "\0".join(("assistant", "", "", text))
    expected = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    assert (
        engine._ignored_dependent_reply_content_fingerprint(
            {"role": "assistant", "content": text},
            text,
        )
        == expected
    )


def test_tool_call_argument_surrogate_does_not_raise():
    engine = _Fingerprint()
    digest = engine._ignored_dependent_reply_content_fingerprint(
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "function": {"name": "write", "arguments": "{\"note\": \"\ud83d\"}"}}
            ],
        },
        "ok",
    )
    assert digest is not None
    assert len(digest) == 16
