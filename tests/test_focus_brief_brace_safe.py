"""Regression tests: focus-brief builders must not treat the focus topic as a
format template.

The auto-derived focus topic is built from raw user message text
(``LCMEngine._derive_auto_focus_topic``). That text can contain literal curly
braces — brace-dense JSON payloads, code snippets, or even a single stray ``{``.

``_build_l1_focus_brief`` / ``_build_l2_focus_brief`` must embed that text
verbatim without ever running ``str.format()`` over it. Running ``.format()``
over untrusted content raises ``ValueError: unmatched '{' in format spec`` (or
``KeyError`` / ``Single '}' encountered`` depending on the brace shape), which
aborts compression and drops the whole turn.
"""

import pytest

from hermes_lcm import escalation


# Brace shapes that all break str.format() when used as the template string.
BRACY_TOPICS = [
    "rate is 3{ per unit",                 # lone open brace
    "rate is 3} per unit",                 # lone close brace
    'data {"k": "v"}',                     # balanced braces
    "plan {tier: {gold",                   # nested/colon -> "unmatched '{' in format spec"
    '[IMPORTANT] {"rule": {"id": "quo-maint-line"}}',  # brace-dense wakeup-style JSON
]


@pytest.mark.parametrize("builder", [
    escalation._build_l1_focus_brief,
    escalation._build_l2_focus_brief,
])
@pytest.mark.parametrize("topic", BRACY_TOPICS)
def test_focus_brief_survives_braces(builder, topic):
    # Must not raise regardless of braces in the focus topic.
    out = builder(topic)
    # The historical-heading markers must still be substituted (the sole real
    # placeholder), and the topic text must survive verbatim.
    assert "## Historical Task Snapshot" in out
    assert "{markers}" not in out  # placeholder actually got filled
    assert topic in out            # focus topic embedded literally, braces intact


@pytest.mark.parametrize("builder", [
    escalation._build_l1_focus_brief,
    escalation._build_l2_focus_brief,
])
def test_focus_brief_empty_topic_returns_blank(builder):
    assert builder("") == ""
    assert builder("   ") == ""
