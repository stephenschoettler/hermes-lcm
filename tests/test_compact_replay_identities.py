"""Large tool replay identities retain exact matching with bounded strings."""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "memory.db")), hermes_home=str(tmp_path / "home"))
    instance._session_id = "example-session"
    try:
        yield instance
    finally:
        instance.shutdown()


def identity(engine, content, role="tool", call="example-call"):
    return engine._message_replay_identity({"role": role, "content": content, "tool_call_id": call})


def test_large_tool_identity_is_bounded_and_content_exact(engine):
    content = "a" * 70_000
    original = identity(engine, content)
    assert len(original[1]) < 200
    assert original == identity(engine, content)
    assert original != identity(engine, content[:-1] + "b")
    assert original != identity(engine, content, call="different-call")
    assert original != identity(engine, original[1])


def test_large_unicode_content_matches_by_utf8_digest(engine):
    content = "🍀" * 70_000
    result = identity(engine, content)
    assert len(result[1]) < 200
    assert "bytes=280000" in result[1]
    assert result != identity(engine, content[:-1] + "🌱")


def test_short_tool_and_nontool_content_keep_original_identity(engine):
    assert identity(engine, "short")[1] == "short"
    assert identity(engine, "a" * 70_000, role="assistant")[1] == "a" * 70_000
