"""Regression tests for non-destructive sensitive-pattern redaction.

Redaction runs on the ingest write path, so a false positive is written into
SQLite and the original characters are unrecoverable. No pattern set is
perfect, so an operator can opt into retaining the pre-redaction text in the
non-indexed ``messages.content_raw`` column. ``messages.content`` keeps only
the redacted form, so FTS, search, summarization, active replay and
externalization are unchanged, and a false positive becomes a rendering
annoyance rather than permanent data loss.
"""

from __future__ import annotations

from pathlib import Path

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.ingest_protection import (
    SENSITIVE_RAW_CONTENT_KEY,
    protect_message_for_ingest,
    sensitive_raw_content_retention_enabled,
)
from hermes_lcm.store import MessageStore


def _sensitive_config(tmp_path: Path, **overrides) -> LCMConfig:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    config.sensitive_patterns_enabled = True
    config.sensitive_patterns = [
        "api_key",
        "bearer_token",
        "password_assignment",
        "private_key",
    ]
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _sensitive_engine(tmp_path: Path, **overrides) -> LCMEngine:
    engine = LCMEngine(
        config=_sensitive_config(tmp_path, **overrides),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "precision-session",
        platform="telegram",
        conversation_id="precision-conversation",
        context_length=200_000,
    )
    return engine


# --- 2. Durability: raw content survives a false positive ------------------

# A residual false positive the tightened patterns still match: a filesystem
# path to where the credential lives, not the credential itself.
RESIDUAL_FALSE_POSITIVE = "api_key: /etc/secrets/openai_v2_key.txt"


def test_raw_content_retention_is_off_by_default(tmp_path):
    config = _sensitive_config(tmp_path)

    assert config.sensitive_retain_raw_content is False
    assert sensitive_raw_content_retention_enabled(config) is False


def test_raw_content_retention_requires_sensitive_patterns_enabled(tmp_path):
    config = _sensitive_config(tmp_path, sensitive_retain_raw_content=True)
    config.sensitive_patterns_enabled = False

    assert sensitive_raw_content_retention_enabled(config) is False


def test_content_raw_column_is_absent_when_retention_is_off(tmp_path):
    engine = _sensitive_engine(tmp_path)

    engine._ingest_messages([{"role": "user", "content": RESIDUAL_FALSE_POSITIVE}])

    columns = {
        row[1]
        for row in engine._store._conn.execute("PRAGMA table_info(messages)").fetchall()
    }
    assert "content_raw" not in columns
    stored = engine._store._conn.execute(
        "SELECT content FROM messages ORDER BY store_id"
    ).fetchone()[0]
    # The pre-existing, documented behaviour: the original text is gone.
    assert "/etc/secrets/openai_v2_key.txt" not in stored


def test_false_positive_stays_recoverable_when_retention_is_on(tmp_path):
    """The core durability guarantee: redaction stops being permanent loss."""
    engine = _sensitive_engine(tmp_path, sensitive_retain_raw_content=True)

    engine._ingest_messages([{"role": "user", "content": RESIDUAL_FALSE_POSITIVE}])

    store_id, content, content_raw = engine._store._conn.execute(
        "SELECT store_id, content, content_raw FROM messages ORDER BY store_id"
    ).fetchone()
    # `content` is unchanged by this feature: still redacted.
    assert "/etc/secrets/openai_v2_key.txt" not in content
    assert "[LCM sensitive redaction:" in content
    # ... and the original characters are recoverable.
    assert content_raw == RESIDUAL_FALSE_POSITIVE
    assert engine._store.get_raw_content(store_id) == RESIDUAL_FALSE_POSITIVE


def test_retained_raw_content_is_never_fts_indexed(tmp_path):
    """messages_fts indexes `content` only, so raw text stays unsearchable."""
    engine = _sensitive_engine(tmp_path, sensitive_retain_raw_content=True)

    engine._ingest_messages([{"role": "user", "content": RESIDUAL_FALSE_POSITIVE}])

    hits = engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", ("secrets",)
    ).fetchone()[0]
    assert hits == 0
    assert engine._store.search("openai_v2_key", session_id=engine.current_session_id) == []


def test_retention_does_not_write_raw_content_for_unmatched_messages(tmp_path):
    """A message no pattern touched costs nothing: `content` is already raw."""
    engine = _sensitive_engine(tmp_path, sensitive_retain_raw_content=True)
    ordinary = "the deploy finished and the smoke tests are green"

    engine._ingest_messages([
        {"role": "user", "content": RESIDUAL_FALSE_POSITIVE},
        {"role": "user", "content": ordinary},
    ])

    rows = engine._store._conn.execute(
        "SELECT content, content_raw FROM messages ORDER BY store_id"
    ).fetchall()
    assert rows[1][0] == ordinary
    assert rows[1][1] is None


def test_retention_marks_a_named_migration_step_not_a_schema_bump(tmp_path):
    """Follows the temporal_rollups_v1 / embeddings_v1 lazy-feature idiom."""
    from hermes_lcm import db_bootstrap

    engine = _sensitive_engine(tmp_path, sensitive_retain_raw_content=True)
    engine._ingest_messages([{"role": "user", "content": RESIDUAL_FALSE_POSITIVE}])

    steps = {
        row[0]
        for row in engine._store._conn.execute(
            "SELECT step_name FROM lcm_migration_state"
        ).fetchall()
    }
    assert "sensitive_raw_content_v1" in steps
    assert (
        db_bootstrap.get_schema_version(engine._store._conn)
        == db_bootstrap.SCHEMA_VERSION
    )


def test_retention_sidecar_key_never_reaches_a_stored_message(tmp_path):
    """The transport key is popped before the INSERT and is not a column."""
    config = _sensitive_config(tmp_path, sensitive_retain_raw_content=True)
    store = MessageStore(
        str(tmp_path / "lcm.db"), ingest_protection_config=config
    )

    store_id = store.append("s", {"role": "user", "content": RESIDUAL_FALSE_POSITIVE})

    stored = store.get(store_id)
    assert SENSITIVE_RAW_CONTENT_KEY not in stored
    columns = {
        row[1] for row in store._conn.execute("PRAGMA table_info(messages)").fetchall()
    }
    assert SENSITIVE_RAW_CONTENT_KEY not in columns
    assert store.get_raw_content(store_id) == RESIDUAL_FALSE_POSITIVE


def test_protect_message_for_ingest_attaches_sidecar_only_when_enabled(tmp_path):
    off = _sensitive_config(tmp_path)
    on = _sensitive_config(tmp_path, sensitive_retain_raw_content=True)
    message = {"role": "user", "content": RESIDUAL_FALSE_POSITIVE}

    assert SENSITIVE_RAW_CONTENT_KEY not in protect_message_for_ingest(
        dict(message), config=off
    )
    protected = protect_message_for_ingest(dict(message), config=on)
    assert protected[SENSITIVE_RAW_CONTENT_KEY] == RESIDUAL_FALSE_POSITIVE


def test_get_raw_content_returns_none_on_a_store_without_the_column(tmp_path):
    """Read-back is safe on an install that never enabled retention."""
    config = _sensitive_config(tmp_path)
    store = MessageStore(str(tmp_path / "lcm.db"), ingest_protection_config=config)

    store_id = store.append("s", {"role": "user", "content": "ordinary text"})

    assert store.get_raw_content(store_id) is None
    assert store.get_raw_content(9999) is None


def test_retention_can_be_enabled_on_an_existing_store(tmp_path):
    """The lazy ALTER migrates a live DB written before retention was on."""
    db_path = str(tmp_path / "lcm.db")
    off = _sensitive_config(tmp_path)
    store = MessageStore(db_path, ingest_protection_config=off)
    store.append("s", {"role": "user", "content": RESIDUAL_FALSE_POSITIVE})
    store.close()

    on = _sensitive_config(tmp_path, sensitive_retain_raw_content=True)
    reopened = MessageStore(db_path, ingest_protection_config=on)
    new_id = reopened.append("s", {"role": "user", "content": RESIDUAL_FALSE_POSITIVE})

    # The pre-existing row stays lossy (redaction is forward-only, as
    # documented), but every row written after the flip is recoverable.
    assert reopened.get_raw_content(1) is None
    assert reopened.get_raw_content(new_id) == RESIDUAL_FALSE_POSITIVE


def test_status_reports_retention_and_lossless_recovery(tmp_path):
    """`lcm_status` / `lcm_doctor` must not claim loss when raw text is kept."""
    from hermes_lcm.ingest_protection import sensitive_pattern_status

    off = sensitive_pattern_status(_sensitive_config(tmp_path))
    assert off["retain_raw_content"] is False
    assert off["lossless_recovery"] is False

    on = sensitive_pattern_status(
        _sensitive_config(tmp_path, sensitive_retain_raw_content=True)
    )
    assert on["retain_raw_content"] is True
    assert on["lossless_recovery"] is True

    disabled = _sensitive_config(tmp_path)
    disabled.sensitive_patterns_enabled = False
    assert sensitive_pattern_status(disabled)["lossless_recovery"] is None
