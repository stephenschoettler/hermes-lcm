"""Regression tests for compaction-scaffold-aware replay identity.

After context compression the host appends a volatile task-list annotation
(``_PRESERVED_TODO_CONTEXT_PREFIX``) to the last user message.  The annotation
changes on every compression cycle (task statuses update), so including it in
the replay identity causes ``_find_reconciled_cursor_for_store_tail`` to fail
suffix matching, fall through to ``cursor=0``, and re-persist every message —
creating duplicates with distinct store_ids and timestamps.

These tests verify that:
1. ``_message_replay_identity`` strips the todo annotation before hashing.
2. Reconciliation after a compression boundary correctly advances the cursor
   even when the todo annotation changed between compaction cycles.
3. ``_is_replayed_context_scaffold_message`` recognises standalone todo
   annotation messages as scaffolding (not durable user content).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

# engine.py imports agent.context_engine at module level; provide a stub
# before the conftest partial-import can poison sys.modules.
from hermes_lcm.config import LCMConfig
from hermes_lcm.reconcile import (
    _PRESERVED_TODO_CONTEXT_PREFIX,
)

_TODO_ANNOTATION_V1 = (
    f"\n\n{_PRESERVED_TODO_CONTEXT_PREFIX}\n"
    "- [>] 1. First task (in_progress)\n"
    "- [ ] 2. Second task (pending)\n"
)

_TODO_ANNOTATION_V2 = (
    f"\n\n{_PRESERVED_TODO_CONTEXT_PREFIX}\n"
    "- [x] 1. First task (completed)\n"
    "- [>] 2. Second task (in_progress)\n"
    "- [ ] 3. Third task (pending)\n"
)


def _make_engine(tmp_path: Path, *, session_id: str = "reconcile-todo"):
    # engine.py imports agent.context_engine at module level; provide a
    # stub here (not at module level, so collection order of other test
    # files is unaffected), then clear any partial import left by conftest.
    if "agent.context_engine" not in sys.modules:
        _agent_mod = ModuleType("agent")
        _agent_mod.__path__ = []
        _ce_mod = ModuleType("agent.context_engine")

        class _StubContextEngine:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self.last_prompt_tokens = 0

            def get_status(self):
                return {}

        _ce_mod.ContextEngine = _StubContextEngine
        sys.modules["agent"] = _agent_mod
        sys.modules["agent.context_engine"] = _ce_mod

    _existing = sys.modules.get("hermes_lcm.engine")
    if _existing is not None and not hasattr(_existing, "LCMEngine"):
        sys.modules.pop("hermes_lcm.engine", None)
    from hermes_lcm.engine import LCMEngine

    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine._session_id = session_id
    return engine


class TestTodoAnnotationIdentity:
    """_message_replay_identity must be stable across todo annotation changes."""

    def test_identity_strips_todo_annotation(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            msg_plain = {"role": "user", "content": "Do the thing"}
            msg_annotated = {
                "role": "user",
                "content": "Do the thing" + _TODO_ANNOTATION_V1,
            }
            msg_annotated_v2 = {
                "role": "user",
                "content": "Do the thing" + _TODO_ANNOTATION_V2,
            }

            id_plain = engine._message_replay_identity(msg_plain)
            id_v1 = engine._message_replay_identity(msg_annotated)
            id_v2 = engine._message_replay_identity(msg_annotated_v2)

            assert id_plain == id_v1, (
                "Identity with todo annotation v1 must match plain content"
            )
            assert id_plain == id_v2, (
                "Identity with todo annotation v2 must match plain content"
            )
            assert id_v1 == id_v2, (
                "Identity must be stable across different todo annotation versions"
            )
        finally:
            engine.shutdown()

    def test_identity_preserves_content_without_annotation(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            msg_a = {"role": "user", "content": "Alpha message"}
            msg_b = {"role": "user", "content": "Beta message"}
            assert (
                engine._message_replay_identity(msg_a)
                != engine._message_replay_identity(msg_b)
            )
        finally:
            engine.shutdown()

    def test_identity_strips_annotation_from_stored_row(self, tmp_path):
        """Stored rows carry the annotation verbatim; identity must still strip."""
        engine = _make_engine(tmp_path)
        try:
            stored = {
                "role": "user",
                "content": "Persisted content" + _TODO_ANNOTATION_V1,
            }
            incoming = {
                "role": "user",
                "content": "Persisted content" + _TODO_ANNOTATION_V2,
            }
            id_stored = engine._message_replay_identity(stored, stored_row=True)
            id_incoming = engine._message_replay_identity(incoming)
            assert id_stored == id_incoming
        finally:
            engine.shutdown()


class TestTodoAnnotationScaffoldDetection:
    """Standalone todo annotation messages must be recognised as scaffolding."""

    def test_standalone_todo_is_scaffold(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            msg = {
                "role": "user",
                "content": _PRESERVED_TODO_CONTEXT_PREFIX + "\n- [ ] task",
            }
            assert engine._is_replayed_context_scaffold_message(msg) is True
        finally:
            engine.shutdown()

    def test_objective_prefix_still_scaffold(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            msg = {
                "role": "user",
                "content": "[Current user objective preserved from compacted history]\nDo X",
            }
            assert engine._is_replayed_context_scaffold_message(msg) is True
        finally:
            engine.shutdown()

    def test_regular_user_message_not_scaffold(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            msg = {"role": "user", "content": "Just a normal question"}
            assert engine._is_replayed_context_scaffold_message(msg) is False
        finally:
            engine.shutdown()


class TestReconcileAfterCompactionWithTodoAnnotation:
    """End-to-end: ingest → compress → re-ingest with changed annotation."""

    def test_no_duplication_when_todo_annotation_changes(self, tmp_path):
        """Simulates the production bug: compaction changes the todo annotation
        on the last user message, then reconciliation must still recognise the
        stored messages and advance the cursor past them."""
        engine = _make_engine(tmp_path)
        try:
            # Phase 1: initial ingest (pre-compaction)
            original_messages = [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "First user request"},
                {"role": "assistant", "content": "First assistant reply"},
                {"role": "user", "content": "Second user request" + _TODO_ANNOTATION_V1},
                {"role": "assistant", "content": "Second assistant reply"},
            ]
            engine._ingest_messages(original_messages)
            count_after_first = engine._store.get_session_count("reconcile-todo")
            assert count_after_first == 5

            # Phase 2: simulate compression boundary + restart.
            # The host rebuilds the message list with a DIFFERENT todo
            # annotation (task statuses changed during compaction).
            replay_engine = _make_engine(tmp_path)
            try:
                replay_engine._ingest_cursor_needs_reconcile = True
                replay_messages = [
                    {"role": "system", "content": "System prompt."},
                    {"role": "user", "content": "First user request"},
                    {"role": "assistant", "content": "First assistant reply"},
                    {"role": "user", "content": "Second user request" + _TODO_ANNOTATION_V2},
                    {"role": "assistant", "content": "Second assistant reply"},
                    {"role": "user", "content": "Brand new question after restart"},
                ]
                replay_engine._ingest_messages(replay_messages)

                count_after_replay = replay_engine._store.get_session_count(
                    "reconcile-todo"
                )
                # Must be 6 (5 original + 1 new), NOT 11 (5 dup + 5 + 1).
                assert count_after_replay == 6, (
                    f"Expected 6 messages (5 original + 1 new), got {count_after_replay}. "
                    "Duplication bug: reconciliation failed to match stored tail "
                    "because the todo annotation changed."
                )

                # Verify no content duplication
                rows = replay_engine._store.get_session_messages("reconcile-todo")
                contents = [r["content"] for r in rows]
                assert contents.count("First user request") == 1
                assert contents.count("First assistant reply") == 1
                assert contents.count("Second assistant reply") == 1
                assert contents.count("Brand new question after restart") == 1
            finally:
                replay_engine.shutdown()
        finally:
            engine.shutdown()

    def test_no_duplication_with_identical_annotation(self, tmp_path):
        """Control: identical annotation should also produce no duplication."""
        engine = _make_engine(tmp_path)
        try:
            original_messages = [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "User request" + _TODO_ANNOTATION_V1},
                {"role": "assistant", "content": "Assistant reply"},
            ]
            engine._ingest_messages(original_messages)
            assert engine._store.get_session_count("reconcile-todo") == 3

            replay_engine = _make_engine(tmp_path)
            try:
                replay_engine._ingest_cursor_needs_reconcile = True
                replay_messages = [
                    {"role": "system", "content": "System prompt."},
                    {"role": "user", "content": "User request" + _TODO_ANNOTATION_V1},
                    {"role": "assistant", "content": "Assistant reply"},
                    {"role": "user", "content": "Follow-up"},
                ]
                replay_engine._ingest_messages(replay_messages)
                assert replay_engine._store.get_session_count("reconcile-todo") == 4
            finally:
                replay_engine.shutdown()
        finally:
            engine.shutdown()
