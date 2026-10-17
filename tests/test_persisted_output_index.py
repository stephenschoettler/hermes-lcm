"""Persisted-output lookup indexes routing metadata without caching payloads."""

import json
from pathlib import Path
from types import SimpleNamespace

from hermes_lcm.config import LCMConfig


def test_inplace_routing_change_is_visible_without_directory_change(tmp_path):
    import hermes_lcm.externalize as externalize

    output_dir = tmp_path / "externalized"
    output_dir.mkdir()
    config = LCMConfig(large_output_externalization_enabled=True, large_output_externalization_path=str(output_dir))
    target = output_dir / "payload.json"
    payload = {"kind": "tool_result", "role": "tool", "session_id": "old-session", "tool_call_id": "old-call", "content": "old content"}
    target.write_text(json.dumps(payload), encoding="utf-8")
    def lookup(session, call):
        return externalize.find_externalized_tool_result_content_for_call(
            tool_call_id=call, session_id=session, config=config, hermes_home=str(tmp_path / "home"),
        )
    assert lookup("old-session", "old-call") == "old content"
    info = output_dir.stat()
    directory_signature = (info.st_mtime_ns, info.st_ctime_ns)
    target.write_text(json.dumps({**payload, "session_id": "new-session", "tool_call_id": "new-call", "content": "new content"}), encoding="utf-8")
    info = output_dir.stat()
    assert (info.st_mtime_ns, info.st_ctime_ns) == directory_signature
    assert lookup("new-session", "new-call") == "new content"
    assert lookup("old-session", "old-call") is None


class TestPersistedOutputIndex:
    def _engine(self, tmp_path):
        output_dir = tmp_path / "externalized"
        config = LCMConfig(large_output_externalization_enabled=True, large_output_externalization_path=str(output_dir))
        return SimpleNamespace(_config=config), output_dir

    def test_persisted_output_lookup_indexes_directory_once(self, tmp_path, monkeypatch):
        import hermes_lcm.externalize as externalize

        engine, output_dir = self._engine(tmp_path)
        output_dir.mkdir()
        for index in range(40):
            (output_dir / f"decoy-{index:02d}.json").write_text(
                json.dumps(
                    {
                        "kind": "tool_result",
                        "role": "tool",
                        "session_id": "other-session",
                        "tool_call_id": f"decoy-{index}",
                        "content": f"decoy content {index}",
                    }
                ),
                encoding="utf-8",
            )
        target_path = output_dir / "target.json"
        target_path.write_text(
            json.dumps(
                {
                    "kind": "tool_result",
                    "role": "tool",
                    "session_id": "ingest-session",
                    "tool_call_id": "call-target",
                    "content": "durable target content",
                }
            ),
            encoding="utf-8",
        )

        original_read_text = Path.read_text
        reads_by_name: dict[str, int] = {}

        def counting_read_text(path, *args, **kwargs):
            if path.parent == output_dir:
                reads_by_name[path.name] = reads_by_name.get(path.name, 0) + 1
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", counting_read_text)

        lookup_kwargs = {
            "tool_call_id": "call-target",
            "session_id": "ingest-session",
            "config": engine._config,
            "hermes_home": str(tmp_path / "hermes"),
        }
        assert (
            externalize.find_externalized_tool_result_content_for_call(**lookup_kwargs)
            == "durable target content"
        )
        reads_after_cold_lookup = dict(reads_by_name)
        assert all(reads_after_cold_lookup.get(f"decoy-{index:02d}.json") == 1 for index in range(40))

        assert (
            externalize.find_externalized_tool_result_content_for_call(**lookup_kwargs)
            == "durable target content"
        )
        assert all(reads_by_name.get(f"decoy-{index:02d}.json") == 1 for index in range(40))
        assert reads_by_name["target.json"] == reads_after_cold_lookup["target.json"] + 1


    def test_persisted_output_index_refreshes_after_internal_write(self, tmp_path, monkeypatch):
        import hermes_lcm.externalize as externalize

        engine, output_dir = self._engine(tmp_path)
        output_dir.mkdir()
        build_calls = 0
        original_build = externalize._build_externalized_tool_result_index

        def counting_build(storage_dir):
            nonlocal build_calls
            build_calls += 1
            return original_build(storage_dir)

        monkeypatch.setattr(externalize, "_build_externalized_tool_result_index", counting_build)
        lookup_kwargs = {
            "tool_call_id": "call-new",
            "session_id": "ingest-session",
            "config": engine._config,
            "hermes_home": str(tmp_path / "hermes"),
        }

        assert externalize.find_externalized_tool_result_content_for_call(**lookup_kwargs) is None
        assert build_calls == 1
        created = externalize.maybe_externalize_payload(
            "new durable tool output",
            kind="tool_result",
            tool_call_id="call-new",
            session_id="ingest-session",
            role="tool",
            config=engine._config,
            hermes_home=str(tmp_path / "hermes"),
            force=True,
        )
        assert created is not None
        assert (
            externalize.find_externalized_tool_result_content_for_call(**lookup_kwargs)
            == "new durable tool output"
        )
        assert build_calls == 2


    def test_persisted_output_index_does_not_hide_concurrent_external_write(self, tmp_path):
        import hermes_lcm.externalize as externalize

        engine, output_dir = self._engine(tmp_path)
        output_dir.mkdir()
        lookup_kwargs = {
            "tool_call_id": "call-external",
            "session_id": "ingest-session",
            "config": engine._config,
            "hermes_home": str(tmp_path / "hermes"),
        }
        assert externalize.find_externalized_tool_result_content_for_call(**lookup_kwargs) is None

        # Simulate a writer outside this process creating the payload between
        # our cold lookup and a normal in-process externalization.
        (output_dir / "external.json").write_text(
            json.dumps(
                {
                    "kind": "tool_result",
                    "role": "tool",
                    "session_id": "ingest-session",
                    "tool_call_id": "call-external",
                    "content": "externally written durable content",
                }
            ),
            encoding="utf-8",
        )
        created = externalize.maybe_externalize_payload(
            "unrelated local output",
            kind="tool_result",
            tool_call_id="call-local",
            session_id="ingest-session",
            role="tool",
            config=engine._config,
            hermes_home=str(tmp_path / "hermes"),
            force=True,
        )
        assert created is not None

        assert (
            externalize.find_externalized_tool_result_content_for_call(**lookup_kwargs)
            == "externally written durable content"
        )


    def test_persisted_output_lookup_falls_back_when_cold_index_never_stabilizes(
        self,
        tmp_path,
        monkeypatch,
    ):
        import hermes_lcm.externalize as externalize

        engine, output_dir = self._engine(tmp_path)
        output_dir.mkdir()
        (output_dir / "target.json").write_text(
            json.dumps(
                {
                    "kind": "tool_result",
                    "role": "tool",
                    "session_id": "ingest-session",
                    "tool_call_id": "call-target",
                    "content": "fallback durable content",
                }
            ),
            encoding="utf-8",
        )
        unstable_index = externalize._ExternalizedToolResultPathIndex(
            directory_signature=None
        )
        monkeypatch.setattr(
            externalize,
            "_build_externalized_tool_result_index",
            lambda _storage_dir: unstable_index,
        )

        assert (
            externalize.find_externalized_tool_result_content_for_call(
                tool_call_id="call-target",
                session_id="ingest-session",
                config=engine._config,
                hermes_home=str(tmp_path / "hermes"),
            )
            == "fallback durable content"
        )




def test_parse_error_is_retried_after_inplace_repair(tmp_path):
    from hermes_lcm import externalize
    output_dir = tmp_path / "externalized"
    output_dir.mkdir()
    target = output_dir / "target.json"
    target.write_text("{partial", encoding="utf-8")
    config = LCMConfig(large_output_externalization_path=str(output_dir))
    kwargs = {"tool_call_id": "example-call", "session_id": "example-session", "config": config}
    assert externalize.find_externalized_tool_result_content_for_call(**kwargs) is None
    target.write_text(json.dumps({"kind": "tool_result", "role": "tool", "tool_call_id": "example-call", "session_id": "example-session", "content": "repaired content"}), encoding="utf-8")
    assert externalize.find_externalized_tool_result_content_for_call(**kwargs) == "repaired content"


def test_process_cache_bounds_the_number_of_directories(tmp_path, monkeypatch):
    from hermes_lcm import externalize
    monkeypatch.setattr(externalize, "_MAX_EXTERNALIZED_TOOL_RESULT_INDEXES", 2)
    monkeypatch.setattr(externalize, "_EXTERNALIZED_TOOL_RESULT_INDEXES", externalize.OrderedDict())
    for i in range(5):
        directory = tmp_path / str(i)
        directory.mkdir()
        assert externalize._externalized_tool_result_candidate_paths(directory, tool_call_id="example-call", session_id="example-session") == ()
    assert len(externalize._EXTERNALIZED_TOOL_RESULT_INDEXES) == 2
    assert str((tmp_path / "4").resolve()) in externalize._EXTERNALIZED_TOOL_RESULT_INDEXES
