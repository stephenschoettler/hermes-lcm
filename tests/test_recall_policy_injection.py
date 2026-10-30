"""Policy suppression must preserve independently compiled evidence bytes."""

import importlib
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from tests.test_packaging_install import _load_plugin_entrypoint_module


@pytest.fixture
def plugin():
    return _load_plugin_entrypoint_module("hermes_lcm_policy_injection")


@pytest.mark.parametrize("mode", ["every_turn", "first_turn", "off"])
@pytest.mark.parametrize("first_turn", [True, False, None])
@pytest.mark.parametrize("evidence_mode", ["legacy_selective", "requirements_v1"])
@pytest.mark.parametrize("evidence", [None, "<evidence>exact source</evidence>"])
def test_policy_and_evidence_matrix(
    plugin, monkeypatch, mode, first_turn, evidence_mode, evidence
):
    engine = SimpleNamespace(
        _config=SimpleNamespace(
            recall_policy_injection=mode,
            preanswer_evidence_enabled=True,
            preanswer_evidence_mode=evidence_mode,
        )
    )
    monkeypatch.setattr(plugin, "_answer_ready_baseline", lambda *_: ())
    if evidence_mode == "requirements_v1":
        contract = importlib.import_module(f"{plugin.__name__}.answer_contract")
        compiler = importlib.import_module(f"{plugin.__name__}.evidence_compiler")
        monkeypatch.setattr(
            contract,
            "compile_answer_contract",
            lambda *_: SimpleNamespace(status="planned"),
        )
        monkeypatch.setattr(
            compiler,
            "compile_preanswer_evidence",
            lambda *_, **__: {"context": evidence},
        )
    else:
        selective = importlib.import_module(f"{plugin.__name__}.selective_recall")
        monkeypatch.setattr(
            selective, "route_selective_recall", lambda *_: {"route": "selective"}
        )
        monkeypatch.setattr(
            selective,
            "build_selective_session_bundle",
            lambda *_, **__: {"context": evidence},
        )
    payload = {
        "user_message": "How long is my commute?",
        "enabled_toolsets": ["context_engine"],
    }
    if first_turn is not None:
        payload["is_first_turn"] = first_turn
    policy = plugin.get_recall_policy()
    parts = (
        [policy]
        if mode == "every_turn" or (mode == "first_turn" and first_turn is True)
        else []
    )
    if evidence:
        parts.append(evidence)
    expected = {"context": "\n\n".join(parts)} if parts else None
    assert plugin._pre_llm_context(engine, policy, payload) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, "every_turn"),
        ("every_turn", "every_turn"),
        ("first_turn", "first_turn"),
        ("off", "off"),
        (" FIRST_TURN ", "first_turn"),
        ("invalid", "every_turn"),
        ("", "every_turn"),
    ],
)
def test_env_parsing(monkeypatch, caplog, raw, expected):
    monkeypatch.delenv("LCM_RECALL_POLICY_INJECTION", raising=False)
    if raw is not None:
        monkeypatch.setenv("LCM_RECALL_POLICY_INJECTION", raw)
    config = LCMConfig.from_env()
    assert config.recall_policy_injection == expected
    invalid = raw in {"invalid", ""}
    assert config.config_sources["recall_policy_injection"] == (
        "default" if raw is None or invalid else "env:LCM_RECALL_POLICY_INJECTION"
    )
    if invalid:
        assert "LCM_RECALL_POLICY_INJECTION" in caplog.text
        assert any(
            "LCM_RECALL_POLICY_INJECTION" in warning
            for warning in config.config_source_warnings
        )


@pytest.mark.parametrize("mode", ["every_turn", "first_turn", "off"])
@pytest.mark.parametrize(
    "path", ["disabled", "toolset_disabled", "empty_question", "ordinary", "error"]
)
def test_no_evidence_paths(plugin, monkeypatch, mode, path):
    engine = SimpleNamespace(
        _config=SimpleNamespace(
            recall_policy_injection=mode,
            preanswer_evidence_enabled=path != "disabled",
            preanswer_evidence_mode="legacy_selective",
        )
    )
    selective = importlib.import_module(f"{plugin.__name__}.selective_recall")

    def route(*_):
        if path == "error":
            raise RuntimeError("evidence unavailable")
        return {"route": "ordinary"}

    monkeypatch.setattr(selective, "route_selective_recall", route)
    payload = {
        "is_first_turn": False,
        "user_message": "" if path == "empty_question" else "Hello",
        "enabled_toolsets": [] if path == "toolset_disabled" else ["context_engine"],
    }
    policy = plugin.get_recall_policy()
    assert plugin._pre_llm_context(engine, policy, payload) == (
        {"context": policy} if mode == "every_turn" else None
    )


@pytest.mark.parametrize("configured", [None, "invalid"])
def test_missing_or_invalid_manual_config_preserves_policy(plugin, configured):
    config = SimpleNamespace()
    if configured is not None:
        config.recall_policy_injection = configured
    policy = plugin.get_recall_policy()
    assert plugin._pre_llm_context(SimpleNamespace(_config=config), policy, {}) == {
        "context": policy
    }


@pytest.mark.parametrize("mode", ["every_turn", "first_turn", "off"])
def test_registered_hook_uses_env_mode_across_turns(
    plugin, monkeypatch, tmp_path, mode
):
    from tests.test_packaging_install import _ensure_agent_context_engine_importable

    _ensure_agent_context_engine_importable(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LCM_RECALL_POLICY_INJECTION", mode)
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_ENABLED", "false")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    hooks = {}

    class Context:
        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            hooks[name] = callback

    ctx = Context()
    plugin.register(ctx)
    try:
        ctx.engine.on_session_start("policy-session", platform="cli")
        policy = plugin.get_recall_policy()
        for first_turn in (True, False):
            result = hooks["pre_llm_call"](
                session_id="policy-session",
                user_message="Hello",
                enabled_toolsets=["context_engine"],
                is_first_turn=first_turn,
            )
            assert result == (
                {"context": policy}
                if mode == "every_turn" or (mode == "first_turn" and first_turn)
                else None
            )
    finally:
        ctx.engine.shutdown()
