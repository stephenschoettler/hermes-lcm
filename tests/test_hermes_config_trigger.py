"""Focused tests for the optional Hermes compression-policy adapter."""

from pathlib import Path

from hermes_lcm.config import LCMConfig, load_hermes_compression_policy, resolve_hermes_model_threshold
from hermes_lcm.engine import LCMEngine


def _write_config(home: Path, body: str) -> None:
    home.mkdir()
    (home / "config.yaml").write_text(body, encoding="utf-8")


def test_model_threshold_matching_is_longest_and_provider_scoped():
    thresholds = {"model": 0.40, "next-model": 0.30, "provider:next-model": 0.20}
    assert resolve_hermes_model_threshold("next-model-v2", thresholds, 0.50, "provider") == 0.20
    assert resolve_hermes_model_threshold("next-model-v2", thresholds, 0.50, "other") == 0.30
    assert resolve_hermes_model_threshold("unmatched", thresholds, 0.50, "provider") == 0.50


def test_env_backed_engine_reads_active_home_and_clone_keeps_policy(tmp_path):
    home = tmp_path / "hermes"
    _write_config(home, """\
lcm:
  trigger_mode: hermes_config
compression:
  threshold: 0.50
  threshold_tokens: 250000
  model_thresholds:
    test-model: 0.30
    test-provider:next-model: 0.20
""")
    engine = LCMEngine(hermes_home=str(home))
    clone = None
    try:
        assert engine._config_from_env is True
        engine.update_model("test-model", 100000, provider="test-provider")
        assert engine._config.trigger_mode == "hermes_config"
        assert engine.threshold_tokens == 75000
        clone = engine.clone_for_agent()
        assert clone._config_from_env is True
        clone.update_model("next-model", 1_000_000, provider="test-provider")
        assert clone.threshold_tokens == 200000
        assert clone.get_status()["trigger_mode"] == "hermes_config"
    finally:
        if clone is not None:
            clone.shutdown()
        engine.shutdown()


def test_explicit_config_remains_legacy_and_missing_cap_is_uncapped(tmp_path):
    home = tmp_path / "hermes"
    _write_config(home, """\
lcm:
  trigger_mode: hermes_config
compression:
  threshold: 0.20
""")
    explicit = LCMConfig(context_threshold=0.90, database_path=str(tmp_path / "explicit.db"))
    engine = LCMEngine(config=explicit, hermes_home=str(home))
    try:
        assert engine._config_from_env is False
        engine.update_model("test-model", 1_000_000, provider="test-provider")
        assert engine._config.trigger_mode == "legacy"
        assert engine.threshold_tokens == 900000
        policy = load_hermes_compression_policy(str(home))
        assert policy.threshold_tokens is None
    finally:
        engine.shutdown()


def test_malformed_profile_keeps_last_good_policy(tmp_path):
    home = tmp_path / "hermes"
    _write_config(home, """\
lcm:
  trigger_mode: hermes_config
compression:
  threshold: 0.20
  threshold_tokens: 250000
""")
    engine = LCMEngine(hermes_home=str(home))
    try:
        engine.update_model("test-model", 1_000_000, provider="test-provider")
        assert engine.threshold_tokens == 200000
        (home / "config.yaml").write_text("this is malformed", encoding="utf-8")
        status = engine.get_status()
        assert status["trigger_mode"] == "hermes_config"
        assert status["threshold_tokens_cap"] == 250000
        assert engine.threshold_tokens == 200000
    finally:
        engine.shutdown()


def test_manual_optin_reads_policy_and_manual_legacy_remains_isolated(tmp_path):
    home = tmp_path / "hermes"
    _write_config(home, """\
lcm:
  trigger_mode: legacy
compression:
  threshold: 0.20
  threshold_tokens: 250000
""")
    config = LCMConfig(trigger_mode="hermes_config", database_path=str(tmp_path / "manual.db"))
    engine = LCMEngine(config=config, hermes_home=str(home))
    try:
        engine.update_model("test-model", 1_000_000, provider="test-provider")
        assert engine.threshold_tokens == 200000
        assert engine.get_status()["trigger_mode_source"] == "manual:LCMConfig.trigger_mode"
        clone = engine.clone_for_agent()
        try:
            assert clone._config_from_env is False
            assert clone.threshold_tokens == 200000
            assert clone.get_status()["trigger_mode_source"] == "manual:LCMConfig.trigger_mode"
        finally:
            clone.shutdown()
    finally:
        engine.shutdown()


def test_hot_reload_and_mode_switch_restore_thresholds(tmp_path):
    home = tmp_path / "hermes"
    _write_config(home, "lcm:\n  trigger_mode: hermes_config\ncompression:\n  threshold: 0.20\n  threshold_tokens: 250000\n")
    engine = LCMEngine(hermes_home=str(home))
    try:
        engine.update_model("test-model", 1_000_000)
        assert engine.threshold_tokens == 200000
        (home / "config.yaml").write_text("lcm:\n  trigger_mode: hermes_config\ncompression:\n  threshold: 0.50\n  threshold_tokens: 1000\n", encoding="utf-8")
        assert engine.should_compress(1000)
        assert engine.threshold_tokens == 1000
        (home / "config.yaml").write_text("lcm:\n  trigger_mode: legacy\n  context_threshold: 0.40\ncompression:\n  threshold: 0.50\n  threshold_tokens: 2000\n", encoding="utf-8")
        assert not engine.should_compress(2000)
        assert engine.threshold_tokens == 400000
        assert engine.threshold_tokens_cap is None
    finally:
        engine.shutdown()


def test_lcm_env_ratio_is_ignored_but_trigger_mode_env_wins(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    _write_config(home, "lcm:\n  trigger_mode: hermes_config\ncompression:\n  threshold: 0.20\n")
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.80")
    engine = LCMEngine(hermes_home=str(home))
    try:
        engine.update_model("test-model", 1_000_000)
        assert engine.threshold_tokens == 200000
    finally:
        engine.shutdown()
    monkeypatch.setenv("LCM_TRIGGER_MODE", "legacy")
    engine = LCMEngine(hermes_home=str(home))
    try:
        engine.update_model("test-model", 1_000_000)
        assert engine.threshold_tokens == 800000
        assert engine.threshold_tokens_cap is None
    finally:
        engine.shutdown()


def test_without_pyyaml_handles_provider_keys_null_sequences_and_colon_values(tmp_path, monkeypatch):
    import hermes_lcm.config as config_module
    monkeypatch.setattr(config_module, "yaml", None)
    home = tmp_path / "hermes"
    _write_config(home, """\
lcm:
  trigger_mode: hermes_config
compression:
  threshold: 0.50
  threshold_tokens: null
  model_thresholds:
    "provider:test-model": 0.20
    provider:next-model: 0.30
plugins:
  enabled:
    - hermes-lcm
model:
  base_url: https://example.invalid:443/v1
  windows_path: C:/Users/example/model
""")
    policy = load_hermes_compression_policy(home)
    assert policy.threshold_tokens is None
    assert policy.model_thresholds == {"provider:test-model": 0.20, "provider:next-model": 0.30}
    cfg = config_module._load_hermes_config_yaml(home)
    assert cfg["model"]["base_url"] == "https://example.invalid:443/v1"
    assert cfg["model"]["windows_path"] == "C:/Users/example/model"


def test_host_cache_invalidation_restores_policy_after_field_clobber(tmp_path):
    home = tmp_path / "hermes"
    _write_config(home, "lcm:\n  trigger_mode: hermes_config\ncompression:\n  threshold: 0.20\n  threshold_tokens: 250000\n")
    engine = LCMEngine(hermes_home=str(home))
    try:
        engine.update_model("test-model", 1_000_000)
        engine.threshold_tokens_cap = 1000
        engine.threshold_percent = 0.80
        engine._threshold_tokens = None
        assert engine.threshold_tokens == 200000
        assert engine.threshold_tokens_cap == 250000
    finally:
        engine.shutdown()


def test_explicit_from_env_config_keeps_dynamic_policy(tmp_path):
    home = tmp_path / "hermes"
    _write_config(home, "lcm:\n  trigger_mode: hermes_config\ncompression:\n  threshold: 0.20\n")
    engine = LCMEngine(config=LCMConfig.from_env(home), hermes_home=str(home))
    try:
        engine.update_model("test-model", 1_000_000)
        assert engine._config_from_env is True
        assert engine.threshold_tokens == 200000
        (home / "config.yaml").write_text("lcm:\n  trigger_mode: hermes_config\ncompression:\n  threshold: 0.30\n", encoding="utf-8")
        engine.get_status()
        assert engine.threshold_tokens == 300000
    finally:
        engine.shutdown()


def test_small_windows_and_stricter_assembly_cap(tmp_path):
    home = tmp_path / "hermes"
    _write_config(home, "lcm:\n  trigger_mode: hermes_config\ncompression:\n  threshold: 0.10\n  threshold_tokens: null\n")
    engine = LCMEngine(hermes_home=str(home))
    try:
        engine.update_model("test-model", 32000)
        assert engine.threshold_tokens == 27200
        engine.update_model("test-model", 100000)
        assert engine.threshold_tokens == 75000
        engine._config.max_assembly_tokens = 50000
        engine.update_model("test-model", 100000)
        assert engine.threshold_tokens == 50000
    finally:
        engine.shutdown()


def test_profile_rebind_changes_policy_without_replacing_derived_settings(tmp_path):
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    _write_config(home_a, "lcm:\n  trigger_mode: hermes_config\ncompression:\n  threshold: 0.20\n")
    _write_config(home_b, "lcm:\n  trigger_mode: hermes_config\ncompression:\n  threshold: 0.30\n")
    engine = LCMEngine(hermes_home=str(home_a))
    try:
        engine.update_model("test-model", 1_000_000)
        old_config = engine._config
        old_summary_model = engine.summary_model
        old_protect_last_n = engine.protect_last_n
        assert engine._rebind_storage_for_home(str(home_b))
        engine.get_status()
        assert engine.threshold_tokens == 300000
        assert engine._config is old_config
        assert engine.summary_model == old_summary_model
        assert engine.protect_last_n == old_protect_last_n
    finally:
        engine.shutdown()


def test_scalar_policy_updates_keep_last_good(tmp_path):
    home = tmp_path / "hermes"
    _write_config(home, "lcm:\n  trigger_mode: hermes_config\ncompression:\n  threshold: 0.20\n")
    engine = LCMEngine(hermes_home=str(home))
    try:
        engine.update_model("test-model", 1_000_000)
        for scalar in ("false", "0", "this is malformed"):
            (home / "config.yaml").write_text(scalar, encoding="utf-8")
            assert engine.get_status()["trigger_mode"] == "hermes_config"
            assert engine.threshold_tokens == 200000
    finally:
        engine.shutdown()


def test_plugin_registration_prototype_and_clone_are_environment_backed(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    home = tmp_path / "hermes"
    _write_config(home, "lcm:\n  trigger_mode: hermes_config\ncompression:\n  threshold: 0.20\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    entry = Path(__file__).parent.parent / "__init__.py"
    spec = importlib.util.spec_from_file_location("hermes_lcm.registration_test", entry)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "hermes_lcm"
    spec.loader.exec_module(module)
    registered = []
    module.register(SimpleNamespace(register_context_engine=registered.append))
    engine = registered[0]
    clone = None
    try:
        assert engine._config_from_env is True
        engine.update_model("test-model", 1_000_000)
        clone = engine.clone_for_agent()
        assert clone._config_from_env is True
        assert clone.threshold_tokens == 200000
    finally:
        if clone is not None:
            clone.shutdown()
        engine.shutdown()
