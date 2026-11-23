"""Regression coverage for behavioral LCM settings in config.yaml."""

from pathlib import Path

from hermes_lcm.config import LCMConfig


def test_config_yaml_reads_bounded_context_guardrails(monkeypatch, tmp_path: Path):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "lcm:\n"
        "  context_threshold: 0.70\n"
        "  fresh_tail_count: 16\n"
        "  fresh_tail_max_tokens: 32000\n"
        "  max_assembly_tokens: 190000\n"
        "  reserve_tokens_floor: 82000\n"
        "  threshold_full_sweep_enabled: true\n"
        "  large_output_externalization_enabled: true\n"
        "  large_output_active_replay_stubbing_enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in (
        "LCM_CONTEXT_THRESHOLD",
        "LCM_FRESH_TAIL_COUNT",
        "LCM_FRESH_TAIL_MAX_TOKENS",
        "LCM_MAX_ASSEMBLY_TOKENS",
        "LCM_RESERVE_TOKENS_FLOOR",
        "LCM_THRESHOLD_FULL_SWEEP_ENABLED",
        "LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED",
        "LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)

    config = LCMConfig.from_env()

    assert config.context_threshold == 0.70
    assert config.fresh_tail_count == 16
    assert config.fresh_tail_max_tokens == 32_000
    assert config.max_assembly_tokens == 190_000
    assert config.reserve_tokens_floor == 82_000
    assert config.threshold_full_sweep_enabled is True
    assert config.large_output_externalization_enabled is True
    assert config.large_output_active_replay_stubbing_enabled is True
    assert config.ignored_config_yaml_lcm_keys == []


def test_environment_keeps_precedence_over_config_yaml(monkeypatch, tmp_path: Path):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "lcm:\n  max_assembly_tokens: 190000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("LCM_MAX_ASSEMBLY_TOKENS", "180000")

    assert LCMConfig.from_env().max_assembly_tokens == 180_000
