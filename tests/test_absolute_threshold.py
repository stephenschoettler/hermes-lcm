"""Absolute compaction threshold override tests."""

import pytest

from hermes_lcm.config import LCMConfig


class TestAbsoluteThresholdTokens:
    def test_absolute_threshold_pins_across_model_windows(self, tmp_path, monkeypatch):
        from hermes_lcm.engine import LCMEngine

        monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", "130000")
        monkeypatch.delenv("LCM_CONTEXT_THRESHOLD", raising=False)

        config = LCMConfig(database_path=str(tmp_path / "abs-threshold.db"))
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        try:
            for window in (262_144, 500_000, 1_000_000):
                engine.update_model(
                    model="coding-model",
                    context_length=window,
                    provider="custom",
                    base_url="https://example.invalid/v1",
                    api_key="test-secret",
                    api_mode="chat",
                )
                assert engine.threshold_tokens == 130_000, (
                    f"window={window} expected absolute 130000, got {engine.threshold_tokens}"
                )
                assert engine.context_length == window
        finally:
            engine.shutdown()

    def test_unset_absolute_keeps_ratio_behavior(self, tmp_path, monkeypatch):
        from hermes_lcm.engine import LCMEngine

        monkeypatch.delenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", raising=False)
        monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.35")

        config = LCMConfig.from_env()
        config.database_path = str(tmp_path / "ratio-threshold.db")
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        try:
            engine.update_model(
                model="coding-model",
                context_length=200_000,
                provider="custom",
                base_url="https://example.invalid/v1",
                api_key="test-secret",
                api_mode="chat",
            )
            assert engine.threshold_tokens == int(200_000 * 0.35)
        finally:
            engine.shutdown()

    def test_absolute_suppresses_codex_gpt55_autoraise(self, tmp_path, monkeypatch):
        from hermes_lcm.engine import LCMEngine

        monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", "130000")
        monkeypatch.delenv("LCM_CONTEXT_THRESHOLD", raising=False)

        config = LCMConfig(
            database_path=str(tmp_path / "abs-autoraise.db"),
            codex_gpt55_autoraise_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        try:
            engine.update_model(
                model="gpt-5.5",
                context_length=400_000,
                provider="openai-codex",
                base_url="https://example.invalid/v1",
                api_key="test-secret",
                api_mode="responses",
            )
            assert engine.threshold_tokens == 130_000
            assert engine._config.codex_gpt55_autoraise_enabled is False
        finally:
            engine.shutdown()

    def test_invalid_absolute_env_falls_back_to_ratio(self, tmp_path, monkeypatch):
        from hermes_lcm.engine import LCMEngine

        monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", "not-a-number")
        monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.40")

        config = LCMConfig.from_env()
        config.database_path = str(tmp_path / "bad-abs.db")
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        try:
            engine.update_model(
                model="coding-model",
                context_length=100_000,
                provider="custom",
                base_url="https://example.invalid/v1",
                api_key="test-secret",
                api_mode="chat",
            )
            assert engine.threshold_tokens == int(100_000 * 0.40)
        finally:
            engine.shutdown()
