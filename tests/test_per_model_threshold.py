"""Tests for per-model compression threshold overrides in LCM."""

import pytest

from hermes_lcm.config import LCMConfig, _parse_model_thresholds_env


class TestParseModelThresholdsEnv:
    def test_basic_parse(self):
        result = _parse_model_thresholds_env("glm-5.2:0.70,glm-5.2-1M:0.25")
        assert result == {"glm-5.2": 0.70, "glm-5.2-1M": 0.25}

    def test_empty_string(self):
        """An explicitly empty value is a deliberate override -- not an error."""
        assert _parse_model_thresholds_env("") == {}
        assert _parse_model_thresholds_env("   ") == {}

    def test_missing_colon_raises(self):
        """A malformed entry must be rejected loudly, not silently skipped.

        Silently dropping it would make an invalid env var indistinguishable
        from an intentionally empty one, which could silently clear a valid
        YAML ``lcm.model_thresholds`` config (see LCMConfig.from_env).
        """
        with pytest.raises(ValueError, match="badentry"):
            _parse_model_thresholds_env("glm-5.2:0.70,badentry,glm-5.2-1M:0.25")

    def test_invalid_float_raises(self):
        with pytest.raises(ValueError, match="abc"):
            _parse_model_thresholds_env("glm-5.2:0.70,bad:abc")

    def test_out_of_range_and_non_finite_values_raise(self):
        for raw in ("zero:0", "negative:-0.1", "too-high:1.01", "nan:nan", "inf:inf"):
            with pytest.raises(ValueError):
                _parse_model_thresholds_env(raw)

    def test_whitespace_stripped(self):
        result = _parse_model_thresholds_env(" glm-5.2 : 0.70 , glm-5.2-1M : 0.25 ")
        assert result == {"glm-5.2": 0.70, "glm-5.2-1M": 0.25}

    def test_trailing_comma_ignored(self):
        """Stray empty segments from a trailing/leading comma are not errors."""
        result = _parse_model_thresholds_env("glm-5.2:0.70,")
        assert result == {"glm-5.2": 0.70}


class TestLCMConfigModelThresholds:
    def test_default_empty(self):
        c = LCMConfig()
        assert c.model_thresholds == {}

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("LCM_MODEL_THRESHOLDS", "glm-5.2:0.70,glm-5.2-1M:0.25")
        c = LCMConfig.from_env()
        assert c.model_thresholds == {"glm-5.2": 0.70, "glm-5.2-1M": 0.25}

    def test_no_env_keeps_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.delenv("LCM_MODEL_THRESHOLDS", raising=False)
        c = LCMConfig.from_env()
        assert c.model_thresholds == {}

    def test_yaml_skips_invalid_keys_and_values(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "lcm:\n"
            "  model_thresholds:\n"
            "    valid: 0.4\n"
            "    \"\": 0.5\n"
            "    zero: 0\n"
            "    too_high: 1.1\n"
            "    boolean: true\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("LCM_MODEL_THRESHOLDS", raising=False)

        c = LCMConfig.from_env()

        assert c.model_thresholds == {"valid": 0.4}

    def test_empty_env_clears_yaml(self, monkeypatch, tmp_path):
        """An explicit LCM_MODEL_THRESHOLDS="" intentionally clears YAML."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "lcm:\n  model_thresholds:\n    glm-5.2: 0.4\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("LCM_MODEL_THRESHOLDS", "")

        c = LCMConfig.from_env()

        assert c.model_thresholds == {}
        assert c.config_sources["model_thresholds"] == "env:LCM_MODEL_THRESHOLDS"

    def test_invalid_env_does_not_clear_yaml(self, monkeypatch, tmp_path):
        """An invalid (unparseable) env var must never silently clear YAML.

        Regression test: the original implementation treated any env var
        value -- valid, empty, or garbage -- the same way: parse it (garbage
        parses to `{}` since no entry has a ':') and unconditionally replace
        `model_thresholds` with the result. That silently discarded a valid
        YAML config whenever the env var happened to be malformed.
        """
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "lcm:\n  model_thresholds:\n    glm-5.2: 0.4\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("LCM_MODEL_THRESHOLDS", "not-a-valid-entry")

        c = LCMConfig.from_env()

        assert c.model_thresholds == {"glm-5.2": 0.4}
        assert c.config_sources["model_thresholds"] == "config_yaml:lcm.model_thresholds"
        assert any(
            "LCM_MODEL_THRESHOLDS" in warning
            for warning in c.config_source_warnings
        )

    def test_invalid_env_with_no_yaml_leaves_empty_and_warns(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv("LCM_MODEL_THRESHOLDS", "garbage-no-colon")

        c = LCMConfig.from_env()

        assert c.model_thresholds == {}
        assert c.config_sources["model_thresholds"] == "default"
        assert any(
            "LCM_MODEL_THRESHOLDS" in warning
            for warning in c.config_source_warnings
        )


class TestRuntimeContextThreshold:
    """Test that _runtime_context_threshold respects model_thresholds."""

    def _make_engine(self, model_thresholds=None):
        """Build a minimal LCM engine with the given model_thresholds."""
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        config = LCMConfig()
        if model_thresholds:
            config.model_thresholds = model_thresholds

        engine = LCMEngine.__new__(LCMEngine)
        engine._config = config
        engine.model = "glm-5.2"
        engine.provider = ""
        engine._context_threshold_autoraised = None
        return engine

    def test_no_overrides_returns_default(self):
        engine = self._make_engine()
        threshold, source, notice = engine._runtime_context_threshold()
        assert threshold == 0.35
        assert notice is None

    def test_exact_match(self):
        engine = self._make_engine({"glm-5.2": 0.70})
        threshold, source, notice = engine._runtime_context_threshold()
        assert threshold == 0.70
        assert "model_thresholds" in source
        assert notice == {"from": 0.35, "to": 0.70}

    def test_match_is_case_insensitive(self, tmp_path):
        """Model-name matching follows repo convention: case-insensitive.

        config.py normalizes model/provider names with ``strip().lower()``
        elsewhere, so a threshold key written as ``GLM-5.2`` must match a
        route model reported as ``glm-5.2-instant`` (and vice versa).
        """
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "case-insensitive.db"),
                model_thresholds={"GLM-5.2": 0.5},
            )
        )
        try:
            engine.update_model(
                model="glm-5.2-instant",
                provider="test",
                context_length=100_000,
            )
            assert engine.context_threshold == 0.5
            assert engine._context_threshold_source == "model_thresholds:GLM-5.2"
        finally:
            engine.shutdown()

    def test_longest_match_wins(self):
        engine = self._make_engine({"glm-5.2": 0.70, "glm-5.2-1M": 0.25})
        engine.model = "glm-5.2-1M"
        threshold, source, notice = engine._runtime_context_threshold()
        assert threshold == 0.25
        assert "glm-5.2-1M" in source

    def test_no_match_returns_default(self):
        engine = self._make_engine({"claude-sonnet-4": 0.60})
        threshold, source, notice = engine._runtime_context_threshold()
        assert threshold == 0.35
        assert notice is None

    def test_override_with_explicit_model_param(self):
        engine = self._make_engine({"glm-5.2-1M": 0.25})
        threshold, source, notice = engine._runtime_context_threshold(model="glm-5.2-1M")
        assert threshold == 0.25

    def test_override_can_lower(self):
        engine = self._make_engine({"small-model": 0.15})
        engine.model = "small-model"
        threshold, _, _ = engine._runtime_context_threshold()
        assert threshold == 0.15

    def test_override_can_raise(self):
        engine = self._make_engine({"big-model": 0.85})
        engine.model = "big-model"
        threshold, _, _ = engine._runtime_context_threshold()
        assert threshold == 0.85

    def test_update_model_recomputes_live_threshold(self, tmp_path):
        from hermes_lcm.engine import LCMEngine

        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "model-threshold.db"),
                model_thresholds={"small-model": 0.2, "large-model": 0.8},
            )
        )
        try:
            engine.update_model(
                model="small-model",
                provider="test",
                context_length=100_000,
            )
            assert engine.threshold_tokens == 20_000
            assert engine._context_threshold_source == "model_thresholds:small-model"

            engine.update_model(
                model="large-model",
                provider="test",
                context_length=100_000,
            )
            assert engine.threshold_tokens == 80_000
            assert engine._context_threshold_source == "model_thresholds:large-model"
        finally:
            engine.shutdown()


class TestModelOverrideNotLabeledAsAutoraise:
    """A per-model override is a distinct mechanism from codex_gpt55_autoraise.

    Regression coverage: the engine used to report both under the same
    ``context_threshold_autoraised`` status field/attribute, which mislabels
    a per-model override -- especially one that *lowers* the effective
    threshold -- as an "autoraise" (a term that specifically means Hermes
    Agent's Codex gpt-5.5 route-specific threshold raise).
    """

    def test_lowering_override_is_not_reported_as_autoraised(self, tmp_path):
        from hermes_lcm.engine import LCMEngine

        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "model-override-lower.db"),
                model_thresholds={"small-model": 0.15},
            )
        )
        try:
            engine.update_model(
                model="small-model",
                provider="test",
                context_length=100_000,
            )
            assert engine.context_threshold == 0.15
            assert engine._context_threshold_source == "model_thresholds:small-model"
            # Must NOT be surfaced as an autoraise notice.
            assert engine._context_threshold_autoraised is None
            assert engine._context_threshold_model_override == {"from": 0.35, "to": 0.15}

            status = engine.get_status()
            assert status["context_threshold_autoraised"] is None
            assert status["context_threshold_model_override"] == {"from": 0.35, "to": 0.15}
        finally:
            engine.shutdown()

    def test_raising_override_is_also_not_reported_as_autoraised(self, tmp_path):
        from hermes_lcm.engine import LCMEngine

        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "model-override-raise.db"),
                model_thresholds={"big-model": 0.85},
            )
        )
        try:
            engine.update_model(
                model="big-model",
                provider="test",
                context_length=100_000,
            )
            assert engine.context_threshold == 0.85
            assert engine._context_threshold_autoraised is None
            assert engine._context_threshold_model_override == {"from": 0.35, "to": 0.85}
        finally:
            engine.shutdown()

    def test_codex_gpt55_autoraise_still_uses_autoraise_field(self, tmp_path):
        """The genuine autoraise case is unaffected by the new field."""
        from hermes_lcm.engine import LCMEngine

        config = LCMConfig(
            context_threshold=0.68,
            database_path=str(tmp_path / "codex-autoraise-still-works.db"),
        )
        config.config_sources["context_threshold"] = "config_yaml:compression.threshold"
        engine = LCMEngine(config=config)
        try:
            engine.update_model(
                model="gpt-5.5",
                provider="openai-codex",
                context_length=400_000,
            )
            assert engine._context_threshold_source == "codex_gpt55_autoraise"
            assert engine._context_threshold_autoraised == {"from": 0.68, "to": 0.85}
            assert engine._context_threshold_model_override is None

            status = engine.get_status()
            assert status["context_threshold_autoraised"] == {"from": 0.68, "to": 0.85}
            assert status["context_threshold_model_override"] is None
        finally:
            engine.shutdown()
