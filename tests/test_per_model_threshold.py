"""Tests for per-model compression threshold overrides in LCM."""

from hermes_lcm.config import LCMConfig, _parse_model_thresholds_env


class TestParseModelThresholdsEnv:
    def test_basic_parse(self):
        result = _parse_model_thresholds_env("glm-5.2:0.70,glm-5.2-1M:0.25")
        assert result == {"glm-5.2": 0.70, "glm-5.2-1M": 0.25}

    def test_empty_string(self):
        assert _parse_model_thresholds_env("") == {}

    def test_missing_colon_skipped(self):
        result = _parse_model_thresholds_env("glm-5.2:0.70,badentry,glm-5.2-1M:0.25")
        assert result == {"glm-5.2": 0.70, "glm-5.2-1M": 0.25}

    def test_invalid_float_skipped(self):
        result = _parse_model_thresholds_env("glm-5.2:0.70,bad:abc")
        assert result == {"glm-5.2": 0.70}

    def test_out_of_range_and_non_finite_values_are_skipped(self):
        result = _parse_model_thresholds_env(
            "zero:0,negative:-0.1,too-high:1.01,nan:nan,inf:inf,valid:1.0"
        )
        assert result == {"valid": 1.0}

    def test_whitespace_stripped(self):
        result = _parse_model_thresholds_env(" glm-5.2 : 0.70 , glm-5.2-1M : 0.25 ")
        assert result == {"glm-5.2": 0.70, "glm-5.2-1M": 0.25}


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