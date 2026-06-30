"""Tests for /lcm preset inspection and dry-run application."""

from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.presets import get_preset, preset_match_confidence, preset_status_payload


_PRESET_ENV_VARS = (
    "LCM_CONTEXT_THRESHOLD",
    "LCM_FRESH_TAIL_COUNT",
    "LCM_LEAF_CHUNK_TOKENS",
    "LCM_CONDENSATION_FANIN",
    "LCM_INCREMENTAL_MAX_DEPTH",
)


def _clear_preset_env(monkeypatch):
    for key in _PRESET_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


def _engine(tmp_path, *, context_length: int = 272_000) -> LCMEngine:
    config = LCMConfig(database_path=str(tmp_path / "lcm_preset.db"))
    hermes_home = tmp_path / "hermes_home"
    engine = LCMEngine(config=config, hermes_home=str(hermes_home))
    engine._session_id = "preset-session"
    engine._session_platform = "cli"
    engine.context_length = context_length
    engine.threshold_tokens = int(context_length * config.context_threshold)
    return engine


def test_lcm_preset_help_lists_inspection_commands(tmp_path):
    engine = _engine(tmp_path)

    result = handle_lcm_command("help", engine)

    assert "- /lcm preset show" in result
    assert "- /lcm preset suggest" in result
    assert "- /lcm preset apply <name> --dry-run" in result


def test_lcm_preset_show_exposes_codex_provenance_without_mutating_config(tmp_path):
    engine = _engine(tmp_path)
    before = (engine._config.context_threshold, engine._config.fresh_tail_count, engine._config.leaf_chunk_tokens)

    result = handle_lcm_command("preset show codex_gpt_long_context", engine)

    assert "LCM preset show" in result
    assert "preset: codex_gpt_long_context" in result
    assert "policy_version: 1" in result
    assert "benchmark_version: 2" in result
    assert "scrubbed_operator_coding_tool_heavy" in result
    assert "codex_pressure_probe:42:4:1000" in result
    assert "score: 92.941" in result
    assert "baseline_score: 82.941" in result
    assert "policy_path: benchmarks/policies/codex_gpt_long_context.yaml" in result
    assert "operator_config_precedence: explicit preset-managed LCM_* overrides win" in result
    assert "runtime_mutation: no" in result
    assert before == (engine._config.context_threshold, engine._config.fresh_tail_count, engine._config.leaf_chunk_tokens)


def test_lcm_preset_show_exposes_spark_provenance_without_mutating_config(tmp_path, monkeypatch):
    _clear_preset_env(monkeypatch)
    engine = _engine(tmp_path, context_length=128_000)
    before = (engine._config.context_threshold, engine._config.fresh_tail_count, engine._config.leaf_chunk_tokens)

    result = handle_lcm_command("preset show codex_spark_context", engine)

    assert "LCM preset show" in result
    assert "preset: codex_spark_context" in result
    assert "policy_version: 1" in result
    assert "benchmark_version: 2" in result
    assert "scrubbed_operator_chatter_repeated_compaction" in result
    assert "spark_pressure_probe:42:4:1000" in result
    assert "score: 92.941" in result
    assert "baseline_score: 82.941" in result
    assert "policy_path: benchmarks/policies/codex_spark_context.yaml" in result
    assert "large context windows near 128k tokens" in result
    assert "LCM_CONTEXT_THRESHOLD=0.75" in result
    assert "LCM_FRESH_TAIL_COUNT=16" in result
    assert "LCM_LEAF_CHUNK_TOKENS=8000" in result
    assert "runtime_mutation: no" in result
    assert before == (engine._config.context_threshold, engine._config.fresh_tail_count, engine._config.leaf_chunk_tokens)


def test_lcm_preset_suggest_reports_explicit_operator_config_precedence(tmp_path, monkeypatch):
    _clear_preset_env(monkeypatch)
    monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "99")
    engine = _engine(tmp_path)
    engine._config.fresh_tail_count = 99

    result = handle_lcm_command("preset suggest", engine)

    assert "LCM preset suggest" in result
    assert "suggested_preset: codex_gpt_long_context" in result
    assert "reason: context-window match for GPT/Codex candidate; verify provider/model family before applying" in result
    assert "match_confidence: context-only" in result
    assert "confidence_reasons:" in result
    assert "provider/model family was not verified by host metadata" in result
    assert "explicit_overrides: LCM_FRESH_TAIL_COUNT" in result
    assert "LCM_FRESH_TAIL_COUNT: keep explicit value 99 (preset 24)" in result
    assert "note: suggestion only; no live config was changed" in result


def test_lcm_preset_suggest_does_not_lower_codex_gpt55_autoraised_threshold(tmp_path, monkeypatch):
    _clear_preset_env(monkeypatch)
    engine = _engine(tmp_path, context_length=400_000)
    engine._config.context_threshold = 0.68
    engine._config.config_sources["context_threshold"] = "config_yaml:compression.threshold"
    engine.update_model(
        model="gpt-5.5",
        provider="openai-codex",
        context_length=400_000,
    )

    result = handle_lcm_command("preset suggest", engine)
    payload = preset_status_payload(engine, environ={})

    assert "LCM_CONTEXT_THRESHOLD: keep runtime auto-raised value 0.85 (preset 0.75)" in result
    assert "LCM_FRESH_TAIL_COUNT=24" in result
    assert "LCM_LEAF_CHUNK_TOKENS=8000" in result
    context_delta = next(item for item in payload["dry_run_delta"] if item["field"] == "context_threshold")
    assert context_delta == {
        "field": "context_threshold",
        "env": "LCM_CONTEXT_THRESHOLD",
        "action": "keep_runtime_autoraised",
        "current_value": 0.85,
        "preset_value": 0.75,
    }


def test_lcm_preset_suggest_reports_spark_for_128k_context_window(tmp_path, monkeypatch):
    _clear_preset_env(monkeypatch)
    engine = _engine(tmp_path, context_length=128_000)

    result = handle_lcm_command("preset suggest", engine)

    assert "LCM preset suggest" in result
    assert "suggested_preset: codex_spark_context" in result
    assert "reason: context-window match for GPT/Codex Spark candidate; verify provider/model family before applying" in result
    assert "match_confidence: context-only" in result
    assert "LCM_CONTEXT_THRESHOLD=0.75" in result
    assert "LCM_FRESH_TAIL_COUNT=16" in result
    assert "LCM_LEAF_CHUNK_TOKENS=8000" in result
    assert "note: suggestion only; no live config was changed" in result


def test_lcm_preset_suggest_reports_benchmark_backed_route_confidence_when_host_metadata_matches(tmp_path, monkeypatch):
    _clear_preset_env(monkeypatch)
    engine = _engine(tmp_path, context_length=128_000)
    engine.update_model(
        model="gpt-5.3-codex-spark",
        provider="openai-codex",
        context_length=128_000,
    )

    result = handle_lcm_command("preset suggest", engine)
    payload = preset_status_payload(engine, environ={})

    assert "suggested_preset: codex_spark_context" in result
    assert "match_confidence: benchmark-backed-route" in result
    assert "provider=openai-codex; model=gpt-5.3-codex-spark" in result
    assert payload["match_confidence"] == "benchmark-backed-route"
    assert any("benchmark evidence: score=92.941" in reason for reason in payload["confidence_reasons"])


def test_lcm_preset_suggest_keeps_non_spark_gpt5_128k_context_only(tmp_path, monkeypatch):
    _clear_preset_env(monkeypatch)
    engine = _engine(tmp_path, context_length=128_000)
    engine.update_model(
        model="gpt-5",
        provider="openai-codex",
        context_length=128_000,
    )

    result = handle_lcm_command("preset suggest", engine)
    payload = preset_status_payload(engine, environ={})

    assert "suggested_preset: codex_spark_context" in result
    assert "match_confidence: context-only" in result
    assert "provider/model family was not verified by host metadata" in result
    assert payload["match_confidence"] == "context-only"
    assert any("provider/model family was not verified" in reason for reason in payload["confidence_reasons"])


def test_lcm_preset_confidence_does_not_treat_spark_route_as_long_context_match(tmp_path, monkeypatch):
    _clear_preset_env(monkeypatch)
    engine = _engine(tmp_path, context_length=272_000)
    engine.update_model(
        model="gpt-5.3-codex-spark",
        provider="openai-codex",
        context_length=272_000,
    )

    assert preset_match_confidence(engine, get_preset("codex_gpt_long_context")) == "context-only"


def test_lcm_preset_suggest_declines_unbenchmarked_context_window(tmp_path):
    engine = _engine(tmp_path, context_length=32_000)

    result = handle_lcm_command("preset suggest", engine)

    assert "LCM preset suggest" in result
    assert "suggested_preset: (none)" in result
    assert "reason: no shipped benchmarked preset matches context_length 32000" in result
    assert "note: run deterministic benchmarks before promoting a runtime preset" in result


def test_lcm_preset_apply_requires_dry_run_and_does_not_mutate_config(tmp_path, monkeypatch):
    _clear_preset_env(monkeypatch)
    engine = _engine(tmp_path)
    before = (engine._config.context_threshold, engine._config.fresh_tail_count, engine._config.leaf_chunk_tokens)

    denied = handle_lcm_command("preset apply codex_gpt_long_context", engine)
    dry_run = handle_lcm_command("preset apply codex_gpt_long_context --dry-run", engine)

    assert "LCM preset apply" in denied
    assert "status: denied" in denied
    assert "error: preset apply is preview-only for now; pass --dry-run" in denied
    assert "LCM preset apply" in dry_run
    assert "status: dry-run" in dry_run
    assert "preset: codex_gpt_long_context" in dry_run
    assert "would_set:" in dry_run
    assert "LCM_CONTEXT_THRESHOLD=0.75" in dry_run
    assert "LCM_FRESH_TAIL_COUNT=24" in dry_run
    assert "LCM_LEAF_CHUNK_TOKENS=8000" in dry_run
    assert "unsupported_runtime_fields: target_after_compaction=0.55" in dry_run
    assert "note: no live config was changed" in dry_run
    assert before == (engine._config.context_threshold, engine._config.fresh_tail_count, engine._config.leaf_chunk_tokens)


def test_lcm_preset_apply_spark_dry_run_uses_benchmarked_128k_values(tmp_path, monkeypatch):
    _clear_preset_env(monkeypatch)
    engine = _engine(tmp_path, context_length=128_000)
    before = (engine._config.context_threshold, engine._config.fresh_tail_count, engine._config.leaf_chunk_tokens)

    result = handle_lcm_command("preset apply codex_spark_context --dry-run", engine)

    assert "LCM preset apply" in result
    assert "status: dry-run" in result
    assert "preset: codex_spark_context" in result
    assert "LCM_CONTEXT_THRESHOLD=0.75" in result
    assert "LCM_FRESH_TAIL_COUNT=16" in result
    assert "LCM_LEAF_CHUNK_TOKENS=8000" in result
    assert "unsupported_runtime_fields: target_after_compaction=0.55" in result
    assert before == (engine._config.context_threshold, engine._config.fresh_tail_count, engine._config.leaf_chunk_tokens)


def test_lcm_preset_apply_dry_run_keeps_explicit_managed_env_values(tmp_path, monkeypatch):
    _clear_preset_env(monkeypatch)
    monkeypatch.setenv("LCM_LEAF_CHUNK_TOKENS", "12000")
    engine = _engine(tmp_path)
    engine._config.leaf_chunk_tokens = 12_000

    result = handle_lcm_command("preset apply codex_gpt_long_context --dry-run", engine)

    assert "status: dry-run" in result
    assert "LCM_LEAF_CHUNK_TOKENS: keep explicit value 12000 (preset 8000)" in result
    assert "LCM_FRESH_TAIL_COUNT=24" in result
    assert engine._config.leaf_chunk_tokens == 12_000


def test_lcm_preset_dry_run_does_not_honor_invalid_env_overrides(tmp_path, monkeypatch):
    _clear_preset_env(monkeypatch)
    monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "abc")
    engine = _engine(tmp_path)

    suggest = handle_lcm_command("preset suggest", engine)
    apply = handle_lcm_command("preset apply codex_gpt_long_context --dry-run", engine)

    assert "explicit_overrides: (none)" in suggest
    assert "invalid_overrides: LCM_FRESH_TAIL_COUNT=abc" in suggest
    for result in (suggest, apply):
        assert "LCM_FRESH_TAIL_COUNT=24 (invalid current value abc ignored by runtime; runtime value 32)" in result
        assert "keep explicit value abc" not in result
