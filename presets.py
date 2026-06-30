"""Shipped model-family preset metadata and dry-run helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping


@dataclass(frozen=True)
class LCMPreset:
    """Inspectable preset metadata.

    Presets are deliberately metadata and dry-run suggestions for now. They do
    not mutate live config and do not override explicit operator settings.
    """

    name: str
    family: str
    description: str
    policy_path: str
    policy_version: str
    runtime_env: Mapping[str, Any]
    unsupported_runtime_fields: Mapping[str, Any] = field(default_factory=dict)
    applies_to: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def policy_key(self) -> str:
        return f"{self.name}@{self.policy_version}"


_FIELD_ENV = {
    "context_threshold": "LCM_CONTEXT_THRESHOLD",
    "fresh_tail_count": "LCM_FRESH_TAIL_COUNT",
    "leaf_chunk_tokens": "LCM_LEAF_CHUNK_TOKENS",
    "condensation_fanin": "LCM_CONDENSATION_FANIN",
    "incremental_max_depth": "LCM_INCREMENTAL_MAX_DEPTH",
}

_FIELD_PARSERS = {
    "context_threshold": float,
    "fresh_tail_count": int,
    "leaf_chunk_tokens": int,
    "condensation_fanin": int,
    "incremental_max_depth": int,
}

_CODEX_GPT_LONG_CONTEXT = LCMPreset(
    name="codex_gpt_long_context",
    family="GPT/Codex long-context",
    description="Benchmark-backed candidate for GPT/Codex-style long-context routes.",
    policy_path="benchmarks/policies/codex_gpt_long_context.yaml",
    policy_version="1",
    runtime_env={
        "context_threshold": 0.75,
        "fresh_tail_count": 24,
        "leaf_chunk_tokens": 8_000,
    },
    unsupported_runtime_fields={
        "target_after_compaction": 0.55,
    },
    applies_to=(
        "Codex/OpenAI-style long-context routes",
        "large context windows near 272k tokens",
        "workloads where repeated compaction risk matters more than keeping a 64-message fresh tail",
    ),
    provenance={
        "benchmark_version": "2",
        "fixture_suite": [
            "long_history_canaries",
            "repeated_compaction_chatter",
            "summary_timeout_probe",
            "summary_refusal_probe",
            "scrubbed_operator_coding_tool_heavy",
            "scrubbed_operator_chatter_repeated_compaction",
            "codex_pressure_probe:42:4:1000",
            "spark_pressure_probe:42:4:1000",
        ],
        "metric_summary": {
            "score": 92.941,
            "baseline_score": 82.941,
            "retrieval_canary_recall": 1.0,
            "baseline_repeated_compaction_risk_count": 4,
            "candidate_repeated_compaction_risk_count": 0,
            "candidate_min_post_compaction_headroom_tokens": 89_288,
        },
        "evidence": "Fresh-main deterministic suite with scrubbed operator-shape replays; aggregate export omits raw transcript content.",
    },
    notes=(
        "Benchmark-only candidate until the preset surface matures. "
        "No live provider tuning or automatic config mutation is performed."
    ),
)

_CODEX_SPARK_CONTEXT = LCMPreset(
    name="codex_spark_context",
    family="GPT/Codex Spark 128k",
    description="Benchmark-backed candidate for GPT-5.3 Codex Spark / 128k Codex-style routes.",
    policy_path="benchmarks/policies/codex_spark_context.yaml",
    policy_version="1",
    runtime_env={
        "context_threshold": 0.75,
        "fresh_tail_count": 16,
        "leaf_chunk_tokens": 8_000,
    },
    unsupported_runtime_fields={
        "target_after_compaction": 0.55,
    },
    applies_to=(
        "GPT-5.3 Codex Spark on Codex OAuth routes",
        "large context windows near 128k tokens",
        "workloads where Spark's smaller effective window needs more post-compaction headroom",
    ),
    provenance={
        "benchmark_version": "2",
        "fixture_suite": [
            "long_history_canaries",
            "repeated_compaction_chatter",
            "summary_timeout_probe",
            "summary_refusal_probe",
            "scrubbed_operator_coding_tool_heavy",
            "scrubbed_operator_chatter_repeated_compaction",
            "codex_pressure_probe:42:4:1000",
            "spark_pressure_probe:42:4:1000",
        ],
        "metric_summary": {
            "score": 92.941,
            "baseline_score": 82.941,
            "retrieval_canary_recall": 1.0,
            "baseline_repeated_compaction_risk_count": 4,
            "candidate_repeated_compaction_risk_count": 0,
            "candidate_min_post_compaction_headroom_tokens": 26_432,
            "candidate_prompt_tokens_after": 69_568,
        },
        "evidence": "Fresh-main deterministic suite with scrubbed operator-shape replays; Spark min headroom stayed above 25k while avoiding repeated-compaction risk.",
    },
    notes=(
        "Benchmark-only candidate for the 128k Codex Spark route. "
        "No live provider tuning or automatic config mutation is performed."
    ),
)


def shipped_presets() -> list[LCMPreset]:
    """Return the shipped, inspectable preset catalog."""

    return [_CODEX_GPT_LONG_CONTEXT, _CODEX_SPARK_CONTEXT]


def get_preset(name: str | None = None) -> LCMPreset | None:
    """Return a preset by name, or the default shipped preset when omitted."""

    selected = (name or _CODEX_GPT_LONG_CONTEXT.name).strip()
    for preset in shipped_presets():
        if preset.name == selected:
            return preset
    return None


def _parse_override_value(field: str, raw: str) -> Any:
    return _FIELD_PARSERS[field](raw)


def _valid_override_value(field: str, raw: str) -> bool:
    try:
        _parse_override_value(field, raw)
    except (TypeError, ValueError):
        return False
    return True


def explicit_operator_overrides(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return parseable runtime preset fields explicitly set by LCM_* env vars."""

    env = environ if environ is not None else os.environ
    return {
        field: env_var
        for field, env_var in _FIELD_ENV.items()
        if env_var in env and _valid_override_value(field, env[env_var])
    }


def invalid_operator_overrides(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return present but unparsable runtime preset env vars."""

    env = environ if environ is not None else os.environ
    return {
        field: env_var
        for field, env_var in _FIELD_ENV.items()
        if env_var in env and not _valid_override_value(field, env[env_var])
    }


def _current_config_value(config: Any, field: str) -> Any:
    return getattr(config, field, "(unknown)")


def preset_match_confidence(engine: Any, preset: LCMPreset | None = None) -> str:
    """Return an honest confidence label for the dry-run recommendation."""

    if preset is None:
        return "none"
    provider = str(getattr(engine, "provider", "") or "").strip().lower()
    model = str(getattr(engine, "model", "") or "").strip().lower()
    is_spark_route = "gpt-5.3-codex-spark" in model
    if preset.name == "codex_spark_context":
        if provider == "openai-codex" and is_spark_route:
            return "benchmark-backed-route"
        return "context-only"
    if provider == "openai-codex" and not is_spark_route and ("codex" in model or "gpt-5" in model):
        return "benchmark-backed-route"
    return "context-only"


def preset_confidence_reasons(engine: Any, preset: LCMPreset | None, reason: str) -> list[str]:
    """Return concise operator-facing reasons for the confidence label."""

    if preset is None:
        return [reason]
    provider = str(getattr(engine, "provider", "") or "").strip() or "(unknown)"
    model = str(getattr(engine, "model", "") or "").strip() or "(unknown)"
    metric_summary = dict(preset.provenance.get("metric_summary") or {})
    reasons = [
        reason,
        f"provider={provider}; model={model}",
        (
            "benchmark evidence: "
            f"score={metric_summary.get('score', '(unknown)')}, "
            f"retrieval_canary_recall={metric_summary.get('retrieval_canary_recall', '(unknown)')}, "
            f"repeated_compaction_risk_count={metric_summary.get('candidate_repeated_compaction_risk_count', '(unknown)')}"
        ),
        "dry-run only; explicit parseable LCM_* operator overrides are preserved",
    ]
    if preset_match_confidence(engine, preset) == "context-only":
        reasons.append("provider/model family was not verified by host metadata; operator must confirm fit before applying env changes")
    return reasons


def preset_env_diff(
    preset: LCMPreset,
    config: Any,
    *,
    environ: Mapping[str, str] | None = None,
    runtime_context_threshold: float | None = None,
    runtime_context_threshold_source: str = "",
) -> list[str]:
    """Render env-var changes a preset would suggest without applying them."""

    env = environ if environ is not None else os.environ
    explicit = explicit_operator_overrides(env)
    invalid = invalid_operator_overrides(env)
    lines: list[str] = []
    for field, value in preset.runtime_env.items():
        env_var = _FIELD_ENV[field]
        if field in explicit:
            current = _parse_override_value(field, env[env_var])
            lines.append(f"{env_var}: keep explicit value {current} (preset {value})")
        elif field in invalid:
            raw = env.get(env_var, "")
            current = _current_config_value(config, field)
            lines.append(
                f"{env_var}={value} "
                f"(invalid current value {raw} ignored by runtime; runtime value {current})"
            )
        elif (
            field == "context_threshold"
            and runtime_context_threshold_source == "codex_gpt55_autoraise"
            and runtime_context_threshold is not None
            and float(runtime_context_threshold) >= float(value)
        ):
            lines.append(
                f"{env_var}: keep runtime auto-raised value {runtime_context_threshold:g} "
                f"(preset {value})"
            )
        else:
            lines.append(f"{env_var}={value}")
    return lines


def _preset_dry_run_delta(
    preset: LCMPreset,
    config: Any,
    *,
    environ: Mapping[str, str] | None = None,
    runtime_context_threshold: float | None = None,
    runtime_context_threshold_source: str = "",
) -> list[dict[str, Any]]:
    """Return structured preset dry-run actions without mutating runtime state."""

    env = environ if environ is not None else os.environ
    explicit = explicit_operator_overrides(env)
    invalid = invalid_operator_overrides(env)
    delta: list[dict[str, Any]] = []
    for field, preset_value in preset.runtime_env.items():
        env_var = _FIELD_ENV[field]
        current = _current_config_value(config, field)
        if field in explicit:
            delta.append({
                "field": field,
                "env": env_var,
                "action": "keep_explicit",
                "current_value": _parse_override_value(field, env[env_var]),
                "preset_value": preset_value,
            })
        elif field in invalid:
            delta.append({
                "field": field,
                "env": env_var,
                "action": "replace_invalid",
                "invalid_value": env.get(env_var, ""),
                "current_value": current,
                "preset_value": preset_value,
            })
        elif (
            field == "context_threshold"
            and runtime_context_threshold_source == "codex_gpt55_autoraise"
            and runtime_context_threshold is not None
            and float(runtime_context_threshold) >= float(preset_value)
        ):
            delta.append({
                "field": field,
                "env": env_var,
                "action": "keep_runtime_autoraised",
                "current_value": runtime_context_threshold,
                "preset_value": preset_value,
            })
        else:
            delta.append({
                "field": field,
                "env": env_var,
                "action": "set",
                "current_value": current,
                "preset_value": preset_value,
            })
    return delta


def preset_status_payload(
    engine: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return read-only, machine-readable preset suggestion metadata."""

    env = environ if environ is not None else os.environ
    preset, reason = suggest_preset_for_engine(engine)
    explicit = explicit_operator_overrides(env)
    invalid = invalid_operator_overrides(env)
    config = getattr(engine, "_config", None)

    explicit_payload = {
        field: {
            "env": env_var,
            "value": _parse_override_value(field, env[env_var]),
        }
        for field, env_var in explicit.items()
    }
    invalid_payload = {
        field: {
            "env": env_var,
            "value": env.get(env_var, ""),
            "runtime_value": _current_config_value(config, field),
            **(
                {"preset_value": preset.runtime_env[field]}
                if preset is not None and field in preset.runtime_env
                else {}
            ),
        }
        for field, env_var in invalid.items()
    }

    payload: dict[str, Any] = {
        "read_only": True,
        "runtime_mutation": False,
        "reason": reason,
        "match_confidence": preset_match_confidence(engine, preset),
        "confidence_reasons": preset_confidence_reasons(engine, preset, reason),
        "suggested_preset": None,
        "provenance": {},
        "explicit_overrides": explicit_payload,
        "invalid_overrides": invalid_payload,
        "dry_run_delta": [],
    }
    if preset is None:
        return payload

    payload["suggested_preset"] = {
        "name": preset.name,
        "family": preset.family,
        "description": preset.description,
        "policy_version": preset.policy_version,
        "policy_path": preset.policy_path,
        "applies_to": list(preset.applies_to),
        "unsupported_runtime_fields": dict(preset.unsupported_runtime_fields),
        "notes": preset.notes,
    }
    payload["provenance"] = dict(preset.provenance)
    payload["dry_run_delta"] = _preset_dry_run_delta(
        preset,
        config,
        environ=env,
        runtime_context_threshold=getattr(engine, "context_threshold", None),
        runtime_context_threshold_source=getattr(engine, "_context_threshold_source", ""),
    )
    return payload


def suggest_preset_for_engine(engine: Any) -> tuple[LCMPreset | None, str]:
    """Return the safest shipped preset suggestion for the current engine state."""

    context_length = int(getattr(engine, "context_length", 0) or 0)
    if context_length >= 200_000:
        return (
            _CODEX_GPT_LONG_CONTEXT,
            "context-window match for GPT/Codex candidate; verify provider/model family before applying",
        )
    if 110_000 <= context_length < 200_000:
        return (
            _CODEX_SPARK_CONTEXT,
            "context-window match for GPT/Codex Spark candidate; verify provider/model family before applying",
        )
    return None, f"no shipped benchmarked preset matches context_length {context_length}"


def unsupported_runtime_fields_text(preset: LCMPreset) -> str:
    if not preset.unsupported_runtime_fields:
        return "(none)"
    return ", ".join(
        f"{key}={value}" for key, value in sorted(preset.unsupported_runtime_fields.items())
    )
