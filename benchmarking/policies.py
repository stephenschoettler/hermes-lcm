"""Policy loading for model-aware LCM benchmark replays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .types import LCMPolicy

try:  # pragma: no cover - exercised when PyYAML is installed
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _REPO_ROOT / candidate


def builtin_policies() -> list[LCMPolicy]:
    """Return the zero-config policy set used by the skeleton harness."""
    return [
        LCMPolicy(
            name="baseline_272k",
            context_length=272_000,
            context_threshold=0.75,
            fresh_tail_count=64,
            leaf_chunk_tokens=20_000,
            notes="Current long-context baseline: 64-message fresh tail, 20k leaf chunks.",
        ),
        LCMPolicy(
            name="codex_gpt_long_context",
            context_length=272_000,
            context_threshold=0.75,
            fresh_tail_count=24,
            leaf_chunk_tokens=8_000,
            target_after_compaction=0.55,
            policy_version="1",
            notes="Initial benchmark candidate for GPT/Codex long-context routes.",
        ),
        LCMPolicy(
            name="codex_spark_context",
            context_length=128_000,
            context_threshold=0.75,
            fresh_tail_count=16,
            leaf_chunk_tokens=8_000,
            target_after_compaction=0.55,
            policy_version="1",
            notes="Benchmark candidate for GPT-5.3 Codex Spark / 128k Codex-style routes.",
        ),
        LCMPolicy(
            name="pressure_smoke",
            context_length=300,
            context_threshold=0.30,
            fresh_tail_count=2,
            leaf_chunk_tokens=24,
            target_after_compaction=0.55,
            policy_version="1",
            notes="Small deterministic pressure policy for benchmark chatter/headroom validation only.",
        ),
    ]


def _parse_scalar(raw: str) -> Any:
    value = raw.strip().strip("'\"")
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _minimal_yaml_mapping(text: str) -> dict[str, Any]:
    """Parse the flat key/value YAML shape used by benchmark policies.

    This is deliberately tiny. It keeps policy loading dependency-light while
    still allowing `.yaml` policy files in minimal installs without PyYAML.
    """
    result: dict[str, Any] = {}
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"unsupported policy YAML line {line_no}: {raw_line!r}")
        key, value = line.split(":", 1)
        result[key.strip()] = _parse_scalar(value)
    return result


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is not None:
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"policy file must contain a mapping: {path}")
        return loaded
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = _minimal_yaml_mapping(text)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"policy file must contain a mapping: {path}")
    return loaded


def load_policy(path: str | Path) -> LCMPolicy:
    """Load one policy from JSON or flat YAML."""
    return LCMPolicy.from_dict(_load_mapping(_resolve_path(path)))


def load_policies(paths: Iterable[str | Path] | None = None) -> list[LCMPolicy]:
    """Load policies from paths, or return built-ins when no paths are supplied."""
    selected = list(paths or [])
    if not selected:
        return builtin_policies()
    return [load_policy(path) for path in selected]
