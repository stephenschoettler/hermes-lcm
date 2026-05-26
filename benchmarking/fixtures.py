"""Fixture loading and deterministic fixture generation for LCM benchmarks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping

from .types import ReplayFixture, SummaryFailureMode, _summary_failure_mode


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAX_SYNTHETIC_MESSAGE_PAIRS = 250
_MAX_SYNTHETIC_FILLER_WORDS = 2_000


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _REPO_ROOT / candidate


def fixture_from_dict(data: Mapping[str, object]) -> ReplayFixture:
    missing = [key for key in ("name", "messages") if key not in data]
    if missing:
        raise ValueError(f"fixture missing required key(s): {', '.join(missing)}")
    if not isinstance(data["messages"], list):
        raise ValueError("fixture messages must be a list")
    return ReplayFixture.from_dict(data)


def load_fixture(path: str | Path) -> ReplayFixture:
    """Load one benchmark fixture JSON file."""
    fixture_path = _resolve_path(path)
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"fixture must contain a JSON object: {fixture_path}")
    return fixture_from_dict(data)


def load_fixtures(paths: Iterable[str | Path], synthetic_specs: Iterable[str] | None = None) -> list[ReplayFixture]:
    fixtures = [load_fixture(path) for path in paths]
    fixtures.extend(parse_synthetic_fixture_spec(spec) for spec in (synthetic_specs or []))
    return fixtures


def parse_synthetic_fixture_spec(spec: str) -> ReplayFixture:
    """Parse `name:pairs:canaries:filler_words` into a deterministic fixture."""
    parts = spec.split(":")
    if len(parts) != 4 or not parts[0].strip():
        raise ValueError("synthetic fixture spec must be name:pairs:canaries:filler_words")
    name = parts[0].strip()
    try:
        message_pairs = int(parts[1])
        canary_count = int(parts[2])
        filler_words = int(parts[3])
    except ValueError as exc:
        raise ValueError("synthetic fixture counts must be integers") from exc
    if message_pairs <= 0:
        raise ValueError("message_pairs must be positive")
    if canary_count < 0:
        raise ValueError("canary_count cannot be negative")
    if canary_count > message_pairs:
        raise ValueError("canary_count cannot exceed message_pairs")
    if message_pairs > _MAX_SYNTHETIC_MESSAGE_PAIRS:
        raise ValueError(f"message_pairs exceeds maximum {_MAX_SYNTHETIC_MESSAGE_PAIRS}")
    if filler_words < 0:
        raise ValueError("filler_words cannot be negative")
    if filler_words > _MAX_SYNTHETIC_FILLER_WORDS:
        raise ValueError(f"filler_words exceeds maximum {_MAX_SYNTHETIC_FILLER_WORDS}")
    return make_synthetic_fixture(
        name=name,
        message_pairs=message_pairs,
        canary_count=canary_count,
        filler_words=filler_words,
    )


def iter_fixture_files(directory: str | Path = "benchmarks/fixtures") -> list[Path]:
    fixture_dir = _resolve_path(directory)
    return sorted(fixture_dir.glob("*.json"))


def _canary_prefix(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return normalized or "SYNTHETIC"


def make_synthetic_fixture(
    *,
    name: str = "synthetic_long_history",
    message_pairs: int = 12,
    canary_count: int = 3,
    filler_words: int = 40,
) -> ReplayFixture:
    """Build a deterministic synthetic long-session fixture.

    The generator avoids randomness so benchmark tests and smoke runs can compare
    exact metrics across policy changes.
    """
    prefix = _canary_prefix(name)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a deterministic LCM benchmark agent."}
    ]
    canaries = []
    filler = " ".join(f"{prefix.lower()}_filler_{idx}" for idx in range(filler_words))
    for idx in range(message_pairs):
        content = f"Turn {idx:04d}. {filler}"
        if idx < canary_count:
            canary_id = f"CANARY_{prefix}_{idx:04d}"
            value = f"VALUE_{prefix}_{idx:04d}"
            content = f"{canary_id} = {value}. {content}"
            canaries.append({"id": canary_id, "value": value, "expected_query": canary_id})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": f"Acknowledged turn {idx:04d}."})
    return ReplayFixture.from_dict({
        "name": name,
        "messages": messages,
        "canaries": canaries,
        "tags": ["synthetic", "deterministic"],
    })


def make_summary_failure_fixture(
    *,
    name: str,
    summary_level: int,
    summary_failure_mode: str | SummaryFailureMode,
    message_pairs: int = 12,
    canary_count: int = 3,
    filler_words: int = 40,
) -> ReplayFixture:
    """Build a synthetic fixture annotated for summary failure-mode benchmarking."""
    fixture = make_synthetic_fixture(
        name=name,
        message_pairs=message_pairs,
        canary_count=canary_count,
        filler_words=filler_words,
    )
    tags = list(dict.fromkeys([*fixture.tags, "summary_failure"]))
    mode = _summary_failure_mode(summary_failure_mode)
    return ReplayFixture(
        name=fixture.name,
        messages=fixture.messages,
        canaries=fixture.canaries,
        tags=tags,
        benchmark_profile={
            "summary_level": int(summary_level),
            "summary_failure_mode": mode.value,
        },
    )
