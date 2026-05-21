"""Serializable types for deterministic LCM benchmark replays."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(frozen=True)
class LCMPolicy:
    name: str
    context_length: int
    context_threshold: float
    fresh_tail_count: int
    leaf_chunk_tokens: int
    condensation_fanin: int = 4
    incremental_max_depth: int = 1
    dynamic_leaf_chunk_enabled: bool = False
    target_after_compaction: float | None = None
    min_turns_between_compactions: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LCMPolicy":
        return cls(
            name=str(data["name"]),
            context_length=int(data["context_length"]),
            context_threshold=float(data["context_threshold"]),
            fresh_tail_count=int(data["fresh_tail_count"]),
            leaf_chunk_tokens=int(data["leaf_chunk_tokens"]),
            condensation_fanin=int(data.get("condensation_fanin", 4)),
            incremental_max_depth=int(data.get("incremental_max_depth", 1)),
            dynamic_leaf_chunk_enabled=_as_bool(data.get("dynamic_leaf_chunk_enabled", False)),
            target_after_compaction=(
                None
                if data.get("target_after_compaction") is None
                else float(data["target_after_compaction"])
            ),
            min_turns_between_compactions=int(data.get("min_turns_between_compactions", 0)),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class Canary:
    id: str
    value: str
    expected_query: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Canary":
        canary_id = str(data["id"])
        return cls(
            id=canary_id,
            value=str(data["value"]),
            expected_query=str(data.get("expected_query") or canary_id),
        )


@dataclass(frozen=True)
class ReplayFixture:
    name: str
    messages: list[dict[str, Any]]
    canaries: list[Canary] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "messages": list(self.messages),
            "canaries": [canary.to_dict() for canary in self.canaries],
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayFixture":
        return cls(
            name=str(data["name"]),
            messages=[dict(message) for message in data["messages"]],
            canaries=[Canary.from_dict(item) for item in data.get("canaries", [])],
            tags=[str(tag) for tag in data.get("tags", [])],
        )


@dataclass
class ReplayMetrics:
    policy_name: str
    fixture_name: str
    prompt_tokens_before: int
    prompt_tokens_after: int
    threshold_tokens: int
    compression_count: int
    compaction_attempts: int
    post_compaction_headroom_tokens: int
    active_canaries_found: int
    retrieval_canaries_found: int
    total_canaries: int
    failures: list[str] = field(default_factory=list)
    database_path: str = ""
    hermes_home: str = ""
    active_message_count: int = 0
    store_messages: int = 0
    dag_nodes: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayMetrics":
        return cls(
            policy_name=str(data["policy_name"]),
            fixture_name=str(data["fixture_name"]),
            prompt_tokens_before=int(data["prompt_tokens_before"]),
            prompt_tokens_after=int(data["prompt_tokens_after"]),
            threshold_tokens=int(data["threshold_tokens"]),
            compression_count=int(data["compression_count"]),
            compaction_attempts=int(data["compaction_attempts"]),
            post_compaction_headroom_tokens=int(data["post_compaction_headroom_tokens"]),
            active_canaries_found=int(data["active_canaries_found"]),
            retrieval_canaries_found=int(data["retrieval_canaries_found"]),
            total_canaries=int(data["total_canaries"]),
            failures=[str(item) for item in data.get("failures", [])],
            database_path=str(data.get("database_path", "")),
            hermes_home=str(data.get("hermes_home", "")),
            active_message_count=int(data.get("active_message_count", 0)),
            store_messages=int(data.get("store_messages", 0)),
            dag_nodes=int(data.get("dag_nodes", 0)),
            elapsed_ms=float(data.get("elapsed_ms", 0.0)),
        )
