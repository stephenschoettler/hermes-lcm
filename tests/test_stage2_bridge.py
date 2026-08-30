from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_lcm.store import MessageStore


BRIDGE_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "qa-harness"
    / "src"
    / "providers"
    / "hermes-lcm"
    / "bridge"
    / "hermes_lcm_bridge.py"
)


class _MockEmbedder:
    provider_id = "mock"
    model_id = "mock-model"
    dim = 2
    last_usage_tokens = 0

    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 0.0]


def _load_bridge_module():
    spec = importlib.util.spec_from_file_location(
        "stage2_real_hermes_lcm_bridge", BRIDGE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_stdout = sys.stdout
    try:
        spec.loader.exec_module(module)
    finally:
        # The bridge redirects library chatter away from its JSONL protocol.
        # Importing it in-process for this probe must not retain that redirect.
        sys.stdout = original_stdout
    return module


def _real_bridge_results(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setenv("LCM_SESSION_EXPAND_V1", "true")
    monkeypatch.setenv("LCM_SESSION_EXPAND_V1_PER_SESSION_TOKENS", "2000")
    module = _load_bridge_module()
    bridge = module.Bridge.__new__(module.Bridge)
    bridge.repo_root = Path(__file__).parents[1]
    bridge.workdir = tmp_path
    bridge.provider_name = "mock"
    bridge.model = "mock-model"
    bridge.embedder = _MockEmbedder()
    bridge.dim = 2
    bridge._order = {}

    container_tag = "stage2-wire-probe"
    db_path = bridge._db_path(container_tag)
    config = bridge._config(db_path)
    assert config.session_expand_v1 is True
    store = MessageStore(str(db_path), ingest_protection_config=config)
    try:
        for turn in range(5):
            store.append(
                "session-a",
                {
                    "role": "user" if turn % 2 == 0 else "assistant",
                    "content": (
                        f"BRIDGE-EXPANSION-SENTINEL turn {turn}. "
                        "kanban dashboard sprint exact evidence. "
                        + "context " * 12
                    ),
                },
                source="benchmark",
            )
    finally:
        store.close()
    bridge._dates_path(container_tag).write_text(
        json.dumps({"session-a": "2024-02-01"}),
        encoding="utf-8",
    )

    return bridge.search(
        {
            "containerTag": container_tag,
            "query": "kanban dashboard sprint exact evidence",
            "limit": 1,
        }
    )


def test_real_bridge_preserves_expansion_past_ranked_limit(
    tmp_path, monkeypatch
):
    response = _real_bridge_results(tmp_path, monkeypatch)

    assert response["ranked_limit"] == 1
    assert response["delivered_result_count"] > response["ranked_limit"]
    assert response["delivered_additional_result_count"] > 0
    expanded = [
        result
        for result in response["results"]
        if result["metadata"]["session_expanded"]
    ]
    assert expanded
    assert all(
        "BRIDGE-EXPANSION-SENTINEL" in result["content"]
        for result in expanded
    )
    assert all(result["metadata"]["exact_ref"] for result in expanded)


def test_real_bridge_flag_off_uses_historical_recall_mode(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LCM_SESSION_EXPAND_V1", raising=False)
    module = _load_bridge_module()
    bridge = module.Bridge.__new__(module.Bridge)
    bridge.repo_root = Path(__file__).parents[1]
    bridge.workdir = tmp_path
    bridge.provider_name = "mock"
    bridge.model = "mock-model"
    bridge.embedder = _MockEmbedder()
    bridge.dim = 2
    bridge._order = {}

    container_tag = "stage2-flag-off-wire-probe"
    db_path = bridge._db_path(container_tag)
    config = bridge._config(db_path)
    assert config.session_expand_v1 is False
    store = MessageStore(str(db_path), ingest_protection_config=config)
    store.close()

    import hermes_lcm.tools as lcm_tools

    captured = {}

    def fake_recall(args, *, engine):
        captured.update(args)
        assert engine._config.session_expand_v1 is False
        return json.dumps({"hits": []})

    monkeypatch.setattr(lcm_tools, "lcm_recall", fake_recall)

    response = bridge.search(
        {
            "containerTag": container_tag,
            "query": "historical control query",
            "limit": 7,
        }
    )

    assert captured == {
        "query": "historical control query",
        "limit": 7,
    }
    assert response["results"] == []
    assert response["delivered_additional_result_count"] == 0


def test_real_bridge_expansion_reaches_build_evidence_card_answer_prompt(
    tmp_path, monkeypatch
):
    harness_root = os.environ.get("HERMES_MB_HARNESS_REPO")
    if not harness_root:
        pytest.skip(
            "set HERMES_MB_HARNESS_REPO to the full memorybench checkout "
            "for the real buildEvidenceCardAnswerPrompt seam probe"
        )
    prompt_module = (
        Path(harness_root).resolve() / "src" / "prompts" / "evidence-cards.ts"
    )
    if not prompt_module.is_file():
        pytest.fail(f"evidence-card prompt module not found: {prompt_module}")

    response = _real_bridge_results(tmp_path, monkeypatch)
    context_path = tmp_path / "bridge-results.json"
    context_path.write_text(
        json.dumps(response["results"], ensure_ascii=False),
        encoding="utf-8",
    )
    script = f"""
import {{ readFileSync }} from "node:fs";
const {{ buildEvidenceCardAnswerPrompt }} = await import({json.dumps(prompt_module.as_uri())});
const context = JSON.parse(readFileSync({json.dumps(str(context_path))}, "utf8"));
const result = buildEvidenceCardAnswerPrompt("What is the exact evidence?", context, "2024-02-02");
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["bun", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    rendered = json.loads(completed.stdout)

    assert "BRIDGE-EXPANSION-SENTINEL" in rendered["prompt"]
    assert (
        rendered["presentation"]["renderedItems"]
        == response["delivered_result_count"]
    )
    assert rendered["presentation"]["unresolvedExactRefs"] == 0
