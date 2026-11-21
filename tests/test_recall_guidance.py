from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_guidance_module():
    package_name = "hermes_lcm_recall_guidance"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        str(REPO_ROOT / "__init__.py"),
        submodule_search_locations=[str(REPO_ROOT)],
    )
    package = importlib.util.module_from_spec(package_spec)
    assert package_spec is not None
    assert package_spec.loader is not None
    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)
    return __import__(f"{package_name}.guidance", fromlist=["guidance"])


def test_recall_policy_is_canonical_bounded_and_benchmark_neutral():
    guidance = _load_guidance_module()
    first = guidance.get_recall_policy()
    second = guidance.get_recall_policy()

    assert first == second
    assert first == guidance.RECALL_POLICY_PATH.read_text(encoding="utf-8").strip()
    assert 0 < len(first.encode("utf-8")) <= 8 * 1024
    assert hashlib.sha256(first.encode("utf-8")).hexdigest() == guidance.recall_policy_sha256()
    lowered = first.lower()
    for forbidden in ("longmemeval", "question_id", "reference answer", "judge output"):
        assert forbidden not in lowered


def test_recall_policy_covers_evidence_safety_and_stable_tool_routing():
    guidance = _load_guidance_module()
    policy = guidance.get_recall_policy()

    for phrase in (
        "summaries are recall cues",
        "newer source-backed evidence",
        "1-3 distinctive terms",
        "lcm_grep",
        "lcm_describe",
        "lcm_expand_query",
        "lcm_recall",
        "lcm_load_session",
        "lcm_recent",
        "lcm_compile_evidence",
        "lcm_evidence_pack",
        "lcm_compute",
        "include_exact_ref=true",
        "FTS5",
        "requested_facets",
        "session_search",
        "recover exact evidence",
        "State uncertainty",
        "Open-cardinality evidence remains incomplete",
    ):
        assert phrase in policy


def test_bundled_skill_has_valid_minimal_frontmatter_and_matching_tool_references():
    skill_root = REPO_ROOT / "skills" / "hermes-lcm"
    skill_text = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    assert skill_text.startswith("---\nname: hermes-lcm\ndescription:")
    frontmatter = skill_text.split("---", 2)[1]
    keys = {
        line.split(":", 1)[0].strip()
        for line in frontmatter.splitlines()
        if ":" in line
    }
    assert keys == {"name", "description"}

    recall_reference = (skill_root / "references" / "recall-tools.md").read_text(
        encoding="utf-8"
    )
    documented = {
        "lcm_grep",
        "lcm_recall",
        "lcm_recent",
        "lcm_load_session",
        "lcm_describe",
        "lcm_expand",
        "lcm_expand_query",
        "lcm_compute",
        "lcm_compile_evidence",
        "lcm_evidence_pack",
    }
    for tool_name in documented:
        assert f"`{tool_name}`" in recall_reference

    schemas_text = (REPO_ROOT / "schemas.py").read_text(encoding="utf-8")
    for tool_name in documented:
        assert f'"name": "{tool_name}"' in schemas_text
