from datetime import datetime, timezone
from pathlib import Path
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import types


EXPECTED_LCM_TOOLS = {
    "lcm_grep",
    "lcm_recall",
    "lcm_query_state",
    "lcm_compute",
    "lcm_compile_evidence",
    "lcm_evidence_pack",
    "lcm_retrieve",
    "lcm_recent",
    "lcm_load_session",
    "lcm_describe",
    "lcm_expand",
    "lcm_expand_query",
    "lcm_status",
    "lcm_inspect",
    "lcm_doctor",
}


def _load_plugin_entrypoint_module(module_name: str):
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(repo_root / "__init__.py"),
        submodule_search_locations=[str(repo_root)],
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_agent_context_engine_importable(monkeypatch):
    """Install the smallest host stub needed for isolated plugin tests."""
    agent_module = sys.modules.get("agent")
    if agent_module is None:
        agent_module = types.ModuleType("agent")
        agent_module.__path__ = []
        monkeypatch.setitem(sys.modules, "agent", agent_module)
    context_engine_module = types.ModuleType("agent.context_engine")

    class ContextEngine:
        def on_session_reset(self):
            return None

    context_engine_module.ContextEngine = ContextEngine
    monkeypatch.setitem(sys.modules, "agent.context_engine", context_engine_module)
    monkeypatch.setattr(
        agent_module,
        "context_engine",
        context_engine_module,
        raising=False,
    )


def _register_plugin_engine(module_name: str):
    module = _load_plugin_entrypoint_module(module_name)

    class _Ctx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
            pass

    ctx = _Ctx()
    module.register(ctx)
    return ctx.engine


def _register_plugin_with_command(
    monkeypatch,
    tmp_path,
    module_name: str,
    session_values: dict[str, str],
):
    """Register the plugin against a host-shaped slash-command context."""
    manager = types.SimpleNamespace(_hooks={})
    fake_plugins = types.SimpleNamespace(get_plugin_manager=lambda: manager)
    fake_hermes_cli = types.SimpleNamespace(plugins=fake_plugins)
    fake_session_context = types.SimpleNamespace(
        get_session_env=lambda name, default="": session_values.get(name, default)
    )
    fake_gateway = types.SimpleNamespace(session_context=fake_session_context)
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", fake_plugins)
    monkeypatch.setitem(sys.modules, "gateway", fake_gateway)
    monkeypatch.setitem(sys.modules, "gateway.session_context", fake_session_context)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / module_name))
    monkeypatch.setenv("LCM_ENABLE_SLASH_COMMAND", "1")

    module = _load_plugin_entrypoint_module(module_name)

    class _Ctx:
        def __init__(self):
            self.engine = None
            self.commands = {}

        def register_context_engine(self, engine):
            self.engine = engine

        def register_command(self, name, handler, description=""):
            self.commands[name] = handler

    ctx = _Ctx()
    module.register(ctx)
    return ctx


def _slash_status_fields(text: str) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for line in text.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }


def test_standalone_install_scripts_exist_and_are_shell_scripts():
    repo_root = Path(__file__).resolve().parent.parent

    install_script = repo_root / "scripts" / "install.sh"
    update_script = repo_root / "scripts" / "update.sh"
    validate_script = repo_root / "scripts" / "validate_release.sh"

    assert install_script.exists(), "scripts/install.sh should exist"
    assert update_script.exists(), "scripts/update.sh should exist"
    assert validate_script.exists(), "scripts/validate_release.sh should exist"
    assert install_script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n")
    assert update_script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n")
    assert validate_script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n")


def test_validate_release_routes_cache_artifacts_outside_checkout():
    repo_root = Path(__file__).resolve().parent.parent
    validate_script = (repo_root / "scripts" / "validate_release.sh").read_text(encoding="utf-8")

    assert "PYTHONPYCACHEPREFIX=\"$OUTPUT_DIR/pycache\"" in validate_script
    assert "PYTEST_ADDOPTS=\"-p no:cacheprovider" in validate_script
    assert "dirty_start=\"$(git status --short" in validate_script
    assert "dirty_end=\"$(git status --short" in validate_script
    assert "validation changed git status" in validate_script
    assert "run_pytest()" in validate_script
    assert "ensure_agent_context_engine_importable()" in validate_script
    assert "run_gate \"focused pytest\" run_pytest" in validate_script
    assert "run_gate \"pytest full\" run_pytest" in validate_script
    assert "run_low_fd_pytest" in validate_script
    assert "ulimit -n 1024 &&" not in validate_script


def test_validate_release_checks_committed_pr_diff_against_origin_main(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    source_script = repo_root / "scripts" / "validate_release.sh"
    true_bin = shutil.which("true")
    assert true_bin is not None

    repo = tmp_path / "repo"
    scripts_dir = repo / "scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(source_script, scripts_dir / "validate_release.sh")

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

    git("init", "-b", "main")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes-test@example.invalid")
    (repo / "README.md").write_text("clean\n", encoding="utf-8")
    git("add", "README.md", "scripts/validate_release.sh")
    git("commit", "-m", "base")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    git("checkout", "-b", "feature")
    (repo / "bad.txt").write_text("committed trailing whitespace  \n", encoding="utf-8")
    git("add", "bad.txt")
    git("commit", "-m", "add bad whitespace")

    output_dir = tmp_path / "validation-output"
    result = subprocess.run(
        ["bash", "scripts/validate_release.sh", "--output", str(output_dir)],
        cwd=repo,
        env={**os.environ, "PYTHON": true_bin},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "FAILED: git diff check (origin/main...HEAD)" in result.stderr
    checklist = output_dir / "validation-checklist.md"
    assert checklist.exists()
    assert "diff_check_range: origin/main...HEAD" in checklist.read_text(encoding="utf-8")


def test_validate_release_checks_last_commit_when_origin_main_missing(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    source_script = repo_root / "scripts" / "validate_release.sh"
    true_bin = shutil.which("true")
    assert true_bin is not None

    repo = tmp_path / "repo"
    scripts_dir = repo / "scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(source_script, scripts_dir / "validate_release.sh")

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

    git("init", "-b", "main")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes-test@example.invalid")
    (repo / "README.md").write_text("clean\n", encoding="utf-8")
    git("add", "README.md", "scripts/validate_release.sh")
    git("commit", "-m", "base")
    (repo / "bad.txt").write_text("committed trailing whitespace  \n", encoding="utf-8")
    git("add", "bad.txt")
    git("commit", "-m", "add bad whitespace")

    output_dir = tmp_path / "validation-output"
    result = subprocess.run(
        ["bash", "scripts/validate_release.sh", "--output", str(output_dir)],
        cwd=repo,
        env={**os.environ, "PYTHON": true_bin},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "FAILED: git diff check (HEAD^...HEAD)" in result.stderr
    checklist = output_dir / "validation-checklist.md"
    assert checklist.exists()
    assert "diff_check_range: HEAD^...HEAD" in checklist.read_text(encoding="utf-8")


def test_plugin_manifest_lists_all_registered_tools():
    repo_root = Path(__file__).resolve().parent.parent
    manifest = (repo_root / "plugin.yaml").read_text(encoding="utf-8")

    for tool_name in EXPECTED_LCM_TOOLS:
        assert f"  - {tool_name}\n" in manifest


def test_install_script_creates_profile_aware_symlink_and_prints_activation_steps(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    hermes_home = tmp_path / "hermes-home"
    env = {
        "HOME": str(tmp_path / "home"),
        "HERMES_HOME": str(hermes_home),
        "HERMES_PROFILE": "sandbox",
    }

    result = subprocess.run(
        ["bash", str(repo_root / "scripts" / "install.sh")],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    target = hermes_home / "profiles" / "sandbox" / "plugins" / "hermes-lcm"
    skill_target = hermes_home / "profiles" / "sandbox" / "skills" / "hermes-lcm"
    assert target.is_symlink()
    assert target.resolve() == repo_root.resolve()
    assert skill_target.is_symlink()
    assert skill_target.resolve() == (repo_root / "skills" / "hermes-lcm").resolve()
    assert "plugins:" in result.stdout
    assert "- hermes-lcm" in result.stdout
    assert "context:" in result.stdout
    assert "engine: lcm" in result.stdout
    assert "Discoverable skill:" in result.stdout
    assert str(skill_target) in result.stdout


def test_install_script_is_idempotent_for_plugin_and_skill_links(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    hermes_home = tmp_path / "hermes-home"
    env = {
        "HOME": str(tmp_path / "home"),
        "HERMES_HOME": str(hermes_home),
    }

    for _ in range(2):
        subprocess.run(
            ["bash", str(repo_root / "scripts" / "install.sh")],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    assert (hermes_home / "plugins" / "hermes-lcm").resolve() == repo_root.resolve()
    assert (hermes_home / "skills" / "hermes-lcm").resolve() == (
        repo_root / "skills" / "hermes-lcm"
    ).resolve()


def test_install_script_preflights_skill_conflict_before_creating_plugin_link(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    hermes_home = tmp_path / "hermes-home"
    skill_target = hermes_home / "skills" / "hermes-lcm"
    skill_target.mkdir(parents=True)
    (skill_target / "SKILL.md").write_text("existing skill\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(repo_root / "scripts" / "install.sh")],
        cwd=repo_root,
        env={
            "HOME": str(tmp_path / "home"),
            "HERMES_HOME": str(hermes_home),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing to replace existing skill path" in result.stderr
    assert not (hermes_home / "plugins" / "hermes-lcm").exists()


def test_install_script_accepts_checkout_already_in_canonical_plugin_path(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    hermes_home = tmp_path / "hermes-home"
    checkout = hermes_home / "plugins" / "hermes-lcm"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "skills" / "hermes-lcm").mkdir(parents=True)
    shutil.copy2(repo_root / "scripts" / "install.sh", checkout / "scripts" / "install.sh")
    (checkout / "skills" / "hermes-lcm" / "SKILL.md").write_text(
        "---\nname: hermes-lcm\ndescription: test\n---\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(checkout / "scripts" / "install.sh")],
        cwd=checkout,
        env={
            "HOME": str(tmp_path / "home"),
            "HERMES_HOME": str(hermes_home),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert checkout.is_dir()
    skill_target = hermes_home / "skills" / "hermes-lcm"
    assert skill_target.is_symlink()
    assert skill_target.resolve() == (checkout / "skills" / "hermes-lcm").resolve()


def test_install_script_refuses_to_replace_existing_non_symlink_path(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    hermes_home = tmp_path / "hermes-home"
    target = hermes_home / "plugins" / "hermes-lcm"
    target.mkdir(parents=True)
    (target / "README.txt").write_text("existing checkout", encoding="utf-8")

    env = {
        "HOME": str(tmp_path / "home"),
        "HERMES_HOME": str(hermes_home),
    }

    result = subprocess.run(
        ["bash", str(repo_root / "scripts" / "install.sh")],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing to replace existing path" in result.stderr
    assert target.is_dir()


def test_lcm_grep_time_filters_use_anyof_not_union_type_arrays():
    engine = _register_plugin_engine("hermes_lcm_schema_shape")
    assert engine is not None
    schemas = {schema["name"]: schema for schema in engine.get_tool_schemas()}
    properties = schemas["lcm_grep"]["parameters"]["properties"]

    for name in ("time_from", "time_to"):
        field = properties[name]
        assert field["anyOf"] == [{"type": "number"}, {"type": "string"}]
        assert "type" not in field


def test_lcm_grep_declares_opt_in_externalized_content_scope():
    engine = _register_plugin_engine("hermes_lcm_externalized_search_schema")
    schema = next(item for item in engine.get_tool_schemas() if item["name"] == "lcm_grep")
    properties = schema["parameters"]["properties"]

    assert properties["content_scope"]["default"] == "history"
    assert properties["content_scope"]["enum"] == ["history", "externalized", "both"]
    assert properties["externalized_refs"]["maxItems"] == 256


def test_plugin_entrypoint_registers_lcm_context_engine():
    engine = _register_plugin_engine("hermes_lcm_packaging_entrypoint")

    assert engine is not None
    assert engine.name == "lcm"
    identity = engine.get_status()["runtime_identity"]
    repo_root = Path(__file__).resolve().parent.parent
    assert identity["plugin_name"] == "hermes-lcm"
    assert identity["plugin_version"] == "0.21.0-rc2"
    assert Path(identity["plugin_path"]) == repo_root
    assert identity["database_path_source"] in {"config.database_path", "hermes_home", "default_home"}
    assert identity["plugin_git_commit"]
    assert identity["plugin_git_commit"] == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    assert "plugin_git_dirty" in identity

    tool_names = {schema["name"] for schema in engine.get_tool_schemas()}
    assert EXPECTED_LCM_TOOLS.issubset(tool_names)


def test_plugin_entrypoint_registers_declared_lcm_tools():
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_tool_registration")
    registered = []

    class _Ctx:
        context_engine_tool_handlers_receive_messages = True

        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
            registered.append(
                {
                    "name": name,
                    "toolset": toolset,
                    "schema": schema,
                    "handler": handler,
                    "description": description,
                    "emoji": emoji,
                }
            )

    ctx = _Ctx()
    module.register(ctx)

    assert ctx.engine is not None
    assert {entry["name"] for entry in registered} == EXPECTED_LCM_TOOLS
    assert {entry["toolset"] for entry in registered} == {"context_engine"}
    for entry in registered:
        assert entry["schema"]["name"] == entry["name"]
        assert entry["description"] == entry["schema"].get("description", "")
        assert callable(entry["handler"])


def test_plugin_entrypoint_skips_registered_lcm_tools_without_message_forwarding():
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_tool_registration_unsafe_host")
    registered = {}

    class _HermesAgentLikeCtx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
            registered[name] = {
                "handler": handler,
                "toolset": toolset,
            }

        def registry_dispatch(self, name, args):
            return registered[name]["handler"](
                args,
                task_id="task-1",
                user_task="find current turn",
            )

    ctx = _HermesAgentLikeCtx()
    module.register(ctx)

    assert ctx.engine is not None
    assert registered == {}
    assert EXPECTED_LCM_TOOLS.issubset({schema["name"] for schema in ctx.engine.get_tool_schemas()})


def test_capability_false_host_log_describes_expected_path_b_fallback(caplog):
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_expected_path_b_fallback")
    registered = []

    class _HermesAgentV016LikeCtx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
            registered.append(name)

    ctx = _HermesAgentV016LikeCtx()
    caplog.set_level(logging.INFO)

    module.register(ctx)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert ctx.engine is not None
    assert registered == []
    assert EXPECTED_LCM_TOOLS.issubset({schema["name"] for schema in ctx.engine.get_tool_schemas()})
    assert "LCM tools are available through context-engine schemas" in messages
    assert "expected Path B fallback" in messages
    assert "tool registration skipped because" not in messages


def test_register_gracefully_degrades_when_host_lacks_register_tool():
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_no_register_tool")

    class _CtxNoTool:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

    ctx = _CtxNoTool()
    module.register(ctx)

    assert ctx.engine is not None
    assert ctx.engine.name == "lcm"


def test_plugin_entrypoint_registers_bundled_skill_and_active_lcm_recall_policy(tmp_path, monkeypatch):
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_skill_and_policy")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    registered_skills = []
    hooks = {}

    class _Ctx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_skill(self, name, path, description=""):
            registered_skills.append((name, Path(path), description))

        def register_hook(self, name, callback):
            hooks.setdefault(name, []).append(callback)

    ctx = _Ctx()
    module.register(ctx)

    assert len(registered_skills) == 1
    name, path, description = registered_skills[0]
    assert name == "hermes-lcm"
    assert path == Path(module.__file__).resolve().parent / "skills" / "hermes-lcm"
    assert (path / "SKILL.md").is_file()
    assert "recall" in description.lower()
    assert len(hooks["pre_llm_call"]) == 1

    policy_hook = hooks["pre_llm_call"][0]
    assert policy_hook(session_id="not-bound") is None

    ctx.engine.on_session_start("active-session", platform="cli")
    first = policy_hook(session_id="active-session")
    second = policy_hook(session_id="active-session")
    assert first == second
    assert first == {"context": module.get_recall_policy()}
    assert "Hermes-LCM Recall Policy" in first["context"]
    assert "lcm_recall" in first["context"]
    assert "lcm_expand_query" in first["context"]
    ctx.engine.shutdown()


def test_pre_llm_hook_disabled_toolset_is_identical_and_routed_adds_exact_session_evidence(
    tmp_path, monkeypatch
):
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_preanswer_hook")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_ENABLED", "true")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    hooks = {}

    class _Ctx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            hooks.setdefault(name, []).append(callback)

    ctx = _Ctx()
    module.register(ctx)
    policy = module.get_recall_policy()
    hook = hooks["pre_llm_call"][0]
    ctx.engine.on_session_start("active-session", platform="cli")
    source_text = "I was preparing to move away from Austin."
    source_id = ctx.engine._store.append(
        "prior-session",
        {
            "role": "user",
            "content": source_text,
            "timestamp": 1_712_275_200,
        },
    )
    current_text = "I moved to Denver and live there now."
    current_id = ctx.engine._store.append(
        "prior-session",
        {"role": "user", "content": current_text, "timestamp": 1_712_275_260},
    )

    disabled = hook(
        session_id="active-session",
        user_message="Where do I live now?",
        enabled_toolsets=[],
    )
    active = hook(
        session_id="active-session",
        user_message="Where do I live now?",
        baseline_refs=[
            {
                "exact_ref": f"lcm:{source_id}:0-{len(source_text)}",
                "quote": source_text,
            }
        ],
        conversation_history=[
            {
                "role": "user",
                "content": "Where do I live now?",
                "timestamp": 1_712_361_600,
            }
        ],
    )

    assert disabled == {"context": policy}
    assert active["context"].startswith(
        policy + "\n\n[Hermes-LCM selective session evidence"
    ), ctx.engine._last_preanswer_evidence_trace
    assert "Denver" in active["context"]
    assert f"lcm:{current_id}:0-{len(current_text)}" in active["context"]
    trace = ctx.engine._last_preanswer_evidence_trace
    assert trace["status"] == "augmented"
    assert trace["provenance"]["selector_calls"] == 0
    assert trace["provenance"]["provider_calls"] == 0
    ctx.engine.shutdown()


def test_pre_llm_hook_ordinary_path_makes_no_recall_or_selector_call(
    tmp_path, monkeypatch
):
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_host_envelope")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_ENABLED", "true")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    hooks = {}

    class _Ctx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            hooks.setdefault(name, []).append(callback)

    ctx = _Ctx()
    module.register(ctx)
    recall_calls = []

    def handle(tool, args, **_kwargs):
        recall_calls.append((tool, args))
        raise AssertionError("ordinary path must make no product call")

    monkeypatch.setattr(ctx.engine, "handle_tool_call", handle)
    ctx.engine.on_session_start("active-session", platform="cli")
    response = hooks["pre_llm_call"][0](
        session_id="active-session",
        user_message="Who owns the Atlas rollout?",
        question_date="2026-07-20",
        enabled_toolsets=["context_engine"],
    )

    assert recall_calls == []
    assert response == {"context": module.get_recall_policy()}
    assert not hasattr(ctx.engine, "_last_preanswer_evidence_trace")
    ctx.engine.shutdown()


def test_pre_llm_hook_routed_path_creates_one_answer_ready_baseline(tmp_path, monkeypatch):
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_selective_baseline")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_ENABLED", "true")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    hooks = {}

    class _Ctx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            hooks.setdefault(name, []).append(callback)

    ctx = _Ctx()
    module.register(ctx)
    lead = "I was preparing the Atlas ownership handoff."
    lead_id = ctx.engine._store.append(
        "prior-session", {"role": "user", "content": lead}
    )
    fact = "Maya took ownership of Atlas on July 15."
    fact_id = ctx.engine._store.append(
        "prior-session", {"role": "user", "content": fact}
    )
    calls = []

    def handle(tool, args, **_kwargs):
        calls.append((tool, args))
        assert tool == "lcm_recall"
        return json.dumps(
            {
                "hits": [
                    {
                        "store_id": lead_id,
                        "content_offset": 0,
                        "content": lead,
                    }
                ]
            }
        )

    monkeypatch.setattr(ctx.engine, "handle_tool_call", handle)
    ctx.engine.on_session_start("active-session", platform="cli")
    response = hooks["pre_llm_call"][0](
        session_id="active-session",
        user_message="When did Maya take ownership of Atlas?",
        question_date="2026-07-20",
        enabled_toolsets=["context_engine"],
    )

    assert len(calls) == 1
    assert calls[0][1]["detail"] == "answer_ready"
    assert f"lcm:{fact_id}:0-{len(fact)}" in response["context"]
    assert ctx.engine._last_preanswer_evidence_trace["status"] == "augmented"
    ctx.engine.shutdown()


def test_pre_llm_hook_selective_compiler_uses_existing_auxiliary_seam_and_fails_open(
    tmp_path, monkeypatch
):
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_selective_compiler")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_ENABLED", "true")
    monkeypatch.setenv("LCM_SELECTIVE_COMPILER_ENABLED", "true")
    monkeypatch.setenv("LCM_SELECTIVE_COMPILER_MODEL", "selector-test-model")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    hooks = {}

    class _Ctx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            hooks.setdefault(name, []).append(callback)

    ctx = _Ctx()
    module.register(ctx)
    first = "The first of the two purchases cost $20."
    second = "The second purchase cost $30."
    # Pin the observation time before the pinned question_date ("2026-07-20"):
    # an unpinned append is observed "now", which crosses the as_of boundary
    # once the wall clock passes the pinned date, so grounding rejects the
    # operands and the hook fails open without the selective-evidence block.
    observed_at = datetime(2026, 7, 19, 9, tzinfo=timezone.utc).timestamp()
    first_id = ctx.engine._store.append(
        "purchase-a", {"role": "user", "content": first, "timestamp": observed_at}
    )
    second_id = ctx.engine._store.append(
        "purchase-b", {"role": "user", "content": second, "timestamp": observed_at}
    )
    refs = [
        {"exact_ref": f"lcm:{first_id}:0-{len(first)}", "quote": first},
        {"exact_ref": f"lcm:{second_id}:0-{len(second)}", "quote": second},
    ]
    selective = importlib.import_module(f"{module.__name__}.selective_compiler")
    calls = []

    def select(prepared, *, model, timeout_seconds):
        calls.append((prepared["request"], model, timeout_seconds))
        return (
            {
                "version": "selective-evidence-selector-v1",
                "selections": [
                    {
                        "handle": "e01",
                        "quote": "$20",
                        "facet": "operand",
                        "value": 20,
                        "unit": "usd",
                    },
                    {
                        "handle": "e02",
                        "quote": "$30",
                        "facet": "operand",
                        "value": 30,
                        "unit": "usd",
                    },
                ],
            },
            {"provider": "test", "model": model, "calls": 1},
        )

    monkeypatch.setattr(selective, "call_selective_auxiliary_selector", select)
    hook = hooks["pre_llm_call"][0]
    ctx.engine.on_session_start("active-session", platform="cli")
    compiled = hook(
        session_id="active-session",
        user_message="What was the total cost of the two purchases?",
        question_date="2026-07-20",
        enabled_toolsets=["context_engine"],
        baseline_refs=refs,
    )

    assert len(calls) == 1, (
        ctx.engine._config.selective_compiler_enabled,
        compiled,
        getattr(ctx.engine, "_last_preanswer_evidence_trace", None),
    )
    assert calls[0][0]["operation"] == "sum"
    assert calls[0][1:] == ("selector-test-model", 8.0)
    assert "lcm-selective-evidence" in compiled["context"]
    assert ctx.engine._last_preanswer_evidence_trace["computation"]["result_value"] == 50

    def fail(*_args, **_kwargs):
        raise TimeoutError("selector deadline")

    monkeypatch.setattr(selective, "call_selective_auxiliary_selector", fail)
    fallback = hook(
        session_id="active-session",
        user_message="What was the total cost of the two purchases?",
        question_date="2026-07-20",
        enabled_toolsets=["context_engine"],
        baseline_refs=refs,
    )
    assert "lcm-selective-evidence" not in fallback["context"]
    assert fallback["context"].startswith(module.get_recall_policy())
    ctx.engine.shutdown()


def test_pre_llm_hook_requirements_mode_uses_compiler_without_selector(
    tmp_path, monkeypatch
):
    _ensure_agent_context_engine_importable(monkeypatch)
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_requirements_v1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_ENABLED", "true")
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_MODE", "requirements_v1")
    monkeypatch.setenv("LCM_SELECTIVE_COMPILER_ENABLED", "true")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    hooks = {}

    class _Ctx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            hooks.setdefault(name, []).append(callback)

    ctx = _Ctx()
    module.register(ctx)
    prompt = "How long is your commute?"
    answer = "My commute is 35 minutes."
    observed_before_question = datetime(
        2026, 7, 19, tzinfo=timezone.utc
    ).timestamp()
    prompt_id = ctx.engine._store.append(
        "prior-session",
        {
            "role": "user",
            "content": prompt,
            "timestamp": observed_before_question,
        },
    )
    answer_id = ctx.engine._store.append(
        "prior-session",
        {
            "role": "assistant",
            "content": answer,
            "timestamp": observed_before_question,
        },
    )
    baseline = [{"exact_ref": f"lcm:{prompt_id}:0-{len(prompt)}", "quote": prompt}]
    product_calls = []

    def handle(tool, args, **_kwargs):
        product_calls.append((tool, args))
        raise AssertionError("adjacent exact closure must not call retrieval")

    monkeypatch.setattr(ctx.engine, "handle_tool_call", handle)
    ctx.engine.on_session_start("active-session", platform="cli")
    response = hooks["pre_llm_call"][0](
        session_id="active-session",
        user_message="How long is my commute?",
        question_date="2026-07-20",
        enabled_toolsets=["context_engine"],
        baseline_refs=baseline,
    )

    assert product_calls == []
    assert "lcm-answer-brief" in response["context"]
    assert "35 minutes" in response["context"]
    assert f"lcm:{answer_id}:" in response["context"]
    trace = ctx.engine._last_preanswer_evidence_trace
    assert trace["state"] == "answer_sufficient"
    assert trace["provenance"]["selector_calls"] == 0
    ctx.engine.shutdown()


def test_pre_llm_hook_renders_internally_recalled_answer_sufficient_fact(
    tmp_path, monkeypatch
):
    _ensure_agent_context_engine_importable(monkeypatch)
    module = _load_plugin_entrypoint_module(
        "hermes_lcm_packaging_requirements_internal_baseline"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_ENABLED", "true")
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_MODE", "requirements_v1")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    hooks = {}

    class _Ctx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            hooks.setdefault(name, []).append(callback)

    ctx = _Ctx()
    module.register(ctx)
    assert ctx.engine is not None
    fact = "You need 15 points to redeem the reward."
    fact_id = ctx.engine._store.append(
        "prior-session", {"role": "user", "content": fact, "timestamp": 100.0}
    )

    def handle(tool, args, **_kwargs):
        assert tool == "lcm_recall"
        return json.dumps(
            {
                "hits": [
                    {
                        "exact_ref": f"lcm:{fact_id}:0-{len(fact)}",
                        "content": fact,
                    }
                ]
            }
        )

    monkeypatch.setattr(ctx.engine, "handle_tool_call", handle)
    ctx.engine.on_session_start("active-session", platform="cli")
    response = hooks["pre_llm_call"][0](
        session_id="active-session",
        user_message="How many points do I need to redeem the reward?",
        enabled_toolsets=["context_engine"],
    )

    assert "lcm-answer-brief" in response["context"]
    assert "15 point" in response["context"]
    assert f"lcm:{fact_id}:9-18" in response["context"]
    trace = ctx.engine._last_preanswer_evidence_trace
    assert trace["state"] == "answer_sufficient"
    assert trace["reason_code"] == "baseline_already_answer_sufficient"
    ctx.engine.shutdown()


def test_pre_llm_hook_requirements_ordinary_and_disabled_toolset_are_byte_identical(
    tmp_path, monkeypatch
):
    _ensure_agent_context_engine_importable(monkeypatch)
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_requirements_noop")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_ENABLED", "true")
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_MODE", "requirements_v1")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    hooks = {}

    class _Ctx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            hooks.setdefault(name, []).append(callback)

    ctx = _Ctx()
    module.register(ctx)

    def fail(*_args, **_kwargs):
        raise AssertionError("no product retrieval is allowed")

    monkeypatch.setattr(ctx.engine, "handle_tool_call", fail)
    ctx.engine.on_session_start("active-session", platform="cli")
    hook = hooks["pre_llm_call"][0]
    policy = module.get_recall_policy()
    ordinary = hook(
        session_id="active-session",
        user_message="Tell me about the Atlas project",
        enabled_toolsets=["context_engine"],
    )
    disabled = hook(
        session_id="active-session",
        user_message="How long is my commute?",
        enabled_toolsets=[],
    )

    assert ordinary == {"context": policy}
    assert disabled == {"context": policy}
    ctx.engine.shutdown()


def test_plugin_entrypoint_gracefully_degrades_without_skill_or_hook_registration(
    tmp_path, monkeypatch
):
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_legacy_guidance_host")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

    class _LegacyCtx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

    ctx = _LegacyCtx()
    module.register(ctx)

    assert ctx.engine is not None
    assert ctx.engine.name == "lcm"
    ctx.engine.shutdown()


def test_register_gracefully_degrades_when_register_tool_hook_raises():
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_register_tool_raises")

    class _CtxRaisesTool:
        context_engine_tool_handlers_receive_messages = True

        def __init__(self):
            self.engine = None
            self.register_tool_calls = []

        def register_context_engine(self, engine):
            self.engine = engine

        def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
            self.register_tool_calls.append(name)
            raise TypeError("host register_tool signature mismatch")

    ctx = _CtxRaisesTool()
    module.register(ctx)

    assert ctx.engine is not None
    assert ctx.engine.name == "lcm"
    assert ctx.register_tool_calls


def test_registered_tool_handlers_route_through_engine_handle_tool_call(monkeypatch):
    module = _load_plugin_entrypoint_module("hermes_lcm_packaging_tool_handler_route")
    registered = {}

    class _Ctx:
        context_engine_tool_handlers_receive_messages = True

        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
            registered[name] = handler

        def registry_dispatch(self, name, args, messages):
            return registered[name](
                args,
                task_id="task-1",
                user_task="find current turn",
                messages=messages,
            )

    ctx = _Ctx()
    module.register(ctx)
    assert ctx.engine is not None
    assert set(registered) == EXPECTED_LCM_TOOLS

    calls = []

    def spy_handle_tool_call(name, args, **kwargs):
        calls.append((name, args, kwargs))
        return f"handled:{name}"

    monkeypatch.setattr(ctx.engine, "handle_tool_call", spy_handle_tool_call)
    messages = [{"role": "user", "content": "find current turn"}]

    for tool_name in registered:
        args = {"query": "current turn"}
        assert ctx.registry_dispatch(tool_name, args, messages) == f"handled:{tool_name}"

    assert {name for name, _, _ in calls} == EXPECTED_LCM_TOOLS
    for name, args, kwargs in calls:
        assert args == {"query": "current turn"}
        assert kwargs["messages"] == messages


def test_git_runtime_identity_preserves_unknown_dirty_state_when_git_probe_fails(tmp_path, monkeypatch):
    module_name = "hermes_lcm_packaging_entrypoint_git_probe_failure"
    _register_plugin_engine(module_name)
    identity_module = sys.modules[f"{module_name}.runtime_identity"]

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)

    def fail_git(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(identity_module.subprocess, "run", fail_git)

    identity = identity_module._git_runtime_identity(checkout)
    assert identity["plugin_git_commit"] == ""
    assert identity["plugin_git_branch"] == ""
    assert identity["plugin_git_dirty"] is None
    assert identity["plugin_git_remote"] == ""


def test_git_runtime_identity_reports_untracked_files_as_dirty(tmp_path, monkeypatch):
    module_name = "hermes_lcm_packaging_entrypoint_git_untracked"
    _register_plugin_engine(module_name)
    identity_module = sys.modules[f"{module_name}.runtime_identity"]

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "untracked.txt").write_text("hi", encoding="utf-8")

    def fake_git(*args, **kwargs):
        cmd = args[0]
        if cmd[-2:] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="?? untracked.txt\n", stderr="")
        if cmd[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        if cmd[-3:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
        if cmd[-4:] == ["config", "--get", "remote.origin.url"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="https://github.com/example/repo.git\n", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(identity_module.subprocess, "run", fake_git)

    identity = identity_module._git_runtime_identity(checkout)

    assert identity["plugin_git_dirty"] is True


def test_plugin_entrypoint_registration_is_repeatable_and_returns_lcm_engine():
    engine = _register_plugin_engine("hermes_lcm_packaging_entrypoint_repeat")

    assert engine is not None
    assert engine.name == "lcm"


def test_register_gracefully_degrades_when_legacy_host_lacks_register_tool():
    """Guard regression: register() must not raise when ctx lacks register_tool."""
    module = _load_plugin_entrypoint_module("hermes_lcm_no_register_tool")

    class _CtxNoTool:
        def __init__(self):
            self.engine = None
        def register_context_engine(self, engine):
            self.engine = engine

    ctx = _CtxNoTool()
    # Must not raise AttributeError on hosts without register_tool
    module.register(ctx)
    assert ctx.engine is not None
    assert ctx.engine.name == "lcm"


def test_register_continues_when_register_tool_raises_type_error():
    """Regression: register() must not abort when register_tool exists but raises TypeError."""
    module = _load_plugin_entrypoint_module("hermes_lcm_type_error_tool")

    class _CtxRaisingTool:
        context_engine_tool_handlers_receive_messages = True

        def __init__(self):
            self.engine = None
        def register_context_engine(self, engine):
            self.engine = engine
        def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
            raise TypeError("host register_tool signature mismatch")

    ctx = _CtxRaisingTool()
    # Must not raise — should log warning and continue
    module.register(ctx)
    assert ctx.engine is not None
    assert ctx.engine.name == "lcm"


def test_registered_tool_handler_forwards_messages_to_engine_handle_tool_call(monkeypatch):
    """Regression: registered tool handlers must forward kwargs (incl. messages=...)
    to engine.handle_tool_call(), preserving equivalent current-turn ingest behavior.

    Note: This test uses a _CtxRecord with **kwargs to simulate a host that
    supports message-forwarding. With the new gating logic, hosts with rigid
    register_tool signatures will NOT have tools registered — they rely on
    the native context-engine path instead.
    """
    module = _load_plugin_entrypoint_module("hermes_lcm_handler_forward")

    registered = {}  # tool_name -> handler

    class _CtxRecord:
        context_engine_tool_handlers_receive_messages = True

        def __init__(self):
            self.engine = None
        def register_context_engine(self, engine):
            self.engine = engine
        def register_tool(self, name, toolset, schema, handler, **kwargs):
            registered[name] = handler

    ctx = _CtxRecord()
    module.register(ctx)
    assert ctx.engine is not None

    # Spy on handle_tool_call
    calls = []
    original_handle = ctx.engine.handle_tool_call
    def spy_handle(name, args, **kwargs):
        calls.append((name, args, kwargs))
        return original_handle(name, args, **kwargs)
    monkeypatch.setattr(ctx.engine, "handle_tool_call", spy_handle)

    # Call each registered handler with messages=... kwarg
    test_messages = [{"role": "user", "content": "test"}]
    for tool_name in ("lcm_grep", "lcm_status", "lcm_inspect", "lcm_doctor"):
        handler = registered.get(tool_name)
        assert handler is not None, f"handler for {tool_name} not registered"
        result = handler({"query": tool_name}, messages=test_messages)
        assert isinstance(result, str), f"{tool_name} handler should return str"
        assert len(result) > 0, f"{tool_name} handler should return non-empty result"

    # Verify handle_tool_call was invoked for each
    called_names = {c[0] for c in calls}
    assert "lcm_grep" in called_names
    assert "lcm_status" in called_names
    assert "lcm_doctor" in called_names

    # Verify messages=... kwarg was forwarded
    for name, args, kwargs in calls:
        assert "messages" in kwargs, f"{name}: messages kwarg not forwarded"
        # Depending on whether engine passes it through, at minimum verify it arrived
        assert kwargs["messages"] == test_messages, f"{name}: messages content mismatch"


def test_slash_status_matches_active_tool_clone_after_rebind_and_side_channel(
    monkeypatch,
    tmp_path,
):
    session_values = {
        "HERMES_SESSION_KEY": "agent:main:discord:dm:42",
    }
    monkeypatch.setenv("LCM_STATELESS_SESSION_PATTERNS", "side-channel")
    ctx = _register_plugin_with_command(
        monkeypatch,
        tmp_path,
        "hermes_lcm_slash_active_clone",
        session_values,
    )
    prototype = ctx.engine
    clone = prototype.clone_for_agent()
    try:
        clone.update_model("gpt-active", 372000, provider="openai-codex")
        clone.on_session_start(
            "discord-session-a",
            platform="discord",
            conversation_id="agent:main:discord:dm:42",
        )

        tool_status = json.loads(clone.handle_tool_call("lcm_status", {}))
        slash_status = _slash_status_fields(ctx.commands["lcm"]("status"))
        assert slash_status["session_id"] == tool_status["session_id"]
        assert (
            slash_status["conversation_id"]
            == tool_status["runtime_identity"]["conversation_id"]
        )
        assert slash_status["model"] == tool_status["model"]
        assert slash_status["provider"] == tool_status["provider"]
        assert int(slash_status["context_length"]) == tool_status["context_length"]
        assert int(slash_status["threshold_tokens"]) == tool_status["threshold_tokens"]
        assert prototype.current_session_id == ""

        clone.on_session_start(
            "discord-session-b",
            platform="discord",
            conversation_id="agent:main:discord:dm:84",
        )
        session_values.update({
            "HERMES_SESSION_KEY": "agent:main:discord:dm:84",
        })
        rebound = _slash_status_fields(ctx.commands["lcm"]("status"))
        assert rebound["session_id"] == "discord-session-b"
        assert rebound["conversation_id"] == "agent:main:discord:dm:84"

        clone.on_session_start(
            "side-channel",
            platform="cron",
            conversation_id="agent:main:cron:tick:1",
        )
        assert clone.bound_session_id == "side-channel"
        assert clone.current_session_id == "discord-session-b"
        side_channel = _slash_status_fields(ctx.commands["lcm"]("status"))
        assert side_channel["session_id"] == "discord-session-b"
        assert side_channel["conversation_id"] == "agent:main:discord:dm:84"
        assert side_channel["side_channel_active"] == "yes"
    finally:
        clone.shutdown()
        prototype.shutdown()


def test_slash_status_stays_unbound_until_lane_has_an_active_runtime(
    monkeypatch,
    tmp_path,
):
    session_values = {
        "HERMES_SESSION_KEY": "agent:main:discord:dm:42",
    }
    ctx = _register_plugin_with_command(
        monkeypatch,
        tmp_path,
        "hermes_lcm_slash_host_context",
        session_values,
    )
    try:
        cold = _slash_status_fields(ctx.commands["lcm"]("status"))
        assert cold["session_id"] == "(unbound)"
        assert cold["model"] == "(uninitialized)"
        assert cold["context_length"] == "(uninitialized)"
        assert cold["threshold_tokens"] == "(uninitialized)"
        assert ctx.engine.current_session_id == ""
    finally:
        ctx.engine.shutdown()


def test_post_llm_hook_resolves_registered_active_clone_without_host_context_compressor(monkeypatch, tmp_path):
    module = _load_plugin_entrypoint_module("hermes_lcm_post_hook_registered_clone")
    manager = types.SimpleNamespace(_hooks={})
    fake_plugins = types.SimpleNamespace(get_plugin_manager=lambda: manager)
    fake_hermes_cli = types.SimpleNamespace(plugins=fake_plugins)
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", fake_plugins)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))

    class _CtxNoTool:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

    ctx = _CtxNoTool()
    module.register(ctx)
    assert ctx.engine is not None

    active_clone = ctx.engine.clone_for_agent()
    active_clone.on_session_start(
        "discord-session",
        platform="discord",
        conversation_id="agent:main:discord:thread:t:t",
    )
    hook = manager._hooks["post_llm_call"][-1]
    history = [{"role": "user", "content": "registered clone canary"}]

    clone_ingests = []
    singleton_ingests = []

    def spy_clone_ingest(messages):
        clone_ingests.append(list(messages))

    def spy_singleton_ingest(messages):
        singleton_ingests.append(list(messages))

    monkeypatch.setattr(active_clone, "ingest", spy_clone_ingest)
    monkeypatch.setattr(ctx.engine, "ingest", spy_singleton_ingest)

    hook(
        session_id="discord-session",
        conversation_id="agent:main:discord:thread:t:t",
        platform="discord",
        conversation_history=history,
    )

    assert clone_ingests == [history]
    assert singleton_ingests == []
    assert active_clone.current_session_id == "discord-session"
    assert active_clone.current_conversation_id == "agent:main:discord:thread:t:t"
    assert ctx.engine.current_session_id == ""
    active_clone.shutdown()
    ctx.engine.shutdown()


def test_post_llm_hook_does_not_foreground_match_side_channel_clone(monkeypatch, tmp_path):
    module = _load_plugin_entrypoint_module("hermes_lcm_post_hook_side_channel_clone")
    manager = types.SimpleNamespace(_hooks={})
    fake_plugins = types.SimpleNamespace(get_plugin_manager=lambda: manager)
    fake_hermes_cli = types.SimpleNamespace(plugins=fake_plugins)
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", fake_plugins)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setenv("LCM_STATELESS_SESSION_PATTERNS", "side-channel")

    class _CtxNoTool:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

    ctx = _CtxNoTool()
    module.register(ctx)
    assert ctx.engine is not None
    active_clone = ctx.engine.clone_for_agent()
    active_clone.on_session_start(
        "foreground-session",
        platform="discord",
        conversation_id="agent:main:discord:dm:42",
    )
    active_clone.on_session_start(
        "side-channel",
        platform="cron",
        conversation_id="agent:main:cron:tick:1",
    )
    assert active_clone.bound_session_id == "side-channel"
    assert active_clone.current_session_id == "foreground-session"

    clone_ingests = []
    singleton_ingests = []
    monkeypatch.setattr(active_clone, "ingest", lambda messages: clone_ingests.append(list(messages)))
    monkeypatch.setattr(ctx.engine, "ingest", lambda messages: singleton_ingests.append(list(messages)))
    history = [{"role": "user", "content": "foreground turn"}]

    manager._hooks["post_llm_call"][-1](
        session_id="foreground-session",
        conversation_id="agent:main:discord:dm:42",
        platform="discord",
        conversation_history=history,
    )

    assert active_clone.bound_session_id == "side-channel"
    assert clone_ingests == []
    assert singleton_ingests == [history]
    active_clone.shutdown()
    ctx.engine.shutdown()


def test_post_llm_hook_ignores_stale_registered_clone_after_rebind(monkeypatch, tmp_path):
    for lookup_mode in ("session", "conversation", "mismatched_conversation"):
        module = _load_plugin_entrypoint_module(f"hermes_lcm_post_hook_stale_{lookup_mode}")
        manager = types.SimpleNamespace(_hooks={})
        fake_plugins = types.SimpleNamespace(get_plugin_manager=lambda: manager)
        fake_hermes_cli = types.SimpleNamespace(plugins=fake_plugins)
        monkeypatch.setitem(sys.modules, "hermes_cli", fake_hermes_cli)
        monkeypatch.setitem(sys.modules, "hermes_cli.plugins", fake_plugins)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / f"hermes_home_{lookup_mode}"))

        class _CtxNoTool:
            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine

        ctx = _CtxNoTool()
        module.register(ctx)
        active_clone = ctx.engine.clone_for_agent()
        active_clone.on_session_start(
            "session-a",
            platform="discord",
            conversation_id="agent:main:discord:thread:a:a",
        )
        active_clone.on_session_start(
            "session-b",
            platform="discord",
            conversation_id="agent:main:discord:thread:b:b",
        )

        clone_ingests = []
        singleton_ingests = []
        monkeypatch.setattr(active_clone, "ingest", lambda messages: clone_ingests.append(list(messages)))
        monkeypatch.setattr(ctx.engine, "ingest", lambda messages: singleton_ingests.append(list(messages)))

        history = [{"role": "user", "content": f"old {lookup_mode}"}]
        kwargs = {
            "conversation_id": "agent:main:discord:thread:a:a",
            "platform": "discord",
            "conversation_history": history,
        }
        if lookup_mode == "session":
            kwargs["session_id"] = "session-a"
        elif lookup_mode == "mismatched_conversation":
            kwargs["session_id"] = "session-b"

        manager._hooks["post_llm_call"][-1](**kwargs)

        if lookup_mode == "mismatched_conversation":
            assert clone_ingests == [history]
            assert singleton_ingests == []
        else:
            assert clone_ingests == []
            assert singleton_ingests == [history]
        assert active_clone.current_session_id == "session-b"
        assert active_clone.current_conversation_id == "agent:main:discord:thread:b:b"

        if lookup_mode == "mismatched_conversation":
            clone_ingests.clear()
            manager._hooks["post_llm_call"][-1](
                session_id="session-b",
                platform="discord",
                conversation_history=[{"role": "user", "content": "session only"}],
            )
            assert clone_ingests == [[{"role": "user", "content": "session only"}]]
        active_clone.shutdown()
        ctx.engine.shutdown()


def test_post_llm_hook_prefers_active_lcm_clone(monkeypatch, tmp_path):
    module = _load_plugin_entrypoint_module("hermes_lcm_post_hook_active_clone")
    manager = types.SimpleNamespace(_hooks={})
    fake_plugins = types.SimpleNamespace(get_plugin_manager=lambda: manager)
    fake_hermes_cli = types.SimpleNamespace(plugins=fake_plugins)
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", fake_plugins)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))

    class _CtxNoTool:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

    ctx = _CtxNoTool()
    module.register(ctx)
    assert ctx.engine is not None

    class _ActiveClone:
        name = "lcm"

        def __init__(self):
            self.current_session_id = ""
            self.current_conversation_id = ""
            self.starts = []
            self.ingested = []

        def on_session_start(self, session_id, **kwargs):
            self.current_session_id = session_id
            self.current_conversation_id = kwargs.get("conversation_id") or session_id
            self.starts.append((session_id, kwargs))

        def ingest(self, history):
            self.ingested.append(list(history))

    active = _ActiveClone()
    hook = manager._hooks["post_llm_call"][-1]
    history = [{"role": "user", "content": "discord lane canary"}]

    hook(
        context_compressor=active,
        session_id="discord-session",
        conversation_id="agent:main:discord:thread:t:t",
        platform="discord",
        conversation_history=history,
    )

    assert active.starts == [
        (
            "discord-session",
            {
                "platform": "discord",
                "conversation_id": "agent:main:discord:thread:t:t",
            },
        )
    ]
    assert active.ingested == [history]
    assert ctx.engine.current_session_id == ""
    ctx.engine.shutdown()


def test_post_llm_hook_rebinds_legacy_singleton_between_gateway_lanes(monkeypatch, tmp_path):
    module = _load_plugin_entrypoint_module("hermes_lcm_post_hook_singleton_rebind")
    manager = types.SimpleNamespace(_hooks={})
    fake_plugins = types.SimpleNamespace(get_plugin_manager=lambda: manager)
    fake_hermes_cli = types.SimpleNamespace(plugins=fake_plugins)
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", fake_plugins)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))

    class _CtxNoTool:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

    ctx = _CtxNoTool()
    module.register(ctx)
    assert ctx.engine is not None
    hook = manager._hooks["post_llm_call"][-1]

    ingests = []

    def spy_ingest(history):
        ingests.append(
            (
                ctx.engine.current_session_id,
                ctx.engine.current_conversation_id,
                ctx.engine.current_session_platform,
                list(history),
            )
        )

    monkeypatch.setattr(ctx.engine, "ingest", spy_ingest)
    hook(
        session_id="discord-topic-a",
        conversation_id="agent:main:discord:thread:a:a",
        platform="discord",
        conversation_history=[{"role": "user", "content": "topic a"}],
    )
    hook(
        session_id="telegram-dm",
        conversation_id="agent:main:telegram:private:1782862480",
        platform="telegram",
        conversation_history=[{"role": "user", "content": "telegram dm"}],
    )

    assert ingests == [
        (
            "discord-topic-a",
            "agent:main:discord:thread:a:a",
            "discord",
            [{"role": "user", "content": "topic a"}],
        ),
        (
            "telegram-dm",
            "agent:main:telegram:private:1782862480",
            "telegram",
            [{"role": "user", "content": "telegram dm"}],
        ),
    ]
    ctx.engine.shutdown()
