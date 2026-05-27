from pathlib import Path
import importlib.util
import subprocess
import sys


EXPECTED_LCM_TOOLS = {
    "lcm_grep",
    "lcm_load_session",
    "lcm_describe",
    "lcm_expand",
    "lcm_expand_query",
    "lcm_status",
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


def test_standalone_install_scripts_exist_and_are_shell_scripts():
    repo_root = Path(__file__).resolve().parent.parent

    install_script = repo_root / "scripts" / "install.sh"
    update_script = repo_root / "scripts" / "update.sh"

    assert install_script.exists(), "scripts/install.sh should exist"
    assert update_script.exists(), "scripts/update.sh should exist"
    assert install_script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n")
    assert update_script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n")


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
    assert target.is_symlink()
    assert target.resolve() == repo_root.resolve()
    assert "plugins:" in result.stdout
    assert "- hermes-lcm" in result.stdout
    assert "context:" in result.stdout
    assert "engine: lcm" in result.stdout


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


def test_plugin_entrypoint_registers_lcm_context_engine():
    engine = _register_plugin_engine("hermes_lcm_packaging_entrypoint")

    assert engine is not None
    assert engine.name == "lcm"
    identity = engine.get_status()["runtime_identity"]
    repo_root = Path(__file__).resolve().parent.parent
    assert identity["plugin_name"] == "hermes-lcm"
    assert identity["plugin_version"] == "0.13.0"
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
    engine_module = sys.modules[f"{module_name}.engine"]

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)

    def fail_git(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(engine_module.subprocess, "run", fail_git)

    identity = engine_module._git_runtime_identity(checkout)
    assert identity["plugin_git_commit"] == ""
    assert identity["plugin_git_branch"] == ""
    assert identity["plugin_git_dirty"] is None
    assert identity["plugin_git_remote"] == ""


def test_git_runtime_identity_reports_untracked_files_as_dirty(tmp_path, monkeypatch):
    module_name = "hermes_lcm_packaging_entrypoint_git_untracked"
    _register_plugin_engine(module_name)
    engine_module = sys.modules[f"{module_name}.engine"]

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "untracked.txt").write_text("hi", encoding="utf-8")

    called = []

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

    monkeypatch.setattr(engine_module.subprocess, "run", fake_git)

    identity = engine_module._git_runtime_identity(checkout)

    assert identity["plugin_git_dirty"] is True


def test_plugin_entrypoint_registration_is_repeatable_and_returns_lcm_engine():
    engine = _register_plugin_engine("hermes_lcm_packaging_entrypoint_repeat")

    assert engine is not None
    assert engine.name == "lcm"


def test_register_gracefully_degrades_when_host_lacks_register_tool():
    """Guard regression: register() must not raise when ctx lacks register_tool."""
    repo_root = Path(__file__).resolve().parent.parent
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
    repo_root = Path(__file__).resolve().parent.parent
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
    for tool_name in ("lcm_grep", "lcm_status", "lcm_doctor"):
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
