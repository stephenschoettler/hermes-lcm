from pathlib import Path
import importlib.util
import subprocess
import sys


def _load_plugin_entrypoint_module(module_name: str):
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(repo_root / "__init__.py"),
        submodule_search_locations=[str(repo_root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _register_plugin_engine(module_name: str):
    module = _load_plugin_entrypoint_module(module_name)

    class _Ctx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

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

    expected_tools = {
        "lcm_grep",
        "lcm_load_session",
        "lcm_describe",
        "lcm_expand",
        "lcm_expand_query",
        "lcm_status",
        "lcm_doctor",
    }
    for tool_name in expected_tools:
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
    assert identity["plugin_version"] == "0.11.1"
    assert Path(identity["plugin_path"]) == repo_root
    assert identity["database_path_source"] in {"config.database_path", "hermes_home", "default_home"}
    assert identity["plugin_git_commit"]
    assert identity["plugin_git_commit"] == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    assert "plugin_git_dirty" in identity

    tool_names = {schema["name"] for schema in engine.get_tool_schemas()}
    assert {
        "lcm_grep",
        "lcm_load_session",
        "lcm_describe",
        "lcm_expand",
        "lcm_expand_query",
        "lcm_status",
        "lcm_doctor",
    }.issubset(tool_names)


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
    module_name = "hermes_lcm_packaging_entrypoint_git_untracked_dirty"
    _register_plugin_engine(module_name)
    engine_module = sys.modules[f"{module_name}.engine"]

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)

    def fake_git(args, **kwargs):
        if "status" in args:
            assert "--untracked-files=no" not in args
            return subprocess.CompletedProcess(args, 0, stdout="?? scratch.txt\n", stderr="")
        if args[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")
        if args[-3:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="main\n", stderr="")
        if args[-4:] == ["config", "--get", "remote.origin.url"]:
            return subprocess.CompletedProcess(args, 0, stdout="https://github.com/example/repo.git\n", stderr="")
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(engine_module.subprocess, "run", fake_git)

    identity = engine_module._git_runtime_identity(checkout)

    assert identity["plugin_git_dirty"] is True


def test_plugin_entrypoint_registration_is_repeatable_and_returns_lcm_engine():
    engine = _register_plugin_engine("hermes_lcm_packaging_entrypoint_repeat")

    assert engine is not None
    assert engine.name == "lcm"
