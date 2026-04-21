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


def test_plugin_entrypoint_registers_lcm_context_engine():
    engine = _register_plugin_engine("hermes_lcm_packaging_entrypoint")

    assert engine is not None
    assert engine.name == "lcm"

    tool_names = {schema["name"] for schema in engine.get_tool_schemas()}
    assert {
        "lcm_grep",
        "lcm_describe",
        "lcm_expand",
        "lcm_expand_query",
        "lcm_status",
        "lcm_doctor",
    }.issubset(tool_names)


def test_plugin_entrypoint_registration_is_repeatable_and_returns_lcm_engine():
    engine = _register_plugin_engine("hermes_lcm_packaging_entrypoint_repeat")

    assert engine is not None
    assert engine.name == "lcm"
