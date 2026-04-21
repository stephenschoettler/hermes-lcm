from pathlib import Path
import subprocess


def _install_repo_as_user_plugin(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    hermes_home = tmp_path / "hermes-home"
    plugin_dir = hermes_home / "plugins" / "hermes-lcm"
    plugin_dir.parent.mkdir(parents=True, exist_ok=True)
    plugin_dir.symlink_to(repo_root, target_is_directory=True)
    (hermes_home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - hermes-lcm\n"
        "context:\n"
        "  engine: lcm\n",
        encoding="utf-8",
    )
    return hermes_home


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


def test_user_plugin_manager_loads_hermes_lcm_as_context_engine(monkeypatch, tmp_path):
    hermes_home = _install_repo_as_user_plugin(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import plugins as plugins_mod

    manager = plugins_mod.PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["hermes-lcm"]
    assert loaded.enabled
    assert loaded.manifest.source == "user"

    engine = manager._context_engine
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


def test_get_plugin_context_engine_returns_lcm_for_enabled_user_plugin(monkeypatch, tmp_path):
    hermes_home = _install_repo_as_user_plugin(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import plugins as plugins_mod

    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    engine = plugins_mod.get_plugin_context_engine()

    assert engine is not None
    assert engine.name == "lcm"
