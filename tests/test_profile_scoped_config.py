"""LCM's config follows the ROUTED profile, not the process the gateway booted as.

The canonical deployment model is ONE gateway per customer serving MANY profiles.
In that shape `os.environ["HERMES_HOME"]` is whatever the process booted with and
never changes, while `get_hermes_home()` follows the context-local override that
`gateway/run.py::_profile_runtime_scope` installs per routed turn.

`_hermes_config_path()` read the ENV. So every routed profile loaded the BOOT
profile's LCM settings -- compaction thresholds, `ignore_session_patterns`,
`assertions_enabled`, and `database_path`, which decides which `lcm.db` the
plugin opens at all.

This is the same seam `hermes_cli/plugins.py::_plugin_manager_scope_key` keys its
per-profile PluginManager on. Every other per-profile resolution already followed
the contextvar; LCM's own config was the one that did not.
"""

from __future__ import annotations

import sys
import types

import pytest

from hermes_lcm import config as lcm_config


@pytest.fixture()
def fake_hermes_cli(monkeypatch):
    """Stand in for `hermes_cli.config.get_hermes_home`.

    The plugin imports it lazily inside the function precisely so it still works
    standalone, so the test installs a module rather than patching an attribute
    that may not exist.
    """
    state = {"home": ""}
    module = types.ModuleType("hermes_cli.config")
    module.get_hermes_home = lambda: state["home"]
    package = types.ModuleType("hermes_cli")
    package.config = module
    monkeypatch.setitem(sys.modules, "hermes_cli", package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", module)
    return state


def test_the_routed_profile_wins_over_the_process_env(monkeypatch, fake_hermes_cli, tmp_path):
    """THE test. Boot as A, route to B, resolve B."""
    boot = tmp_path / "profile-A"
    routed = tmp_path / "profile-B"
    monkeypatch.setenv("HERMES_HOME", str(boot))
    fake_hermes_cli["home"] = str(routed)

    resolved = lcm_config._hermes_config_path()

    assert resolved == routed / "config.yaml", (
        "LCM resolved the BOOT profile's config while serving a different routed "
        "profile -- every profile would share the boot profile's LCM settings"
    )


def test_the_env_is_still_the_fallback_standalone(monkeypatch, tmp_path):
    """Outside a gateway `hermes_cli` may not be importable at all."""
    monkeypatch.setitem(sys.modules, "hermes_cli", None)  # forces ImportError
    monkeypatch.setitem(sys.modules, "hermes_cli.config", None)
    home = tmp_path / "standalone"
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert lcm_config._hermes_config_path() == home / "config.yaml"


def test_an_empty_context_home_falls_back_rather_than_resolving_to_root(
    monkeypatch, fake_hermes_cli, tmp_path
):
    """`Path("")` is `.` -- a blank override must not silently mean the cwd."""
    home = tmp_path / "profile-A"
    monkeypatch.setenv("HERMES_HOME", str(home))
    fake_hermes_cli["home"] = "   "

    assert lcm_config._hermes_config_path() == home / "config.yaml"


def test_a_raising_get_hermes_home_falls_back(monkeypatch, tmp_path):
    """A host that blows up resolving the home must not take LCM down with it."""
    module = types.ModuleType("hermes_cli.config")

    def explode():
        raise RuntimeError("host in a bad state")

    module.get_hermes_home = explode
    package = types.ModuleType("hermes_cli")
    package.config = module
    monkeypatch.setitem(sys.modules, "hermes_cli", package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", module)

    home = tmp_path / "profile-A"
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert lcm_config._hermes_config_path() == home / "config.yaml"


def test_a_pinned_database_path_no_longer_claims_it_rebound(caplog) -> None:
    """It relabels the home and keeps the FILE. Do not call that 'rebound'.

    An explicit `database_path` is an operator override that outranks the
    profile -- correct for a single-profile deployment, an isolation hazard under
    a multiplexed gateway where it collapses every routed profile onto one store.
    The old line logged "LCM rebound Hermes home" at INFO, which is the one thing
    it is not: an operator grepping for a mis-binding would read it as
    confirmation the switch happened.
    """
    import inspect

    from hermes_lcm.engine import LCMEngine

    source = inspect.getsource(LCMEngine._rebind_storage_for_home)
    pinned_branch = source[source.index("if self._config.database_path:"):]
    pinned_branch = pinned_branch[: pinned_branch.index("db_path = self._resolve_db_path")]

    assert "logger.warning" in pinned_branch, (
        "the pinned-database_path branch must warn; it does not switch stores"
    )
    assert "did NOT switch stores" in pinned_branch
    assert "rebound Hermes home" not in pinned_branch, (
        "the misleading success wording is back"
    )
