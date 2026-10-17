"""Test configuration for hermes-lcm plugin tests.

Patches the plugin modules so they can be imported both as a package
(relative imports during plugin loading) and directly during testing.
"""
import os
import sys
import importlib
from pathlib import Path

import pytest

# Make the repo root importable (for agent.context_engine etc.)
repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Register the plugin directory as a proper package
plugin_dir = Path(__file__).resolve().parent.parent
pkg_name = "hermes_lcm"

if pkg_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(plugin_dir / "__init__.py"),
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(plugin_dir)]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod
    # Don't exec the module (it tries to register with ctx)
    # Just make submodules importable

    # Register each submodule
    for py_file in plugin_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        sub_name = f"{pkg_name}.{py_file.stem}"
        if sub_name not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(
                sub_name, str(py_file),
                submodule_search_locations=[],
            )
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sub_mod.__package__ = pkg_name
            sys.modules[sub_name] = sub_mod
            setattr(mod, py_file.stem, sub_mod)
            try:
                sub_spec.loader.exec_module(sub_mod)
            except Exception:
                pass  # some modules may fail (e.g. engine needs agent)


@pytest.fixture(autouse=True)
def _isolate_lcm_env(monkeypatch):
    """Run every test against LCMConfig defaults, not the developer's shell.

    Production code reads ambient configuration through ``LCMConfig.from_env()``
    (engine, vector store, and the import/backfill scripts). Hermes exports its
    own ``LCM_*`` settings into any shell that sources ``~/.hermes/.env``, so
    without this fixture the suite silently tests whatever the current machine
    happens to be configured for: the same commit passes or fails depending on
    the environment it runs in, and CI cannot reproduce a local failure.

    Concretely, ``LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED=1`` in the ambient
    env flips whole-message externalization on, so the lossless-import tests
    saw a ``media_payload`` covering the entire message instead of the expected
    payload-only ``ingest_payload``, and failed on content they never set.

    Tests that want a specific value set it explicitly with monkeypatch.setenv.
    """
    for name in [key for key in os.environ if key.startswith("LCM_")]:
        monkeypatch.delenv(name, raising=False)
