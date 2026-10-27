"""Registration contract test, not proof of an actual host unload/runtime round trip."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def test_post_llm_callback_uses_owned_host_registration(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    name = "hermes_lcm_owned_post_llm_reconciliation"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    manager = SimpleNamespace(_hooks={})
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", SimpleNamespace(get_plugin_manager=lambda: manager))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "synthetic-home"))

    class OwnedContext:
        def __init__(self):
            self.engine = None
            self.hooks = []
        def register_context_engine(self, engine):
            self.engine = engine
        def register_hook(self, event, callback):
            self.hooks.append((event, callback))

    ctx = OwnedContext()
    try:
        module.register(ctx)
        assert len([callback for event, callback in ctx.hooks if event == "post_llm_call"]) == 1
        assert manager._hooks.get("post_llm_call", []) == []
    finally:
        if ctx.engine is not None:
            ctx.engine.shutdown()
