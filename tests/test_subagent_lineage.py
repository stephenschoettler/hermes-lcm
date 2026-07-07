"""WS5.7: explicit subagent-lineage signal takes precedence over the frame walk.

Hosts that expose the plugin hook bus fire ``subagent_start`` / ``subagent_stop``
with an explicit ``child_session_id`` -> ``parent_session_id`` linkage. LCM records
that linkage and uses it to identify a subagent session from the host's own signal
instead of walking the call stack and reading private agent attributes. The frame
walk stays as a fallback for hosts that do not fire these hooks.
"""

import importlib.util
import sys
from pathlib import Path

import hermes_lcm.aux_session as aux
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _clear_lineage():
    with aux._SUBAGENT_LINEAGE_LOCK:
        aux._SUBAGENT_LINEAGE_BY_SESSION_ID.clear()


def _load_plugin_module(name: str):
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        name, str(repo_root / "__init__.py"), submodule_search_locations=[str(repo_root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_record_query_stop_roundtrip():
    _clear_lineage()
    aux.record_subagent_start(
        {
            "child_session_id": "child-1",
            "parent_session_id": "parent-1",
            "child_subagent_id": "sa-1",
            "child_role": "reviewer",
        }
    )
    assert aux.explicit_subagent_lineage("child-1") == {
        "parent_session_id": "parent-1",
        "child_subagent_id": "sa-1",
        "parent_subagent_id": "",
        "role": "reviewer",
    }
    assert aux.explicit_subagent_lineage("unknown-session") == {}
    aux.record_subagent_stop({"child_session_id": "child-1"})
    assert aux.explicit_subagent_lineage("child-1") == {}
    _clear_lineage()


def test_missing_child_session_id_is_ignored():
    _clear_lineage()
    aux.record_subagent_start({"child_session_id": "", "parent_session_id": "p"})
    aux.record_subagent_start({})
    assert aux._SUBAGENT_LINEAGE_BY_SESSION_ID == {}


def test_explicit_parent_takes_precedence_over_frame_walk(tmp_path):
    _clear_lineage()
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lineage.db")))

    # No explicit signal and no auxiliary caller frame -> the frame walk yields "".
    assert engine._in_process_parent_session_id({}, session_id="child-9") == ""

    # An explicit subagent_start linkage is used without walking the stack.
    aux.record_subagent_start(
        {"child_session_id": "child-9", "parent_session_id": "parent-9"}
    )
    assert engine._in_process_parent_session_id({}, session_id="child-9") == "parent-9"

    # An explicit kwargs parent_session_id still wins over everything.
    assert (
        engine._in_process_parent_session_id(
            {"parent_session_id": "kw-parent"}, session_id="child-9"
        )
        == "kw-parent"
    )

    # After the subagent stops, detection falls back to the frame walk again.
    aux.record_subagent_stop({"child_session_id": "child-9"})
    assert engine._in_process_parent_session_id({}, session_id="child-9") == ""
    _clear_lineage()


def test_register_subscribes_to_subagent_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    module = _load_plugin_module("hermes_lcm_ws57_hooks")
    captured = {}

    class _Ctx:
        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            captured[name] = callback

    module.register(_Ctx())

    # register() imports the plugin's aux_session submodule (via the engine and
    # the hook wiring), so it is now available under the synthetic package name.
    aux_mod = sys.modules["hermes_lcm_ws57_hooks.aux_session"]
    aux_mod._SUBAGENT_LINEAGE_BY_SESSION_ID.clear()

    assert "subagent_start" in captured
    assert "subagent_stop" in captured

    captured["subagent_start"](
        child_session_id="child-h", parent_session_id="parent-h", child_role="reviewer"
    )
    assert aux_mod.explicit_subagent_lineage("child-h")["parent_session_id"] == "parent-h"

    captured["subagent_stop"](child_session_id="child-h")
    assert aux_mod.explicit_subagent_lineage("child-h") == {}


def test_register_without_hook_bus_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    module = _load_plugin_module("hermes_lcm_ws57_nohooks")

    class _CtxNoHooks:
        def register_context_engine(self, engine):
            self.engine = engine

    # Must not raise on hosts without a plugin hook bus (legacy frame-walk path).
    module.register(_CtxNoHooks())
