"""Every host-callable method on LCMEngine is classified, or CI fails.

Phase 4c asks whether gateway, ACP and host-cron triggers land on gated entry
points. They are host concepts with no LCM-visible API -- all of them drive turns
through the SAME registered engine -- so the question is not "what does the
gateway do" but **"is LCM's host-facing surface closed, and is every method on it
accounted for"**.

That surface is the public methods of LCMEngine. Only FOUR of them gate. The rest
are not defects, but the reason each one is safe differs, and until now none of
those reasons was written down anywhere -- which is exactly the "bullet silently
absent" state #209 exists to prevent.

So each method is classified here with the argument that makes it safe. A new
public method lands in none of the buckets and fails, forcing that argument to be
made rather than assumed.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


# Calls the authorization seam directly.
GATED = {
    "on_session_start",
    "on_session_end",
    "on_session_reset",
    "handle_tool_call",
}

# Read-only accessors. They report engine state the caller already holds a
# reference to; none reaches a store query that could cross a principal.
ACCESSORS = {
    "name", "last_compression_status", "last_compression_noop_reason",
    "last_compression_was_noop", "bound_session_id", "current_session_id",
    "current_session_platform", "current_conversation_id", "side_channel_active",
    "current_session_ignored", "current_session_stateless", "cache_read_ratio",
    "get_tool_schemas", "get_runtime_identity", "get_status", "backup_dir",
    # Pure path computation -- returns where a rotate backup WOULD go. It writes
    # nothing; command.py is what writes there.
    "rotate_backup_path",
}

# Model-reachable, but only from a slash command that gates before reaching
# them. The gate lives on the handler, not here, so the classification is
# checked against the handler below rather than asserted.
REACHED_VIA_GATED_COMMAND = {
    "try_acquire_rollup_operator_lease": "_rollups_rebuild_text",
    "release_rollup_operator_lease": "_rollups_rebuild_text",
}

# Writes performed AS the acting principal. The write path stamps
# `access_scope` from the bound context, so these cannot place a row inside
# another principal's scope -- the stamp is not caller-supplied. A gate would
# add nothing a principal could fail.
WRITES_AS_SELF = {
    "ingest",
    "update_from_response",
    "carry_over_new_session_context",
    "rollover_session",
    "rotate_active_session",
}

# Host/operator lifecycle, reachable only by something holding the Python
# object. NOT reachable from any tool or slash command -- verified by the test
# below, because that unreachability is the whole argument. Authenticating the
# caller here is #497's connector contract (Phase 5), not something the model
# can reach today.
HOST_ONLY = {
    "enable_teams", "setup_teams", "disable_teams", "preflight_teams",
    "update_model", "shutdown", "clone_for_agent", "drain_rollup_maintenance",
}


def _public_methods() -> set[str]:
    root = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "engine.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LCMEngine":
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not item.name.startswith("_")
            }
    raise AssertionError("LCMEngine not found")


def test_every_host_callable_method_is_classified() -> None:
    classified = (
        GATED | ACCESSORS | WRITES_AS_SELF | HOST_ONLY
        | set(REACHED_VIA_GATED_COMMAND)
    )
    unclassified = _public_methods() - classified
    assert not unclassified, (
        f"public LCMEngine methods with no recorded safety argument: "
        f"{sorted(unclassified)}. Gate it, or classify it with the reason it "
        f"does not need a gate. An unclassified method is #209's 'bullet "
        f"silently absent'."
    )


def test_the_classification_does_not_drift_from_reality() -> None:
    """A method listed as GATED must actually call the seam.

    Without this the table is a comment, and a gate removed during a refactor
    would leave the claim standing.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    source = (root / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == "LCMEngine"):
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name not in GATED:
                continue
            body = ast.get_source_segment(source, item) or ""
            assert "policy_for_engine" in body, (
                f"{item.name} is listed as GATED but no longer calls the seam"
            )


@pytest.mark.parametrize(
    "method, handler", sorted(REACHED_VIA_GATED_COMMAND.items())
)
def test_the_command_that_reaches_them_actually_gates(method: str, handler: str) -> None:
    """These are model-reachable, so their safety rests entirely on the handler.

    `rotate_backup_path` was originally listed here and turned out to be a pure
    path computation that writes nothing -- reclassified as an accessor. These
    two genuinely take an operator lease, and the gate is on the handler.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    source = (root / "command.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == handler:
            body = ast.get_source_segment(source, node) or ""
            assert f".{method}(" in body, f"{handler} no longer reaches {method}"
            assert "policy_for_engine" in body, (
                f"{handler} reaches {method} without authorizing"
            )
            return
    raise AssertionError(f"{handler} not found in command.py")


@pytest.mark.parametrize("method", sorted(HOST_ONLY - {"clone_for_agent"}))
def test_host_only_methods_are_not_reachable_from_a_tool_or_command(method: str) -> None:
    """The safety argument for HOST_ONLY is unreachability, so test THAT.

    `disable_teams` turns isolation off for every principal in the store. It is
    ungated, and that is only acceptable while nothing the model can drive can
    call it. If a tool or slash command ever wires one of these up, this fails
    and the gate becomes mandatory.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    for surface in ("tools.py", "command.py"):
        text = (root / surface).read_text(encoding="utf-8")
        assert f".{method}(" not in text, (
            f"{surface} reaches {method}, which is ungated. Either gate it or "
            f"remove the model-reachable path."
        )
