"""Guard the host's context-engine discovery heuristic.

Hermes discovers user-installed context engines by scanning the first
8192 characters of the plugin's __init__.py for a marker string
("register_context_engine" or "ContextEngine").  If an edit ever pushes
both markers past that window, the engine silently stops loading — no
error, no ingest, no log line (see the 2026-10 outage where nothing was
ingested for two weeks because the marker sat at index ~15300).

This test pins at least one marker inside the window so the whole
failure class fails loudly in CI instead of silently in production.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO_ROOT / "__init__.py"

# Hermes host reads only the first 8192 chars of __init__.py when
# deciding whether a directory is a context-engine plugin
# (plugins/context_engine/__init__.py :: _is_context_engine_dir).
HOST_DISCOVERY_WINDOW = 8192
HOST_MARKERS = ("register_context_engine", "ContextEngine")


def test_discovery_marker_within_host_window():
    source = ENTRYPOINT.read_text(encoding="utf-8-sig")
    window = source[:HOST_DISCOVERY_WINDOW]
    hits = [marker for marker in HOST_MARKERS if marker in window]
    assert hits, (
        f"neither {HOST_MARKERS!r} appears in the first "
        f"{HOST_DISCOVERY_WINDOW} characters of __init__.py — the Hermes "
        f"host will not discover this directory as a context engine and "
        f"LCM will silently stop ingesting.  Keep a marker phrase in the "
        f"module docstring."
    )


def test_entrypoint_readable():
    # A missing/unreadable entrypoint is the same silent-death failure
    # mode as a moved marker; guard it in the same place.
    assert ENTRYPOINT.is_file(), f"plugin entrypoint missing: {ENTRYPOINT}"
    ENTRYPOINT.read_text(encoding="utf-8-sig")
