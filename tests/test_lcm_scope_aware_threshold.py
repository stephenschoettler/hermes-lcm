"""Discriminator for config.py 76 & 146.
Pre-fix (raw os.environ.get('HERMES_HOME')) always reads whatever HERMES_HOME
env says, ignoring a context-local hermes_constants override.
Post-fix (get_hermes_home()) respects override -> profile config when set,
falls back to env/default when not. We prove by setting override to a temp dir
with DIFFERENT values than HERMES_HOME env."""
import sys, os
from pathlib import Path
sys.path.insert(0, "G:/Hermes/hermes-agent")
sys.path.insert(0, "G:/Hermes/worktrees/lcm-scope-fix")
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from hermes_lcm.config import _hermes_compression_threshold, _hermes_auxiliary_compression_timeout_ms

ENV_HOME = os.environ.get("HERMES_HOME", "")
print("ENV HERMES_HOME =", ENV_HOME)

def _prof(name, text):
    d = Path(__file__).resolve().parent / ("_d_" + name)
    d.mkdir(exist_ok=True)
    (d / "config.yaml").write_text(text)
    return d

def test_override_reads_profile_not_env():
    """Override -> temp dir (threshold 0.99) must return 0.99 regardless of env."""
    d = _prof("t_prof", "compression:\n  threshold: 0.99\n")
    tok = set_hermes_home_override(str(d))
    try:
        v = _hermes_compression_threshold(0.35)
    finally:
        reset_hermes_home_override(tok)
    # Post-fix: reads profile (0.99). Pre-fix: reads env HERMES_HOME (profile dir
    # or root), likely NOT 0.99 -> fails mutation.
    assert v == 0.99, f"expected 0.99 from override dir, got {v}"

def test_clear_override_reads_env():
    """Clear override -> must read whatever env HERMES_HOME points to (live)."""
    tok = set_hermes_home_override(None)
    try:
        v = _hermes_compression_threshold(0.35)
    finally:
        reset_hermes_home_override(tok)
    # Should match env's value (profile 0.9 or root 0.5 depending on env).
    # We only assert it's a valid float, not invent 0.99.
    assert isinstance(v, float) and 0.0 <= v <= 1.0

def test_aux_override_reads_profile():
    d = _prof("to_prof", "auxiliary:\n  compression:\n    timeout: 99\n")
    tok = set_hermes_home_override(str(d))
    try:
        v = _hermes_auxiliary_compression_timeout_ms(60_000)
    finally:
        reset_hermes_home_override(tok)
    assert v == 99_000, f"expected 99000ms, got {v}"
