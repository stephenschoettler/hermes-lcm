"""Tests for path containment checks - validates boundary enforcement."""
import tempfile
from pathlib import Path
import pytest


def test_path_containment_within_allowed_base(monkeypatch):
    """Test that hermes_home within allowed base is accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set allowed base to tmpdir using monkeypatch
        monkeypatch.setenv("LCM_HERMES_BASE_DIR", tmpdir)

        from hermes_lcm.command import _state_db_path_for_engine

        # Create a mock engine with hermes_home inside allowed base
        hermes_home = str(Path(tmpdir) / "hermes")

        class MockEngine:
            _hermes_home = hermes_home

        engine = MockEngine()
        # Should succeed without raising
        path = _state_db_path_for_engine(engine)
        assert path.is_absolute()
        assert str(path).startswith(tmpdir)


def test_path_containment_outside_allowed_base(monkeypatch):
    """Test that hermes_home outside allowed base raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set allowed base to tmpdir
        monkeypatch.setenv("LCM_HERMES_BASE_DIR", tmpdir)

        from hermes_lcm.command import _state_db_path_for_engine

        # Create a mock engine with hermes_home outside allowed base
        class MockEngine:
            _hermes_home = "/etc"

        engine = MockEngine()
        # Should raise ValueError
        with pytest.raises(ValueError, match="not within allowed base"):
            _state_db_path_for_engine(engine)


def test_engine_state_db_path_outside_allowed_base(monkeypatch):
    """Test LCMEngine._state_db_path with engine method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("LCM_HERMES_BASE_DIR", tmpdir)

        from hermes_lcm.engine import LCMEngine

        # Create a mock store with db_path
        class MockStore:
            db_path = str(Path(tmpdir) / "lcm.db")

        # Create engine with hermes_home outside allowed base
        engine = LCMEngine.__new__(LCMEngine)
        engine._hermes_home = "/etc"
        engine._store = MockStore()

        # Should raise ValueError
        with pytest.raises(ValueError, match="not within allowed base"):
            engine._state_db_path()
