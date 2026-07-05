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


def test_state_db_path_fallback_outside_allowed_base_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("LCM_HERMES_BASE_DIR", str(tmp_path / "allowed"))

    from hermes_lcm.command import _state_db_path_for_engine as command_state_db_path
    from hermes_lcm.tools import _state_db_path_for_engine as tools_state_db_path

    class MockStore:
        db_path = str(tmp_path / "outside" / "lcm.db")

    class MockEngine:
        _hermes_home = ""
        _store = MockStore()

    for state_db_path_for_engine in (command_state_db_path, tools_state_db_path):
        with pytest.raises(ValueError, match="not within allowed base"):
            state_db_path_for_engine(MockEngine())


def test_engine_state_db_path_fallback_outside_allowed_base_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("LCM_HERMES_BASE_DIR", str(tmp_path / "allowed"))

    from hermes_lcm.engine import LCMEngine

    class MockStore:
        db_path = str(tmp_path / "outside" / "lcm.db")

    engine = LCMEngine.__new__(LCMEngine)
    engine._hermes_home = ""
    engine._store = MockStore()

    with pytest.raises(ValueError, match="not within allowed base"):
        engine._state_db_path()


def _resolved(p) -> Path:
    return Path(str(p)).expanduser().resolve()


def test_externalization_path_outside_hermes_home_warns_but_does_not_break(monkeypatch, tmp_path, caplog):
    import logging
    from hermes_lcm.externalize import get_large_output_storage_dir, _WARNED_EXTERNALIZATION_PATHS

    monkeypatch.delenv("LCM_HERMES_BASE_DIR", raising=False)
    _WARNED_EXTERNALIZATION_PATHS.clear()
    outside = tmp_path / "other-volume" / "payloads"

    class Config:
        large_output_externalization_path = str(outside)

    with caplog.at_level(logging.WARNING):
        path = get_large_output_storage_dir(
            Config(), hermes_home=str(tmp_path / "hermes"), create=False
        )

    assert path == _resolved(outside)
    assert any("outside the hermes_home base" in r.message for r in caplog.records)


def test_externalization_path_within_hermes_home_does_not_warn(monkeypatch, tmp_path, caplog):
    import logging
    from hermes_lcm.externalize import get_large_output_storage_dir, _WARNED_EXTERNALIZATION_PATHS

    monkeypatch.delenv("LCM_HERMES_BASE_DIR", raising=False)
    _WARNED_EXTERNALIZATION_PATHS.clear()
    hermes_home = tmp_path / "hermes"
    inside = hermes_home / "custom-outputs"

    class Config:
        large_output_externalization_path = str(inside)

    with caplog.at_level(logging.WARNING):
        path = get_large_output_storage_dir(Config(), hermes_home=str(hermes_home), create=False)

    assert path == _resolved(inside)
    assert not any("outside the hermes_home base" in r.message for r in caplog.records)


def test_externalization_path_strict_containment_when_base_set(monkeypatch, tmp_path):
    from hermes_lcm.externalize import get_large_output_storage_dir

    monkeypatch.setenv("LCM_HERMES_BASE_DIR", str(tmp_path / "allowed"))

    class Config:
        large_output_externalization_path = str(tmp_path / "elsewhere" / "payloads")

    with pytest.raises(ValueError):
        get_large_output_storage_dir(
            Config(), hermes_home=str(tmp_path / "allowed" / "hermes"), create=False
        )
