"""Tests for path handling security - containment validation."""
import tempfile
from pathlib import Path
import os
import pytest


def test_configured_externalization_path_outside_allowed_base_rejected():
    """Test that configured externalization paths outside allowed base are rejected.

    When LCM_HERMES_BASE_DIR is set, config.large_output_externalization_path
    must be validated against the allowed base.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        allowed_base = tmpdir
        os.environ["LCM_HERMES_BASE_DIR"] = allowed_base

        try:
            from hermes_lcm.externalize import get_large_output_storage_dir

            # Create a config with large_output_externalization_path OUTSIDE allowed base
            class FakeConfig:
                large_output_externalization_path = "/tmp/evil-external"
                hermes_home = None

            # This should raise ValueError because the configured path is outside allowed base
            with pytest.raises(ValueError, match="not within allowed base"):
                get_large_output_storage_dir(FakeConfig(), "", create=False)
        finally:
            del os.environ["LCM_HERMES_BASE_DIR"]


def test_configured_externalization_path_inside_allowed_base_accepted():
    """Test that configured externalization paths inside allowed base work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        allowed_base = tmpdir
        os.environ["LCM_HERMES_BASE_DIR"] = allowed_base

        try:
            from hermes_lcm.externalize import get_large_output_storage_dir

            # Create a config with path inside allowed base
            internal_path = Path(tmpdir) / "internal-external"

            class FakeConfig:
                large_output_externalization_path = str(internal_path)
                hermes_home = None

            path = get_large_output_storage_dir(FakeConfig(), "", create=False)
            assert path.is_absolute()
            assert str(path).startswith(tmpdir)
        finally:
            del os.environ["LCM_HERMES_BASE_DIR"]


def test_hermes_home_outside_allowed_base_rejected(monkeypatch):
    """Test that hermes_home outside allowed base raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        allowed_base = tmpdir
        monkeypatch.setenv("LCM_HERMES_BASE_DIR", allowed_base)

        from hermes_lcm.command import _state_db_path_for_engine

        # Create a mock engine with hermes_home outside allowed base
        class MockEngine:
            _hermes_home = "/etc"

        engine = MockEngine()
        with pytest.raises(ValueError, match="not within allowed base"):
            _state_db_path_for_engine(engine)
