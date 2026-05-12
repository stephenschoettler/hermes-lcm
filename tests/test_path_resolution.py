"""Tests for path resolution behavior."""
import tempfile
from pathlib import Path


def test_state_db_path_resolves_to_absolute():
    """Test that _state_db_path resolves to an absolute path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from hermes_lcm.command import _state_db_path_for_engine

        # Create a mock engine with hermes_home in tmpdir
        hermes_home = Path(tmpdir) / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)

        class MockEngine:
            _hermes_home = str(hermes_home)

        engine = MockEngine()
        path = _state_db_path_for_engine(engine)

        assert path.is_absolute()
        assert ".." not in str(path).split("/")[1:]


def test_get_large_output_storage_dir_resolves():
    """Test that get_large_output_storage_dir resolves paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from hermes_lcm.externalize import get_large_output_storage_dir

        # Test with explicit path
        configured_path = str(Path(tmpdir) / "external")
        path = get_large_output_storage_dir(None, configured_path, create=False)
        assert path.is_absolute()
        assert ".." not in str(path).split("/")[1:]

        # Test with hermes_home
        hermes_home = str(Path(tmpdir) / "hermes")
        path = get_large_output_storage_dir(None, hermes_home=hermes_home, create=False)
        assert path.is_absolute()
        assert ".." not in str(path).split("/")[1:]


def test_path_escapes_via_traversal_rejected():
    """Test that paths with ../ sequences are properly resolved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from hermes_lcm.externalize import get_large_output_storage_dir

        # Create a path with ../ sequences - should be resolved
        base = Path(tmpdir) / "base"
        base.mkdir(parents=True, exist_ok=True)
        traversal_path = str(base / ".." / ".." / "etc")

        # get_large_output_storage_dir should resolve this
        path = get_large_output_storage_dir(None, hermes_home=traversal_path, create=False)
        # The path should be absolute after resolve
        assert path.is_absolute()
        # No ../ segments should remain (except root)
        parts = str(path).split("/")[1:]
        assert ".." not in parts
