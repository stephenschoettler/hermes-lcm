"""Cross-repository JSON fixture corpus loading with fail-loud path resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import AccessContextV1


FIXTURE_ROOT_NAME = "access_context_v1"
FIXTURE_KINDS = frozenset({"positive", "negative", "delegation", "revocation", "derivation"})


class FixtureCorpusNotFound(FileNotFoundError):
    """Raised when the shared corpus is absent instead of silently empty."""


class FixtureFormatError(ValueError):
    """Raised when a fixture does not carry the required envelope."""


def fixture_root(root: str | Path | None = None) -> Path:
    """Resolve corpus paths from a checkout or an installed package layout."""

    if root is not None:
        candidate = Path(root)
        if candidate.name != FIXTURE_ROOT_NAME:
            candidate = candidate / FIXTURE_ROOT_NAME
        if candidate.is_dir():
            return candidate
    package_root = Path(__file__).resolve().parent
    candidates = (
        package_root.parent / "tests" / "fixtures" / FIXTURE_ROOT_NAME,
        package_root / "tests" / "fixtures" / FIXTURE_ROOT_NAME,
        package_root.parent.parent / "tests" / "fixtures" / FIXTURE_ROOT_NAME,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FixtureCorpusNotFound(f"AccessContextV1 fixture corpus not found; searched: {searched}")


def fixture_paths(kind: str | None = None, *, root: str | Path | None = None) -> tuple[Path, ...]:
    base = fixture_root(root)
    if kind is None:
        directories = [base / name for name in sorted(FIXTURE_KINDS)]
    else:
        if kind not in FIXTURE_KINDS:
            raise ValueError(f"unknown fixture kind: {kind}")
        directories = [base / kind]
    paths = tuple(sorted(path for directory in directories if directory.is_dir() for path in directory.glob("*.json")))
    if not paths:
        raise FixtureCorpusNotFound(f"AccessContextV1 fixture corpus is empty under {base}")
    return paths


def load_fixture(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    if not fixture_path.is_file():
        raise FixtureCorpusNotFound(f"fixture not found: {fixture_path}")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FixtureFormatError(f"fixture must be a JSON object: {fixture_path}")
    for field in ("contract_revision", "description", "context"):
        if field not in payload:
            raise FixtureFormatError(f"fixture missing {field}: {fixture_path}")
    if payload["contract_revision"] != "v1":
        # Unknown revisions are valid negative vectors; the validator, not the
        # loader, decides their denial.  We only require the envelope field.
        if not isinstance(payload["contract_revision"], str):
            raise FixtureFormatError(f"fixture revision must be text: {fixture_path}")
    if not isinstance(payload["description"], str) or not payload["description"].strip():
        raise FixtureFormatError(f"fixture description must be one line: {fixture_path}")
    if payload["context"] is not None and not isinstance(payload["context"], dict):
        raise FixtureFormatError(f"fixture context must be an object: {fixture_path}")
    return payload


def load_context(path: str | Path) -> AccessContextV1 | None:
    context = load_fixture(path)["context"]
    return None if context is None else AccessContextV1.from_payload(context)


def load_corpus(*, kind: str | None = None, root: str | Path | None = None) -> tuple[dict[str, Any], ...]:
    paths = fixture_paths(kind, root=root)
    return tuple(load_fixture(path) for path in paths)
