"""Round-two CLI and configuration regression tests for LongMemEval medium."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmarking.longmemeval as lme
from tests.conftest import load_cli as _load_cli


@pytest.mark.parametrize("value", ["+2", "-2", " 2", "2 ", " 2 ", "\t2", "2\n"])
def test_embedding_batch_size_rejects_noncanonical_integer_text(monkeypatch, value):
    monkeypatch.setenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", value)

    with pytest.raises(ValueError, match=r"must match \^\[0-9\]\+\$"):
        lme._embedding_batch_size()


@pytest.mark.parametrize("value", ["1", "2", "002", "64"])
def test_embedding_batch_size_accepts_plain_ascii_digits(monkeypatch, value):
    monkeypatch.setenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", value)

    assert lme._embedding_batch_size() == int(value)


@pytest.mark.parametrize("relative_output", [Path("."), Path("reports"), Path("reports/run")])
def test_run_rejects_output_equal_to_or_inside_prepared_dir(
    tmp_path, monkeypatch, relative_output
):
    cli = _load_cli()
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    sentinel = prepared_dir / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    output_dir = prepared_dir / relative_output

    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("prepared loading and scoring must not start")

    monkeypatch.setattr(cli, "load_prepared_dataset", _unexpected_call)
    monkeypatch.setattr(cli, "run_harness", _unexpected_call)
    args = cli._parse_args(
        [
            "run",
            "--prepared-dir",
            str(prepared_dir),
            "--dataset-label",
            "m",
            "--output",
            str(output_dir),
            "--allow-external-output",
        ]
    )

    with pytest.raises(SystemExit, match=r"--output equal to or inside --prepared-dir"):
        cli._cmd_run(args)

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    if relative_output != Path("."):
        assert not output_dir.exists()


@pytest.mark.parametrize("error", [OSError("prepared file disappeared"), ValueError("prepared checksum mismatch")])
def test_run_converts_lazy_prepared_iterator_errors_to_system_exit(tmp_path, monkeypatch, error):
    cli = _load_cli()

    class _Prepared:
        source_sha256 = "0" * 64
        manifest_sha256 = "1" * 64
        question_count = 0

        def validate_question_ids(self, *, limit=None):
            return None

        def selected_question_ids(self, *, limit=None):
            return ()

        def iter_questions(self, *, limit=None):
            def _lazy():
                raise error
                yield  # pragma: no cover - keeps this a generator

            return _lazy()

    monkeypatch.setattr(cli, "load_prepared_dataset", lambda *_args, **_kwargs: _Prepared())

    def _consume(questions, **_kwargs):
        return list(questions)

    monkeypatch.setattr(cli, "run_harness", _consume)
    args = cli._parse_args(
        [
            "run",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--dataset-label",
            "m",
            "--output",
            str(tmp_path / "output"),
            "--allow-external-output",
        ]
    )

    with pytest.raises(SystemExit, match="prepared"):
        cli._cmd_run(args)


def test_run_preflights_prepared_qids_before_scoring(tmp_path, monkeypatch):
    cli = _load_cli()
    scoring_started = False

    class _Prepared:
        source_sha256 = "0" * 64
        manifest_sha256 = "1" * 64
        question_count = 2

        def validate_question_ids(self, *, limit=None):
            raise ValueError("prepared question sequence ended early: expected 'q1'")

        def iter_questions(self, *, limit=None):
            raise AssertionError("iterator should not be consumed after failed preflight")

    monkeypatch.setattr(cli, "load_prepared_dataset", lambda *_args, **_kwargs: _Prepared())

    def _unexpected(*_args, **_kwargs):
        nonlocal scoring_started
        scoring_started = True
        raise AssertionError("scoring must not start")

    monkeypatch.setattr(cli, "run_harness", _unexpected)
    args = cli._parse_args(
        [
            "run",
            "--prepared-dir",
            str(tmp_path / "prepared"),
            "--dataset-label",
            "m",
            "--output",
            str(tmp_path / "output"),
            "--allow-external-output",
        ]
    )

    with pytest.raises(SystemExit, match="ended early"):
        cli._cmd_run(args)
    assert scoring_started is False


def test_direct_dataset_digest_change_fails_closed_on_resume(tmp_path):
    cli = _load_cli()
    dataset = tmp_path / lme.DATASET_COORDS["s"]["file"]
    row = {
        "question_id": "q0",
        "question_type": "single-session-user",
        "question": "what is the code?",
        "answer": "ALPHA",
        "question_date": "2023-01-01",
        "haystack_session_ids": ["s0"],
        "haystack_dates": ["2023-01-01"],
        "haystack_sessions": [[{"role": "user", "content": "the code is ALPHA"}]],
        "answer_session_ids": ["s0"],
    }
    dataset.write_text(json.dumps([row]) + "\n", encoding="utf-8")
    output = tmp_path / "output"
    base_args = [
        "run",
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--allow-external-output",
    ]
    assert cli.main(base_args) == 0

    row["haystack_sessions"][0][0]["content"] = "the code is BETA"
    dataset.write_text(json.dumps([row]) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"configuration mismatch.*source_sha256"):
        cli.main([*base_args, "--resume"])
