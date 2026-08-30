"""Medium-tier preparation, provenance, and batching regression tests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmarking.longmemeval as lme
from tests.conftest import load_cli as _load_cli
_BANKED_METRICS = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "results"
    / "longmemeval-v3-500q-fastembed-metrics.json"
)
_REAL_DATASET = os.environ.get("LME_M_REAL_DATASET")
_REAL_DATASET_PATH = Path(_REAL_DATASET) if _REAL_DATASET else None

# The banked golden SHA below was recorded on Darwin arm64.
_GOLDEN_PLATFORM = ("darwin", "arm64")
_GOLDEN_REGEN_COMMAND = (
    "uv run --with pytest --with ijson python3 -m pytest "
    "tests/test_longmemeval_medium.py::test_small_default_cli_report_is_byte_identical_to_golden"
)


def _raw_question(index: int) -> dict:
    evidence_id = f"q{index}-evidence"
    session_ids = [f"q{index}-first", evidence_id, f"q{index}-last"]
    sessions = [
        [{"role": "user", "content": f"ordinary note {index}"}],
        [
            {
                "role": "user",
                "content": f"locker passcode is MEDIUM{index}",
                "has_answer": True,
            }
        ],
        [{"role": "assistant", "content": f"closing note {index}"}],
    ]
    return {
        "question_id": f"q{index}",
        "question_type": "single-session-user",
        "question": f"what is the locker passcode MEDIUM{index}",
        "answer": f"MEDIUM{index}",
        "question_date": "2023-01-01",
        "haystack_session_ids": session_ids,
        "haystack_dates": ["2023-01-01"] * len(session_ids),
        "haystack_sessions": sessions,
        "answer_session_ids": [evidence_id],
    }


def _write_dataset(directory: Path, label: str = "m", count: int = 3) -> tuple[Path, list[dict]]:
    rows = [_raw_question(index) for index in range(count)]
    path = directory / lme.DATASET_COORDS[label]["file"]
    path.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    return path, rows


def test_prepare_streams_and_writes_checksum_manifest(tmp_path, monkeypatch):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source, rows = _write_dataset(tmp_path)
    prepared_dir = tmp_path / "prepared"

    def _full_parse_forbidden(*_args, **_kwargs):
        raise AssertionError("prepare must not call json.loads on the corpus")

    monkeypatch.setattr(lme.json, "loads", _full_parse_forbidden)
    manifest = lme.prepare_dataset(source, prepared_dir, dataset_label="m")

    assert manifest["dataset_label"] == "m"
    assert manifest["source_file"] == "longmemeval_m"
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["question_count"] == len(rows)
    assert [entry["question_id"] for entry in manifest["questions"]] == ["q0", "q1", "q2"]
    for entry, row in zip(manifest["questions"], rows):
        payload = (prepared_dir / entry["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert payload == lme._canonical_json_bytes(row)
    assert (prepared_dir / "manifest.json").is_file()

    malformed_root = tmp_path / "malformed"
    malformed_root.mkdir()
    malformed_source = malformed_root / lme.DATASET_COORDS["m"]["file"]
    malformed_source.write_text(json.dumps([_raw_question(0)])[:-1], encoding="utf-8")
    malformed_output = malformed_root / "prepared"
    with pytest.raises(ValueError, match="invalid LongMemEval dataset JSON"):
        lme.prepare_dataset(malformed_source, malformed_output, dataset_label="m")
    assert not malformed_output.exists()
    assert not list(malformed_root.glob(".prepared.prepare-*"))

    reserved_root = tmp_path / "reserved"
    reserved_root.mkdir()
    reserved_source, reserved_rows = _write_dataset(reserved_root, count=2)
    reserved_rows[1]["question_id"] = "Manifest"
    reserved_source.write_text(json.dumps(reserved_rows) + "\n", encoding="utf-8")
    reserved_output = reserved_root / "prepared"
    with pytest.raises(ValueError, match="unsafe question_id"):
        lme.prepare_dataset(reserved_source, reserved_output, dataset_label="m")
    assert not reserved_output.exists()

    existing_root = tmp_path / "existing-empty"
    existing_root.mkdir()
    existing_source, _rows = _write_dataset(existing_root, count=1)
    existing_output = existing_root / "prepared"
    existing_output.mkdir()
    lme.prepare_dataset(existing_source, existing_output, dataset_label="m")
    assert sorted(path.name for path in existing_output.iterdir()) == [
        "manifest.json",
        "q0.json",
    ]


def test_prepare_rejects_missing_question_shape_before_publish(tmp_path):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source, rows = _write_dataset(tmp_path, count=1)
    rows[0].pop("question_type")
    source.write_text(json.dumps(rows), encoding="utf-8")
    prepared_dir = tmp_path / "prepared"

    with pytest.raises(ValueError, match=r"question 'q0'.*'question_type'"):
        lme.prepare_dataset(source, prepared_dir, dataset_label="m")

    assert not prepared_dir.exists()


def test_prepare_rejects_non_list_collection_fields_before_publish(tmp_path):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source, rows = _write_dataset(tmp_path, count=1)
    rows[0]["haystack_session_ids"] = None
    source.write_text(json.dumps(rows), encoding="utf-8")
    prepared_dir = tmp_path / "prepared"

    with pytest.raises(ValueError, match=r"question 'q0'.*'haystack_session_ids' must be a list"):
        lme.prepare_dataset(source, prepared_dir, dataset_label="m")

    assert not prepared_dir.exists()


def test_prepare_rejects_trailing_content_after_array(tmp_path):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source, rows = _write_dataset(tmp_path, count=1)
    source.write_text(json.dumps(rows) + json.dumps([_raw_question(1)]), encoding="utf-8")
    prepared_dir = tmp_path / "prepared"

    # Read-ahead backends (yajl2, python) raise their own trailing-garbage
    # JSONError before drain() sees the tail; drain()'s check covers backends
    # that stop at the array close. Either way prepare fails closed.
    with pytest.raises(
        ValueError,
        match="invalid LongMemEval dataset JSON|trailing content after the top-level array",
    ):
        lme.prepare_dataset(source, prepared_dir, dataset_label="m")

    assert not prepared_dir.exists()


def test_prepare_rejects_null_haystack_session_before_publish(tmp_path):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source, rows = _write_dataset(tmp_path, count=1)
    rows[0]["haystack_sessions"] = [None]
    source.write_text(json.dumps(rows), encoding="utf-8")
    prepared_dir = tmp_path / "prepared"

    with pytest.raises(ValueError, match=r"haystack_sessions\[0\] must be a list of messages"):
        lme.prepare_dataset(source, prepared_dir, dataset_label="m")

    assert not prepared_dir.exists()


def test_prepare_rejects_casefold_question_id_collision(tmp_path):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source, rows = _write_dataset(tmp_path, count=2)
    rows[1]["question_id"] = "Q0"
    source.write_text(json.dumps(rows), encoding="utf-8")
    prepared_dir = tmp_path / "prepared"

    with pytest.raises(ValueError, match=r"duplicate question_id in dataset: 'Q0'"):
        lme.prepare_dataset(source, prepared_dir, dataset_label="m")

    assert not prepared_dir.exists()


@pytest.mark.parametrize(
    ("root", "root_pattern"),
    [({"item": [_raw_question(0)]}, r"got object"), (17, r"got scalar \(number\)")],
)
def test_prepare_rejects_non_array_roots(tmp_path, root, root_pattern):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source = tmp_path / lme.DATASET_COORDS["m"]["file"]
    source.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"dataset root must be a JSON array; {root_pattern}"):
        lme.prepare_dataset(source, tmp_path / "prepared", dataset_label="m")


@pytest.mark.skipif(
    _REAL_DATASET_PATH is None or not _REAL_DATASET_PATH.is_file(),
    reason="set LME_M_REAL_DATASET to an existing real longmemeval_m JSON file",
)
def test_real_medium_prepare_first_three_round_trip(tmp_path):
    ijson = pytest.importorskip("ijson")
    first_three = []
    with _REAL_DATASET_PATH.open("rb") as source:
        for row in ijson.items(source, "item", use_float=True):
            first_three.append(row)
            if len(first_three) == 3:
                break
    assert len(first_three) == 3

    source = tmp_path / lme.DATASET_COORDS["m"]["file"]
    source.write_text(json.dumps(first_three) + "\n", encoding="utf-8")
    prepared_dir = tmp_path / "prepared"
    manifest = lme.prepare_dataset(source, prepared_dir, dataset_label="m")
    prepared = lme.load_prepared_dataset(prepared_dir, dataset_label="m")
    questions = list(prepared.iter_questions())

    assert [question.question_id for question in questions] == [
        str(row["question_id"]) for row in first_three
    ]
    for row, question, entry in zip(first_three, questions, manifest["questions"]):
        assert question.question_type == str(row["question_type"])
        assert question.question == str(row["question"])
        assert question.haystack_session_ids == [str(value) for value in row["haystack_session_ids"]]
        assert question.haystack_sessions == row["haystack_sessions"]
        assert question.answer_session_ids == [str(value) for value in row["answer_session_ids"]]
        assert lme.sha256_file(prepared_dir / entry["file"]) == entry["sha256"]


def test_prepared_manifest_fails_closed_on_label_count_and_content_mismatch(tmp_path):
    pytest.importorskip("ijson", reason="prepare path requires ijson; the run env installs it explicitly")
    source, _rows = _write_dataset(tmp_path)
    prepared_dir = tmp_path / "prepared"
    lme.prepare_dataset(source, prepared_dir, dataset_label="m")

    with pytest.raises(ValueError, match="dataset label mismatch"):
        lme.load_prepared_dataset(prepared_dir, dataset_label="s")

    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["question_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="question_count mismatch"):
        lme.load_prepared_dataset(prepared_dir, dataset_label="m")

    manifest["question_count"] -= 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (prepared_dir / "q1.json").write_text("{}", encoding="utf-8")
    prepared = lme.load_prepared_dataset(prepared_dir, dataset_label="m")
    with pytest.raises(ValueError, match="checksum mismatch"):
        list(prepared.iter_questions())

    guard_cases = [
        ("schema_version", "schema_version"),
        ("source_file", "source_file"),
        ("source_sha256", "source_sha256 is invalid"),
        ("question_sha256", "invalid prepared question checksum"),
        ("extra_file", "file set does not match manifest"),
        ("missing_file", "question file not found"),
        ("duplicate", "duplicate prepared question entry"),
    ]
    for case, message in guard_cases:
        case_root = tmp_path / f"guard-{case}"
        case_root.mkdir()
        case_source, _rows = _write_dataset(case_root, count=2)
        case_prepared = case_root / "prepared"
        lme.prepare_dataset(case_source, case_prepared, dataset_label="m")
        case_manifest_path = case_prepared / "manifest.json"
        case_manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))

        if case == "schema_version":
            case_manifest["schema_version"] += 1
        elif case == "source_file":
            case_manifest["source_file"] = "longmemeval_s"
        elif case == "source_sha256":
            case_manifest["source_sha256"] = "not-a-digest"
        elif case == "question_sha256":
            case_manifest["questions"][0]["sha256"] = "ABC"
        elif case == "extra_file":
            (case_prepared / "extra.json").write_text("{}", encoding="utf-8")
        elif case == "missing_file":
            (case_prepared / case_manifest["questions"][0]["file"]).unlink()
        elif case == "duplicate":
            case_manifest["questions"].append(dict(case_manifest["questions"][0]))
            case_manifest["question_count"] += 1

        if case not in {"extra_file", "missing_file"}:
            case_manifest_path.write_text(json.dumps(case_manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            lme.load_prepared_dataset(case_prepared, dataset_label="m")


def test_direct_dataset_label_must_match_filename(tmp_path):
    source, _rows = _write_dataset(tmp_path, label="m")
    with pytest.raises(ValueError, match="requires filename 'longmemeval_s'"):
        lme.validate_dataset_path_label(source, "s")
    lme.validate_dataset_path_label(source, "m")


def test_cli_requires_exactly_one_run_source_and_exposes_prepare():
    cli = _load_cli()
    prepared = cli._parse_args(
        [
            "prepare", "--dataset", "longmemeval_m", "--prepared-dir", "prepared",
            "--dataset-label", "m",
        ]
    )
    assert prepared.command == "prepare"
    assert prepared.dataset_label == "m"
    resumed = cli._parse_args(
        [
            "run", "--dataset", "longmemeval_s", "--output", "out", "--resume",
        ]
    )
    assert resumed.resume is True
    with pytest.raises(SystemExit):
        cli._parse_args(["run", "--output", "out"])
    with pytest.raises(SystemExit):
        cli._parse_args(
            [
                "run", "--dataset", "longmemeval_s", "--prepared-dir", "prepared",
                "--output", "out",
            ]
        )


def test_fetch_routes_the_selected_dataset_label(tmp_path, monkeypatch):
    cli = _load_cli()
    captured = {}

    def _fake_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=_fake_download),
    )
    args = cli._parse_args(
        ["fetch", "--dataset-label", "m", "--output", str(tmp_path / "download")]
    )

    assert cli._cmd_fetch(args) == 0
    assert captured["filename"] == lme.DATASET_COORDS["m"]["file"]
    assert captured["repo_id"] == lme.DATASET_COORDS["m"]["repo_id"]
    assert captured["revision"] == lme.DATASET_COORDS["m"]["revision"]


def test_prepare_requires_opt_in_for_external_output(tmp_path, monkeypatch):
    cli = _load_cli()
    source = tmp_path / "longmemeval_m"
    prepared_dir = tmp_path / "prepared"
    argv = [
        "prepare",
        "--dataset", str(source),
        "--prepared-dir", str(prepared_dir),
        "--dataset-label", "m",
    ]

    with pytest.raises(SystemExit, match="Refusing output outside repo"):
        cli._cmd_prepare(cli._parse_args(argv))

    captured = {}

    def _fake_prepare(dataset, output, *, dataset_label):
        captured.update(dataset=dataset, output=output, dataset_label=dataset_label)
        return {"prepared": True}

    monkeypatch.setattr(cli, "prepare_dataset", _fake_prepare)
    assert cli._cmd_prepare(cli._parse_args([*argv, "--allow-external-output"])) == 0
    assert captured == {
        "dataset": source,
        "output": prepared_dir.resolve(),
        "dataset_label": "m",
    }


def test_run_rejects_direct_medium_dataset(tmp_path):
    cli = _load_cli()
    source, _rows = _write_dataset(tmp_path)
    args = cli._parse_args(
        [
            "run", "--dataset", str(source), "--dataset-label", "m",
            "--output", str(tmp_path / "output"), "--allow-external-output",
        ]
    )

    with pytest.raises(SystemExit, match="prepare.*--prepared-dir"):
        cli._cmd_run(args)


def test_direct_dataset_hashes_the_same_single_read_it_parses(tmp_path, monkeypatch):
    source, rows = _write_dataset(tmp_path, label="s", count=1)
    expected_bytes = source.read_bytes()
    original_read_bytes = Path.read_bytes
    read_count = 0

    def _counted_read(path):
        nonlocal read_count
        if path == source:
            read_count += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", _counted_read)
    questions, digest = lme.load_questions_with_sha256(source)

    assert read_count == 1
    assert questions == [lme.parse_question(rows[0])]
    assert digest == hashlib.sha256(expected_bytes).hexdigest()


def _zero_timing(monkeypatch):
    monkeypatch.setattr(lme.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(lme.time, "strftime", lambda *_args, **_kwargs: "2000-01-01T00:00:00Z")
    monkeypatch.setattr(lme, "_timed", lambda fn: (fn(), 0.0))


def test_prepared_and_dataset_runs_have_identical_metrics(tmp_path, monkeypatch):
    pytest.importorskip("ijson", reason="prepared-run equivalence requires the prepare path (ijson); run env installs it explicitly")
    monkeypatch.delenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", raising=False)
    _zero_timing(monkeypatch)
    source, _rows = _write_dataset(tmp_path)
    prepared_dir = tmp_path / "prepared"
    lme.prepare_dataset(source, prepared_dir, dataset_label="m")
    prepared = lme.load_prepared_dataset(prepared_dir, dataset_label="m")

    direct_tmp = tmp_path / "direct-run"
    prepared_tmp = tmp_path / "prepared-run"
    direct_tmp.mkdir()
    prepared_tmp.mkdir()
    direct = lme.run_harness(
        lme.load_questions(source),
        provider_name="stub",
        model="",
        tmp_dir=direct_tmp,
        dataset_label="m",
        source_sha256=lme.sha256_file(source),
    )
    from_prepared = lme.run_harness(
        prepared.iter_questions(),
        provider_name="stub",
        model="",
        tmp_dir=prepared_tmp,
        question_count=prepared.question_count,
        dataset_label="m",
        source_sha256=prepared.source_sha256,
        manifest_sha256=prepared.manifest_sha256,
    )

    assert from_prepared["dataset"].pop("manifest_sha256") == prepared.manifest_sha256
    assert from_prepared["ingest"]["embedding_batch_size"] == lme.EMBED_BATCH_SIZE
    assert json.dumps(from_prepared, sort_keys=True) == json.dumps(direct, sort_keys=True)


def test_medium_report_records_inherited_embedding_batch_size(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", "2")
    report = lme.run_harness(
        [],
        provider_name="stub",
        model="",
        tmp_dir=tmp_path,
        reuse_db_template=False,
        question_count=0,
        dataset_label="m",
    )

    assert report["ingest"]["embedding_batch_size"] == 2


def test_run_harness_rejects_count_and_digest_mismatches(tmp_path):
    question = lme.parse_question(_raw_question(0))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="question count mismatch"):
        lme.run_harness(
            [question],
            provider_name="stub",
            model="",
            tmp_dir=run_dir,
            reuse_db_template=False,
            question_count=2,
            dataset_label="m",
        )
    with pytest.raises(ValueError, match="source_sha256"):
        lme.run_harness(
            [],
            provider_name="stub",
            model="",
            tmp_dir=run_dir,
            reuse_db_template=False,
            source_sha256="invalid",
        )
    with pytest.raises(ValueError, match="manifest_sha256"):
        lme.run_harness(
            [],
            provider_name="stub",
            model="",
            tmp_dir=run_dir,
            reuse_db_template=False,
            manifest_sha256="invalid",
        )


class _RecordingStub(lme.StubEmbedder):
    def __init__(self):
        super().__init__()
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        batch = list(texts)
        self.calls.append(batch)
        return super().embed_documents(batch)


def test_batched_embedding_preserves_order_and_single_call_values(monkeypatch):
    texts = ["alpha", "bravo", "charlie", "delta", "echo"]
    expected = [lme.StubEmbedder().embed_documents([text])[0] for text in texts]
    recorder = _RecordingStub()
    monkeypatch.setenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", "2")

    actual = lme._embed_in_batches(recorder, texts)

    assert recorder.calls == [["alpha", "bravo"], ["charlie", "delta"], ["echo"]]
    assert actual == expected


@pytest.mark.parametrize(("value", "message"), [("abc", "integer"), ("0", "positive")])
def test_embedding_batch_size_rejects_invalid_environment_values(monkeypatch, value, message):
    monkeypatch.setenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", value)
    with pytest.raises(ValueError, match=message):
        lme._embedding_batch_size()


def test_embedding_batches_reject_provider_vector_count_mismatch():
    class _DroppingStub(lme.StubEmbedder):
        def embed_documents(self, texts):
            return super().embed_documents(texts)[:-1]

    with pytest.raises(ValueError, match="returned 1 vectors for 2 texts"):
        lme._embed_in_batches(_DroppingStub(), ["alpha", "bravo"], batch_size=2)


def test_evaluate_question_keeps_session_insertion_order_when_summary_embedding_is_batched(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", "2")
    raw = _raw_question(7)
    question = lme.parse_question(raw)
    recorder = _RecordingStub()

    lme.evaluate_question(
        question,
        recorder,
        provider_name="stub",
        tmp_dir=tmp_path,
        embeddings_enabled=True,
    )

    expected_summaries = [
        lme.deterministic_session_summary(session) for session in raw["haystack_sessions"]
    ]
    assert recorder.calls[-2:] == [expected_summaries[:2], expected_summaries[2:]]
    with closing(sqlite3.connect(tmp_path / "q7.db")) as conn:
        with conn:
            inserted = conn.execute(
                "SELECT session_id, created_at FROM summary_nodes ORDER BY node_id"
            ).fetchall()
    assert inserted == list(zip(raw["haystack_session_ids"], [1.0, 2.0, 3.0]))


# Deliberately do not round floats here: byte identity is the compatibility anchor
# for the single-platform banked run. Cross-platform libm last-bit drift is a known
# caveat, but rounding the emitted metrics would itself change the banked schema.
def test_small_default_cli_report_is_byte_identical_to_golden(tmp_path, monkeypatch):
    """Freeze CLI-emitted bytes; regenerate only from the pinned banked platform.

    The CLI run, dataset-block, and structural assertions execute on EVERY
    platform (so CI enforces them); only the final byte-identity hash is
    gated to the pinned banked platform.
    """
    cli = _load_cli()
    monkeypatch.delenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", raising=False)
    _zero_timing(monkeypatch)
    source, _rows = _write_dataset(tmp_path, label="s", count=1)
    output_dir = tmp_path / "output"
    temp_root = tmp_path / "tmp-root"
    temp_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp_root))
    monkeypatch.setenv("HERMES_HOME", str(temp_root / "hermes-home"))
    monkeypatch.setattr(tempfile, "tempdir", None)
    captured_tmp_dir = None
    original_run_harness = cli.run_harness

    def _capture_tmp_dir(*args, **kwargs):
        nonlocal captured_tmp_dir
        captured_tmp_dir = Path(kwargs["tmp_dir"])
        return original_run_harness(*args, **kwargs)

    monkeypatch.setattr(cli, "run_harness", _capture_tmp_dir)
    assert cli.main(
        [
            "run", "--dataset", str(source), "--provider", "stub",
            "--output", str(output_dir), "--allow-external-output", "--json",
        ]
    ) == 0

    report_bytes = (output_dir / "longmemeval_metrics.json").read_bytes()
    report = json.loads(report_bytes)
    banked_report = json.loads(_BANKED_METRICS.read_bytes())
    assert captured_tmp_dir is not None
    assert captured_tmp_dir.parent == temp_root
    assert report["dataset"] == banked_report["dataset"]
    assert "source_sha256" not in report["dataset"]
    assert "manifest_sha256" not in report["dataset"]
    assert "embedding_batch_size" not in report["ingest"]
    if (sys.platform, platform.machine()) != _GOLDEN_PLATFORM:
        pytest.skip(
            "byte-identity enforced only on the pinned platform (Darwin arm64) — "
            f"dataset-block/structural assertions ran; regenerate with: {_GOLDEN_REGEN_COMMAND}"
        )
    # This full-report byte hash is intentional. If it legitimately breaks, run the
    # CLI on the pinned platform, verify the dataset block field-by-field, then re-bank
    # the hash and golden file together in the same commit.
    assert hashlib.sha256(report_bytes).hexdigest() == (
        "b8952714d53f1ae819770c513d42421cdf6396bced2dc03f2aa8ca8b2209bc07"
    )
