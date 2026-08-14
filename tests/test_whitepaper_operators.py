from __future__ import annotations

import csv
import json
import sqlite3
import threading
from pathlib import Path

from hermes_lcm.file_registry import FileRegistry, union_file_ids
from hermes_lcm.operators import AgenticMap, LLMMap, OperatorStore


OBJECT_SCHEMA = {
    "type": "object",
    "required": ["value"],
    "additionalProperties": False,
    "properties": {"value": {"type": "integer"}},
}


def _jsonl(path: Path, values: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def test_file_registry_keeps_only_path_and_type_aware_json_summary(tmp_path: Path):
    source = tmp_path / "large.json"
    marker = "payload-that-must-not-be-copied-into-sqlite"
    source.write_text(
        json.dumps({"users": [{"id": 1, "secret": marker}], "active": True}),
        encoding="utf-8",
    )
    db_path = tmp_path / "lcm.db"
    registry = FileRegistry(db_path)

    first = registry.register(source)
    second = registry.register(source)

    assert first.file_id == second.file_id
    assert first.file_id.startswith("file_")
    assert first.path == str(source.resolve())
    assert first.mime_type == "application/json"
    assert first.size_bytes == source.stat().st_size
    assert first.token_count > 0
    assert "users" in first.exploration_summary
    assert "active" in first.exploration_summary
    assert marker not in first.exploration_summary
    assert marker.encode() not in db_path.read_bytes()


def test_file_registry_summarizes_csv_sqlite_python_and_text(tmp_path: Path):
    registry = FileRegistry(tmp_path / "lcm.db")

    csv_path = tmp_path / "rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "score"])
        writer.writerow(["Ada", 10])
        writer.writerow(["Lin", 20])

    sqlite_path = tmp_path / "catalog.sqlite"
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT NOT NULL)")

    python_path = tmp_path / "service.py"
    python_path.write_text(
        "class Service:\n"
        "    def run(self, count: int) -> str:\n"
        "        return str(count)\n\n"
        "def helper(value):\n"
        "    return value\n",
        encoding="utf-8",
    )

    text_path = tmp_path / "notes.txt"
    text_path.write_text("first line\nsecond line\nthird line\n", encoding="utf-8")

    csv_record = registry.register(csv_path)
    sqlite_record = registry.register(sqlite_path)
    python_record = registry.register(python_path)
    text_record = registry.register(text_path)

    assert "name" in csv_record.exploration_summary
    assert "2 data rows" in csv_record.exploration_summary
    assert "books" in sqlite_record.exploration_summary
    assert "title TEXT" in sqlite_record.exploration_summary
    assert "class Service" in python_record.exploration_summary
    assert "run(self, count: int)" in python_record.exploration_summary
    assert "3 lines" in text_record.exploration_summary
    assert "6 words" in text_record.exploration_summary


def test_file_registry_summarizes_jsonl_sql_and_generic_code(tmp_path: Path):
    registry = FileRegistry(tmp_path / "lcm.db")
    jsonl_path = tmp_path / "events.jsonl"
    _jsonl(jsonl_path, [{"kind": "open", "count": 1}, {"kind": "close", "count": 2}])
    sql_path = tmp_path / "schema.sql"
    sql_path.write_text(
        "CREATE TABLE accounts (id INTEGER);\nCREATE VIEW account_ids AS SELECT id FROM accounts;\n",
        encoding="utf-8",
    )
    code_path = tmp_path / "worker.js"
    code_path.write_text(
        "export class Worker {}\nexport function execute(item) { return item; }\n",
        encoding="utf-8",
    )

    jsonl_record = registry.register(jsonl_path)
    sql_record = registry.register(sql_path)
    code_record = registry.register(code_path)

    assert "2 records" in jsonl_record.exploration_summary
    assert "count: integer" in jsonl_record.exploration_summary
    assert "accounts" in sql_record.exploration_summary
    assert "account_ids" in sql_record.exploration_summary
    assert "class Worker" in code_record.exploration_summary
    assert "function execute(item)" in code_record.exploration_summary


def test_file_id_union_propagates_lineage_without_collecting_unrelated_ids():
    values = [
        {"file_id": "file_a", "source_ids": [10, 11]},
        {"file_ids": ["file_b", "file_a"]},
        {"metadata": {"file_ids": ["file_c"]}, "id": "not-a-file"},
    ]

    assert union_file_ids(*values) == ("file_a", "file_b", "file_c")


def test_file_id_union_finds_ids_in_serialized_tool_results():
    assert union_file_ids(
        '{"output_file_id":"file_0123456789abcdef"}'
    ) == ("file_0123456789abcdef",)


def test_llm_map_runs_jsonl_concurrently_and_persists_completed_state(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _jsonl(input_path, [{"n": value} for value in range(12)])
    lock = threading.Lock()
    calls: dict[int, int] = {}

    def executor(*, item, prompt, attempt, validation_error):
        assert prompt == "double"
        assert validation_error is None
        with lock:
            calls[item["n"]] = calls.get(item["n"], 0) + 1
        return {"value": item["n"] * 2}

    operator = LLMMap(tmp_path / "lcm.db", executor=executor)
    result = operator.run(
        input_path=input_path,
        output_path=output_path,
        prompt="double",
        output_schema=OBJECT_SCHEMA,
    )

    assert result.status == "completed"
    assert result.total == 12
    assert result.completed == 12
    assert result.failed == 0
    assert result.concurrency == 16
    assert result.output_file_id is not None
    assert operator.file_registry.get(result.output_file_id).path == str(output_path)
    assert calls == {value: 1 for value in range(12)}
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["output"]["value"] for row in rows] == [value * 2 for value in range(12)]
    assert all(row["status"] == "completed" for row in rows)

    stored = operator.status(result.batch_id)
    assert stored == result


def test_llm_map_retries_schema_failures_with_validation_feedback(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _jsonl(input_path, [{"n": 4}])
    feedback: list[str | None] = []

    def executor(*, item, prompt, attempt, validation_error):
        feedback.append(validation_error)
        if attempt == 1:
            return {"value": "wrong type"}
        return {"value": item["n"]}

    result = LLMMap(tmp_path / "lcm.db", executor=executor).run(
        input_path=input_path,
        output_path=output_path,
        prompt="extract",
        output_schema=OBJECT_SCHEMA,
        max_retries=2,
        concurrency=1,
    )

    assert result.status == "completed"
    assert result.completed == 1
    assert feedback[0] is None
    assert "integer" in feedback[1]
    row = json.loads(output_path.read_text())
    assert row["attempts"] == 2
    assert row["output"] == {"value": 4}


def test_operator_store_claim_is_atomic_and_resume_requeues_abandoned_work(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _jsonl(input_path, [{"n": 1}, {"n": 2}])
    db_path = tmp_path / "lcm.db"
    store = OperatorStore(db_path)
    batch_id = store.create_batch(
        kind="llm_map",
        input_path=input_path,
        output_path=output_path,
        prompt="copy",
        output_schema=OBJECT_SCHEMA,
        concurrency=2,
        max_retries=0,
    )

    claimed: list[int] = []
    barrier = threading.Barrier(3)

    def claim(worker: str) -> None:
        barrier.wait()
        item = store.claim_next(batch_id, worker)
        assert item is not None
        claimed.append(item.item_index)

    threads = [threading.Thread(target=claim, args=(f"w{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(claimed) == [0, 1]
    assert store.requeue_running(batch_id) == 2

    calls: list[int] = []

    def executor(*, item, prompt, attempt, validation_error):
        calls.append(item["n"])
        return {"value": item["n"]}

    result = LLMMap(db_path, executor=executor).resume(batch_id)
    assert result.status == "completed"
    assert sorted(calls) == [1, 2]


def test_agentic_map_passes_enforced_read_only_capability(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _jsonl(input_path, [{"path": "one"}])
    observed: list[bool] = []

    def subagent(*, item, prompt, attempt, validation_error, read_only):
        observed.append(read_only)
        return {"value": 1}

    operator = AgenticMap(tmp_path / "lcm.db", executor=subagent)
    result = operator.run(
        input_path=input_path,
        output_path=output_path,
        prompt="inspect",
        output_schema=OBJECT_SCHEMA,
        read_only=True,
    )

    assert result.status == "completed"
    assert observed == [True]


def test_malformed_input_and_permanent_failure_are_durable(tmp_path: Path):
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
    operator = LLMMap(tmp_path / "lcm.db", executor=lambda **_: {"value": 1})

    try:
        operator.run(
            input_path=malformed,
            output_path=tmp_path / "out.jsonl",
            prompt="x",
            output_schema=OBJECT_SCHEMA,
        )
    except ValueError as exc:
        assert "line 2" in str(exc)
    else:
        raise AssertionError("malformed JSONL was accepted")

    valid = tmp_path / "valid.jsonl"
    _jsonl(valid, [{"n": 1}])
    failed = LLMMap(
        tmp_path / "failed.db",
        executor=lambda **_: {"value": "invalid"},
    ).run(
        input_path=valid,
        output_path=tmp_path / "failed-output.jsonl",
        prompt="x",
        output_schema=OBJECT_SCHEMA,
        max_retries=1,
    )
    assert failed.status == "completed_with_errors"
    assert failed.failed == 1
    row = json.loads((tmp_path / "failed-output.jsonl").read_text())
    assert row["status"] == "failed"
    assert row["attempts"] == 2
    assert "integer" in row["error"]
