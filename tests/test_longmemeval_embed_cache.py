"""Content-hash embedding cache and LongMemEval cache CLI regressions."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import benchmarking.longmemeval as lme
from tests.conftest import load_cli as _load_cli


class _CountingProvider:
    provider_id = "voyage"
    dim = 2

    def __init__(self, model_id: str = "voyage-test", *, offset: float = 0.0):
        self.model_id = model_id
        self.offset = offset
        self.calls = 0
        self.documents: list[list[str]] = []

    def embed_documents(self, texts):
        batch = [str(text) for text in texts]
        self.calls += 1
        self.documents.append(batch)
        return [
            [self.offset + float(len(text)), self.offset + float(sum(map(ord, text)))]
            for text in batch
        ]

    def embed_query(self, text):
        return [self.offset, float(len(str(text)))]


class TestCacheHit:
    def test_hit_skips_provider_call(self, tmp_path):
        raw = _CountingProvider()
        cached = lme.ContentHashEmbeddingCache(raw, tmp_path / "embeddings.db")

        expected = cached.embed_documents(["same document"])
        actual = cached.embed_documents(["same document"])

        assert actual == expected
        assert raw.calls == 1
        assert cached.hits == 1
        assert cached.misses == 1


class TestCacheMiss:
    def test_miss_populates_sqlite_row(self, tmp_path):
        raw = _CountingProvider()
        path = tmp_path / "embeddings.db"
        cached = lme.ContentHashEmbeddingCache(raw, path)

        vector = cached.embed_documents(["new document"])[0]

        assert raw.calls == 1
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                "SELECT provider, model, content_sha256, vector_dim, length(vector_f64_le) "
                "FROM embedding_cache"
            ).fetchone()
        assert row == (
            "voyage",
            "voyage-test",
            lme.ContentHashEmbeddingCache.content_sha256("new document"),
            len(vector),
            len(vector) * 8,
        )


class TestCacheIdentity:
    def test_provider_and_model_are_part_of_key(self, tmp_path):
        path = tmp_path / "embeddings.db"
        first = _CountingProvider("model-a", offset=1.0)
        second = _CountingProvider("model-b", offset=2.0)
        other_provider = _CountingProvider("model-a", offset=3.0)

        vector_a = lme.ContentHashEmbeddingCache(first, path).embed_documents(["shared"])
        vector_b = lme.ContentHashEmbeddingCache(second, path).embed_documents(["shared"])
        vector_other = lme.ContentHashEmbeddingCache(
            other_provider, path, provider_id="other"
        ).embed_documents(["shared"])

        assert first.calls == second.calls == other_provider.calls == 1
        assert vector_a != vector_b != vector_other
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT count(*) FROM embedding_cache").fetchone()[0] == 3


class TestCacheEnvGate:
    def test_unset_env_returns_provider_object_unchanged(self, monkeypatch):
        raw = _CountingProvider()
        monkeypatch.delenv(lme.EMBED_CACHE_ENV, raising=False)

        resolved = lme._maybe_cache_harness_provider(raw, provider_name="voyage")

        assert resolved is raw


class TestCacheConcurrentWriters:
    def test_two_threads_use_wal_without_corruption(self, tmp_path):
        path = tmp_path / "embeddings.db"
        barrier = threading.Barrier(2)

        class _RacingProvider(_CountingProvider):
            def embed_documents(self, texts):
                barrier.wait(timeout=5)
                return super().embed_documents(texts)

        left_raw = _RacingProvider(offset=10.0)
        right_raw = _RacingProvider(offset=20.0)
        left = lme.ContentHashEmbeddingCache(left_raw, path)
        right = lme.ContentHashEmbeddingCache(right_raw, path)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(left.embed_documents, ["raced document"]),
                executor.submit(right.embed_documents, ["raced document"]),
            ]
            results = [future.result(timeout=10) for future in futures]

        assert results[0] == results[1]
        with sqlite3.connect(path) as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert connection.execute("SELECT count(*) FROM embedding_cache").fetchone()[0] == 1


def _question() -> lme.Question:
    return lme.Question(
        question_id="q0",
        question_type="single-session-user",
        question="what was said?",
        haystack_session_ids=["s0"],
        haystack_sessions=[[{"role": "user", "content": "cache this exact content"}]],
        answer_session_ids=["s0"],
    )


def test_prewarm_is_resumable_and_skips_all_warm_units(tmp_path):
    raw = _CountingProvider()
    cached = lme.ContentHashEmbeddingCache(raw, tmp_path / "embeddings.db")

    first = lme.prewarm_embedding_cache([_question()], cached)
    calls_after_first = raw.calls
    second = lme.prewarm_embedding_cache([_question()], cached)

    assert first["unique_request_units"] >= 1
    assert first["populated"] == first["unique_request_units"]
    assert second["already_cached"] == second["unique_request_units"]
    assert second["populated"] == 0
    assert raw.calls == calls_after_first


def test_cache_cli_subcommands_parse_without_execution():
    cli = _load_cli()

    prewarm = cli._parse_args(
        [
            "prewarm-cache",
            "--prepared-dir",
            "prepared",
            "--shards-manifest",
            "shards",
            "--model",
            "voyage-3-large",
        ]
    )
    probe = cli._parse_args(
        [
            "determinism-probe",
            "--prepared-dir",
            "prepared",
            "--shards-manifest",
            "shards",
            "--model",
            "voyage-3-large",
        ]
    )

    assert prewarm.command == "prewarm-cache"
    assert probe.command == "determinism-probe"
    assert probe.sample_size == 20


def test_empty_cache_env_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, "")

    with pytest.raises(ValueError, match="non-empty SQLite path"):
        lme._maybe_cache_harness_provider(_CountingProvider(), provider_name="voyage")


def test_cache_database_creation_syncs_parent_directory(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(lme, "_fsync_parent_directory", synced.append)
    path = tmp_path / "embeddings.db"

    lme.ContentHashEmbeddingCache(_CountingProvider(), path)

    assert synced == [path]


def test_report_discloses_cache_stats_only_when_env_is_set(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lme, "evaluate_question", lambda question, *_args, **_kwargs: {
            **{
                arm: {
                    "recall@1": 1.0,
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "ndcg@10": 1.0,
                    "latency_ms": 1.0,
                    "turn": {
                        "recall@1": 1.0,
                        "recall@5": 1.0,
                        "recall@10": 1.0,
                        "ndcg@10": 1.0,
                        "session_granularity": False,
                    },
                }
                for arm in lme.ARMS
            },
            "ingest_ms": 1.0,
        },
    )
    monkeypatch.delenv(lme.EMBED_CACHE_ENV, raising=False)
    without_cache = lme.run_harness(
        [_question()], provider_name="stub", model="", tmp_dir=tmp_path, reuse_db_template=False
    )
    assert "embed_cache" not in without_cache["ingest"]

    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.db"))
    with_cache = lme.run_harness(
        [_question()], provider_name="stub", model="", tmp_dir=tmp_path, reuse_db_template=False
    )
    assert with_cache["ingest"]["embed_cache"] == {"hits": 0, "misses": 0}


def test_fastembed_prewarm_resolves_with_run_path_warmup(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.db"))
    monkeypatch.setattr(cli, "_prepared_shard_questions", lambda _args: [])
    calls = []

    class _Provider:
        pass

    def _resolve(*args, **kwargs):
        calls.append((args, kwargs))
        return _Provider()

    monkeypatch.setattr(cli, "resolve_harness_provider", _resolve)
    monkeypatch.setattr(cli, "prewarm_embedding_cache", lambda *_args, **_kwargs: {})
    args = cli._parse_args(
        [
            "prewarm-cache",
            "--prepared-dir",
            "prepared",
            "--shards-manifest",
            "shards",
            "--provider",
            "fastembed",
            "--model",
            "local-model",
        ]
    )

    assert cli._cmd_prewarm_cache(args) == 0
    assert calls == [(("fastembed", "local-model"), {"timeout": 300.0, "warmup": True})]
