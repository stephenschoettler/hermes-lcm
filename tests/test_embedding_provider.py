from __future__ import annotations

import json
import logging
import threading
import time
from types import SimpleNamespace

import pytest

import hermes_lcm.command as command_mod
import hermes_lcm.embedding_provider as provider_mod
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.embedding_provider import (
    EmbeddingCircuitBreaker,
    EmbeddingProviderError,
    EmbeddingSpendGuard,
    FastembedProvider,
    HttpResponse,
    OpenAICompatibleProvider,
    OllamaProvider,
    ProviderCircuitOpen,
    ProviderNotWarmedUp,
    ProviderPreDispatchError,
    ProviderRateLimited,
    ProviderUnavailable,
    VoyageError,
    VoyageProvider,
    resolve_provider,
)
from hermes_lcm.vector_store import VectorStore


def _response(status: int, payload, headers=None) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers=headers or {},
        body=json.dumps(payload).encode(),
    )


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _voyage_success(count: int, dim: int = 3) -> HttpResponse:
    return _response(
        200,
        {
            "data": [
                {"index": index, "embedding": [float(index + 1)] * dim}
                for index in range(count)
            ]
        },
    )


def test_voyage_batch_bin_packing_boundary(monkeypatch):
    token_counts = {"a": 24_000, "b": 24_000, "c": 24_000, "d": 1}
    monkeypatch.setattr(provider_mod, "count_tokens", token_counts.__getitem__)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_success(3), _voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport)

    vectors = provider.embed_documents(["a", "b", "c", "d"])

    assert len(vectors) == 4
    assert [call["payload"]["input"] for call in transport.calls] == [
        ["a", "b", "c"],
        ["d"],
    ]
    assert all(call["payload"]["truncation"] is False for call in transport.calls)


def test_voyage_flat_embeddings_record_provider_billed_tokens(monkeypatch):
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    document_response = _voyage_success(1)
    document_payload = json.loads(document_response.body)
    document_payload["usage"] = {"total_tokens": 17}
    query_response = _voyage_success(1)
    query_payload = json.loads(query_response.body)
    query_payload["usage"] = {"total_tokens": 5}
    provider = VoyageProvider(
        "voyage-test",
        transport=FakeTransport(
            _response(200, document_payload),
            _response(200, query_payload),
        ),
    )

    provider.embed_documents(["document"])
    assert provider.last_usage_tokens == 17
    provider.embed_query("query")
    assert provider.last_usage_tokens == 5


def test_voyage_batch_splits_on_item_count_cap(monkeypatch):
    # Many tiny documents stay well under the token budget but exceed Voyage's
    # 1000-item per-request cap, so the batch must split.
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_success(1000), _voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport)

    docs = [f"doc-{index}" for index in range(1001)]
    vectors = provider.embed_documents(docs)

    assert len(vectors) == 1001
    assert [len(call["payload"]["input"]) for call in transport.calls] == [1000, 1]


def test_voyage_item_cap_is_configurable(monkeypatch):
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_success(2), _voyage_success(2), _voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport, max_batch_items=2)

    provider.embed_documents([f"doc-{index}" for index in range(5)])
    assert [len(call["payload"]["input"]) for call in transport.calls] == [2, 2, 1]


def test_voyage_absolute_deadline_bounds_total_retry_time(monkeypatch):
    # A tiny budget must return in ~that budget: the backoff between attempts
    # would blow the deadline, so no retry stacks up despite max_attempts>1.
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    slept: list[float] = []
    # Every attempt returns a retryable 500; without an absolute deadline this
    # would sleep 0.5 + 1.0s across three attempts.
    transport = FakeTransport(
        _response(500, {"error": "x"}),
        _response(500, {"error": "x"}),
        _response(500, {"error": "x"}),
    )
    provider = VoyageProvider("voyage-test", transport=transport, sleeper=slept.append)

    with pytest.raises(VoyageError):
        provider.embed_query_interactive("q", timeout=0.02)

    # One attempt, zero backoff sleeps (the 0.5s backoff would exceed 0.02s).
    assert len(transport.calls) == 1
    assert slept == []


def test_voyage_normal_embed_query_bounds_total_retry_time(monkeypatch):
    # Maintainer repro (B1): normal embed_query() with timeout=0.02 + 3 retryable
    # failures made three attempts and took ~1.501s (the 0.5s + 1.0s backoffs sat
    # outside any total budget). The NORMAL path must get ONE absolute deadline
    # too, just like the interactive path: a tiny budget returns in ~that budget.
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    slept: list[float] = []
    transport = FakeTransport(
        _response(500, {"error": "x"}),
        _response(500, {"error": "x"}),
        _response(500, {"error": "x"}),
    )
    provider = VoyageProvider(
        "voyage-test", transport=transport, timeout=0.02, sleeper=slept.append
    )

    with pytest.raises(VoyageError):
        provider.embed_query("q")

    # One attempt, zero backoff sleeps (the 0.5s backoff would exceed 0.02s).
    assert len(transport.calls) == 1
    assert slept == []


def test_ollama_normal_embed_query_bounds_total_retry_time():
    # B1 for Ollama's normal path: a tiny timeout budget bounds all attempts +
    # backoff so retryable network failures cannot stack backoff sleeps.
    slept: list[float] = []
    transport = FakeTransport(OSError("boom"), OSError("boom"), OSError("boom"))
    provider = OllamaProvider(
        "model", transport=transport, timeout=0.02, sleeper=slept.append
    )

    with pytest.raises(EmbeddingProviderError):
        provider.embed_query("q")

    assert len(transport.calls) == 1
    assert slept == []


def test_voyage_item_cap_clamped_to_hard_limit(monkeypatch):
    # Maintainer repro (B2): configuring max_batch_items=2000 emitted a request
    # with 1,001 inputs. Voyage's hard 1,000 cap is authoritative regardless of
    # config: clamp down so no request ever exceeds 1000 inputs.
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_success(1000), _voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport, max_batch_items=2000)

    assert provider.max_batch_items == 1000
    provider.embed_documents([f"doc-{index}" for index in range(1001)])

    sizes = [len(call["payload"]["input"]) for call in transport.calls]
    assert sizes == [1000, 1]
    assert all(size <= 1000 for size in sizes)


def test_voyage_over_cap_document_is_skipped_and_reported(monkeypatch, caplog):
    monkeypatch.setattr(
        provider_mod,
        "count_tokens",
        lambda text: 24_301 if text == "too large" else 2,
    )
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport)

    with caplog.at_level(logging.WARNING):
        vectors = provider.embed_documents(["keep", "too large"])

    assert len(vectors) == 1
    assert provider.last_skipped_documents == [1]
    assert transport.calls[0]["payload"]["input"] == ["keep"]
    assert "index=1" in caplog.text


def test_voyage_uses_asymmetric_input_types(monkeypatch):
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_success(2), _voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport)

    provider.embed_documents(["first", "second"])
    provider.embed_query("question")

    assert [call["payload"]["input_type"] for call in transport.calls] == [
        "document",
        "query",
    ]


def test_voyage_429_honors_retry_after_and_caps_budget(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    sleeps = []
    retry = _response(429, {"error": "slow down"}, {"Retry-After": "2.5"})
    transport = FakeTransport(retry, _voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport, sleeper=sleeps.append)

    assert provider.embed_query("question")
    assert sleeps == [2.5]

    capped_transport = FakeTransport(
        _response(429, {"error": "slow down"}, {"Retry-After": "61"})
    )
    capped = VoyageProvider("voyage-test", transport=capped_transport, sleeper=sleeps.append)
    with pytest.raises(VoyageError, match="429") as exc_info:
        capped.embed_query("question")
    assert exc_info.value.kind == "rate_limit"
    assert sleeps == [2.5]


def test_voyage_5xx_fails_closed_without_ambiguous_resend(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    server_error = _response(503, {"error": "unavailable"})
    transport = FakeTransport(server_error, server_error, server_error)
    sleeps = []
    provider = VoyageProvider("voyage-test", transport=transport, sleeper=sleeps.append)

    with pytest.raises(VoyageError) as exc_info:
        provider.embed_query("question")

    assert exc_info.value.kind == "server_error"
    assert len(transport.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    ("status", "kind"),
    [(400, "bad_request"), (401, "auth"), (403, "auth")],
)
def test_voyage_4xx_is_classified_without_retry(monkeypatch, status, kind):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_response(status, {"error": "rejected"}))
    provider = VoyageProvider("voyage-test", transport=transport, sleeper=lambda _delay: None)

    with pytest.raises(VoyageError) as exc_info:
        provider.embed_query("question")

    assert exc_info.value.kind == kind
    assert len(transport.calls) == 1


def test_voyage_error_logging_scrubs_echoed_inputs(monkeypatch, caplog):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    body = {
        "error": {"kind": "invalid_request", "message": "secret error content"},
        "detail": "private detail content",
        "msg": "private message content",
        "input": "private prompt",
        "nested": {"texts": ["secret text"], "documents": ["secret doc"]},
    }
    provider = VoyageProvider("voyage-test", transport=FakeTransport(_response(400, body)))

    with caplog.at_level(logging.WARNING), pytest.raises(VoyageError):
        provider.embed_query("question")

    for private_text in (
        "secret error content",
        "private detail content",
        "private message content",
        "private prompt",
        "secret text",
        "secret doc",
    ):
        assert private_text not in caplog.text
    assert "status=400" in caplog.text
    assert "REDACTED" in caplog.text


def test_interactive_http_calls_use_one_attempt_and_explicit_timeout(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    voyage_transport = FakeTransport(OSError("offline"), _voyage_success(1))
    voyage = VoyageProvider(
        "voyage-test",
        transport=voyage_transport,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VoyageError, match="network"):
        voyage.embed_query_interactive("question", timeout=0.125)

    # The absolute deadline equals the budget, so the backoff after the first
    # failure would blow it and no retry is attempted: exactly one HTTP call,
    # each attempt bounded by (approximately) the remaining budget.
    assert len(voyage_transport.calls) == 1
    assert voyage_transport.calls[0]["timeout"] == pytest.approx(0.125, abs=0.02)

    ollama_transport = FakeTransport(OSError("offline"), _response(200, {}))
    ollama = OllamaProvider(
        "model",
        transport=ollama_transport,
        sleeper=lambda _delay: None,
    )
    with pytest.raises(EmbeddingProviderError, match="network"):
        ollama.embed_query_interactive("question", timeout=0.25)

    assert len(ollama_transport.calls) == 1
    assert ollama_transport.calls[0]["timeout"] == pytest.approx(0.25, abs=0.02)


def test_voyage_network_errors_are_classified(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(OSError("offline"), OSError("offline"), OSError("offline"))
    provider = VoyageProvider("voyage-test", transport=transport, sleeper=lambda _delay: None)

    with pytest.raises(VoyageError) as exc_info:
        provider.embed_query("question")

    assert exc_info.value.kind == "network"


def test_ollama_request_shape_and_ambiguous_network_failure_is_not_retried():
    transport = FakeTransport(OSError("starting"), _response(200, {"embeddings": [[1, 2]]}))
    sleeps = []
    provider = OllamaProvider(
        "nomic-embed-text",
        base_url="http://ollama.internal:11434/",
        transport=transport,
        sleeper=sleeps.append,
    )

    with pytest.raises(EmbeddingProviderError, match="network"):
        provider.embed_query("hello")
    assert len(transport.calls) == 1
    assert transport.calls[0]["url"] == "http://ollama.internal:11434/api/embed"
    assert transport.calls[0]["payload"] == {
        "model": "nomic-embed-text",
        "input": ["hello"],
        "truncate": False,
    }
    assert sleeps == []

    http_error = FakeTransport(_response(500, {"error": "broken"}))
    provider = OllamaProvider("model", transport=http_error, sleeper=sleeps.append)
    with pytest.raises(EmbeddingProviderError, match="500"):
        provider.embed_query("hello")
    assert len(http_error.calls) == 1


class FakeFastembedModel:
    constructions = []

    def __init__(self, **kwargs):
        self.constructions.append(kwargs)
        self.embed_calls = []
        self.query_calls = []
        if kwargs["local_files_only"]:
            raise RuntimeError("not cached")

    def embed(self, texts):
        texts = list(texts)
        self.embed_calls.append(texts)
        return ([1.0, float(index)] for index, _text in enumerate(texts))

    def query_embed(self, texts):
        # Distinct marker vector so tests can prove the query API is used for
        # queries instead of the generic passage embed().
        texts = list(texts)
        self.query_calls.append(texts)
        return ([2.0, float(index)] for index, _text in enumerate(texts))


def test_fastembed_not_warmed_never_downloads(monkeypatch, tmp_path):
    FakeFastembedModel.constructions = []
    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: FakeFastembedModel)
    provider = FastembedProvider("local-model", cache_dir=tmp_path)

    with pytest.raises(ProviderNotWarmedUp, match="/lcm embed warmup"):
        provider.embed_query("hello")

    assert FakeFastembedModel.constructions == [{
        "model_name": "local-model",
        "cache_dir": str(tmp_path),
        "local_files_only": True,
    }]


def test_fastembed_warmup_is_only_download_path(monkeypatch, tmp_path):
    FakeFastembedModel.constructions = []
    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: FakeFastembedModel)
    provider = FastembedProvider("local-model", cache_dir=tmp_path)

    # warmup() embeds a query, so it goes through the query-specific API.
    assert provider.warmup() == [2.0, 0.0]
    assert provider.dim == 2
    assert FakeFastembedModel.constructions[0]["local_files_only"] is False


def test_fastembed_documents_and_queries_use_distinct_apis(monkeypatch, tmp_path):
    FakeFastembedModel.constructions = []
    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: FakeFastembedModel)
    provider = FastembedProvider("local-model", cache_dir=tmp_path)
    provider.warmup()  # force download-path construction
    model = provider._model

    docs = provider.embed_documents(["a", "b"])
    query = provider.embed_query("q")

    # Documents go through embed(); the query goes through query_embed().
    assert docs == [[1.0, 0.0], [1.0, 1.0]]
    assert query == [2.0, 0.0]
    assert model.embed_calls[-1] == ["a", "b"]
    assert model.query_calls[-1] == ["q"]


def test_fastembed_absent_dependency_is_clean(monkeypatch):
    def missing():
        raise ImportError("no fastembed")

    monkeypatch.setattr(provider_mod, "_load_fastembed", missing)
    provider = FastembedProvider("local-model")

    with pytest.raises(ProviderUnavailable, match="not installed"):
        provider.embed_query("hello")


def test_resolve_provider_strings_and_dormant_defaults(monkeypatch):
    defaults = LCMConfig()
    assert resolve_provider(defaults) is None

    config = LCMConfig(
        embedding_provider="  VOYAGEAI ",
        embedding_model="voyage-3-lite",
        embedding_query_timeout_s=1.25,
    )
    assert isinstance(resolve_provider(config), VoyageProvider)

    config.embedding_provider = "ollama"
    config.ollama_base_url = "http://custom:1234"
    ollama = resolve_provider(config)
    assert isinstance(ollama, OllamaProvider)
    assert ollama.base_url == "http://custom:1234"

    config.embedding_provider = "fast-embed"
    assert isinstance(resolve_provider(config), FastembedProvider)

    config.embedding_provider = "unsupported"
    with pytest.raises(ProviderUnavailable, match="Unsupported"):
        resolve_provider(config)

    config.embedding_provider = ""
    with pytest.raises(ProviderUnavailable, match="must both be set"):
        resolve_provider(config)


def test_resolve_provider_for_backfill_bypasses_spend_guard():
    config = LCMConfig(embedding_provider="ollama", embedding_model="model-x")

    interactive = resolve_provider(config)
    assert interactive.spend_guard.max_calls > 0

    backfill = resolve_provider(config, for_backfill=True)
    # A disabled (max_calls=0) guard never blocks, so a bulk backfill of many
    # batches cannot trip the interactive per-minute guard mid-run.
    assert backfill.spend_guard.max_calls == 0
    for _ in range(interactive.spend_guard.max_calls + 5):
        backfill.spend_guard.record_call()
    assert backfill.spend_guard.allows() is True

    # The circuit breaker is untouched — only the spend guard is relaxed.
    assert backfill.breaker.allows() is True


@pytest.mark.parametrize("provider_name", ["voyage", "ollama", "fastembed"])
def test_resolve_provider_for_backfill_uses_separate_bulk_timeout(provider_name):
    config = LCMConfig(
        embedding_provider=provider_name,
        embedding_model="model-x",
        embedding_query_timeout_s=0.02,
        embedding_backfill_timeout_s=12.5,
    )

    interactive = resolve_provider(config)
    backfill = resolve_provider(config, for_backfill=True)

    assert interactive.timeout == 0.02
    assert backfill.timeout == 12.5


def test_fastembed_backfill_is_not_bound_by_interactive_query_timeout(monkeypatch):
    class SlowBulkModel:
        def __init__(self, **_kwargs):
            pass

        def embed(self, texts):
            time.sleep(0.03)
            return [[1.0, 0.0] for _text in texts]

        def query_embed(self, texts):
            return [[0.0, 1.0] for _text in texts]

    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: SlowBulkModel)
    provider = resolve_provider(
        LCMConfig(
            embedding_provider="fastembed",
            embedding_model="local-model",
            embedding_query_timeout_s=0.01,
            embedding_backfill_timeout_s=0.2,
        ),
        for_backfill=True,
    )

    batches = list(
        provider.embed_document_batches(
            ["ordinary backfill document"],
            before_dispatch=lambda _indexes: None,
        )
    )

    assert [batch.indexes for batch in batches] == [(0,)]
    assert [list(vector) for vector in batches[0].vectors] == [[1.0, 0.0]]


def test_embedding_config_defaults_and_environment(monkeypatch):
    defaults = LCMConfig()
    assert defaults.embedding_provider == ""
    assert defaults.embedding_model == ""
    assert defaults.ollama_base_url == "http://localhost:11434"
    assert defaults.embedding_query_timeout_s == 3.0
    assert defaults.embedding_backfill_timeout_s == 120.0
    assert defaults.embedding_max_batch_items == 1000

    monkeypatch.setenv("LCM_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("LCM_EMBEDDING_MODEL", "model-a")
    monkeypatch.setenv("LCM_OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("LCM_EMBEDDING_QUERY_TIMEOUT_S", "4.5")
    monkeypatch.setenv("LCM_EMBEDDING_BACKFILL_TIMEOUT_S", "45.0")
    monkeypatch.setenv("LCM_EMBEDDING_MAX_BATCH_ITEMS", "500")
    configured = LCMConfig.from_env()
    assert configured.embedding_provider == "ollama"
    assert configured.embedding_model == "model-a"
    assert configured.ollama_base_url == "http://ollama:11434"
    assert configured.embedding_query_timeout_s == 4.5
    assert configured.embedding_backfill_timeout_s == 45.0
    assert configured.embedding_max_batch_items == 500


class FakeWarmupProvider:
    provider_id = "ollama"
    model_id = "model-a"

    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def embed_query(self, text):
        self.calls.append(text)
        return self.vector


def _command_engine(tmp_path):
    return SimpleNamespace(
        _config=LCMConfig(
            database_path=str(tmp_path / "warmup.db"), embeddings_enabled=True
        ),
        _store=SimpleNamespace(db_path=tmp_path / "warmup.db"),
    )


def test_warmup_command_probes_and_registers_profile(monkeypatch, tmp_path):
    provider = FakeWarmupProvider([0.1, 0.2, 0.3])
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config: provider)
    engine = _command_engine(tmp_path)
    engine._config.embedding_provider = "ollama"
    engine._config.embedding_model = "model-a"
    stale_provider = FakeWarmupProvider([9.0])
    engine._lcm_embedding_provider_cache = (("ollama", "model-a"), stale_provider)

    result = handle_lcm_command("embed warmup", engine)

    assert "status: ready" in result
    assert "provider: ollama" in result
    assert "model: model-a" in result
    assert "dim: 3" in result
    assert provider.calls == ["warmup"]
    assert engine._lcm_embedding_provider_cache == (
        ("ollama", "model-a"),
        provider,
    )
    store = VectorStore(engine._store.db_path)
    try:
        rows = store.connection.execute(
            "SELECT provider, dim, task FROM lcm_embedding_profile "
            "WHERE model_name = ? ORDER BY task",
            ("model-a",),
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("ollama", 3, "chunk"),
            ("ollama", 3, "summary"),
        ]
    finally:
        store.close()


def test_warmup_command_new_dim_is_a_distinct_identity_no_clobber(monkeypatch, tmp_path):
    # A different dim is a different canonical identity, so warmup registers a
    # new active profile rather than clobbering (or erroring against) the old
    # one — the dim-2 profile and its vectors survive for a future switch back.
    engine = _command_engine(tmp_path)
    store = VectorStore(engine._store.db_path)
    store.register_profile("model-a", "ollama", 2)
    store.close()
    monkeypatch.setattr(
        command_mod,
        "resolve_provider",
        lambda _config: FakeWarmupProvider([0.1, 0.2, 0.3]),
    )

    result = handle_lcm_command("embed warmup", engine)

    assert "status: ready" in result
    assert "dim: 3" in result
    assert "Traceback" not in result
    store = VectorStore(engine._store.db_path)
    try:
        rows = store.connection.execute(
            "SELECT dim, active FROM lcm_embedding_profile "
            "WHERE model_name = 'model-a' AND task = 'summary' ORDER BY dim"
        ).fetchall()
        assert [tuple(r) for r in rows] == [(2, 0), (3, 1)]
    finally:
        store.close()


def test_warmup_registers_voyage_context_chunk_profile(monkeypatch, tmp_path):
    summary = FakeWarmupProvider([0.1, 0.2, 0.3])
    summary.provider_id = "voyage"
    summary.model_id = "voyage-3"
    chunk = FakeWarmupProvider([0.1, 0.2, 0.3, 0.4])
    chunk.provider_id = "voyage"
    chunk.model_id = "voyage-context-4"

    def resolve(config):
        return chunk if config.embedding_model == "voyage-context-4" else summary

    monkeypatch.setattr(command_mod, "resolve_provider", resolve)
    engine = _command_engine(tmp_path)
    engine._config.embedding_provider = "voyage"
    engine._config.embedding_model = "voyage-3"

    result = handle_lcm_command("embed warmup", engine)

    assert "status: ready" in result
    assert "chunk_model: voyage-context-4" in result
    assert "chunk_dim: 4" in result
    assert summary.calls == ["warmup"]
    assert chunk.calls == ["warmup"]
    store = VectorStore(engine._store.db_path)
    try:
        rows = store.connection.execute(
            "SELECT model_name, dim, task FROM lcm_embedding_profile "
            "WHERE active = 1 ORDER BY task"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("voyage-context-4", 4, "chunk"),
            ("voyage-3", 3, "summary"),
        ]
    finally:
        store.close()


def test_summary_profile_lookup_ignores_newer_chunk_profile(tmp_path):
    engine = _command_engine(tmp_path)
    store = VectorStore(engine._store.db_path)
    try:
        summary_identity = store.register_profile("summary-model", "voyage", 3)
        store.register_profile("chunk-model", "voyage", 4, task="chunk")
        row = command_mod._embedding_current_profile(store.connection)
        assert row is not None
        assert row["identity_hash"] == summary_identity
        assert row["task"] == "summary"
    finally:
        store.close()


def test_warmup_command_fastembed_uses_explicit_download(monkeypatch, tmp_path):
    FakeFastembedModel.constructions = []
    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: FakeFastembedModel)
    provider = FastembedProvider("local-model", cache_dir=tmp_path / "cache")
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config: provider)
    engine = _command_engine(tmp_path)

    result = handle_lcm_command("embed warmup", engine)

    assert "status: ready" in result
    assert "download: ready" in result
    assert "provider: fastembed" in result
    assert "no per-call API charge" in result
    assert FakeFastembedModel.constructions[0]["local_files_only"] is False


def test_circuit_breaker_opens_then_cools_down():
    breaker = EmbeddingCircuitBreaker(failure_threshold=2, cooldown_seconds=10)
    transport = FakeTransport(
        _response(400, {"error": "bad"}),
        _response(400, {"error": "bad"}),
    )
    provider = OllamaProvider("model", transport=transport, breaker=breaker)

    with pytest.raises(EmbeddingProviderError):
        provider.embed_query("one")
    with pytest.raises(EmbeddingProviderError):
        provider.embed_query("two")
    with pytest.raises(ProviderCircuitOpen):
        provider.embed_query("three")
    assert len(transport.calls) == 2
    assert breaker.allows(now=breaker._open_until + 0.1) is True


def test_voyage_auth_errors_never_open_circuit_breaker(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "invalid-key")
    breaker = EmbeddingCircuitBreaker(failure_threshold=2, cooldown_seconds=10)
    transport = FakeTransport(
        _response(401, {"error": "invalid key"}),
        _response(401, {"error": "invalid key"}),
        _response(401, {"error": "invalid key"}),
    )
    provider = VoyageProvider(
        "voyage-test",
        transport=transport,
        breaker=breaker,
    )

    for _ in range(3):
        with pytest.raises(VoyageError) as exc_info:
            provider.embed_query("query")
        assert exc_info.value.kind == "auth"

    assert len(transport.calls) == 3
    assert breaker.allows() is True


def test_spend_guard_rate_limits_provider_calls():
    guard = EmbeddingSpendGuard(max_calls=1, window_seconds=100, backoff_seconds=50)
    transport = FakeTransport(_response(200, {"embeddings": [[1, 2]]}))
    provider = OllamaProvider("model", transport=transport, spend_guard=guard)

    assert provider.embed_query("one") == [1.0, 2.0]
    with pytest.raises(ProviderRateLimited):
        provider.embed_query("two")
    assert len(transport.calls) == 1


def test_resolve_provider_query_path_guard_is_generous_and_unthrottled():
    # Regression for #123: the query path (for_backfill=False) must NOT inherit
    # the strict 60/60s default that gutted retrieval; it gets the generous
    # configurable guard and survives a tight loop of 100+ back-to-back embeds.
    config = LCMConfig(embedding_provider="voyage", embedding_model="voyage-4")
    provider = resolve_provider(config, for_backfill=False)
    guard = provider.spend_guard
    assert guard.max_calls == 600
    assert guard.window_seconds == 60.0
    assert guard.backoff_seconds == 60.0

    now = 0.0
    for _ in range(120):
        assert guard.allows(now=now) is True
        guard.record_call(now=now)
        now += 0.001  # 120 calls inside one 60s window, as the benchmark does
    assert guard.allows(now=now) is True


def test_resolve_provider_query_guard_is_configurable():
    # Constructor-arg surface: the LCMConfig fields thread straight into the
    # query-path guard so a benchmark harness can widen or disable it.
    config = LCMConfig(
        embedding_provider="voyage",
        embedding_model="voyage-4",
        embedding_query_spend_max_calls=5,
        embedding_query_spend_window_seconds=30.0,
        embedding_query_spend_backoff_seconds=15.0,
    )
    provider = resolve_provider(config, for_backfill=False)
    assert provider.spend_guard.max_calls == 5
    assert provider.spend_guard.window_seconds == 30.0
    assert provider.spend_guard.backoff_seconds == 15.0


def test_query_guard_env_override(monkeypatch):
    # Env surface: LCM_EMBEDDING_QUERY_SPEND_* overrides the defaults.
    monkeypatch.setenv("LCM_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("LCM_EMBEDDING_MODEL", "voyage-4")
    monkeypatch.setenv("LCM_EMBEDDING_QUERY_SPEND_MAX_CALLS", "1200")
    monkeypatch.setenv("LCM_EMBEDDING_QUERY_SPEND_WINDOW_SECONDS", "90")
    monkeypatch.setenv("LCM_EMBEDDING_QUERY_SPEND_BACKOFF_SECONDS", "45")
    config = LCMConfig.from_env()
    assert config.embedding_query_spend_max_calls == 1200
    assert config.embedding_query_spend_window_seconds == 90.0
    assert config.embedding_query_spend_backoff_seconds == 45.0
    provider = resolve_provider(config, for_backfill=False)
    assert provider.spend_guard.max_calls == 1200


def test_resolve_provider_backfill_path_stays_exempt():
    # Backfill's bulk contract is unchanged: max_calls=0 => allows() always
    # True, record_call() a no-op, even after many calls.
    config = LCMConfig(embedding_provider="voyage", embedding_model="voyage-4")
    provider = resolve_provider(config, for_backfill=True)
    guard = provider.spend_guard
    assert guard.max_calls == 0
    for _ in range(100):
        guard.record_call()
    assert guard.allows() is True


def test_query_path_rate_limit_surfaces_typed_reason(monkeypatch):
    # Zero-discard: when the guard does trip, the caller receives the TYPED
    # ProviderRateLimited with the exact operator-readable message the live
    # probe recorded -- never a bare swallowed counter bump. The rejection is
    # pre-network (no transport call is made for the throttled attempt).
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    guard = EmbeddingSpendGuard(max_calls=1, window_seconds=100, backoff_seconds=50)
    transport = FakeTransport(_voyage_success(1))
    provider = VoyageProvider("voyage", transport=transport, spend_guard=guard)

    assert provider.embed_query("first")
    with pytest.raises(
        ProviderRateLimited,
        match="voyage embedding call-rate guard is cooling down",
    ):
        provider.embed_query("second")
    assert len(transport.calls) == 1


def test_voyage_document_splits_share_one_absolute_deadline(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    calls: list[float] = []

    def transport(**kwargs):
        timeout = float(kwargs["timeout"])
        calls.append(timeout)
        delay = 0.015
        if timeout < delay:
            time.sleep(max(0.0, timeout))
            raise TimeoutError("request exceeded remaining budget")
        time.sleep(delay)
        return _voyage_success(1, dim=2)

    provider = VoyageProvider(
        "voyage-test",
        transport=transport,
        timeout=0.02,
        max_batch_items=1,
        sleeper=lambda _delay: None,
    )
    started = time.monotonic()
    with pytest.raises(VoyageError, match="(network error|deadline exceeded)"):
        provider.embed_documents(["first", "second"])
    elapsed = time.monotonic() - started
    assert 1 <= len(calls) <= 2
    if len(calls) == 2:
        assert calls[1] < calls[0]
    assert elapsed < 0.06


def test_voyage_token_preprocessing_is_inside_absolute_deadline(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

    def slow_count(_text):
        time.sleep(0.1)
        return 1

    monkeypatch.setattr(provider_mod, "count_tokens", slow_count)
    transport = FakeTransport(_voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport, timeout=0.02)

    started = time.monotonic()
    with pytest.raises(VoyageError, match="document preprocessing"):
        provider.embed_documents(["slow"])

    assert time.monotonic() - started < 0.08
    assert transport.calls == []


def test_voyage_does_not_dispatch_next_split_after_deadline(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    transport = FakeTransport(_voyage_success(1), _voyage_success(1))
    provider = VoyageProvider(
        "voyage-test", transport=transport, timeout=0.02, max_batch_items=1
    )

    batches = provider.embed_document_batches(["first", "second"])
    assert next(batches).indexes == (0,)
    time.sleep(0.03)
    with pytest.raises(VoyageError, match="deadline exceeded"):
        next(batches)
    assert len(transport.calls) == 1


def test_voyage_yields_accepted_subbatch_before_later_failure(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    transport = FakeTransport(
        _voyage_success(1, dim=2),
        _response(400, {"error": "bad second request"}),
    )
    provider = VoyageProvider(
        "voyage-test", transport=transport, max_batch_items=1
    )
    batches = provider.embed_document_batches(["first", "second"])
    first = next(batches)
    assert first.indexes == (0,)
    assert first.vectors == ((1.0, 1.0),)
    with pytest.raises(VoyageError, match="400"):
        next(batches)


def test_document_dispatch_callback_runs_once_per_actual_request(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    transport = FakeTransport(_voyage_success(1), _voyage_success(1))
    provider = VoyageProvider(
        "voyage-test", transport=transport, max_batch_items=1
    )
    dispatched: list[tuple[int, ...]] = []

    batches = list(
        provider.embed_document_batches(
            ["first", "second"], before_dispatch=dispatched.append
        )
    )

    assert dispatched == [(0,), (1,)]
    assert [batch.indexes for batch in batches] == [(0,), (1,)]
    assert len(transport.calls) == 2


def test_voyage_rechecks_deadline_after_dispatch_callback(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setattr(provider_mod, "count_tokens", lambda _text: 1)
    transport = FakeTransport(_voyage_success(1))
    provider = VoyageProvider("voyage-test", transport=transport, timeout=0.01)
    dispatched: list[tuple[int, ...]] = []

    def slow_dispatch(indexes):
        dispatched.append(indexes)
        time.sleep(0.03)

    with pytest.raises(ProviderPreDispatchError) as exc_info:
        list(
            provider.embed_document_batches(
                ["document"], before_dispatch=slow_dispatch
            )
        )

    assert exc_info.value.kind == "pre_dispatch"
    assert dispatched == [(0,)]
    assert transport.calls == []


def test_ollama_rechecks_deadline_after_dispatch_callback():
    transport = FakeTransport(_response(200, {"embeddings": [[1.0, 0.0]]}))
    provider = OllamaProvider("model", transport=transport, timeout=0.01)

    def slow_dispatch(_indexes):
        time.sleep(0.03)

    with pytest.raises(ProviderPreDispatchError):
        list(
            provider.embed_document_batches(
                ["document"], before_dispatch=slow_dispatch
            )
        )
    assert transport.calls == []


@pytest.mark.parametrize("provider_name", ["voyage", "ollama"])
def test_http_provider_does_not_decode_response_after_deadline(
    monkeypatch, provider_name
):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    decode_calls = 0
    original_decode = provider_mod._response_json

    def counted_decode(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(provider_mod, "_response_json", counted_decode)

    def slow_transport(**_kwargs):
        time.sleep(0.03)
        if provider_name == "voyage":
            return _voyage_success(1)
        return _response(200, {"embeddings": [[1.0, 0.0]]})

    provider = (
        VoyageProvider("voyage-test", transport=slow_transport, timeout=0.01)
        if provider_name == "voyage"
        else OllamaProvider("model", transport=slow_transport, timeout=0.01)
    )

    with pytest.raises(EmbeddingProviderError, match="response processing"):
        provider.embed_query("too late")

    assert decode_calls == 0


def test_voyage_slow_response_decode_is_inside_absolute_deadline(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    decode_started = threading.Event()
    original_decode = provider_mod._response_json

    def slow_decode(*args, **kwargs):
        decode_started.set()
        time.sleep(0.05)
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(provider_mod, "_response_json", slow_decode)
    provider = VoyageProvider(
        "voyage-test", transport=FakeTransport(_voyage_success(1)), timeout=0.01
    )

    started = time.monotonic()
    with pytest.raises(VoyageError) as exc_info:
        provider.embed_query("slow decode")

    assert exc_info.value.kind == "timeout"
    assert decode_started.is_set()
    assert time.monotonic() - started < 0.04
    # Let the side-effect-free parser worker exit before monkeypatch teardown.
    time.sleep(0.06)


def test_voyage_slow_error_body_scrub_is_bounded_and_never_resent(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    scrub_started = threading.Event()
    original_scrub = provider_mod._scrub_response_body

    def slow_scrub(body):
        scrub_started.set()
        time.sleep(0.05)
        return original_scrub(body)

    monkeypatch.setattr(provider_mod, "_scrub_response_body", slow_scrub)
    transport = FakeTransport(
        _response(503, {"error": "ambiguous"}),
        _voyage_success(1),
    )
    provider = VoyageProvider("voyage-test", transport=transport, timeout=0.01)

    started = time.monotonic()
    with pytest.raises(VoyageError) as exc_info:
        provider.embed_query("slow error body")

    assert exc_info.value.kind == "server_error"
    assert scrub_started.is_set()
    assert len(transport.calls) == 1
    assert time.monotonic() - started < 0.04
    time.sleep(0.06)


def test_voyage_timeout_after_possible_acceptance_is_not_resent(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(
        TimeoutError("response lost after acceptance"),
        _voyage_success(1),
    )
    provider = VoyageProvider(
        "voyage-test", transport=transport, sleeper=lambda _delay: None
    )

    with pytest.raises(VoyageError) as exc_info:
        provider.embed_query("possibly accepted")

    assert exc_info.value.kind == "network"
    assert len(transport.calls) == 1


def test_ollama_timeout_after_possible_acceptance_is_not_resent():
    transport = FakeTransport(
        TimeoutError("response lost after acceptance"),
        _response(200, {"embeddings": [[1.0, 0.0]]}),
    )
    provider = OllamaProvider("model", transport=transport)

    with pytest.raises(EmbeddingProviderError, match="network"):
        provider.embed_query("possibly accepted")

    assert len(transport.calls) == 1


def test_fastembed_normal_operation_is_deadline_bounded(monkeypatch, tmp_path):
    class SlowFastembedModel:
        def __init__(self, **_kwargs):
            pass

        def query_embed(self, texts):
            time.sleep(0.05)
            return ([1.0, 0.0] for _ in texts)

        def embed(self, texts):
            time.sleep(0.05)
            return ([1.0, 0.0] for _ in texts)

    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: SlowFastembedModel)
    provider = FastembedProvider("local", cache_dir=tmp_path, timeout=0.01)
    started = time.monotonic()
    with pytest.raises(EmbeddingProviderError, match="deadline exceeded"):
        provider.embed_query("slow")
    assert time.monotonic() - started < 0.04


def test_fastembed_timeout_capacity_is_bounded_until_worker_exits(monkeypatch, tmp_path):
    calls = 0
    worker_started = threading.Event()
    release_worker = threading.Event()

    class SlowFastembedModel:
        def __init__(self, **_kwargs):
            pass

        def query_embed(self, texts):
            nonlocal calls
            calls += 1
            worker_started.set()
            assert release_worker.wait(timeout=1.0)
            return ([1.0, 0.0] for _ in texts)

    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: SlowFastembedModel)
    monkeypatch.setattr(
        provider_mod, "_LOCAL_EMBED_WORKER_SLOTS", threading.BoundedSemaphore(1)
    )
    first = FastembedProvider("local", cache_dir=tmp_path, timeout=0.01)
    second = FastembedProvider("local", cache_dir=tmp_path, timeout=0.01)

    try:
        with pytest.raises(EmbeddingProviderError, match="deadline exceeded"):
            first.embed_query("slow")
        assert worker_started.is_set()
        started = time.monotonic()
        with pytest.raises(EmbeddingProviderError, match="worker capacity exhausted"):
            second.embed_query("must-not-start")

        assert time.monotonic() - started < 0.03
        assert calls == 1
    finally:
        release_worker.set()


def test_fastembed_preflight_and_capacity_fail_before_dispatch(monkeypatch, tmp_path):
    FakeFastembedModel.constructions = []
    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: FakeFastembedModel)
    provider = FastembedProvider("missing", cache_dir=tmp_path, timeout=0.01)
    dispatched: list[tuple[int, ...]] = []

    with pytest.raises(ProviderNotWarmedUp):
        list(
            provider.embed_document_batches(
                ["document"], before_dispatch=dispatched.append
            )
        )
    assert dispatched == []

    class CachedFastembedModel(FakeFastembedModel):
        def __init__(self, **kwargs):
            self.embed_calls = []
            self.query_calls = []

    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: CachedFastembedModel)
    monkeypatch.setattr(
        provider_mod, "_LOCAL_EMBED_WORKER_SLOTS", threading.BoundedSemaphore(0)
    )
    provider = FastembedProvider("cached", cache_dir=tmp_path, timeout=0.01)
    with pytest.raises(EmbeddingProviderError, match="worker capacity exhausted"):
        list(
            provider.embed_document_batches(
                ["document"], before_dispatch=dispatched.append
            )
        )
    assert dispatched == []


def test_fastembed_does_not_encode_after_dispatch_callback_exhausts_deadline(
    monkeypatch, tmp_path
):
    encode_calls = 0

    class CachedFastembedModel:
        def __init__(self, **_kwargs):
            pass

        def embed(self, texts):
            nonlocal encode_calls
            encode_calls += 1
            return ([1.0, 0.0] for _ in texts)

        query_embed = embed

    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: CachedFastembedModel)
    provider = FastembedProvider("cached", cache_dir=tmp_path, timeout=0.01)

    def slow_dispatch(_indexes):
        time.sleep(0.03)

    with pytest.raises(EmbeddingProviderError, match="deadline exceeded"):
        list(
            provider.embed_document_batches(
                ["document"], before_dispatch=slow_dispatch
            )
        )
    time.sleep(0.04)
    assert encode_calls == 0


def test_warmup_command_is_inert_when_embeddings_disabled(monkeypatch, tmp_path):
    provider = FakeWarmupProvider([0.1, 0.2])
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config: provider)
    engine = _command_engine(tmp_path)
    engine._config.embeddings_enabled = False
    result = handle_lcm_command("embed warmup", engine)
    assert "status: disabled" in result
    assert provider.calls == []
    assert not engine._store.db_path.exists()


def _voyage_rerank_response(rows) -> HttpResponse:
    """rows: iterable of (index, relevance_score)."""
    return _response(
        200,
        {"data": [{"index": index, "relevance_score": score} for index, score in rows]},
    )


def test_voyage_rerank_happy_path_orders_by_relevance(monkeypatch):
    """F2-voyage-rerank-provider-untested: parses (index, score) and returns them
    ordered by descending relevance under one transport call."""
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_rerank_response([(0, 0.10), (1, 0.90), (2, 0.50)]))
    provider = VoyageProvider("voyage-test", transport=transport)

    ranked = provider.rerank("q", ["a", "b", "c"], timeout=5.0)

    assert ranked == [(1, 0.90), (2, 0.50), (0, 0.10)]
    assert len(transport.calls) == 1
    assert transport.calls[0]["url"].endswith("/v1/rerank")


def test_voyage_rerank_drops_out_of_range_index(monkeypatch):
    """An index outside the documents range is skipped, not returned or crashed."""
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_voyage_rerank_response([(0, 0.8), (5, 0.99), (1, 0.2)]))
    provider = VoyageProvider("voyage-test", transport=transport)

    ranked = provider.rerank("q", ["a", "b"], timeout=5.0)

    assert ranked == [(0, 0.8), (1, 0.2)]  # index 5 dropped


def test_voyage_rerank_non_2xx_raises(monkeypatch):
    """A non-2xx response raises a VoyageError (callers treat it as skip-rerank)."""
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport(_response(500, {"error": "boom"}))
    provider = VoyageProvider("voyage-test", transport=transport)

    with pytest.raises(VoyageError):
        provider.rerank("q", ["a", "b"], timeout=5.0)


def test_voyage_rerank_empty_documents_short_circuits(monkeypatch):
    """No documents => [] with NO transport call (no wasted API request)."""
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    transport = FakeTransport()  # no responses queued; must not be called
    provider = VoyageProvider("voyage-test", transport=transport)

    assert provider.rerank("q", [], timeout=5.0) == []
    assert transport.calls == []


def _openai_compatible_success(count: int, dim: int = 3) -> HttpResponse:
    return _response(
        200,
        {
            "data": [
                {"index": index, "embedding": [float(index + 1)] * dim}
                for index in range(count)
            ]
        },
    )


def test_openai_compatible_request_shape_and_success(monkeypatch):
    monkeypatch.setenv("LCM_EMBEDDING_API_KEY", "test-key")
    transport = FakeTransport(_openai_compatible_success(2))
    provider = OpenAICompatibleProvider(
        "BAAI/bge-m3",
        base_url="https://api.example.com/v1/",
        transport=transport,
    )

    vectors = provider.embed_documents(["first", "second"])

    assert len(vectors) == 2
    assert provider.dim == 3
    assert len(transport.calls) == 1
    call = transport.calls[0]
    # Trailing slash on the base URL is stripped before joining.
    assert call["url"] == "https://api.example.com/v1/embeddings"
    assert call["payload"] == {
        "model": "BAAI/bge-m3",
        "input": ["first", "second"],
    }
    assert call["headers"]["Authorization"] == "Bearer test-key"


def test_openai_compatible_query_and_interactive(monkeypatch):
    monkeypatch.setenv("LCM_EMBEDDING_API_KEY", "test-key")
    transport = FakeTransport(
        _openai_compatible_success(1, dim=4),
        _openai_compatible_success(1, dim=4),
    )
    provider = OpenAICompatibleProvider(
        "bge-m3", base_url="http://localhost:8080/v1", transport=transport
    )

    assert provider.embed_query("question") == [1.0, 1.0, 1.0, 1.0]
    assert provider.embed_query_interactive("q", timeout=5.0) == [1.0, 1.0, 1.0, 1.0]

    assert [call["url"] for call in transport.calls] == [
        "http://localhost:8080/v1/embeddings",
        "http://localhost:8080/v1/embeddings",
    ]


def test_openai_compatible_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("LCM_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    transport = FakeTransport(_openai_compatible_success(1))
    provider = OpenAICompatibleProvider(
        "bge-m3", base_url="https://api.example.com/v1", transport=transport
    )

    with pytest.raises(EmbeddingProviderError, match="LCM_EMBEDDING_API_KEY"):
        provider.embed_query("q")

    assert transport.calls == []


def test_openai_compatible_siliconflow_key_fallback(monkeypatch):
    monkeypatch.delenv("LCM_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("SILICONFLOW_API_KEY", "fallback-key")
    transport = FakeTransport(_openai_compatible_success(1))
    provider = OpenAICompatibleProvider(
        "bge-m3", base_url="https://api.example.com/v1", transport=transport
    )

    provider.embed_query("q")

    assert transport.calls[0]["headers"]["Authorization"] == "Bearer fallback-key"


def test_openai_compatible_empty_base_url_raises():
    with pytest.raises(ValueError, match="LCM_EMBEDDING_BASE_URL"):
        OpenAICompatibleProvider("bge-m3", base_url="   ")


def test_openai_compatible_empty_model_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        OpenAICompatibleProvider("  ", base_url="https://api.example.com/v1")


def test_openai_compatible_network_error_is_not_retried(monkeypatch):
    monkeypatch.setenv("LCM_EMBEDDING_API_KEY", "test-key")
    transport = FakeTransport(OSError("boom"))
    provider = OpenAICompatibleProvider(
        "bge-m3", base_url="https://api.example.com/v1", transport=transport
    )

    with pytest.raises(EmbeddingProviderError, match="network error"):
        provider.embed_documents(["doc"])

    # The transport start makes an automatic identical resend unsafe.
    assert len(transport.calls) == 1


def test_openai_compatible_response_count_mismatch_raises(monkeypatch):
    monkeypatch.setenv("LCM_EMBEDDING_API_KEY", "test-key")
    transport = FakeTransport(_openai_compatible_success(1))  # 1 vector for 2 texts
    provider = OpenAICompatibleProvider(
        "bge-m3", base_url="https://api.example.com/v1", transport=transport
    )

    with pytest.raises(EmbeddingProviderError, match="different number of embeddings"):
        provider.embed_documents(["a", "b"])


def test_resolve_provider_openai_compatible(monkeypatch):
    config = LCMConfig(
        embedding_provider="openai-compatible",
        embedding_model="BAAI/bge-m3",
        embedding_base_url="https://api.example.com/v1",
    )
    provider = resolve_provider(config)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model_id == "BAAI/bge-m3"
    assert provider.base_url == "https://api.example.com/v1"

    config.embedding_provider = "siliconflow"
    assert isinstance(resolve_provider(config), OpenAICompatibleProvider)
