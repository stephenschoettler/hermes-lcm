from __future__ import annotations

import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

import hermes_lcm.command as command_mod
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.embedding_provider import (
    EmbeddedDocumentBatch,
    ProviderPreDispatchError,
    VoyageError,
)
from hermes_lcm.vector_store import EmbeddingPublishOutcome, VectorStore


class FakeProvider:
    provider_id = "ollama"
    model_id = "model-a"

    def __init__(self, *, dim: int = 2):
        self.dim = dim
        self.calls: list[list[str]] = []
        self.last_skipped_documents: list[int] = []

    def embed_documents(self, texts):
        current = list(texts)
        self.calls.append(current)
        self.last_skipped_documents = []
        return [[float(index + 1), 1.0] for index, _text in enumerate(current)]


@pytest.fixture(autouse=True)
def deterministic_token_count(monkeypatch):
    monkeypatch.setattr(command_mod, "count_tokens", lambda text: len(str(text)))


def _engine(tmp_path, *, enabled: bool = True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "backfill.db"
    config = LCMConfig(
        database_path=str(db_path),
        embeddings_enabled=enabled,
        embedding_provider="ollama",
        embedding_model="model-a",
    )
    return SimpleNamespace(
        _config=config,
        _store=SimpleNamespace(db_path=db_path),
    )


def _seed(engine, count: int, *, register: bool = True) -> list[int]:
    dag = SummaryDAG(engine._store.db_path)
    try:
        node_ids = [
            dag.add_node(SummaryNode(
                session_id="session-a",
                depth=0,
                summary=f"summary-{index}",
                source_token_count=100 + index,
                created_at=float(index + 1),
                latest_at=float(index + 1),
            ))
            for index in range(count)
        ]
        dag.add_node(SummaryNode(
            session_id="session-a",
            depth=1,
            summary="not a leaf",
            created_at=10_000.0,
        ))
    finally:
        dag.close()
    if register:
        store = VectorStore(engine._store.db_path, config=engine._config)
        try:
            store.register_profile("model-a", "ollama", 2)
        finally:
            store.close()
    return node_ids


def _meta_ids(engine) -> list[str]:
    conn = sqlite3.connect(engine._store.db_path)
    try:
        return [
            str(row[0])
            for row in conn.execute(
                "SELECT embedded_id FROM lcm_embedding_meta ORDER BY CAST(embedded_id AS INTEGER)"
            ).fetchall()
        ]
    finally:
        conn.close()


def _claim_value(engine):
    conn = sqlite3.connect(engine._store.db_path)
    try:
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (command_mod._EMBEDDING_BACKFILL_CLAIM_KEY,),
        ).fetchone()
        return None if row is None else str(row[0])
    finally:
        conn.close()


def _mark_uncertain(engine, node_ids: list[int]) -> str:
    """Create deterministic durable uncertainty for full-command retry tests."""
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        conn = store.connection
        command_mod._ensure_inflight_table(conn)
        identity = str(conn.execute(
            "SELECT identity_hash FROM lcm_embedding_profile WHERE active=1"
        ).fetchone()[0])
        conn.executemany(
            "INSERT INTO lcm_embedding_backfill_inflight("
            "embedded_id, identity_hash, lease_id, generation, claimed_at, "
            "state, request_id, updated_at, last_error) "
            "VALUES (?, ?, 'prior', 1, 1, 'uncertain', 'prior-request', ?, "
            "'remote acceptance unknown')",
            (
                (str(node_id), identity, float(index + 1))
                for index, node_id in enumerate(node_ids)
            ),
        )
        return identity
    finally:
        store.close()


def _inflight_rows(engine) -> list[tuple[str, str]]:
    conn = sqlite3.connect(engine._store.db_path)
    try:
        return [
            (str(row[0]), str(row[1]))
            for row in conn.execute(
                "SELECT embedded_id, state "
                "FROM lcm_embedding_backfill_inflight ORDER BY updated_at, embedded_id"
            ).fetchall()
        ]
    finally:
        conn.close()


def test_dry_run_reports_counts_tokens_and_cost_without_calls_or_writes(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    _seed(engine, 3)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    before = engine._store.db_path.read_bytes()

    result = handle_lcm_command("embed backfill --limit 2", engine)

    assert "status: dry-run" in result
    assert "pending: 3" in result
    assert "selected: 2" in result
    assert "estimated_tokens: 18" in result
    assert "estimated_cost_usd: $0.000000" in result
    assert "tokens_consumed: 0" in result
    assert provider.calls == []
    assert _meta_ids(engine) == []
    assert engine._store.db_path.read_bytes() == before

    voyage_engine = _engine(tmp_path / "voyage")
    _seed(voyage_engine, 2)
    conn = sqlite3.connect(voyage_engine._store.db_path)
    conn.execute(
        """
        UPDATE lcm_embedding_profile
        SET model_name = 'voyage-4-lite', provider = 'voyage'
        WHERE model_name = 'model-a'
        """
    )
    conn.commit()
    conn.close()
    voyage_engine._config.embedding_provider = "voyage"
    voyage_engine._config.embedding_model = "voyage-4-lite"
    monkeypatch.setattr(
        command_mod,
        "count_tokens",
        lambda text: 10_000 if str(text).endswith("0") else 30_000,
    )

    voyage_result = handle_lcm_command("embed backfill", voyage_engine)

    assert "estimated_tokens: 40000" in voyage_result
    assert "estimated_batches: 1" in voyage_result
    assert "estimated_cost_usd: $0.000200" in voyage_result
    assert provider.calls == []


def test_empty_leaf_summaries_are_not_pending_or_sent_to_provider(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    dag = SummaryDAG(engine._store.db_path)
    try:
        dag.add_node(SummaryNode(
            session_id="session-a",
            depth=0,
            summary="",
            created_at=3.0,
            latest_at=3.0,
        ))
        dag.add_node(SummaryNode(
            session_id="session-a",
            depth=0,
            summary="   \n\t",
            created_at=2.0,
            latest_at=2.0,
        ))
        dag.add_node(SummaryNode(
            session_id="session-a",
            depth=0,
            summary="\v\f",
            created_at=1.5,
            latest_at=1.5,
        ))
        valid_id = dag.add_node(SummaryNode(
            session_id="session-a",
            depth=0,
            summary="useful summary",
            created_at=1.0,
            latest_at=1.0,
        ))
    finally:
        dag.close()
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile("model-a", "ollama", 2)
    finally:
        store.close()
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    preview = handle_lcm_command("embed backfill", engine)
    applied = handle_lcm_command("embed backfill --apply", engine)

    assert "pending: 1" in preview
    assert "selected: 1" in preview
    assert "embedded: 1" in applied
    assert "remaining: 0" in applied
    assert provider.calls == [["useful summary"]]
    assert _meta_ids(engine) == [str(valid_id)]


def test_retry_uncertain_does_not_resend_empty_summary(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    dag = SummaryDAG(engine._store.db_path)
    try:
        blank_id = dag.add_node(SummaryNode(
            session_id="session-a",
            depth=0,
            summary="\t\n",
            created_at=2.0,
            latest_at=2.0,
        ))
        valid_id = dag.add_node(SummaryNode(
            session_id="session-a",
            depth=0,
            summary="retry this summary",
            created_at=1.0,
            latest_at=1.0,
        ))
    finally:
        dag.close()
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile("model-a", "ollama", 2)
    finally:
        store.close()
    _mark_uncertain(engine, [blank_id, valid_id])
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command("embed backfill --apply --retry-uncertain", engine)

    assert "selected: 1" in result
    assert "embedded: 1" in result
    assert provider.calls == [["retry this summary"]]
    assert _meta_ids(engine) == [str(valid_id)]
    assert _inflight_rows(engine) == [(str(blank_id), "uncertain")]


def test_apply_batches_records_correct_meta_and_is_idempotent(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 35)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    first = handle_lcm_command("embed backfill --apply", engine)

    assert "embedded: 35" in first
    assert "remaining: 0" in first
    assert [len(batch) for batch in provider.calls] == [32, 3]
    assert _meta_ids(engine) == [str(node_id) for node_id in node_ids]
    conn = sqlite3.connect(engine._store.db_path)
    try:
        rows = conn.execute(
            """
            SELECT m.embedded_kind, p.model_name, p.provider, m.source_token_count
            FROM lcm_embedding_meta m
            JOIN lcm_embedding_profile p ON p.identity_hash = m.identity_hash
            ORDER BY CAST(m.embedded_id AS INTEGER)
            """
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        ("summary", "model-a", "ollama", 100 + index) for index in range(35)
    ]

    second = handle_lcm_command("embed backfill --apply", engine)
    assert "selected: 0" in second
    assert "embedded: 0" in second
    assert [len(batch) for batch in provider.calls] == [32, 3]


def test_apply_limit_embeds_newest_rows_first(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 4)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command("embed backfill --limit 2 --apply", engine)

    assert "embedded: 2" in result
    assert "remaining: 2" in result
    assert _meta_ids(engine) == [str(node_ids[-2]), str(node_ids[-1])]


def test_accepted_then_local_publish_failure_becomes_uncertain(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 3)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    original = VectorStore.publish_embedding_under_lease

    def fail_one(self, embedded_id, kind, model, vector, **kwargs):
        if str(embedded_id) == str(node_ids[1]):
            raise sqlite3.OperationalError("synthetic row failure")
        return original(self, embedded_id, kind, model, vector, **kwargs)

    monkeypatch.setattr(VectorStore, "publish_embedding_under_lease", fail_one)

    result = handle_lcm_command("embed backfill --apply", engine)

    assert "embedded: 1" in result
    assert "uncertain_remote_acceptance: 2" in result
    assert f"node_id={node_ids[1]} reason=record_error:synthetic row failure" in result
    assert _meta_ids(engine) == [str(node_ids[-1])]

    # Normal retry performs no provider calls for uncertain remote acceptance.
    healthy = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: healthy)
    retry = handle_lcm_command("embed backfill --apply", engine)
    assert healthy.calls == []
    assert "uncertain_remote_acceptance: 2" in retry


def test_auth_error_aborts_immediately_and_releases_claim(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, 35)
    provider = FakeProvider()

    def auth_error(_texts):
        provider.calls.append(["attempt"])
        raise VoyageError("auth", "bad credentials", status_code=401)

    provider.provider_id = "voyage"
    engine._config.embedding_provider = "voyage"
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile("model-a", "voyage", 2)
    finally:
        store.close()
    provider.embed_documents = auth_error
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command("embed backfill --apply", engine)

    assert "status: error" in result
    assert "provider authentication failed" in result
    assert len(provider.calls) == 1
    assert _claim_value(engine) is None


def test_transient_provider_error_skips_batch_and_continues(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, 33)
    provider = FakeProvider()
    original = provider.embed_documents

    def transient_once(texts):
        if not provider.calls:
            provider.calls.append(list(texts))
            raise VoyageError("network", "temporary network failure")
        return original(texts)

    provider.embed_documents = transient_once
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command("embed backfill --apply", engine)

    # A failed batch must NOT be reported as complete — the run only partially
    # embedded the discovered work.
    assert "status: partial" in result
    assert "embedded: 1" in result
    assert "failed: 32" in result
    assert "remaining: 0" in result
    assert "uncertain_remote_acceptance: 32" in result
    assert len(provider.calls) == 2
    assert _claim_value(engine) is None


def test_provider_overcap_rows_are_skipped_and_left_pending(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 3)
    provider = FakeProvider()

    def skip_middle(texts):
        provider.calls.append(list(texts))
        provider.last_skipped_documents = [1]
        return [[1.0, 1.0], [2.0, 1.0]]

    provider.embed_documents = skip_middle
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command("embed backfill --apply", engine)

    assert "embedded: 2" in result
    assert "skipped_overcap: 1" in result
    assert "remaining: 1" in result
    assert f"node_id={node_ids[1]} reason=provider_document_token_cap" in result


def test_fresh_claim_refuses_second_worker_but_stale_claim_is_overridden(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    _seed(engine, 1)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    conn = sqlite3.connect(engine._store.db_path)
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?)",
        (
            command_mod._EMBEDDING_BACKFILL_CLAIM_KEY,
            json.dumps({"owner": "first", "claimed_at": time.time()}),
        ),
    )
    conn.commit()
    conn.close()

    refused = handle_lcm_command("embed backfill --apply", engine)
    assert "status: refused" in refused
    assert "holds the lease" in refused
    assert provider.calls == []

    conn = sqlite3.connect(engine._store.db_path)
    conn.execute(
        "UPDATE metadata SET value = ? WHERE key = ?",
        (
            json.dumps({"owner": "stale", "claimed_at": time.time() - 601}),
            command_mod._EMBEDDING_BACKFILL_CLAIM_KEY,
        ),
    )
    conn.commit()
    conn.close()
    applied = handle_lcm_command("embed backfill --apply", engine)
    assert "status: complete" in applied
    assert "embedded: 1" in applied
    assert _claim_value(engine) is None


def test_apply_claims_before_discovery_and_skips_already_embedded(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 3)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    # Another writer embeds the newest row before this run claims + discovers.
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.record_embedding(
            str(node_ids[-1]),
            "summary",
            "model-a",
            [1.0, 1.0],
            identity=store.capture_identity("model-a", provider="ollama"),
        )
    finally:
        store.close()

    result = handle_lcm_command("embed backfill --apply", engine)

    # Discovery runs AFTER the lease is claimed, so the already-embedded newest
    # row is excluded rather than re-sent to the provider.
    assert "selected: 2" in result
    assert "embedded: 2" in result
    sent = [doc for batch in provider.calls for doc in batch]
    assert "summary-2" not in sent
    assert set(_meta_ids(engine)) == {str(node_id) for node_id in node_ids}


def test_heartbeat_lease_blocks_takeover_until_expiry(tmp_path):
    db_path = tmp_path / "lease.db"
    store = VectorStore(db_path)
    try:
        conn = store.connection
        command_mod._ensure_inflight_table(conn)
        lease = command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0, now=1_000.0
        )
        assert lease is not None
        # A second worker cannot take a live lease within the TTL.
        assert command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0, now=1_100.0
        ) is None
        # A heartbeat near the original expiry extends the lease.
        assert lease.renew(now=1_590.0, force=True) is True
        assert command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0, now=1_595.0
        ) is None
        # Only once the lease is truly expired (past last heartbeat + TTL) can a
        # second worker steal it — and the original can no longer renew.
        stolen = command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0, now=1_590.0 + 601.0
        )
        assert stolen is not None
        assert lease.renew(now=1_590.0 + 602.0, force=True) is False
    finally:
        store.close()


def test_inflight_row_requires_explicit_uncertain_retry_after_crash(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 2)
    provider = FakeProvider()

    def crash(texts):
        provider.calls.append(list(texts))
        raise VoyageError("network", "provider crashed mid-batch")

    provider.embed_documents = crash
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    first = handle_lcm_command("embed backfill --apply", engine)
    # Nothing recorded; both rows are left marked in_flight.
    assert "status: partial" in first
    assert "embedded: 0" in first
    assert "in_flight: 2" in first
    assert _meta_ids(engine) == []

    healthy = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: healthy)
    second = handle_lcm_command("embed backfill --apply", engine)
    # A normal retry is fail-closed: no repeat provider charge.
    assert "status: partial" in second
    assert "embedded: 0" in second
    assert "uncertain_remote_acceptance: 2" in second
    assert healthy.calls == []

    authorized = handle_lcm_command(
        "embed backfill --apply --retry-uncertain", engine
    )
    assert "status: complete" in authorized
    assert "embedded: 2" in authorized
    assert "in_flight: 0" in authorized
    assert _meta_ids(engine) == [str(node_id) for node_id in node_ids]


def test_retry_uncertain_limit_binds_exact_old_row_before_new_ordinary(
    monkeypatch, tmp_path
):
    """The authorized row cannot be deleted then displaced by newer work."""
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 2)
    _mark_uncertain(engine, [node_ids[0]])
    retry_provider = FakeProvider()
    monkeypatch.setattr(
        command_mod, "resolve_provider", lambda _config, **_kw: retry_provider
    )

    first = handle_lcm_command(
        "embed backfill --apply --retry-uncertain --limit 1", engine
    )

    assert "status: complete" in first
    assert retry_provider.calls == [["summary-0"]]
    assert _meta_ids(engine) == [str(node_ids[0])]
    assert _inflight_rows(engine) == []

    ordinary_provider = FakeProvider()
    monkeypatch.setattr(
        command_mod, "resolve_provider", lambda _config, **_kw: ordinary_provider
    )
    second = handle_lcm_command("embed backfill --apply --limit 1", engine)

    assert "status: complete" in second
    assert ordinary_provider.calls == [["summary-1"]]
    assert _meta_ids(engine) == [str(node_id) for node_id in node_ids]


def test_retry_uncertain_partial_publish_preserves_unused_authorization(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 2)
    _mark_uncertain(engine, node_ids)

    class SplitRejectedProvider(FakeProvider):
        def embed_document_batches(self, texts, *, before_dispatch):
            before_dispatch((0,))
            self.calls.append([str(texts[0])])
            yield EmbeddedDocumentBatch((0,), ((1.0, 0.0),))
            before_dispatch((1,))
            self.calls.append([str(texts[1])])
            raise ProviderPreDispatchError("second request rejected before transport")

    provider = SplitRejectedProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command(
        "embed backfill --apply --retry-uncertain --limit 2", engine
    )

    assert "status: partial" in result
    assert "embedded: 1" in result
    assert _meta_ids(engine) == [str(node_ids[0])]
    assert _inflight_rows(engine) == [(str(node_ids[1]), "uncertain")]

    ordinary = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: ordinary)
    followup = handle_lcm_command("embed backfill --apply", engine)
    assert ordinary.calls == []
    assert "uncertain_remote_acceptance: 1" in followup


def test_retry_uncertain_definitive_failure_preserves_marker(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 1)
    _mark_uncertain(engine, node_ids)

    class RejectedProvider(FakeProvider):
        def embed_document_batches(self, texts, *, before_dispatch):
            before_dispatch((0,))
            self.calls.append(list(texts))
            raise ProviderPreDispatchError("rejected before transport")
            yield  # pragma: no cover - keeps this an iterator

    provider = RejectedProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command(
        "embed backfill --apply --retry-uncertain --limit 1", engine
    )

    assert "status: partial" in result
    assert _meta_ids(engine) == []
    assert _inflight_rows(engine) == [(str(node_ids[0]), "uncertain")]

    ordinary = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: ordinary)
    handle_lcm_command("embed backfill --apply", engine)
    assert ordinary.calls == []


def test_retry_uncertain_lease_loss_before_dispatch_preserves_marker(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 1)
    _mark_uncertain(engine, node_ids)
    successor_claim = json.dumps(
        {"owner": "successor", "generation": 99, "heartbeat_at": time.time()},
        sort_keys=True,
    )

    class StealingProvider(FakeProvider):
        def embed_document_batches(self, texts, *, before_dispatch):
            steal = sqlite3.connect(engine._store.db_path)
            try:
                steal.execute(
                    "UPDATE metadata SET value=? WHERE key=?",
                    (successor_claim, command_mod._EMBEDDING_BACKFILL_CLAIM_KEY),
                )
                steal.commit()
            finally:
                steal.close()
            before_dispatch((0,))
            yield EmbeddedDocumentBatch((0,), ((1.0, 0.0),))

    provider = StealingProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command(
        "embed backfill --apply --retry-uncertain --limit 1", engine
    )

    assert "stop_reason: lease_lost" in result
    assert _meta_ids(engine) == []
    assert _inflight_rows(engine) == [(str(node_ids[0]), "uncertain")]
    assert json.loads(_claim_value(engine))["owner"] == "successor"


def test_retry_uncertain_budget_expiry_preserves_unselected_marker(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 1)
    _mark_uncertain(engine, node_ids)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    monkeypatch.setenv("LCM_EMBEDDING_BACKFILL_BUDGET_S", "1")
    calls = {"n": 0}

    def fake_monotonic():
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 1_000.0

    monkeypatch.setattr(command_mod.time, "monotonic", fake_monotonic)

    result = handle_lcm_command(
        "embed backfill --apply --retry-uncertain --limit 1", engine
    )

    assert "stop_reason: op_budget_exhausted" in result
    assert provider.calls == []
    assert _inflight_rows(engine) == [(str(node_ids[0]), "uncertain")]


def test_stale_owner_after_provider_call_does_not_publish(monkeypatch, tmp_path):
    """C1: a lease stolen DURING the blocking provider call.

    Maintainer repro: the provider call outlives the lease, a successor acquires
    it, the old worker returns and PUBLISHED the embedding + reported ``complete``
    while the successor's claim remained. The post-provider owner CAS must make
    the stale owner discard its result (publish nothing, exit cleanly) and leave
    the successor's claim intact.
    """
    engine = _engine(tmp_path)
    _seed(engine, 3)

    successor_claim = json.dumps(
        {"owner": "successor-owner", "generation": 99, "heartbeat_at": time.time()},
        sort_keys=True,
    )

    class StealingProvider(FakeProvider):
        def embed_documents(self, texts):
            # Emulate the provider call outliving the lease: a successor steals
            # the claim (overwrites the owner) while this call is in flight.
            steal = sqlite3.connect(engine._store.db_path)
            try:
                steal.execute(
                    "UPDATE metadata SET value = ? WHERE key = ?",
                    (successor_claim, command_mod._EMBEDDING_BACKFILL_CLAIM_KEY),
                )
                steal.commit()
            finally:
                steal.close()
            return super().embed_documents(texts)

    provider = StealingProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command("embed backfill --apply", engine)

    # The stale owner published nothing: no meta rows written.
    assert _meta_ids(engine) == []
    assert "embedded: 0" in result
    assert "stop_reason: lease_lost" in result
    # The successor's claim is intact — the stale owner did NOT release it.
    claim = json.loads(_claim_value(engine))
    assert claim["owner"] == "successor-owner"


def test_lease_takeover_between_batches_stops_stale_worker(monkeypatch, tmp_path):
    """A lease stolen BETWEEN network batches: the committed batch survives, the
    next batch publishes nothing.

    Each accepted network batch now publishes under ONE ``BEGIN IMMEDIATE``, so
    ownership is verified per row via CAS inside one serialized transaction and a
    takeover is observed at batch boundaries — never mid-transaction (the steal
    serializes after the batch commit). The contract the previous
    row-interleaved version asserted is preserved and strengthened: a stale
    worker can neither clobber successor state nor tear a half-written batch. The
    first (committed) batch stands; the second, dispatched after the steal, loses
    the post-provider owner CAS and writes nothing.
    """
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 2)
    successor_claim = json.dumps(
        {"owner": "successor", "generation": 99, "heartbeat_at": time.time()},
        sort_keys=True,
    )

    class TakeoverBetweenBatches(FakeProvider):
        def embed_document_batches(self, texts, *, before_dispatch):
            before_dispatch((0,))
            yield EmbeddedDocumentBatch((0,), ((1.0, 0.0),))
            # The first network batch has COMMITTED. A successor now steals the
            # lease before the second batch is dispatched/published.
            steal = sqlite3.connect(engine._store.db_path)
            try:
                steal.execute(
                    "UPDATE metadata SET value=? WHERE key=?",
                    (successor_claim, command_mod._EMBEDDING_BACKFILL_CLAIM_KEY),
                )
                steal.commit()
            finally:
                steal.close()
            before_dispatch((1,))
            yield EmbeddedDocumentBatch((1,), ((1.0, 0.0),))

    provider = TakeoverBetweenBatches()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command("embed backfill --apply", engine)

    # The first batch committed fully; the second published nothing.
    assert "embedded: 1" in result
    assert "stop_reason: lease_lost" in result
    assert _meta_ids(engine) == [str(node_ids[-1])]
    conn = sqlite3.connect(engine._store.db_path)
    try:
        version = conn.execute(
            "SELECT data_version FROM lcm_embedding_profile WHERE active=1"
        ).fetchone()[0]
    finally:
        conn.close()
    # Exactly one committed vector — no stale-worker second write.
    assert version == 1
    # The successor's claim is intact — the stale worker did NOT overwrite it.
    assert json.loads(_claim_value(engine))["owner"] == "successor"


def test_network_batch_publishes_in_a_single_transaction(monkeypatch, tmp_path):
    """F5: an accepted network batch opens ONE transaction, not one per row.

    The whole point of batching — one fsync per provider batch instead of one
    per accepted vector. A trace callback counts the ``BEGIN IMMEDIATE`` and
    ``SAVEPOINT`` statements the batch publish issues: one BEGIN for the whole
    batch, each of the five rows nested under its own savepoint.
    """
    engine = _engine(tmp_path)
    _seed(engine, 5)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    stats = {"begin": 0, "savepoint": 0}

    class TracingStore(VectorStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._begins = 0
            self._saves = 0
            self._conn.set_trace_callback(self._trace)

        def _trace(self, sql):
            head = sql.lstrip().upper()
            if head.startswith("BEGIN IMMEDIATE"):
                self._begins += 1
            elif head.startswith("SAVEPOINT"):
                self._saves += 1

        def publish_embedding_batch_under_lease(self, rows, **kwargs):
            before = (self._begins, self._saves)
            try:
                return super().publish_embedding_batch_under_lease(rows, **kwargs)
            finally:
                stats["begin"] += self._begins - before[0]
                stats["savepoint"] += self._saves - before[1]

    monkeypatch.setattr(command_mod, "VectorStore", TracingStore)
    result = handle_lcm_command("embed backfill --apply", engine)

    assert "embedded: 5" in result
    # Five accepted vectors, ONE transaction — the 5x fsync amplification is gone.
    assert stats["begin"] == 1
    assert stats["savepoint"] == 5


class _CrashOnCommitConnection:
    """Proxies a sqlite3 connection but raises on ``commit`` — emulating a process
    killed after the batch's writes but before the transaction lands."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def commit(self):
        raise RuntimeError("crash before batch commit")


def test_crash_between_provider_return_and_commit_is_all_or_nothing(
    monkeypatch, tmp_path
):
    """A crash after the provider returns but before the batch COMMIT publishes
    nothing — the whole batch rolls back, leaving every row recoverably in-flight.

    A failing COMMIT reproduces the durable state of a process killed mid-batch:
    the transaction never lands, so no half-written vectors survive.
    """
    engine = _engine(tmp_path)
    _seed(engine, 3)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    class CrashStore(VectorStore):
        def publish_embedding_batch_under_lease(self, rows, **kwargs):
            real = self._conn
            self._conn = _CrashOnCommitConnection(real)
            try:
                return super().publish_embedding_batch_under_lease(rows, **kwargs)
            finally:
                self._conn = real

    monkeypatch.setattr(command_mod, "VectorStore", CrashStore)
    result = handle_lcm_command("embed backfill --apply", engine)

    # None half-published: the entire batch rolled back.
    assert _meta_ids(engine) == []
    assert "embedded: 0" in result
    # All three dispatched rows survive in-flight for recovery — none lost.
    rows = _inflight_rows(engine)
    assert len(rows) == 3
    assert {state for _id, state in rows} <= {"dispatched", "uncertain"}
    # FIX-4: a commit crash is a LOCAL storage failure, not a provider error.
    # Every dispatched row must be counted as failed and labeled local_error
    # (distinct from provider_error), so the operator-facing count + reason are
    # accurate rather than silently under-counting these unpublished rows.
    assert "failed: 3" in result
    assert "reason=local_error:" in result
    assert "reason=provider_error:" not in result


def test_batch_commits_published_rows_before_a_superseded_row(monkeypatch, tmp_path):
    """A row superseded mid-batch stops the batch WITHOUT aborting the rows already
    published in the same transaction — mixed per-row outcomes, no lost work.
    """
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 3)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    original = VectorStore.publish_embedding_under_lease
    seen: list[str] = []

    def supersede_second(self, embedded_id, kind, model, vector, **kwargs):
        seen.append(str(embedded_id))
        if len(seen) == 2:
            # The active identity is superseded for the 2nd row only; it writes
            # nothing while the 1st row's publication is already in the batch.
            return EmbeddingPublishOutcome.IDENTITY_SUPERSEDED
        return original(self, embedded_id, kind, model, vector, **kwargs)

    monkeypatch.setattr(
        VectorStore, "publish_embedding_under_lease", supersede_second
    )
    result = handle_lcm_command("embed backfill --apply", engine)

    assert "embedded: 1" in result
    assert "stop_reason: identity_superseded" in result
    # The first row committed; the third was never attempted after the stop.
    assert _meta_ids(engine) == [str(node_ids[-1])]


def test_active_identity_switch_quarantines_accepted_request_and_releases_claim(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    _seed(engine, 35)

    class SwitchingProvider(FakeProvider):
        def embed_documents(self, texts):
            # The request has been durably marked dispatched. Emulate an
            # operator activating B while the accepted A request is in flight.
            switch = VectorStore(engine._store.db_path, config=engine._config)
            try:
                switch.register_profile("model-a", "voyage", 2)
            finally:
                switch.close()
            return super().embed_documents(texts)

    provider = SwitchingProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)

    result = handle_lcm_command("embed backfill --apply", engine)

    assert "status: partial" in result
    assert "embedded: 0" in result
    assert "stop_reason: identity_superseded" in result
    assert "stop_reason: lease_lost" not in result
    assert "uncertain_remote_acceptance: 32" in result
    assert len(provider.calls) == 1
    assert len(provider.calls[0]) == 32
    assert _meta_ids(engine) == []
    assert _claim_value(engine) is None
    conn = sqlite3.connect(engine._store.db_path)
    try:
        rows = conn.execute(
            "SELECT state, last_error FROM lcm_embedding_backfill_inflight "
            "ORDER BY embedded_id"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 32
    assert {str(row[0]) for row in rows} == {"uncertain"}
    assert all("identity superseded" in str(row[1]) for row in rows)


def test_accepted_provider_subbatch_survives_later_subbatch_failure(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 2)

    class SplitProvider(FakeProvider):
        def embed_document_batches(self, texts, *, before_dispatch):
            before_dispatch((0,))
            self.calls.append([str(texts[0])])
            yield EmbeddedDocumentBatch((0,), ((1.0, 0.0),))
            before_dispatch((1,))
            self.calls.append([str(texts[1])])
            raise VoyageError("bad_request", "second request rejected")

    provider = SplitProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    first = handle_lcm_command("embed backfill --apply", engine)
    assert "embedded: 1" in first
    assert "uncertain_remote_acceptance: 0" in first
    assert "remaining: 1" in first
    assert _meta_ids(engine) == [str(node_ids[-1])]

    healthy = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: healthy)
    second = handle_lcm_command("embed backfill --apply", engine)
    assert healthy.calls == [["summary-0"]]
    assert "status: complete" in second
    assert _meta_ids(engine) == [str(node_id) for node_id in node_ids]


def test_accepted_split_survives_later_ambiguous_timeout(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 2)

    class SplitProvider(FakeProvider):
        def embed_document_batches(self, texts, *, before_dispatch):
            before_dispatch((0,))
            self.calls.append([str(texts[0])])
            yield EmbeddedDocumentBatch((0,), ((1.0, 0.0),))
            before_dispatch((1,))
            self.calls.append([str(texts[1])])
            raise VoyageError("network", "ambiguous timeout")

    provider = SplitProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    first = handle_lcm_command("embed backfill --apply", engine)

    assert "embedded: 1" in first
    assert "uncertain_remote_acceptance: 1" in first
    assert _meta_ids(engine) == [str(node_ids[-1])]
    healthy = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: healthy)
    second = handle_lcm_command("embed backfill --apply", engine)
    assert healthy.calls == []
    assert "uncertain_remote_acceptance: 1" in second


def test_definitive_pre_dispatch_expiry_is_safe_for_automatic_retry(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    _seed(engine, 2)

    class ExpiredBeforeTransport(FakeProvider):
        def embed_document_batches(self, texts, *, before_dispatch):
            indexes = tuple(range(len(texts)))
            before_dispatch(indexes)
            raise ProviderPreDispatchError("deadline expired before transport")
            yield  # pragma: no cover - keeps this an iterator

    first_provider = ExpiredBeforeTransport()
    monkeypatch.setattr(
        command_mod, "resolve_provider", lambda _config, **_kw: first_provider
    )
    first = handle_lcm_command("embed backfill --apply", engine)
    assert "embedded: 0" in first
    assert "uncertain_remote_acceptance: 0" in first
    assert "in_flight: 0" in first

    healthy = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: healthy)
    second = handle_lcm_command("embed backfill --apply", engine)
    assert "status: complete" in second
    assert "embedded: 2" in second
    assert len(healthy.calls) == 1


def test_inflight_maintenance_processes_only_one_bounded_chunk(tmp_path):
    store = VectorStore(tmp_path / "bounded-inflight.db")
    try:
        conn = store.connection
        command_mod._ensure_inflight_table(conn)
        identity = "identity"
        # VectorStore connections use autocommit, so batch this large fixture
        # explicitly instead of forcing one durable transaction per row.
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO lcm_embedding_backfill_inflight("
            "embedded_id, identity_hash, lease_id, generation, claimed_at, "
            "state, updated_at) VALUES (?, ?, 'stale', 1, 1, 'claimed', 1)",
            ((str(index), identity) for index in range(250_001)),
        )
        conn.commit()
        lease = command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0
        )
        assert lease is not None
        plan = [
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT rowid, embedded_id "
                "FROM lcm_embedding_backfill_inflight "
                "WHERE identity_hash=? AND state=? "
                "AND (lease_id IS NOT ? OR generation IS NOT ?) "
                "ORDER BY updated_at, embedded_id LIMIT ?",
                (identity, "claimed", lease.lease_id, lease.generation, 256),
            )
        ]
        assert any("idx_lcm_embedding_inflight_maintenance" in row for row in plan)
        assert not any("USE TEMP B-TREE" in row for row in plan)

        command_mod._prepare_inflight_for_lease(
            conn,
            identity,
            lease,
        )

        remaining = conn.execute(
            "SELECT COUNT(*) FROM lcm_embedding_backfill_inflight"
        ).fetchone()[0]
        assert remaining == 250_001 - 256
        lease.release()
    finally:
        store.close()


def test_retry_uncertain_limit_authorizes_only_deterministic_rows(tmp_path):
    engine = _engine(tmp_path)
    node_ids = _seed(engine, 10)
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        conn = store.connection
        command_mod._ensure_inflight_table(conn)
        identity = str(conn.execute(
            "SELECT identity_hash FROM lcm_embedding_profile WHERE active=1"
        ).fetchone()[0])
        conn.executemany(
            "INSERT INTO lcm_embedding_backfill_inflight("
            "embedded_id, identity_hash, lease_id, generation, claimed_at, "
            "state, updated_at) VALUES (?, ?, 'stale', 1, 1, 'uncertain', ?)",
            (
                (str(node_id), identity, float(index))
                for index, node_id in enumerate(node_ids)
            ),
        )
        lease = command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0
        )
        assert lease is not None

        command_mod._prepare_inflight_for_lease(
            conn,
            identity,
            lease,
        )

        pending, rows = command_mod._embedding_authorized_uncertain_rows(
            conn, identity, 3
        )
        selected = [
            str(row["node_id"])
            for row in rows
        ]
        remaining = [
            str(row[0])
            for row in conn.execute(
                "SELECT embedded_id FROM lcm_embedding_backfill_inflight "
                "ORDER BY updated_at, embedded_id"
            )
        ]
        assert pending == 10
        assert selected == [str(node_id) for node_id in node_ids[:3]]
        assert set(remaining) == {str(node_id) for node_id in node_ids}
        lease.release()
    finally:
        store.close()


def test_inflight_maintenance_fails_closed_after_successor_takeover(tmp_path):
    store = VectorStore(tmp_path / "successor-inflight.db")
    try:
        conn = store.connection
        command_mod._ensure_inflight_table(conn)
        lease = command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0
        )
        assert lease is not None
        conn.execute(
            "INSERT INTO lcm_embedding_backfill_inflight("
            "embedded_id, identity_hash, lease_id, generation, claimed_at, "
            "state, request_id, updated_at) "
            "VALUES ('row', 'identity', 'successor', 99, 1, "
            "'dispatched', 'successor-request', 1)"
        )
        conn.execute(
            "UPDATE metadata SET value=? WHERE key=?",
            (
                json.dumps(
                    {
                        "owner": "successor",
                        "generation": 99,
                        "heartbeat_at": time.time(),
                    },
                    sort_keys=True,
                ),
                command_mod._EMBEDDING_BACKFILL_CLAIM_KEY,
            ),
        )

        with pytest.raises(command_mod._BackfillLeaseLost):
            command_mod._prepare_inflight_for_lease(
                conn,
                "identity",
                lease,
            )

        row = conn.execute(
            "SELECT lease_id, generation, state, request_id "
            "FROM lcm_embedding_backfill_inflight WHERE embedded_id='row'"
        ).fetchone()
        assert tuple(row) == ("successor", 99, "dispatched", "successor-request")
    finally:
        store.close()


def test_inflight_maintenance_exact_snapshot_cannot_mutate_replaced_row(tmp_path):
    store = VectorStore(tmp_path / "snapshot-inflight.db")
    try:
        conn = store.connection
        command_mod._ensure_inflight_table(conn)
        lease = command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0
        )
        assert lease is not None
        conn.executemany(
            "INSERT INTO lcm_embedding_backfill_inflight("
            "embedded_id, identity_hash, lease_id, generation, claimed_at, "
            "state, request_id, updated_at) VALUES (?, 'identity', 'old', 1, "
            "1, ?, ?, 1)",
            [
                ("trigger", "claimed", None),
                ("replaced", "dispatched", "old-request"),
            ],
        )
        conn.execute(
            "CREATE TEMP TRIGGER replace_dispatched_after_delete "
            "AFTER DELETE ON lcm_embedding_backfill_inflight "
            "WHEN OLD.embedded_id = 'trigger' BEGIN "
            "UPDATE lcm_embedding_backfill_inflight "
            "SET lease_id='successor', generation=99, request_id='successor-request' "
            "WHERE embedded_id='replaced'; END"
        )

        command_mod._prepare_inflight_for_lease(
            conn,
            "identity",
            lease,
        )

        row = conn.execute(
            "SELECT lease_id, generation, state, request_id "
            "FROM lcm_embedding_backfill_inflight WHERE embedded_id='replaced'"
        ).fetchone()
        assert tuple(row) == ("successor", 99, "dispatched", "successor-request")
        lease.release()
    finally:
        store.close()


def test_inflight_schema_repairs_legacy_shape_and_malformed_index(tmp_path):
    store = VectorStore(tmp_path / "legacy-inflight.db")
    try:
        conn = store.connection
        conn.execute(
            "CREATE TABLE lcm_embedding_backfill_inflight("
            "embedded_id TEXT, identity_hash TEXT, lease_id TEXT, "
            "generation INTEGER, claimed_at REAL, "
            "PRIMARY KEY(embedded_id, identity_hash))"
        )
        conn.execute(
            "INSERT INTO lcm_embedding_backfill_inflight VALUES "
            "('row', 'identity', 'old-owner', 1, 10)"
        )

        command_mod._ensure_inflight_table(conn)
        row = conn.execute(
            "SELECT state, updated_at FROM lcm_embedding_backfill_inflight"
        ).fetchone()
        assert tuple(row) == ("uncertain", 10.0)
        conn.execute("DROP INDEX idx_lcm_embedding_inflight_maintenance")
        conn.execute(
            "CREATE INDEX idx_lcm_embedding_inflight_maintenance "
            "ON lcm_embedding_backfill_inflight("
            "identity_hash COLLATE NOCASE, state, updated_at DESC, embedded_id) "
            "WHERE state = 'claimed'"
        )

        command_mod._ensure_inflight_table(conn)
        columns = tuple(
            row[2]
            for row in conn.execute(
                "PRAGMA index_info(idx_lcm_embedding_inflight_maintenance)"
            )
        )
        assert columns == ("identity_hash", "state", "updated_at", "embedded_id")
        before = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='lcm_embedding_backfill_inflight'"
        ).fetchone()[0]
        traced: list[str] = []
        conn.set_trace_callback(traced.append)
        command_mod._ensure_inflight_table(conn)
        conn.set_trace_callback(None)
        after = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='lcm_embedding_backfill_inflight'"
        ).fetchone()[0]
        assert after == before
        assert not any(
            statement.lstrip().upper().startswith(("CREATE ", "DROP ", "ALTER "))
            for statement in traced
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM lcm_embedding_backfill_inflight"
        ).fetchone()[0] == 1
    finally:
        store.close()


@pytest.mark.parametrize(
    ("state_check", "extra_check"),
    [
        (
            "CHECK(state IN ('claimed', 'dispatched', 'uncertain'))",
            "CHECK(lease_id = 'forced')",
        ),
        (
            "CHECK(state IN ('uncertain', 'dispatched', 'claimed'))",
            "",
        ),
        (
            "CHECK(state IN ('CLAIMED', 'dispatched', 'uncertain'))",
            "",
        ),
        (
            "CHECK(state IN ('claimed ', 'dispatched', 'uncertain'))",
            "",
        ),
    ],
)
def test_inflight_schema_repairs_noncanonical_complete_check_set(
    tmp_path, state_check, extra_check
):
    store = VectorStore(tmp_path / "check-fingerprint.db")
    try:
        conn = store.connection
        conn.execute(
            "CREATE TABLE lcm_embedding_backfill_inflight("
            "embedded_id TEXT, identity_hash TEXT, lease_id TEXT, "
            "generation INTEGER, claimed_at REAL, "
            "state TEXT NOT NULL DEFAULT 'claimed' "
            f"{state_check}, request_id TEXT, updated_at REAL, last_error TEXT, "
            f"{extra_check}{',' if extra_check else ''} "
            "PRIMARY KEY(embedded_id, identity_hash))"
        )

        command_mod._ensure_inflight_table(conn)

        sql = str(
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' "
                "AND name='lcm_embedding_backfill_inflight'"
            ).fetchone()[0]
        ).lower()
        assert sql.count("check") == 1
        assert "forced" not in sql
        assert sql.index("'claimed'") < sql.index("'dispatched'") < sql.index("'uncertain'")
        lease = command_mod._acquire_embedding_backfill_lease(
            conn, ttl_s=600.0, heartbeat_s=60.0
        )
        assert lease is not None
        command_mod._mark_inflight(conn, "identity", lease, ["row"])
        lease.release()
    finally:
        store.close()


def test_inflight_schema_rejects_incompatible_primary_key(tmp_path):
    store = VectorStore(tmp_path / "bad-inflight.db")
    try:
        conn = store.connection
        conn.execute(
            "CREATE TABLE lcm_embedding_backfill_inflight("
            "embedded_id TEXT PRIMARY KEY, identity_hash TEXT, lease_id TEXT, "
            "generation INTEGER, claimed_at REAL)"
        )
        with pytest.raises(RuntimeError, match="incompatible.*column"):
            command_mod._ensure_inflight_table(conn)
    finally:
        store.close()


def test_concurrent_first_run_inflight_schema_creation_is_idempotent(tmp_path):
    db_path = tmp_path / "concurrent-inflight.db"
    sqlite3.connect(db_path).close()
    begin_barrier = threading.Barrier(2)
    outcomes: list[str] = []
    outcome_lock = threading.Lock()

    class BeginBarrierConnection:
        def __init__(self, conn):
            self._conn = conn
            self._synchronized = False

        @property
        def in_transaction(self):
            return self._conn.in_transaction

        def execute(self, sql, params=()):
            if sql == "BEGIN IMMEDIATE" and not self._synchronized:
                self._synchronized = True
                begin_barrier.wait(timeout=2)
            return self._conn.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    def initialize() -> None:
        conn = sqlite3.connect(db_path, timeout=3, check_same_thread=False)
        try:
            command_mod._ensure_inflight_table(BeginBarrierConnection(conn))
            outcome = "success"
        except Exception as exc:  # pragma: no cover - assertion reports detail
            outcome = f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()
        with outcome_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=initialize) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert sorted(outcomes) == ["success", "success"]


@pytest.mark.parametrize("shared_connection", [False, True])
def test_embedding_purge_removes_orphaned_inflight_markers(
    tmp_path, shared_connection
):
    store = VectorStore(tmp_path / f"purge-inflight-{shared_connection}.db")
    try:
        command_mod._ensure_inflight_table(store.connection)
        store.connection.execute(
            "INSERT INTO lcm_embedding_backfill_inflight("
            "embedded_id, identity_hash, state, updated_at) "
            "VALUES ('7', 'identity', 'uncertain', 1)"
        )
        store.connection.commit()

        if shared_connection:
            VectorStore.purge_embedding_batch_on_connection(store.connection, [7])
        else:
            store.purge_embeddings_for_nodes([7])

        assert store.connection.execute(
            "SELECT COUNT(*) FROM lcm_embedding_backfill_inflight "
            "WHERE embedded_id = '7'"
        ).fetchone()[0] == 0
    finally:
        store.close()


def test_operation_budget_stops_run_between_batches(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    _seed(engine, 40)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    monkeypatch.setenv("LCM_EMBEDDING_BACKFILL_BUDGET_S", "1")

    calls = {"n": 0}

    def fake_monotonic():
        calls["n"] += 1
        # First call is the run start; every later call is well past the budget.
        return 0.0 if calls["n"] == 1 else 1_000.0

    monkeypatch.setattr(command_mod.time, "monotonic", fake_monotonic)

    result = handle_lcm_command("embed backfill --apply", engine)

    assert "stop_reason: op_budget_exhausted" in result
    assert "status: partial" in result
    assert "embedded: 0" in result
    assert provider.calls == []


def test_disabled_and_missing_profile_refuse_cleanly(tmp_path):
    disabled = _engine(tmp_path / "disabled", enabled=False)
    assert "embeddings are disabled" in handle_lcm_command(
        "embed backfill", disabled
    )

    missing = _engine(tmp_path / "missing")
    _seed(missing, 1, register=False)
    result = handle_lcm_command("embed backfill --apply", missing)
    assert "status: refused" in result
    assert "no current embedding profile" in result
    assert "/lcm embed warmup" in result
