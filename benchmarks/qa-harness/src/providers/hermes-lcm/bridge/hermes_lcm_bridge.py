#!/usr/bin/env python3
"""JSON-line bridge exposing hermes-lcm as a memorybench Provider backend.

The TypeScript ``HermesLcmProvider`` spawns this script in ``serve`` mode and
speaks newline-delimited JSON over stdin/stdout: one request object per line,
one response object per line. It implements the four stateful provider methods
(initialize / ingest / search / clear); ``awaitIndexing`` is a no-op on the TS
side because ingest is fully synchronous here.

Design contract (faithful to ``benchmarking/longmemeval.py`` in the hermes-lcm
repo, which this imports rather than reimplements):

* ``ingest`` accumulates ONE harness session at a time into a per-container LCM
  store on disk (the harness calls ``provider.ingest([session], ...)`` in a loop),
  preserving session ids/order, building the SAME deterministic per-session
  summary the in-house harness uses, recording summary + conversational-chunk
  embeddings. Embeds are batched per call.
* ``search`` invokes the PRODUCTION ``tools.lcm_recall`` over that store through a
  ``SimpleNamespace`` engine with a fresh, dataset-disjoint ``current_session_id``
  (so the scope prior never silently lifts an evidence session), then maps each
  hit -> ``{content, metadata}`` and returns the top-k the harness asks for.

Fairness: the bridge only ever sees what the harness hands it (the session
messages + the query). No dataset-specific logic, no evidence peeking.

The hermes-lcm plugin repo is NEVER modified: it is made importable via the same
``sys.path`` + package-spec bootstrap the repo's own harness uses.

Environment:
    HERMES_LCM_REPO                path to the hermes-lcm checkout (required)
    HERMES_MB_WORKDIR              base dir for per-container LCM dbs (required)
    HERMES_MB_PROVIDER            embedding provider: fastembed (default) | voyage
    HERMES_MB_MODEL              embedding model id (default per provider)
    LCM_LONGMEMEVAL_FASTEMBED_CACHE  fastembed model cache dir
    VOYAGE_API_KEY               required when HERMES_MB_PROVIDER=voyage
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_DEFAULT_MODELS = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "voyage": "voyage-context-3",
}

# Preserve the real stdout for protocol responses, then redirect stdout to
# stderr so any library chatter (model downloads, warnings) can never corrupt
# the newline-delimited JSON channel.
_RESPONSE_OUT = sys.stdout
sys.stdout = sys.stderr


def _log(message: str) -> None:
    print(f"[hermes-lcm-bridge] {message}", file=sys.stderr, flush=True)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-._") or "container"


class Bridge:
    def __init__(self) -> None:
        repo = os.environ.get("HERMES_LCM_REPO")
        if not repo:
            raise RuntimeError("HERMES_LCM_REPO is not set")
        self.repo_root = Path(repo).resolve()
        if not self.repo_root.is_dir():
            raise RuntimeError(f"HERMES_LCM_REPO does not exist: {self.repo_root}")

        workdir = os.environ.get("HERMES_MB_WORKDIR")
        if not workdir:
            raise RuntimeError("HERMES_MB_WORKDIR is not set")
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.provider_name = (os.environ.get("HERMES_MB_PROVIDER") or "fastembed").strip().lower()
        if self.provider_name in {"fast-embed"}:
            self.provider_name = "fastembed"
        self.model = os.environ.get("HERMES_MB_MODEL") or _DEFAULT_MODELS.get(
            self.provider_name, ""
        )
        if not self.model:
            raise RuntimeError(f"no embedding model for provider {self.provider_name!r}")

        # Make the plugin importable exactly the way the repo's own harness does.
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        from benchmarking.longmemeval import (  # noqa: E402
            _ensure_hermes_lcm_package,
            deterministic_session_summary,
            resolve_harness_provider,
        )

        _ensure_hermes_lcm_package()
        self._deterministic_session_summary = deterministic_session_summary
        self._resolve_harness_provider = resolve_harness_provider

        # Lazily populated on initialize().
        self.embedder: Any = None
        self.dim: int = 0
        # Per-container monotonic session order (recency prior in lcm_recall).
        self._order: dict[str, int] = {}

    # -- lifecycle ------------------------------------------------------------

    def initialize(self, _req: dict[str, Any]) -> dict[str, Any]:
        if self.provider_name == "voyage" and not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("HERMES_MB_PROVIDER=voyage but VOYAGE_API_KEY is unset")
        # Warm the embedder once so the model download/load happens here, not
        # inside a per-question path, and .dim is populated.
        self.embedder = self._resolve_harness_provider(self.provider_name, self.model)
        self.dim = int(self.embedder.dim)
        self.model = self.embedder.model_id
        _log(
            f"initialized provider={self.provider_name} model={self.model} dim={self.dim} "
            f"workdir={self.workdir}"
        )
        return {
            "ok": True,
            "provider": self.provider_name,
            "model": self.model,
            "dim": self.dim,
            "embeddings_enabled": True,
        }

    # -- helpers --------------------------------------------------------------

    def _db_path(self, container_tag: str) -> Path:
        return self.workdir / f"{_safe(container_tag)}.db"

    def _dates_path(self, container_tag: str) -> Path:
        # A sidecar mapping session_id -> harness-provided session date. The
        # plugin's append_batch stamps ingest wall-clock time (it takes no
        # per-message timestamp and must not be modified), so the real session
        # date -- data the harness gives EVERY provider -- is preserved here and
        # surfaced onto each search hit's metadata for temporal questions.
        return self.workdir / f"{_safe(container_tag)}.dates.json"

    def _load_dates(self, container_tag: str) -> dict[str, Any]:
        path = self._dates_path(container_tag)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _config(self, db_path: Path):
        from hermes_lcm.config import LCMConfig

        # The experiment flag and its token budget are environment-controlled.
        # Building a bare LCMConfig here silently discarded those overrides, so
        # the real bridge could never exercise the same treatment as direct
        # lcm_recall fixtures.
        config = LCMConfig.from_env()
        config.database_path = str(db_path)
        config.embeddings_enabled = True
        config.embedding_provider = self.provider_name
        config.embedding_model = self.model
        return config

    # -- ingest ---------------------------------------------------------------

    def ingest(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.embedder is None:
            raise RuntimeError("ingest before initialize")
        container_tag = str(req["containerTag"])
        session = req["session"]
        session_id = str(session["sessionId"])
        session_meta = session.get("metadata") or {}
        session_date = session_meta.get("date") or session_meta.get("formattedDate")
        messages = [
            {
                "role": str(m.get("role", "user")),
                "content": str(m.get("content", "")),
            }
            for m in session.get("messages", [])
        ]

        from hermes_lcm.chunking import iter_message_chunks
        from hermes_lcm.dag import SummaryDAG, SummaryNode
        from hermes_lcm.store import MessageStore
        from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore

        db_path = self._db_path(container_tag)
        config = self._config(db_path)
        # Opening these bootstraps the schema on first touch and re-opens
        # idempotently thereafter, so successive sessions accumulate.
        store = MessageStore(str(db_path), ingest_protection_config=config)
        dag = SummaryDAG(str(db_path))
        vector_store = VectorStore(str(db_path), config=config)
        try:
            vector_store.register_profile(self.model, self.provider_name, self.dim)
            identity = vector_store.capture_identity(self.model, provider=self.provider_name)
            vector_store.register_profile(self.model, self.provider_name, self.dim, task="chunk")
            chunk_identity = EmbeddingIdentity.canonical(
                self.provider_name, self.model, "", self.dim, "float32", "little", "chunk"
            )

            store_ids: list[int] = []
            if messages:
                store_ids = store.append_batch(
                    session_id, messages, source="benchmark", conversation_id=session_id
                )
                rows = [
                    {"store_id": sid, "role": m["role"], "content": m["content"]}
                    for sid, m in zip(store_ids, messages)
                ]
                chunk_texts: list[str] = []
                chunk_meta: list[Any] = []
                for chunk in iter_message_chunks(rows, policy="conversational"):
                    chunk_texts.append(chunk.text)
                    chunk_meta.append(chunk)
                if chunk_texts:
                    chunk_vectors = self.embedder.embed_documents(chunk_texts)
                    for chunk, vector in zip(chunk_meta, chunk_vectors):
                        vector_store.record_chunk_embedding(
                            chunk.chunk_id, self.model, vector,
                            store_id=chunk.store_id, chunk_index=chunk.chunk_index,
                            char_start=chunk.char_start, char_end=chunk.char_end,
                            token_estimate=chunk.token_estimate, identity=chunk_identity,
                        )

            summary_text = self._deterministic_session_summary(messages)
            order = self._order.get(container_tag, 0) + 1
            self._order[container_tag] = order
            node_id = dag.add_node(
                SummaryNode(
                    session_id=session_id,
                    depth=0,
                    summary=summary_text,
                    token_count=len(summary_text.split()),
                    source_token_count=sum(len(m["content"].split()) for m in messages),
                    source_type="messages",
                    created_at=float(order),
                )
            )
            summary_vector = self.embedder.embed_documents([summary_text])[0]
            vector_store.record_embedding(
                str(node_id), "summary", self.model, summary_vector, identity=identity
            )
        finally:
            vector_store.close()
            dag.close()
            store.close()

        if session_date:
            dates = self._load_dates(container_tag)
            dates[session_id] = session_date
            self._dates_path(container_tag).write_text(
                json.dumps(dates), encoding="utf-8"
            )

        return {"ok": True, "documentIds": [str(sid) for sid in store_ids] or [session_id]}

    # -- search ---------------------------------------------------------------

    def search(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.embedder is None:
            raise RuntimeError("search before initialize")
        container_tag = str(req["containerTag"])
        query = str(req.get("query", ""))
        limit = int(req.get("limit", 25))

        import hermes_lcm.tools as lcm_tools
        from hermes_lcm.dag import SummaryDAG
        from hermes_lcm.store import MessageStore
        from hermes_lcm.vector_store import VectorStore

        db_path = self._db_path(container_tag)
        config = self._config(db_path)
        dates = self._load_dates(container_tag)
        store = MessageStore(str(db_path), ingest_protection_config=config)
        dag = SummaryDAG(str(db_path))
        vector_store = VectorStore(str(db_path), config=config)  # noqa: F841 (keeps db warm)
        try:
            # A probe current-session id disjoint from any dataset session id
            # (the harness uses "<qid>-session-<i>"); the scope prior may boost
            # the current conversation, so it must NOT be an evidence session.
            fresh_session = f"__hermes_lcm_recall_probe__{container_tag}"
            engine = SimpleNamespace(
                _config=config,
                _store=store,
                _dag=dag,
                _hermes_home=str(self.workdir),
                current_session_id=fresh_session,
                _session_occurrence_dates=dates,
            )
            cache_key = (self.provider_name.strip().lower(), str(self.embedder.model_id).strip())
            engine._lcm_embedding_provider_cache = (cache_key, self.embedder)

            recall_request = {
                "query": query,
                "limit": limit,
            }
            if config.session_expand_v1:
                recall_request["detail"] = "answer_ready"
            payload = json.loads(
                lcm_tools.lcm_recall(
                    recall_request,
                    engine=engine,
                )
            )
        finally:
            vector_store.close()
            dag.close()
            store.close()

        if "error" in payload:
            raise RuntimeError(f"lcm_recall error: {payload['error']}")

        results: list[dict[str, Any]] = []
        for hit in payload.get("hits", []):
            session_id = hit.get("session_id")
            metadata = {
                "session_id": session_id,
                "date": dates.get(str(session_id)),
                "kind": hit.get("kind"),
                "score": hit.get("score"),
                "arms": hit.get("arms"),
                "from_current_session": hit.get("from_current_session"),
                "role": hit.get("role"),
                "session_expanded": bool(hit.get("session_expanded")),
                "exact_ref": hit.get("exact_ref"),
                "content_offset": hit.get("content_offset"),
                "content_returned_chars": hit.get("content_returned_chars"),
            }
            if hit.get("kind") == "summary":
                metadata["node_id"] = hit.get("node_id")
            else:
                metadata["store_id"] = hit.get("store_id")
                if hit.get("chunk_span"):
                    metadata["chunk_span"] = hit.get("chunk_span")
            results.append({
                "content": hit.get("content") or hit.get("snippet") or "",
                "metadata": metadata,
            })

        return {
            "ok": True,
            # ``limit`` bounds the ranked tier. Stage-2 additions intentionally
            # follow that prefix, so slicing here would erase the treatment at
            # the wire seam before the answer prompt sees it.
            "results": results,
            "ranked_limit": limit,
            "delivered_result_count": len(results),
            "delivered_additional_result_count": sum(
                bool(result["metadata"].get("session_expanded"))
                for result in results
            ),
            "degraded": payload.get("degraded", False),
            "degraded_reason": payload.get("degraded_reason"),
        }

    # -- clear ----------------------------------------------------------------

    def clear(self, req: dict[str, Any]) -> dict[str, Any]:
        container_tag = str(req["containerTag"])
        db_path = self._db_path(container_tag)
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(db_path) + suffix)
            if candidate.exists():
                candidate.unlink()
        dates_path = self._dates_path(container_tag)
        if dates_path.exists():
            dates_path.unlink()
        self._order.pop(container_tag, None)
        return {"ok": True}

    # -- dispatch -------------------------------------------------------------

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        cmd = req.get("cmd")
        if cmd == "initialize":
            return self.initialize(req)
        if cmd == "ingest":
            return self.ingest(req)
        if cmd == "search":
            return self.search(req)
        if cmd == "clear":
            return self.clear(req)
        if cmd == "ping":
            return {"ok": True}
        raise RuntimeError(f"unknown cmd: {cmd!r}")


def main() -> int:
    bridge = Bridge()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": f"bad json: {exc}"}), file=_RESPONSE_OUT, flush=True)
            continue
        try:
            response = bridge.handle(req)
        except Exception as exc:  # noqa: BLE001 - report every failure loudly
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), file=_RESPONSE_OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
