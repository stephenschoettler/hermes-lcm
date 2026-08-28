"""Embedding provider abstractions for optional semantic retrieval.

Providers are deliberately independent from the engine.  In particular, model
downloads and dimension discovery happen only through the explicit warmup
command; resolving a provider never performs network or disk-heavy work.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from .config import LCMConfig
from .tokens import count_tokens

logger = logging.getLogger(__name__)

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
# Contextualized-chunk models (voyage-context-*) live ONLY on this endpoint;
# posting them to the flat /v1/embeddings above returns HTTP 400 (live-proven on
# the real archive: zero rows written, clean lease behavior held).
_VOYAGE_CONTEXT_URL = "https://api.voyageai.com/v1/contextualizedembeddings"
_VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
# lcm_recall's cross-encoder rerank model. A lite model keeps the single extra
# API call cheap and inside the latency-sensitive recall deadline.
_VOYAGE_RERANK_MODEL = "rerank-2.5-lite"
_OPENAI_COMPATIBLE_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
_VOYAGE_MAX_BATCH_TOKENS = 80_000
_VOYAGE_MAX_DOCUMENT_TOKENS = 27_000
# Voyage caps a single request at 1000 input items regardless of token count;
# a batch bounded only by tokens can still exceed it (many short documents), so
# an item cap is enforced alongside the token budget.
_VOYAGE_MAX_BATCH_ITEMS = 1000
_VOYAGE_TOKEN_SAFETY_FACTOR = 0.9
_VOYAGE_BATCH_TOKEN_BUDGET = int(
    _VOYAGE_MAX_BATCH_TOKENS * _VOYAGE_TOKEN_SAFETY_FACTOR
)
_VOYAGE_DOCUMENT_TOKEN_BUDGET = int(
    _VOYAGE_MAX_DOCUMENT_TOKENS * _VOYAGE_TOKEN_SAFETY_FACTOR
)
# voyage-context-* /v1/contextualizedembeddings caps. Per the official docs
# (2026-07) a single request allows at most 1,000 inputs (inner lists), 120K
# total tokens across all inputs, and 16K total chunks across all inputs. The
# chunk-grouping spec additionally bounds one document (a single inputs inner
# list = one message's chunks) at 120K tokens and one chunk at 32K tokens.
# A document above the per-doc budget is split into contiguous sub-documents;
# a single chunk above the per-chunk cap is non-embeddable and skipped by the
# backfill path (chunking already caps a chunk well below 32K, so this is only
# a defensive guard).
_VOYAGE_CONTEXT_MAX_CHUNK_TOKENS = 32_000
_VOYAGE_CONTEXT_MAX_DOCUMENT_TOKENS = 120_000
_VOYAGE_CONTEXT_MAX_REQUEST_TOKENS = 120_000
_VOYAGE_CONTEXT_MAX_REQUEST_CHUNKS = 16_000
_VOYAGE_CONTEXT_DOCUMENT_TOKEN_BUDGET = int(
    _VOYAGE_CONTEXT_MAX_DOCUMENT_TOKENS * _VOYAGE_TOKEN_SAFETY_FACTOR
)
_VOYAGE_CONTEXT_REQUEST_TOKEN_BUDGET = int(
    _VOYAGE_CONTEXT_MAX_REQUEST_TOKENS * _VOYAGE_TOKEN_SAFETY_FACTOR
)
_MAX_ATTEMPTS = 3
_RETRY_AFTER_BUDGET_S = 60.0
_DEFAULT_FASTEMBED_CACHE = Path.home() / ".cache" / "fastembed"
_LOCAL_EMBED_MAX_WORKERS = 4
_LOCAL_EMBED_WORKER_SLOTS = threading.BoundedSemaphore(_LOCAL_EMBED_MAX_WORKERS)

BeforeDispatch = Callable[[tuple[int, ...]], None]

# Default model per provider for the raw-history CHUNK corpus. Voyage maps to
# voyage-context-4 (its contextualized-chunks model); local providers reuse the
# configured model unchanged (local-first posture, plain per-chunk embedding).
_DEFAULT_CHUNK_MODELS = {"voyage": "voyage-context-4", "voyageai": "voyage-context-4"}


def default_chunk_model(provider: str, configured_model: str) -> str:
    """Return the chunk-corpus model for a provider.

    Voyage gets the contextualized voyage-context-4 default; every other
    provider reuses the configured summary model so the local-first posture is
    unchanged (fastembed/ollama chunk-embed with the same local model).
    """
    configured = str(configured_model or "").strip()
    key = str(provider or "").strip().lower()
    mapped = _DEFAULT_CHUNK_MODELS.get(key)
    if mapped:
        # An explicit context model is the operator's stated chunk-model intent
        # and wins over the mapping; the voyage-context-4 default applies ONLY
        # when the configured model is a plain (non-context) voyage model. This
        # keeps the dry-run display equal to what apply resolves for the chunk
        # corpus (single resolution path through this function).
        if _is_voyage_context_model(configured):
            return configured
        return mapped
    return configured


def _is_voyage_context_model(model: str) -> bool:
    """True for Voyage contextualized-chunk models (voyage-context-3/-4/...).

    These are served ONLY by the ``/v1/contextualizedembeddings`` endpoint; the
    flat ``/v1/embeddings`` endpoint rejects them with HTTP 400.
    """
    return str(model or "").strip().lower().startswith("voyage-context")


def _plan_contextualized_requests(
    token_counts: "Sequence[Sequence[int]]",
    *,
    doc_token_budget: int,
    request_token_budget: int,
    request_chunk_budget: int,
    max_inputs: int,
) -> list[list[list[tuple[int, int]]]]:
    """Plan contextualized requests from per-document chunk token counts.

    ``token_counts[d]`` is the list of chunk token counts for document ``d``
    (oversize single chunks already removed by the caller). Any document whose
    summed tokens exceed ``doc_token_budget`` is split into contiguous
    sub-documents, each still an inputs inner-list (context is preserved within
    a sub-document). Sub-documents are then greedily packed into requests
    bounded by ``request_token_budget`` (total tokens), ``request_chunk_budget``
    (total chunks) and ``max_inputs`` (inner lists).

    Returns a list of requests; each request is a list of sub-documents; each
    sub-document is a list of ``(doc_index, chunk_index)`` positions, so the
    caller maps every planned chunk straight back to its source text/vector.
    """
    subdocs: list[list[tuple[int, int]]] = []
    for doc_index, tokens in enumerate(token_counts):
        current: list[tuple[int, int]] = []
        current_tokens = 0
        for chunk_index, tok in enumerate(tokens):
            if current and current_tokens + int(tok) > doc_token_budget:
                subdocs.append(current)
                current = []
                current_tokens = 0
            current.append((doc_index, chunk_index))
            current_tokens += int(tok)
        if current:
            subdocs.append(current)
    max_inputs = max(1, int(max_inputs))
    requests: list[list[list[tuple[int, int]]]] = []
    current_req: list[list[tuple[int, int]]] = []
    req_tokens = 0
    req_chunks = 0
    for sub in subdocs:
        sub_tokens = sum(int(token_counts[d][c]) for d, c in sub)
        sub_chunks = len(sub)
        if current_req and (
            req_tokens + sub_tokens > request_token_budget
            or req_chunks + sub_chunks > request_chunk_budget
            or len(current_req) + 1 > max_inputs
        ):
            requests.append(current_req)
            current_req = []
            req_tokens = 0
            req_chunks = 0
        current_req.append(sub)
        req_tokens += sub_tokens
        req_chunks += sub_chunks
    if current_req:
        requests.append(current_req)
    return requests


def embed_contextualized(
    provider: "EmbeddingProvider",
    chunks_by_doc: "Sequence[Sequence[str]]",
) -> list[list[list[float]]]:
    """Embed per-document chunk lists, contextualizing when the provider can.

    Returns one vector list per input document, aligned to that document's
    chunks. If the provider exposes ``embed_contextualized`` (Voyage), that is
    used; otherwise the chunks are flattened, embedded with the plain
    ``embed_documents`` path (ollama/fastembed local-first fallback), and
    regrouped by document so the return shape is identical either way.
    """
    method = getattr(provider, "embed_contextualized", None)
    if callable(method):
        return method(chunks_by_doc)
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for chunks in chunks_by_doc:
        start = len(flat)
        flat.extend(str(chunk) for chunk in chunks)
        spans.append((start, len(flat)))
    vectors = provider.embed_documents(flat) if flat else []
    return [[list(vector) for vector in vectors[start:end]] for start, end in spans]


class EmbeddingProvider(Protocol):
    """Minimal provider contract consumed by later embedding workers."""

    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_document_batches(
        self,
        texts: Sequence[str],
        *,
        before_dispatch: BeforeDispatch | None = None,
    ) -> Iterator["EmbeddedDocumentBatch"]: ...

    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingProviderError(RuntimeError):
    """Base class for operator-readable provider failures."""


class ProviderUnavailable(EmbeddingProviderError):
    """The configured provider or its optional dependency is unavailable."""


class ProviderNotWarmedUp(EmbeddingProviderError):
    """A local embedding model has not been explicitly downloaded yet."""


class ProviderCircuitOpen(EmbeddingProviderError):
    """Calls are temporarily blocked after repeated provider failures."""


class ProviderRateLimited(EmbeddingProviderError):
    """The local embedding call-rate guard is temporarily open."""


class ProviderPreDispatchError(EmbeddingProviderError):
    """Definitive outcome: durable marking ran, but provider I/O never started."""

    kind = "pre_dispatch"
    transport_started = False


class VoyageError(EmbeddingProviderError):
    """A classified Voyage API failure."""

    def __init__(self, kind: str, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


HttpTransport = Callable[..., HttpResponse]


@dataclass(frozen=True)
class EmbeddedDocumentBatch:
    """One provider-accepted request, preserving original document indexes."""

    indexes: tuple[int, ...]
    vectors: tuple[tuple[float, ...], ...]


def _run_blocking_with_deadline(
    call: Callable[[], Any], *, timeout: float, provider: str
) -> Any:
    """Bound a local blocking encoder without waiting for an overrun worker."""
    budget = max(0.0, float(timeout))
    if budget <= 0:
        raise EmbeddingProviderError(f"{provider} operation deadline exceeded")
    deadline = time.monotonic() + budget
    worker_slots = _LOCAL_EMBED_WORKER_SLOTS
    if not worker_slots.acquire(timeout=budget):
        raise EmbeddingProviderError(f"{provider} worker capacity exhausted")
    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            result.put((True, call()))
        except BaseException as exc:  # propagated in the caller thread
            result.put((False, exc))
        finally:
            # A timed-out caller deliberately leaves the slot held until its
            # daemon worker really exits. Repeated timeouts therefore cannot
            # accumulate an unbounded number of live local encoders.
            worker_slots.release()

    worker = threading.Thread(
        target=run, name=f"lcm-{provider.lower()}-embedding", daemon=True
    )
    try:
        worker.start()
    except BaseException:
        worker_slots.release()
        raise
    worker.join(max(0.0, deadline - time.monotonic()))
    if worker.is_alive():
        raise EmbeddingProviderError(f"{provider} operation deadline exceeded")
    ok, value = result.get_nowait()
    if not ok:
        raise value
    return value


def _default_http_transport(
    *,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> HttpResponse:
    """POST JSON using urllib while preserving HTTP error response bodies."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status=int(response.status),
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status=int(exc.code),
            headers=dict(exc.headers.items()) if exc.headers else {},
            body=exc.read(),
        )


@dataclass
class EmbeddingCircuitBreaker:
    """Small process-local circuit breaker, separate from summarization."""

    failure_threshold: int = 2
    cooldown_seconds: float = 300.0
    _failures: int = 0
    _open_until: float = 0.0

    def allows(self, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        if self._open_until and current >= self._open_until:
            self._open_until = 0.0
            self._failures = 0
        return current >= self._open_until

    def record_success(self) -> None:
        self._failures = 0
        self._open_until = 0.0

    def record_failure(self, *, now: float | None = None) -> None:
        self._failures += 1
        if self._failures >= max(1, int(self.failure_threshold or 1)):
            current = time.monotonic() if now is None else now
            self._open_until = current + max(0.0, float(self.cooldown_seconds or 0.0))


@dataclass
class EmbeddingSpendGuard:
    """Sliding-window call guard that bounds accidental embedding spend."""

    max_calls: int = 60
    window_seconds: float = 60.0
    backoff_seconds: float = 60.0
    _calls: list[float] = field(default_factory=list)
    _backoff_until: float = 0.0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._calls = [called_at for called_at in self._calls if called_at >= cutoff]

    def allows(self, *, now: float | None = None) -> bool:
        if self.max_calls <= 0:
            return True
        current = time.monotonic() if now is None else now
        if current < self._backoff_until:
            return False
        self._prune(current)
        return len(self._calls) < self.max_calls

    def record_call(self, *, now: float | None = None) -> None:
        if self.max_calls <= 0:
            return
        current = time.monotonic() if now is None else now
        self._prune(current)
        self._calls.append(current)
        if len(self._calls) >= self.max_calls:
            self._calls.clear()
            self._backoff_until = current + max(0.0, self.backoff_seconds)

    def clear(self) -> None:
        self._calls.clear()
        self._backoff_until = 0.0


class _ResilientProvider:
    provider_id = "unknown"

    def __init__(
        self,
        *,
        breaker: EmbeddingCircuitBreaker | None = None,
        spend_guard: EmbeddingSpendGuard | None = None,
    ) -> None:
        self.breaker = breaker or EmbeddingCircuitBreaker()
        self.spend_guard = spend_guard or EmbeddingSpendGuard()

    def _guarded(self, call: Callable[[], Any]) -> Any:
        if not self.breaker.allows():
            raise ProviderCircuitOpen(
                f"{self.provider_id} embedding circuit is cooling down"
            )
        if not self.spend_guard.allows():
            raise ProviderRateLimited(
                f"{self.provider_id} embedding call-rate guard is cooling down"
            )
        self.spend_guard.record_call()
        try:
            result = call()
        except Exception as exc:
            # Authentication failures are definitive operator/configuration
            # errors, not evidence that the provider is temporarily
            # unavailable. Keeping them out of the breaker preserves the
            # actionable VoyageError on every attempt instead of eventually
            # replacing it with a misleading cooling-down error.
            if not (isinstance(exc, VoyageError) and exc.kind == "auth"):
                self.breaker.record_failure()
            raise
        self.breaker.record_success()
        return result


def _response_json(response: HttpResponse, *, provider: str) -> dict[str, Any]:
    try:
        decoded = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbeddingProviderError(
            f"{provider} returned an invalid JSON response"
        ) from exc
    if not isinstance(decoded, dict):
        raise EmbeddingProviderError(f"{provider} returned an invalid response shape")
    return decoded


def _coerce_vectors(raw: Any, *, provider: str) -> list[list[float]]:
    if not isinstance(raw, list):
        raise EmbeddingProviderError(f"{provider} response did not contain embeddings")
    vectors: list[list[float]] = []
    for vector in raw:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, (list, tuple)):
            raise EmbeddingProviderError(f"{provider} returned an invalid embedding")
        try:
            vectors.append([float(value) for value in vector])
        except (TypeError, ValueError) as exc:
            raise EmbeddingProviderError(
                f"{provider} returned a non-numeric embedding"
            ) from exc
    return vectors


def _scrub_response_body(body: bytes) -> str:
    """Keep useful API errors while removing fields likely to echo inputs."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "<unparseable response body suppressed>"

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): (
                    "[REDACTED]"
                    if any(
                        marker in str(key).strip().lower()
                        for marker in ("input", "texts", "documents")
                    )
                    else scrub(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, str):
            return "[REDACTED]"
        return value

    return json.dumps(scrub(payload), ensure_ascii=True, sort_keys=True)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value)
    return None


def _retry_after_seconds(value: str | None, *, now: float | None = None) -> float:
    if not value:
        return 0.0
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            current = time.time() if now is None else now
            return max(0.0, parsed.timestamp() - current)
        except (TypeError, ValueError, OverflowError):
            return 0.0


class VoyageProvider(_ResilientProvider):
    provider_id = "voyage"

    def __init__(
        self,
        model: str,
        *,
        transport: HttpTransport | None = None,
        timeout: float = 3.0,
        sleeper: Callable[[float], None] = time.sleep,
        breaker: EmbeddingCircuitBreaker | None = None,
        spend_guard: EmbeddingSpendGuard | None = None,
        max_batch_items: int = _VOYAGE_MAX_BATCH_ITEMS,
    ) -> None:
        super().__init__(breaker=breaker, spend_guard=spend_guard)
        self._model_id = str(model).strip()
        if not self._model_id:
            raise ValueError("Voyage embedding model must not be empty")
        # Context models are dispatched through the contextualized endpoint; the
        # flat _request payload path is guarded against them (see _request).
        self._is_context = _is_voyage_context_model(self._model_id)
        # Populated from every successful response's usage.total_tokens so a
        # caller can report provider-billed document and query tokens.
        self.last_usage_tokens = 0
        self._transport = transport or _default_http_transport
        self.timeout = float(timeout)
        self._sleep = sleeper
        self._dim = 0
        # Voyage rejects any request above 1,000 input items; that hard limit is
        # authoritative regardless of config, so clamp down to it. A config value
        # above the cap (e.g. 2000) would otherwise emit a >1000-input request.
        self.max_batch_items = max(1, min(int(max_batch_items), _VOYAGE_MAX_BATCH_ITEMS))
        self.last_skipped_documents: list[int] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _error(
        self, response: HttpResponse, *, include_body: bool = True
    ) -> VoyageError:
        status = response.status
        if status in {401, 403}:
            kind = "auth"
        elif status == 429:
            kind = "rate_limit"
        elif 400 <= status < 500:
            kind = "bad_request"
        else:
            kind = "server_error"
        scrubbed = (
            _scrub_response_body(response.body)
            if include_body
            else "[SKIPPED: response-processing deadline]"
        )
        logger.warning("Voyage embedding error status=%s body=%s", status, scrubbed)
        return VoyageError(kind, f"Voyage embedding request failed ({status})", status_code=status)

    def _request(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
        timeout: float | None = None,
        max_attempts: int = _MAX_ATTEMPTS,
        retry_after_budget_s: float = _RETRY_AFTER_BUDGET_S,
        deadline_budget_s: float | None = None,
        before_transport: Callable[[], None] | None = None,
        url: str = _VOYAGE_URL,
        payload: Mapping[str, Any] | None = None,
        parse: Callable[[HttpResponse], Any] | None = None,
    ) -> Any:
        """POST to Voyage under one absolute deadline across attempts + backoff.

        ``payload``/``parse``/``url`` are injected by the contextualized path
        (:meth:`_contextualized_request`); when omitted this builds and parses
        the flat ``/v1/embeddings`` request unchanged and returns
        ``list[list[float]]`` ordered to ``texts``.
        """
        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not api_key:
            raise VoyageError("auth", "VOYAGE_API_KEY is not set")
        if payload is None:
            # Guard: a context model must NEVER be posted to the flat endpoint
            # (live-proven HTTP 400). Callers route context models through the
            # contextualized endpoint, which injects its own payload/url below.
            if self._is_context:
                raise VoyageError(
                    "bad_request",
                    f"{self.model_id} is a contextualized model; use the "
                    "contextualized endpoint",
                )
            payload = {
                "model": self.model_id,
                "input": list(texts),
                "input_type": input_type,
                "truncation": False,
            }
        request_timeout = self.timeout if timeout is None else max(0.001, float(timeout))
        attempts = max(1, int(max_attempts))
        retry_after_spent = 0.0
        last_error: VoyageError | None = None
        # ONE monotonic deadline across every attempt AND its backoff sleep, so
        # a tiny budget returns in ~that budget rather than accumulating
        # per-attempt timeouts plus exponential backoff.
        deadline = (
            time.monotonic() + max(0.0, float(deadline_budget_s))
            if deadline_budget_s is not None
            else None
        )

        def _remaining() -> float | None:
            return None if deadline is None else deadline - time.monotonic()

        for attempt in range(attempts):
            remaining = _remaining()
            if remaining is not None:
                if remaining <= 0:
                    raise last_error or VoyageError(
                        "timeout", "Voyage operation deadline exceeded"
                    )
            if before_transport is not None:
                before_transport()
            # The durable callback is intentionally the final preflight. It may
            # block on SQLite, so recompute the budget AFTER it returns and do
            # not begin billable transport with a stale timeout.
            remaining = _remaining()
            if remaining is not None:
                if remaining <= 0:
                    raise ProviderPreDispatchError(
                        "Voyage operation deadline exceeded before transport"
                    )
                attempt_timeout = min(request_timeout, remaining)
            else:
                attempt_timeout = request_timeout
            try:
                response = self._transport(
                    url=url,
                    payload=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=attempt_timeout,
                )
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                last_error = VoyageError("network", f"Voyage network error: {exc}")
                # Once transport starts, a timeout/network exception cannot
                # prove the remote rejected the request. Never automatically
                # resend an identical potentially-billable operation.
                raise last_error from exc

            if 200 <= response.status < 300:
                # Transport completion does not stop the operation clock. A
                # large/malicious response can make JSON decode, sorting, and
                # vector coercion block, so keep the complete success parser
                # inside the same absolute budget and do not publish a result
                # that only became available after expiry.
                remaining = _remaining()
                if remaining is not None and remaining <= 0:
                    raise VoyageError(
                        "timeout",
                        "Voyage operation deadline exceeded before response processing",
                    )

                def parse_success() -> list[list[float]]:
                    data = _response_json(response, provider="Voyage")
                    rows = data.get("data")
                    if not isinstance(rows, list):
                        raise EmbeddingProviderError(
                            "Voyage response did not contain embedding data"
                        )
                    ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
                    parsed = _coerce_vectors(
                        [
                            row.get("embedding")
                            for row in ordered
                            if isinstance(row, dict)
                        ],
                        provider="Voyage",
                    )
                    if len(parsed) != len(texts):
                        raise EmbeddingProviderError(
                            "Voyage returned a different number of embeddings than requested"
                        )
                    usage = data.get("usage")
                    if isinstance(usage, dict):
                        try:
                            self.last_usage_tokens = int(usage.get("total_tokens", 0))
                        except (TypeError, ValueError):
                            pass
                    return parsed

                runner = (
                    (lambda: parse(response)) if parse is not None else parse_success
                )
                try:
                    vectors = (
                        runner()
                        if remaining is None
                        else _run_blocking_with_deadline(
                            runner,
                            timeout=remaining,
                            provider="Voyage response processing",
                        )
                    )
                except EmbeddingProviderError as exc:
                    if deadline is not None and _remaining() <= 0:
                        raise VoyageError(
                            "timeout",
                            "Voyage operation deadline exceeded during response processing",
                        ) from exc
                    raise
                if deadline is not None and _remaining() <= 0:
                    raise VoyageError(
                        "timeout",
                        "Voyage operation deadline exceeded during response processing",
                    )
                return vectors

            remaining = _remaining()
            if remaining is not None and remaining <= 0:
                raise self._error(response, include_body=False)
            if remaining is None:
                last_error = self._error(response)
            else:
                try:
                    last_error = _run_blocking_with_deadline(
                        lambda: self._error(response),
                        timeout=remaining,
                        provider="Voyage error response processing",
                    )
                except EmbeddingProviderError:
                    # The status code itself is already authoritative even if
                    # diagnostic body scrubbing cannot finish in-budget. Do
                    # not wait for it, and never turn a 5xx into an auto-retry.
                    last_error = self._error(response, include_body=False)
            if deadline is not None and _remaining() <= 0:
                raise self._error(response, include_body=False)
            # A 429 is an authoritative rejection and is safe to retry. A 5xx
            # may follow remote acceptance, so it is deliberately fail-closed.
            retryable = response.status == 429
            if not retryable or attempt + 1 >= attempts:
                raise last_error
            if response.status == 429:
                delay = _retry_after_seconds(_header(response.headers, "Retry-After"))
                remaining_retry_after = retry_after_budget_s - retry_after_spent
                if delay > remaining_retry_after:
                    raise last_error
                retry_after_spent += delay
            else:
                delay = min(25.0, 0.5 * (2 ** attempt))
            if not self._sleep_within_deadline(delay, deadline):
                raise last_error
        raise last_error or VoyageError("network", "Voyage request failed")

    def _parse_contextualized(
        self, response: HttpResponse, *, expected: Sequence[int]
    ) -> list[list[list[float]]]:
        """Parse the nested contextualized response, order-preserving.

        The response ``data`` is one object per input list (outer ``index``),
        each nesting its own ``data`` list of ``{embedding, index, ...}`` chunk
        objects. Both levels are sorted by ``index`` so a KNN hit maps straight
        back to its ``(document, chunk)`` position regardless of wire order.
        ``usage.total_tokens`` is recorded for provider-billed accounting.
        """
        data = _response_json(response, provider="Voyage")
        rows = data.get("data")
        if not isinstance(rows, list):
            raise EmbeddingProviderError(
                "Voyage contextualized response did not contain embedding data"
            )
        ordered_outer = sorted(
            (row for row in rows if isinstance(row, dict)),
            key=lambda row: int(row.get("index", 0)),
        )
        if len(ordered_outer) != len(expected):
            raise EmbeddingProviderError(
                "Voyage returned a different number of documents than requested"
            )
        result: list[list[list[float]]] = []
        for outer, want in zip(ordered_outer, expected):
            inner = outer.get("data")
            if not isinstance(inner, list):
                raise EmbeddingProviderError(
                    "Voyage contextualized document did not contain embeddings"
                )
            ordered_inner = sorted(
                (item for item in inner if isinstance(item, dict)),
                key=lambda item: int(item.get("index", 0)),
            )
            vectors = _coerce_vectors(
                [item.get("embedding") for item in ordered_inner], provider="Voyage"
            )
            if len(vectors) != want:
                raise EmbeddingProviderError(
                    "Voyage returned a different number of chunk embeddings than requested"
                )
            result.append(vectors)
        usage = data.get("usage")
        if isinstance(usage, dict):
            try:
                self.last_usage_tokens = int(usage.get("total_tokens", 0))
            except (TypeError, ValueError):
                pass
        return result

    def _contextualized_request(
        self,
        inputs: Sequence[Sequence[str]],
        *,
        input_type: str,
        timeout: float | None = None,
        max_attempts: int = _MAX_ATTEMPTS,
        retry_after_budget_s: float = _RETRY_AFTER_BUDGET_S,
        deadline_budget_s: float | None = None,
        before_transport: Callable[[], None] | None = None,
    ) -> list[list[list[float]]]:
        """POST one contextualized request; returns per-document vector lists.

        Reuses the flat path's retry/deadline/cap discipline (:meth:`_request`)
        with an injected ``inputs`` payload and nested parser, so a context
        model shares the same absolute-deadline and fail-closed semantics.
        """
        groups = [[str(chunk) for chunk in group] for group in inputs]
        expected = [len(group) for group in groups]
        payload = {
            "model": self.model_id,
            "inputs": groups,
            "input_type": input_type,
        }
        return self._request(
            (),
            input_type=input_type,
            timeout=timeout,
            max_attempts=max_attempts,
            retry_after_budget_s=retry_after_budget_s,
            deadline_budget_s=deadline_budget_s,
            before_transport=before_transport,
            url=_VOYAGE_CONTEXT_URL,
            payload=payload,
            parse=lambda response: self._parse_contextualized(
                response, expected=expected
            ),
        )

    def _embed_document_request(
        self, texts: Sequence[str], **kwargs: Any
    ) -> list[list[float]]:
        """Embed a batch of documents, routing context models correctly.

        Non-context models keep the flat endpoint unchanged. Context models go
        through the contextualized endpoint as one single-chunk input list per
        document (each document = one inner list; true cross-chunk grouping is
        available via :meth:`embed_contextualized`), so the flat 400 is avoided
        while the batch contract (one vector per input document) is preserved.
        """
        if not self._is_context:
            return self._request(texts, input_type="document", **kwargs)
        nested = self._contextualized_request(
            [[text] for text in texts], input_type="document", **kwargs
        )
        return [group[0] for group in nested]

    def embed_contextualized(
        self, chunks_by_doc: Sequence[Sequence[str]]
    ) -> list[list[list[float]]]:
        """Embed per-document chunk lists via the contextualized endpoint.

        One inner list per source document (conversation/message group); the
        return nests one vector per chunk, aligned to each document's chunks.
        Non-context Voyage models have no contextualized endpoint, so they fall
        back to a flat per-chunk embedding regrouped by document — the module
        ``embed_contextualized`` helper handles that dispatch for local
        providers; here (Voyage) a context model uses the real endpoint.
        """
        groups = [[str(chunk) for chunk in group] for group in chunks_by_doc]
        if not groups:
            return []
        if not self._is_context:
            # A non-context voyage model: embed the flattened chunks and regroup.
            flat = [chunk for group in groups for chunk in group]
            vectors = self.embed_documents(flat) if flat else []
            result: list[list[list[float]]] = []
            offset = 0
            for group in groups:
                result.append(
                    [list(vector) for vector in vectors[offset:offset + len(group)]]
                )
                offset += len(group)
            return result
        # A context model: split oversize documents and pack the resulting
        # sub-documents into cap-respecting requests, reassembling each chunk's
        # vector back into its source document/chunk position.
        token_counts = [[count_tokens(chunk) for chunk in group] for group in groups]
        plan = _plan_contextualized_requests(
            token_counts,
            doc_token_budget=_VOYAGE_CONTEXT_DOCUMENT_TOKEN_BUDGET,
            request_token_budget=_VOYAGE_CONTEXT_REQUEST_TOKEN_BUDGET,
            request_chunk_budget=_VOYAGE_CONTEXT_MAX_REQUEST_CHUNKS,
            max_inputs=self.max_batch_items,
        )
        result: list[list[list[float]]] = [[[] for _ in group] for group in groups]
        for request in plan:
            inputs = [[groups[d][c] for d, c in sub] for sub in request]
            nested = self._guarded(
                lambda inputs=inputs: self._contextualized_request(
                    inputs, input_type="document", deadline_budget_s=self.timeout
                )
            )
            for sub, sub_vectors in zip(request, nested):
                for (d, c), vector in zip(sub, sub_vectors):
                    result[d][c] = list(vector)
        return result

    @property
    def supports_contextualized_grouping(self) -> bool:
        """True when this model can contextualize a document's chunks together.

        Only voyage-context-* models actually gain cross-chunk context; a plain
        voyage model keeps the flat per-chunk backfill path unchanged.
        """
        return self._is_context

    def embed_chunk_group_batches(
        self,
        groups: Sequence[Sequence[tuple[int, str]]],
        *,
        before_dispatch: BeforeDispatch | None = None,
    ) -> Iterator[EmbeddedDocumentBatch]:
        """Contextualized chunk embedding, grouped one document per message.

        Each group is one document — a message's chunks — as ``(flat_index,
        text)`` pairs, where ``flat_index`` is the chunk's position in the
        caller's flat backfill batch. A document's chunks are sent together in
        one inputs inner-list so voyage-context-* contextualizes them across the
        message, then each yielded :class:`EmbeddedDocumentBatch` still carries
        per-chunk flat indexes + vectors so the caller publishes every chunk as
        its own independent row (the batch-publish + lease/inflight path is
        unchanged).

        Oversize documents (> the per-doc token budget) are split into
        contiguous sub-documents; a single chunk above the per-chunk cap is
        recorded in ``last_skipped_documents`` and never dispatched. Requests
        respect the 120K-token / 16K-chunk / 1,000-input caps.
        """
        deadline = time.monotonic() + max(0.0, self.timeout)
        self.last_skipped_documents = []

        def _remaining() -> float:
            return deadline - time.monotonic()

        # Tokenize and drop oversize single chunks (defensive; chunking caps a
        # chunk well below the per-chunk limit). Surviving chunks keep their
        # (flat_index, text) identity and per-document grouping.
        documents: list[list[tuple[int, str]]] = []
        token_counts: list[list[int]] = []
        for group in groups:
            kept: list[tuple[int, str]] = []
            kept_tokens: list[int] = []
            for flat_index, text in group:
                if _remaining() <= 0:
                    raise VoyageError(
                        "timeout",
                        "Voyage operation deadline exceeded during chunk grouping",
                    )
                prepared = str(text)

                def prepare(value=prepared) -> int:
                    return count_tokens(value)

                tokens = _run_blocking_with_deadline(
                    prepare,
                    timeout=_remaining(),
                    provider="Voyage chunk preprocessing",
                )
                if tokens > _VOYAGE_CONTEXT_MAX_CHUNK_TOKENS:
                    self.last_skipped_documents.append(int(flat_index))
                    logger.warning(
                        "Skipping Voyage chunk index=%d tokens=%d limit=%d",
                        int(flat_index),
                        tokens,
                        _VOYAGE_CONTEXT_MAX_CHUNK_TOKENS,
                    )
                    continue
                kept.append((int(flat_index), prepared))
                kept_tokens.append(tokens)
            if kept:
                documents.append(kept)
                token_counts.append(kept_tokens)

        if not documents:
            return

        plan = _plan_contextualized_requests(
            token_counts,
            doc_token_budget=_VOYAGE_CONTEXT_DOCUMENT_TOKEN_BUDGET,
            request_token_budget=_VOYAGE_CONTEXT_REQUEST_TOKEN_BUDGET,
            request_chunk_budget=_VOYAGE_CONTEXT_MAX_REQUEST_CHUNKS,
            max_inputs=self.max_batch_items,
        )
        for request in plan:
            if _remaining() <= 0:
                raise VoyageError("timeout", "Voyage operation deadline exceeded")
            inputs = [[documents[d][c][1] for d, c in sub] for sub in request]
            flat_indexes = tuple(
                documents[d][c][0] for sub in request for d, c in sub
            )
            if before_dispatch is not None:
                dispatch = partial(before_dispatch, flat_indexes)
            else:
                dispatch = None
            nested = self._guarded(
                lambda inputs=inputs, dispatch=dispatch, budget=_remaining(): (
                    self._contextualized_request(
                        inputs,
                        input_type="document",
                        max_attempts=1 if before_dispatch is not None else _MAX_ATTEMPTS,
                        deadline_budget_s=budget,
                        before_transport=dispatch,
                    )
                )
            )
            vectors = [vector for sub_vectors in nested for vector in sub_vectors]
            self._remember_dim(vectors)
            yield EmbeddedDocumentBatch(
                flat_indexes,
                tuple(tuple(vector) for vector in vectors),
            )

    def _sleep_within_deadline(self, delay: float, deadline: float | None) -> bool:
        """Sleep ``delay`` only if it fits the absolute deadline; else give up.

        Returns True if the caller may retry (the sleep happened within budget),
        False if the backoff would blow the deadline and the caller should stop.
        """
        if deadline is None:
            self._sleep(delay)
            return True
        remaining = deadline - time.monotonic()
        if delay >= remaining:
            return False
        self._sleep(delay)
        return True

    def _remember_dim(self, vectors: Sequence[Sequence[float]]) -> None:
        for vector in vectors:
            if not vector:
                raise EmbeddingProviderError("Voyage returned an empty embedding")
            if self._dim and len(vector) != self._dim:
                raise EmbeddingProviderError("Voyage embedding dimension changed within the process")
            self._dim = len(vector)

    def embed_document_batches(
        self,
        texts: Sequence[str],
        *,
        before_dispatch: BeforeDispatch | None = None,
    ) -> Iterator[EmbeddedDocumentBatch]:
        deadline = time.monotonic() + max(0.0, self.timeout)
        accepted: list[tuple[int, str, int]] = []
        self.last_skipped_documents = []
        for index, text in enumerate(texts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VoyageError(
                    "timeout",
                    "Voyage operation deadline exceeded during document preprocessing",
                )

            def prepare_document(value=text) -> tuple[str, int]:
                prepared = str(value)
                return prepared, count_tokens(prepared)

            try:
                text, tokens = _run_blocking_with_deadline(
                    prepare_document,
                    timeout=remaining,
                    provider="Voyage document preprocessing",
                )
            except EmbeddingProviderError as exc:
                raise VoyageError(
                    "timeout",
                    "Voyage operation deadline exceeded during document preprocessing",
                ) from exc
            if time.monotonic() >= deadline:
                raise VoyageError(
                    "timeout",
                    "Voyage operation deadline exceeded during document preprocessing",
                )
            if tokens > _VOYAGE_DOCUMENT_TOKEN_BUDGET:
                self.last_skipped_documents.append(index)
                logger.warning(
                    "Skipping Voyage document index=%d tokens=%d limit=%d",
                    index,
                    tokens,
                    _VOYAGE_DOCUMENT_TOKEN_BUDGET,
                )
                continue
            accepted.append((index, text, tokens))

        batch_indexes: list[int] = []
        batch: list[str] = []
        batch_tokens = 0
        for index, text, tokens in accepted:
            # Flush before the batch would exceed EITHER the token budget or the
            # provider's per-request item cap, so e.g. 1001 short documents split
            # into at least two requests.
            if batch and (
                batch_tokens + tokens > _VOYAGE_BATCH_TOKEN_BUDGET
                or len(batch) >= self.max_batch_items
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VoyageError("timeout", "Voyage operation deadline exceeded")
                indexes = tuple(batch_indexes)
                if before_dispatch is not None:
                    dispatch = partial(before_dispatch, indexes)
                else:
                    dispatch = None
                vectors = self._guarded(
                    lambda current=tuple(batch), budget=remaining: self._embed_document_request(
                        current,
                        max_attempts=1 if before_dispatch is not None else _MAX_ATTEMPTS,
                        deadline_budget_s=budget,
                        before_transport=dispatch,
                    )
                )
                self._remember_dim(vectors)
                yield EmbeddedDocumentBatch(
                    indexes,
                    tuple(tuple(vector) for vector in vectors),
                )
                batch_indexes = []
                batch = []
                batch_tokens = 0
            batch_indexes.append(index)
            batch.append(text)
            batch_tokens += tokens
        if batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VoyageError("timeout", "Voyage operation deadline exceeded")
            indexes = tuple(batch_indexes)
            if before_dispatch is not None:
                dispatch = partial(before_dispatch, indexes)
            else:
                dispatch = None
            vectors = self._guarded(
                lambda current=tuple(batch), budget=remaining: self._embed_document_request(
                    current,
                    max_attempts=1 if before_dispatch is not None else _MAX_ATTEMPTS,
                    deadline_budget_s=budget,
                    before_transport=dispatch,
                )
            )
            self._remember_dim(vectors)
            yield EmbeddedDocumentBatch(
                indexes,
                tuple(tuple(vector) for vector in vectors),
            )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            list(vector)
            for batch in self.embed_document_batches(texts)
            for vector in batch.vectors
        ]

    def _query_request(
        self, text: str, *, timeout: float | None, deadline_budget_s: float,
        retry_after_budget_s: float = _RETRY_AFTER_BUDGET_S,
    ) -> list[float]:
        """Embed one query vector, routing context models to the right endpoint.

        Per the Voyage docs a context model embeds a query on the SAME
        contextualized endpoint with ``input_type="query"`` and a single-item
        inner list (``inputs=[[query]]``); non-context models keep the flat
        ``/v1/embeddings`` query path unchanged.
        """
        if self._is_context:
            nested = self._contextualized_request(
                [[str(text)]], input_type="query", timeout=timeout,
                retry_after_budget_s=retry_after_budget_s,
                deadline_budget_s=deadline_budget_s,
            )
            return nested[0][0]
        return self._request(
            (str(text),), input_type="query", timeout=timeout,
            retry_after_budget_s=retry_after_budget_s,
            deadline_budget_s=deadline_budget_s,
        )[0]

    def embed_query(self, text: str) -> list[float]:
        # Normal (non-interactive) query embedding also gets ONE absolute
        # deadline across every attempt AND its backoff sleeps, so a tiny
        # configured timeout returns in ~that budget instead of stacking
        # per-attempt timeouts plus exponential backoff.
        vector = self._guarded(
            lambda: self._query_request(
                text, timeout=None, deadline_budget_s=self.timeout
            )
        )
        self._remember_dim([vector])
        return vector

    def embed_query_interactive(self, text: str, *, timeout: float) -> list[float]:
        """Embed a latency-sensitive query under one absolute time budget.

        Retries are allowed only insofar as they fit ``timeout`` — the single
        monotonic deadline covers every attempt plus any backoff, so a tiny
        budget returns within that budget instead of stacking per-attempt waits.
        """
        vector = self._guarded(
            lambda: self._query_request(
                text, timeout=timeout, retry_after_budget_s=0.0,
                deadline_budget_s=timeout,
            )
        )
        self._remember_dim([vector])
        return vector

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
        timeout: float,
        model: str = _VOYAGE_RERANK_MODEL,
    ) -> list[tuple[int, float]]:
        """Cross-encoder rerank ``documents`` against ``query`` in one API call.

        Returns ``(original_index, relevance_score)`` pairs ordered by descending
        relevance under a single absolute ``timeout`` budget. Callers treat any
        raised error as "skip rerank" and fall back to their prior order.
        """
        docs = [str(document) for document in documents]
        if not docs:
            return []
        budget = max(0.001, float(timeout))
        deadline = time.monotonic() + budget
        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not api_key:
            raise VoyageError("auth", "VOYAGE_API_KEY is not set")
        payload: dict[str, Any] = {
            "model": str(model),
            "query": str(query),
            "documents": docs,
            "truncation": True,
        }
        if top_k is not None:
            payload["top_k"] = int(top_k)
        response = self._transport(
            url=_VOYAGE_RERANK_URL,
            payload=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=budget,
        )
        if not (200 <= response.status < 300):
            raise self._error(response, include_body=False)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VoyageError("timeout", "Voyage rerank deadline exceeded")

        def parse() -> list[tuple[int, float]]:
            data = _response_json(response, provider="Voyage rerank")
            rows = data.get("data")
            if not isinstance(rows, list):
                raise EmbeddingProviderError(
                    "Voyage rerank response did not contain result data"
                )
            ranked: list[tuple[int, float]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                index = int(row.get("index", -1))
                if not 0 <= index < len(docs):
                    continue
                ranked.append((index, float(row.get("relevance_score", 0.0))))
            ranked.sort(key=lambda item: (-item[1], item[0]))
            return ranked

        return _run_blocking_with_deadline(
            parse, timeout=remaining, provider="Voyage rerank response processing"
        )


class OllamaProvider(_ResilientProvider):
    provider_id = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        transport: HttpTransport | None = None,
        timeout: float = 3.0,
        sleeper: Callable[[float], None] = time.sleep,
        breaker: EmbeddingCircuitBreaker | None = None,
        spend_guard: EmbeddingSpendGuard | None = None,
    ) -> None:
        super().__init__(breaker=breaker, spend_guard=spend_guard)
        self._model_id = str(model).strip()
        if not self._model_id:
            raise ValueError("Ollama embedding model must not be empty")
        self.base_url = str(base_url).strip().rstrip("/") or "http://localhost:11434"
        self._transport = transport or _default_http_transport
        self.timeout = float(timeout)
        self._sleep = sleeper
        self._dim = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _request(
        self,
        texts: Sequence[str],
        *,
        timeout: float | None = None,
        max_attempts: int = _MAX_ATTEMPTS,
        deadline_budget_s: float | None = None,
        before_transport: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        request_timeout = self.timeout if timeout is None else max(0.001, float(timeout))
        attempts = max(1, int(max_attempts))
        deadline = (
            time.monotonic() + max(0.0, float(deadline_budget_s))
            if deadline_budget_s is not None
            else None
        )
        for attempt in range(attempts):
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EmbeddingProviderError("Ollama operation deadline exceeded")
            if before_transport is not None:
                before_transport()
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProviderPreDispatchError(
                        "Ollama operation deadline exceeded before transport"
                    )
                attempt_timeout = min(request_timeout, remaining)
            else:
                attempt_timeout = request_timeout
            try:
                response = self._transport(
                    url=f"{self.base_url}/api/embed",
                    # truncate=false so oversized input fails loudly instead of
                    # being silently truncated to a misleading embedding.
                    payload={
                        "model": self.model_id,
                        "input": list(texts),
                        "truncate": False,
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=attempt_timeout,
                )
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                # Remote/local daemon acceptance is ambiguous after transport
                # begins, so an automatic identical resend is not safe.
                raise EmbeddingProviderError(f"Ollama network error: {exc}") from exc
            if not 200 <= response.status < 300:
                raise EmbeddingProviderError(
                    f"Ollama embedding request failed ({response.status})"
                )
            remaining = (
                None if deadline is None else deadline - time.monotonic()
            )
            if remaining is not None and remaining <= 0:
                raise EmbeddingProviderError(
                    "Ollama operation deadline exceeded before response processing"
                )

            def parse_success() -> list[list[float]]:
                payload = _response_json(response, provider="Ollama")
                raw = payload.get("embeddings")
                if raw is None and "embedding" in payload:
                    raw = [payload["embedding"]]
                parsed = _coerce_vectors(raw, provider="Ollama")
                if len(parsed) != len(texts):
                    raise EmbeddingProviderError(
                        "Ollama returned a different number of embeddings than requested"
                    )
                return parsed

            vectors = (
                parse_success()
                if remaining is None
                else _run_blocking_with_deadline(
                    parse_success,
                    timeout=remaining,
                    provider="Ollama response processing",
                )
            )
            if deadline is not None and time.monotonic() >= deadline:
                raise EmbeddingProviderError(
                    "Ollama operation deadline exceeded during response processing"
                )
            return vectors
        raise EmbeddingProviderError("Ollama embedding request failed")

    def _embed(
        self,
        texts: Sequence[str],
        *,
        timeout: float | None = None,
        max_attempts: int = _MAX_ATTEMPTS,
        deadline_budget_s: float | None = None,
        before_transport: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._guarded(
            lambda: self._request(
                tuple(str(text) for text in texts),
                timeout=timeout,
                max_attempts=max_attempts,
                deadline_budget_s=deadline_budget_s,
                before_transport=before_transport,
            )
        )
        for vector in vectors:
            if not vector:
                raise EmbeddingProviderError("Ollama returned an empty embedding")
            if self._dim and len(vector) != self._dim:
                raise EmbeddingProviderError("Ollama embedding dimension changed within the process")
            self._dim = len(vector)
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # Normal operations also get ONE absolute deadline across attempts +
        # backoff (self.timeout), so retryable failures cannot stack backoff
        # sleeps beyond the configured budget.
        return self._embed(texts, deadline_budget_s=self.timeout)

    def embed_document_batches(
        self,
        texts: Sequence[str],
        *,
        before_dispatch: BeforeDispatch | None = None,
    ) -> Iterator[EmbeddedDocumentBatch]:
        normalized = tuple(str(text) for text in texts)
        if not normalized:
            return
        indexes = tuple(range(len(normalized)))
        vectors = self._embed(
            normalized,
            max_attempts=1 if before_dispatch is not None else _MAX_ATTEMPTS,
            deadline_budget_s=self.timeout,
            before_transport=(
                (lambda: before_dispatch(indexes))
                if before_dispatch is not None
                else None
            ),
        )
        yield EmbeddedDocumentBatch(
            indexes,
            tuple(tuple(vector) for vector in vectors),
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embed((text,), deadline_budget_s=self.timeout)[0]

    def embed_query_interactive(self, text: str, *, timeout: float) -> list[float]:
        """Embed a latency-sensitive query under one absolute time budget."""
        return self._embed(
            (text,), timeout=timeout, deadline_budget_s=timeout
        )[0]

def _load_fastembed():
    """Import the optional dependency in one patchable location."""
    from fastembed import TextEmbedding

    return TextEmbedding


class OpenAICompatibleProvider(_ResilientProvider):
    """OpenAI-compatible ``/v1/embeddings`` provider (SiliconFlow, etc.).

    Local patch (Vocllum): hermes-lcm ships only voyage/ollama/fastembed.
    This provider reuses any OpenAI-format embeddings endpoint — e.g.
    SiliconFlow's ``BAAI/bge-m3`` — so the operator can share the same
    embedding service already used by OpenViking instead of running a
    second local model server.

    Env contract (via LCMConfig):
      LCM_EMBEDDING_PROVIDER=openai-compatible
      LCM_EMBEDDING_MODEL=BAAI/bge-m3
      LCM_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
      LCM_EMBEDDING_API_KEY=...   (falls back to SILICONFLOW_API_KEY)
    """

    provider_id = "openai-compatible"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "",
        api_key_env: str = "LCM_EMBEDDING_API_KEY",
        transport: HttpTransport | None = None,
        timeout: float = 3.0,
        sleeper: Callable[[float], None] = time.sleep,
        breaker: EmbeddingCircuitBreaker | None = None,
        spend_guard: EmbeddingSpendGuard | None = None,
    ) -> None:
        super().__init__(breaker=breaker, spend_guard=spend_guard)
        self._model_id = str(model).strip()
        if not self._model_id:
            raise ValueError("OpenAI-compatible embedding model must not be empty")
        self.base_url = str(base_url).strip().rstrip("/")
        if not self.base_url:
            raise ValueError(
                "LCM_EMBEDDING_BASE_URL must be set for openai-compatible provider"
            )
        self.api_key_env = str(api_key_env).strip() or "LCM_EMBEDDING_API_KEY"
        self._transport = transport or _default_http_transport
        self.timeout = float(timeout)
        self._sleep = sleeper
        self._dim = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _request(
        self,
        texts: Sequence[str],
        *,
        timeout: float | None = None,
        max_attempts: int = _MAX_ATTEMPTS,
        deadline_budget_s: float | None = None,
        before_transport: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            # Convenience fallback: SiliconFlow key name used by OpenViking.
            api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
        if not api_key:
            raise EmbeddingProviderError(
                f"{self.api_key_env} (or SILICONFLOW_API_KEY) is not set"
            )
        request_timeout = self.timeout if timeout is None else max(0.001, float(timeout))
        attempts = max(1, int(max_attempts))
        deadline = (
            time.monotonic() + max(0.0, float(deadline_budget_s))
            if deadline_budget_s is not None
            else None
        )
        for attempt in range(attempts):
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EmbeddingProviderError(
                        "OpenAI-compatible operation deadline exceeded"
                    )
            if before_transport is not None:
                before_transport()
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProviderPreDispatchError(
                        "OpenAI-compatible operation deadline exceeded before transport"
                    )
                attempt_timeout = min(request_timeout, remaining)
            else:
                attempt_timeout = request_timeout
            try:
                response = self._transport(
                    url=f"{self.base_url}/embeddings",
                    payload={
                        "model": self.model_id,
                        "input": list(texts),
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=attempt_timeout,
                )
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                # Transport start makes an automatic identical resend unsafe.
                raise EmbeddingProviderError(
                    f"OpenAI-compatible network error: {exc}"
                ) from exc
            if not 200 <= response.status < 300:
                raise EmbeddingProviderError(
                    f"OpenAI-compatible embedding request failed ({response.status})"
                )
            remaining = (
                None if deadline is None else deadline - time.monotonic()
            )
            if remaining is not None and remaining <= 0:
                raise EmbeddingProviderError(
                    "OpenAI-compatible operation deadline exceeded before response processing"
                )

            def parse_success() -> list[list[float]]:
                payload = _response_json(response, provider="OpenAI-compatible")
                raw = payload.get("data")
                if isinstance(raw, list):
                    ordered = sorted(
                        raw, key=lambda row: int(row.get("index", 0))
                        if isinstance(row, dict)
                        else 0
                    )
                    rows = [
                        row.get("embedding")
                        for row in ordered
                        if isinstance(row, dict)
                    ]
                else:
                    rows = None
                parsed = _coerce_vectors(rows, provider="OpenAI-compatible")
                if len(parsed) != len(texts):
                    raise EmbeddingProviderError(
                        "OpenAI-compatible returned a different number of embeddings than requested"
                    )
                return parsed

            vectors = (
                parse_success()
                if remaining is None
                else _run_blocking_with_deadline(
                    parse_success,
                    timeout=remaining,
                    provider="OpenAI-compatible response processing",
                )
            )
            if deadline is not None and time.monotonic() >= deadline:
                raise EmbeddingProviderError(
                    "OpenAI-compatible operation deadline exceeded during response processing"
                )
            return vectors
        raise EmbeddingProviderError("OpenAI-compatible embedding request failed")

    def _embed(
        self,
        texts: Sequence[str],
        *,
        timeout: float | None = None,
        max_attempts: int = _MAX_ATTEMPTS,
        deadline_budget_s: float | None = None,
        before_transport: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._guarded(
            lambda: self._request(
                tuple(str(text) for text in texts),
                timeout=timeout,
                max_attempts=max_attempts,
                deadline_budget_s=deadline_budget_s,
                before_transport=before_transport,
            )
        )
        for vector in vectors:
            if not vector:
                raise EmbeddingProviderError(
                    "OpenAI-compatible returned an empty embedding"
                )
            if self._dim and len(vector) != self._dim:
                raise EmbeddingProviderError(
                    "OpenAI-compatible embedding dimension changed within the process"
                )
            self._dim = len(vector)
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, deadline_budget_s=self.timeout)

    def embed_document_batches(
        self,
        texts: Sequence[str],
        *,
        before_dispatch: BeforeDispatch | None = None,
    ) -> Iterator[EmbeddedDocumentBatch]:
        normalized = tuple(str(text) for text in texts)
        if not normalized:
            return
        indexes = tuple(range(len(normalized)))
        vectors = self._embed(
            normalized,
            max_attempts=1 if before_dispatch is not None else _MAX_ATTEMPTS,
            deadline_budget_s=self.timeout,
            before_transport=(
                (lambda: before_dispatch(indexes))
                if before_dispatch is not None
                else None
            ),
        )
        yield EmbeddedDocumentBatch(
            indexes,
            tuple(tuple(vector) for vector in vectors),
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embed((text,), deadline_budget_s=self.timeout)[0]

    def embed_query_interactive(self, text: str, *, timeout: float) -> list[float]:
        """Embed a latency-sensitive query under one absolute time budget."""
        return self._embed(
            (text,), timeout=timeout, deadline_budget_s=timeout
        )[0]

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
        timeout: float,
        model: str = "",
    ) -> list[tuple[int, float]]:
        """Rerank documents through a Cohere-compatible ``/rerank`` endpoint."""
        docs = [str(document) for document in documents]
        if not docs:
            return []
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int)
        ):
            raise ValueError("top_k must be an integer or None")
        if top_k is not None and top_k <= 0:
            return []

        budget = max(0.001, float(timeout))
        deadline = time.monotonic() + budget
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            response = self._transport(
                url=f"{self.base_url}/rerank",
                payload={
                    "model": str(model) or _OPENAI_COMPATIBLE_RERANK_MODEL,
                    "query": str(query),
                    "documents": docs,
                    "top_n": len(docs) if top_k is None else min(top_k, len(docs)),
                },
                headers=headers,
                timeout=budget,
            )
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise EmbeddingProviderError(
                f"OpenAI-compatible rerank network error: {exc}"
            ) from exc
        if not 200 <= response.status < 300:
            raise EmbeddingProviderError(
                f"OpenAI-compatible rerank request failed ({response.status})"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EmbeddingProviderError("OpenAI-compatible rerank deadline exceeded")

        def parse() -> list[tuple[int, float]]:
            payload = _response_json(response, provider="OpenAI-compatible rerank")
            rows = payload.get("results")
            if not isinstance(rows, list):
                raise EmbeddingProviderError(
                    "OpenAI-compatible rerank response did not contain result data"
                )
            ranked: list[tuple[int, float]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    raw_index = row["index"]
                    raw_score = row["relevance_score"]
                    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                        continue
                    index = int(raw_index)
                    score = float(raw_score)
                except (KeyError, TypeError, ValueError):
                    continue
                if not 0 <= index < len(docs):
                    continue
                ranked.append((index, score))
            ranked.sort(key=lambda item: (-item[1], item[0]))
            return ranked

        return _run_blocking_with_deadline(
            parse,
            timeout=remaining,
            provider="OpenAI-compatible rerank response processing",
        )


class FastembedProvider(_ResilientProvider):
    provider_id = "fastembed"

    def __init__(
        self,
        model: str,
        *,
        cache_dir: str | Path | None = None,
        timeout: float = 3.0,
        breaker: EmbeddingCircuitBreaker | None = None,
        spend_guard: EmbeddingSpendGuard | None = None,
    ) -> None:
        super().__init__(breaker=breaker, spend_guard=spend_guard)
        self._model_id = str(model).strip()
        if not self._model_id:
            raise ValueError("FastEmbed embedding model must not be empty")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_FASTEMBED_CACHE
        self.timeout = float(timeout)
        self._model: Any = None
        self._dim = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _construct(self, *, allow_download: bool) -> Any:
        try:
            text_embedding = _load_fastembed()
        except ImportError as exc:
            raise ProviderUnavailable(
                "FastEmbed is not installed; install the optional fastembed dependency"
            ) from exc
        try:
            return text_embedding(
                model_name=self.model_id,
                cache_dir=str(self.cache_dir),
                local_files_only=not allow_download,
            )
        except Exception as exc:
            if allow_download:
                raise EmbeddingProviderError(
                    f"FastEmbed warmup failed for {self.model_id}: {exc}"
                ) from exc
            raise ProviderNotWarmedUp(
                f"FastEmbed model {self.model_id!r} is not cached locally; "
                "run /lcm embed warmup"
            ) from exc

    def _ensure_local(self) -> Any:
        if self._model is None:
            self._model = self._construct(allow_download=False)
        return self._model

    def _embed(
        self,
        texts: Sequence[str],
        *,
        query: bool = False,
        before_dispatch: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        deadline = time.monotonic() + max(0.0, self.timeout)

        def encode() -> list[list[float]]:
            return _run_blocking_with_deadline(
                lambda: self._encode_local(
                    texts,
                    query=query,
                    before_dispatch=before_dispatch,
                    deadline=deadline,
                ),
                timeout=deadline - time.monotonic(),
                provider="FastEmbed",
            )

        vectors = self._guarded(encode)
        if len(vectors) != len(texts):
            raise EmbeddingProviderError(
                "FastEmbed returned a different number of embeddings than requested"
            )
        for vector in vectors:
            if not vector:
                raise EmbeddingProviderError("FastEmbed returned an empty embedding")
            if self._dim and len(vector) != self._dim:
                raise EmbeddingProviderError("FastEmbed embedding dimension changed within the process")
            self._dim = len(vector)
        return vectors

    def _encode_local(
        self,
        texts: Sequence[str],
        *,
        query: bool,
        before_dispatch: Callable[[], None] | None = None,
        deadline: float | None = None,
    ) -> list[list[float]]:
        model = self._ensure_local()
        if deadline is not None and time.monotonic() >= deadline:
            raise ProviderPreDispatchError(
                "FastEmbed operation deadline exceeded before dispatch"
            )
        if before_dispatch is not None:
            before_dispatch()
        if deadline is not None and time.monotonic() >= deadline:
            raise ProviderPreDispatchError(
                "FastEmbed operation deadline exceeded before encode"
            )
        # FastEmbed models are asymmetric: queries and passages get different
        # instruction prefixes. Use the query-specific API for queries.
        encode = model.query_embed if query else model.embed
        return _coerce_vectors(list(encode(list(texts))), provider="FastEmbed")

    def warmup(self) -> list[float]:
        """The only path allowed to download a missing FastEmbed model."""
        if self._model is None:
            self._model = self._construct(allow_download=True)
        return self.embed_query("warmup")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(tuple(str(text) for text in texts), query=False)

    def embed_document_batches(
        self,
        texts: Sequence[str],
        *,
        before_dispatch: BeforeDispatch | None = None,
    ) -> Iterator[EmbeddedDocumentBatch]:
        normalized = tuple(str(text) for text in texts)
        if not normalized:
            return
        indexes = tuple(range(len(normalized)))
        vectors = self._embed(
            normalized,
            query=False,
            before_dispatch=(
                (lambda: before_dispatch(indexes))
                if before_dispatch is not None
                else None
            ),
        )
        yield EmbeddedDocumentBatch(
            indexes,
            tuple(tuple(vector) for vector in vectors),
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embed((str(text),), query=True)[0]


def resolve_provider(
    config: LCMConfig, *, for_backfill: bool = False
) -> EmbeddingProvider | None:
    """Resolve inert embedding config without making provider calls.

    ``for_backfill`` selects the bulk-operation contract. It bypasses the
    interactive per-minute spend guard and uses the separate backfill timeout:
    bulk ``lcm embed backfill --apply`` embeds thousands of documents (e.g.
    ~1920 docs at batch 32 → ~60 provider calls), and neither its call volume
    nor a normal document batch/local model load should be governed by the
    latency-sensitive query policy. The backfill worker retains its own
    operation budget and renewable lease.

    The query path (``for_backfill=False``) gets an explicit, configurable
    sliding-window guard from ``embedding_query_spend_{max_calls,window_seconds,
    backoff_seconds}`` (env ``LCM_EMBEDDING_QUERY_SPEND_*``), defaulting to a
    generous ceiling. The historical implicit default here was the strict
    ``EmbeddingSpendGuard()`` (60/60s/60s) -- a backfill-economy control that
    silently gutted retrieval when a tight query loop crossed 60 calls/window
    and every further call was rejected pre-network with ``ProviderRateLimited``
    (451 query embeds ≈ 9k tokens ≈ negligible spend). ``max_calls=0`` disables
    the guard entirely; the circuit breaker still trips on real failures.
    """
    provider = str(getattr(config, "embedding_provider", "") or "").strip().lower()
    model = str(getattr(config, "embedding_model", "") or "").strip()
    if not provider and not model:
        return None
    if not provider or not model:
        raise ProviderUnavailable(
            "LCM_EMBEDDING_PROVIDER and LCM_EMBEDDING_MODEL must both be set"
        )
    # max_calls=0 disables the sliding-window guard (allows() always True,
    # record_call() a no-op); the circuit breaker still trips on failures.
    if for_backfill:
        spend_guard = EmbeddingSpendGuard(max_calls=0)
    else:
        spend_guard = EmbeddingSpendGuard(
            max_calls=int(getattr(config, "embedding_query_spend_max_calls", 600)),
            window_seconds=float(
                getattr(config, "embedding_query_spend_window_seconds", 60.0)
            ),
            backoff_seconds=float(
                getattr(config, "embedding_query_spend_backoff_seconds", 60.0)
            ),
        )
    timeout_field = (
        "embedding_backfill_timeout_s"
        if for_backfill
        else "embedding_query_timeout_s"
    )
    timeout_default = 120.0 if for_backfill else 3.0
    timeout = float(getattr(config, timeout_field, timeout_default))
    if provider in {"voyage", "voyageai"}:
        return VoyageProvider(
            model,
            timeout=timeout,
            spend_guard=spend_guard,
            max_batch_items=int(
                getattr(config, "embedding_max_batch_items", _VOYAGE_MAX_BATCH_ITEMS)
            ),
        )
    if provider == "ollama":
        return OllamaProvider(
            model,
            base_url=getattr(config, "ollama_base_url", "http://localhost:11434"),
            timeout=timeout,
            spend_guard=spend_guard,
        )
    if provider in {"fastembed", "fast-embed"}:
        return FastembedProvider(
            model, timeout=timeout, spend_guard=spend_guard
        )
    if provider in {"openai-compatible", "openai", "siliconflow"}:
        return OpenAICompatibleProvider(
            model,
            base_url=getattr(config, "embedding_base_url", ""),
            api_key_env=getattr(config, "embedding_api_key_env", "LCM_EMBEDDING_API_KEY"),
            timeout=timeout,
            spend_guard=spend_guard,
        )
    raise ProviderUnavailable(
        f"Unsupported embedding provider {provider!r}; use voyage, ollama, "
        "fastembed, or openai-compatible (SiliconFlow)"
    )


def fastembed_download_size_note(model_id: str) -> str:
    """Return a deliberately approximate, operator-facing download size note."""
    known = {
        "BAAI/bge-small-en-v1.5": "about 130 MB",
        "sentence-transformers/all-MiniLM-L6-v2": "about 90 MB",
    }
    return known.get(model_id, "model-dependent; verify available disk space")
