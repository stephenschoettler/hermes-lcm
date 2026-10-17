# Spec: Reranking via OpenAI-Compatible Provider (self-hosted rerankers)

Status: draft for build (Codex) · Target repo: `hermes-lcm` (branch on top of PR #519's `OpenAICompatibleProvider`) · Author: Mandy · Date: 2026-08-28

## 1. Goal

Enable `lcm_recall`'s cross-encoder rerank stage against **any OpenAI-compatible endpoint that also serves a Cohere-format `/rerank` route** (e.g. [michaelfeil/infinity](https://github.com/michaelfeil/infinity), which implements `/rerank` aligned with the Cohere API). Today the rerank stage is hard-locked to the Voyage cloud provider; this spec unlocks local/self-hosted rerankers with zero external API dependency.

Non-goals:
- No changes to embedding behavior, identity hashing, or backfill logic.
- No changes to `VoyageProvider` — its `rerank()` stays as-is.
- No new HTTP transport; reuse the existing `_default_http_transport` / injectable-transport pattern.

## 2. Current state (verified against main @ 10cbb78, v0.21.0-rc2)

| Fact | Location |
|---|---|
| Rerank gate requires `provider_id == "voyage"` | `tools.py:4539–4544` (`_lcm_recall_rerank`) |
| Rerank call: `provider.rerank(query, documents, top_k=..., timeout=...)` | `tools.py:4550–4557` |
| Voyage rerank returns `list[tuple[int, float]]` sorted by `(-score, index)` | `embedding_provider.py:1258–1319` |
| Rerank toggle: `rerank_enabled` / env `LCM_RERANK_ENABLED`, default `False` | `config.py:373, 611` |
| Local-provider set (skips `--confirm-raw-text`): `{"fastembed", "ollama"}` | `command.py:4498` |
| PR #519 adds `OpenAICompatibleProvider` (embeddings only, no `rerank` method) | PR #519 diff |

**Build sequencing (answers Codex's context question):** the PR #519 `OpenAICompatibleProvider` is applied to the working branch BEFORE this spec is implemented (local patch until upstream merges). The implementer builds `rerank()` against the real class in-tree — constructor shape, URL construction, api-key chain, and deadline helpers come from that code, not from this spec.

## 3. Contracts

### 3.1 `OpenAICompatibleProvider.rerank`

Signature and semantics are contract-identical to `VoyageProvider.rerank`:

```python
def rerank(
    self,
    query: str,
    documents: Sequence[str],
    *,
    top_k: int | None = None,
    timeout: float,
    model: str = "",
) -> list[tuple[int, float]]:
```

- Returns `(original_index, relevance_score)` pairs **ordered by descending relevance** (tie-break by ascending original index). Callers treat any raised error as "skip rerank" and keep prior order.
- Empty `documents` → return `[]` **without any transport call** (mirror Voyage's short-circuit; no wasted request).
- `model`: `""` (default) means "use `_OPENAI_COMPATIBLE_RERANK_MODEL`"; any non-empty string overrides. This keeps the setting alive (Codex finding: otherwise `config.rerank_model` would be dead config, because `_lcm_recall_rerank` never passed `model=`).
- `top_k` validation: `top_k <= 0` → return `[]` without a transport call; a non-integer `top_k` raises `ValueError` (programmer error → fail loud). Positive values map to `top_n` per §3.2.
- `top_k` is LCM-side API surface. On the wire it maps to Infinity/Cohere's `top_n` (see §3.2).

### 3.2 Wire format (Cohere shape, as implemented by Infinity `0.0.77`)

Request — `POST {base_url}/rerank`:

```json
{
  "model": "BAAI/bge-reranker-v2-m3",
  "query": "Where is Munich?",
  "documents": ["Munich is in Germany.", "The sky is blue."],
  "top_n": 2
}
```

- `model`: the rerank model id. Default constant `_OPENAI_COMPATIBLE_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"`, overridable per call via `model=` and globally via new config field `rerank_model` (env `LCM_RERANK_MODEL`, default `""` meaning "use the constant"). NOTE: Cohere/Infinity use **`top_n`**, not `top_k` — do not copy Voyage's payload key.
- `top_n`: `len(documents)` when `top_k` is `None`, else `min(top_k, len(documents))`. Rationale: LCM's caller always wants the full window scored back; Infinity defaults to returning all results anyway, but explicit is safer across Cohere-compatible servers.

Response — parse `results`:

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.97, "document": null}
  ]
}
```

- Map to `[(index, relevance_score), ...]`. Drop rows whose `index` is out of `[0, len(documents))` (mirror Voyage's `test_voyage_rerank_drops_out_of_range_index`).
- Sort by `(-score, index)`.
- Malformed-row policy (Codex finding): non-dict rows → skipped; missing, `null`, or non-numeric `relevance_score` → row skipped (a null score is a server bug — never silently score it 0.0); missing or non-coercible `index` → row skipped. Duplicate indexes are preserved at the provider level (mirror Voyage's dumb parser); the recall reorder already dedupes (tools.py `seen` set) — document this split in the PR.

### 3.3 Auth header

Read the API key via the same env chain as embeddings (`LCM_EMBEDDING_API_KEY`, falling back to `SILICONFLOW_API_KEY`), **but**: if no key is set, send the request **without** an `Authorization` header instead of raising. Rationale: self-hosted endpoints (infinity on LAN) need no key; raising would force a dummy-key convention. This is deliberately more lenient than #519's embedding path and should be called out in the PR description (embeddings keep #519's stricter contract; rerank is local-first).

### 3.4 Config additions (`config.py`)

```python
_EnvFieldSpec("rerank_model", "LCM_RERANK_MODEL", str),
# class field:
rerank_model: str = ""
```

`resolve_provider` wiring is unchanged (rerank rides on the existing `OpenAICompatibleProvider` instance; no new provider id).

Call-site rule (closes the §3.2/§4 gap Codex flagged): `_lcm_recall_rerank` forwards `model=config.rerank_model` **only when non-empty** — otherwise each provider keeps its own default (voyage → `rerank-2.5-lite`, openai-compatible → `BAAI/bge-reranker-v2-m3`). `LCM_RERANK_MODEL` therefore overrides whichever provider serves rerank; unset keeps both defaults and never forwards an empty string into a provider payload.

## 4. Gate change (`tools.py` `_lcm_recall_rerank`)

Replace the voyage-only gate:

```python
if (
    provider is None
    or getattr(provider, "provider_id", "") != "voyage"
    or not hasattr(provider, "rerank")
):
    return ordered, "skipped: rerank requires the voyage provider"
```

with capability-based gating:

```python
if provider is None or not hasattr(provider, "rerank"):
    return ordered, "skipped: provider does not support rerank"
```

Everything else (window slicing, `_run_within_deadline`, permutation-only reorder, `skipped: <reason>` / `applied` / `disabled` status strings, RERANK-1 score-scale rule) stays byte-identical. The ONE deliberate change beyond the gate: the `provider.rerank(...)` call gains `model=config.rerank_model` **only when that config value is non-empty** (see §3.4 — empty default preserves Voyage's constant).

## 5. Error & edge semantics

1. Any exception inside `rerank()` (network, non-2xx, parse, deadline) propagates to the existing `except` in `tools.py` → `skipped: {exc}`, order unchanged. Do not swallow errors inside the provider.
2. Non-2xx → raise `EmbeddingProviderError(f"OpenAI-compatible rerank request failed ({status})")`.
3. Transport `OSError/TimeoutError/URLError` → wrap as `EmbeddingProviderError("OpenAI-compatible rerank network error: ...")`, **no automatic identical resend** (same rationale as #519's embedding path).
4. `results` missing or not a list → raise `EmbeddingProviderError("OpenAI-compatible rerank response did not contain result data")`.
5. Deadline exceeded at any stage → raise (same deadline discipline as `VoyageProvider.rerank`).
6. `rerank_enabled=False` (default) → status `disabled`, no call. Unchanged.
7. **Empty-snippet policy (review finding):** `_lcm_recall_rerank` builds documents from `entry["hit"].get("snippet")`, which can be empty (e.g. under chunk hits). Do NOT send empty strings to the cross-encoder. Implementation is an **eligible-index map**, not a plain filter (Codex finding: returned rerank indexes refer to the eligible subset, not to `head`): build `eligible = [(head_pos, doc) for head_pos, entry in enumerate(head) if str(entry["hit"].get("snippet") or "").strip()]`, send only the docs, map returned document-positions back through `eligible` to head positions before the reorder. Slot-preserving merge — worked example: head `[A, empty, B, C]`, eligible rerank order `[C, B, A]` → result `[C, empty, B, A]` (the empty row keeps its slot). Make this exact example the executable assertion in §6 test 10. Status notes partial coverage: `applied (n=42/50)` when any row was skipped.
8. **All-empty case (Codex finding):** if every row in the window has an empty snippet, make NO provider call and return `skipped: no non-empty snippets to rerank` — never `applied (n=0/N)`.

## 6. Tests (mirror the Voyage rerank suite, `tests/test_embedding_provider.py`)

Use the existing `FakeTransport` pattern; all tests inject transport, no real network.

1. `test_openai_compatible_rerank_request_shape_and_success` — asserts URL `{base}/rerank`, payload `{"model", "query", "documents", "top_n"}` with `top_n == len(documents)`, parses `results` into descending `[(index, score)]`.
2. `test_openai_compatible_rerank_top_k_maps_to_top_n` — `top_k=2` over 5 docs → wire `top_n=2`; response may return fewer rows than documents. The fake response is deliberately UNsorted and the assertion checks the provider output is sorted by `(-score, index)` (Codex finding: "reflect wire order" contradicted the §3.2 sort contract — the test must prove the sort happens, not inherit wire order).
3. `test_openai_compatible_rerank_drops_out_of_range_index` — index 5 over 2 docs is dropped, not crashed.
4. `test_openai_compatible_rerank_non_2xx_raises` — 500 → `EmbeddingProviderError`.
5. `test_openai_compatible_rerank_empty_documents_short_circuits` — `[]` with zero transport calls.
6. `test_openai_compatible_rerank_no_api_key_sends_no_auth_header` — key envs unset → no `Authorization` header, request still succeeds (fake 200).
7. `test_openai_compatible_rerank_network_error_not_retried` — `OSError` → single call, raises.
8. `test_rerank_applies_with_openai_compatible_provider` (in `tests/test_lcm_recall.py`) — engine with `rerank_enabled=True` and an openai-compatible provider exposing `rerank` → provenance `applied`, order permuted, score stays RRF-scale (mirror `test_rerank_does_not_splice_voyage_score_onto_rrf_scale`).
9. `test_rerank_gate_is_capability_based` — a non-voyage provider **without** `rerank` → `skipped: provider does not support rerank`; update the existing voyage-gate test accordingly.
10. `test_rerank_passes_through_empty_snippets` (in `tests/test_lcm_recall.py`) — window `[A, empty, B, C]` with eligible rerank order `[C, B, A]` → result `[C, empty, B, A]` (the §5.7 worked example, verbatim); empty row is NOT in the rerank documents; status reads `applied (n=3/4)`.
11. `test_rerank_all_empty_snippets_skips_call` — every snippet empty → zero transport calls, status `skipped: no non-empty snippets to rerank`.
12. `test_openai_compatible_rerank_skips_malformed_rows` — response containing a non-dict row, a `null` score, and a missing index → those rows skipped, valid rows parsed; duplicate indexes preserved.

## 7. Docs

- `docs/embeddings-setup.md`, PR #519's "Option 4" section: add subsection "Reranking via the same endpoint" — show infinity serving `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3`, env `LCM_RERANK_ENABLED=true`, optional `LCM_RERANK_MODEL`, note `rerank` requires no API key for local endpoints.
- `config.py` comment block at `rerank_enabled` (line ~608): reword "voyage rerank-2.5-lite" → "provider-capable rerank (voyage rerank-2.5-lite, or a Cohere-compatible local endpoint via the openai-compatible provider)".
- `config.py` comment at `proactive_recall_min_score` (Codex finding): currently claims rerank scores dominate the score scale — factually wrong (rerank only permutes; scores stay RRF-scale, RERANK-1). Correct that comment in the same PR; it contradicts this spec's score semantics.

## 8. Exit criteria

- [ ] All new tests pass; existing suite green (`uv run --extra test pytest tests/ -q`).
- [ ] `grep -n "voyage provider" tools.py` returns nothing (gate message gone).
- [ ] Voyage rerank tests untouched and passing (no behavioral change to voyage path).
- [ ] `provenance.rerank` semantics unchanged: `disabled` / `skipped: ...` / `applied`.
- [ ] Docs updated.

## 9. Deliberate non-changes (do not touch)

- `_LOCAL_EMBEDDING_PROVIDERS` — `openai-compatible` must NOT be added: the provider string cannot distinguish LAN infinity from cloud SiliconFlow, so the raw-text confirmation gate stays conservative.
- Voyage constants, URL, parser.
- `benchmarking/longmemeval.py` — its separate rerank path ( Voyage-locked `RERANK_MODE_VOYAGE`) stays untouched; this spec covers ONLY the runtime recall path (`tools.py`).
- Identity hashing, backfill, chunker.