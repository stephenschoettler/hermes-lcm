# Embeddings setup — free and local options

Semantic and hybrid retrieval are **opt-in** and need an embedding provider. You have three good
options, two of which cost nothing. If you configure nothing, nothing changes — retrieval stays
FTS-only exactly as before.

## TL;DR

| Option | Cost | Signup | Install | Best for |
|---|---|---|---|---|
| Voyage AI | free tier (200M tokens for the voyage-4 group), then ~$0.02–0.12 per million tokens | yes (no credit card) | none | best quality, zero local footprint |
| fastembed | $0 | no | `pip install fastembed` (ONNX, no torch) | local default — no accounts, no daemon |
| Ollama | $0 | no | Ollama app/daemon | you already run Ollama |
| OpenAI-compatible | provider pricing | yes (per provider) | none | reuse an existing `/v1/embeddings` service (SiliconFlow, llama.cpp, vLLM) |

All providers feed the same store. Switching provider/model is one config change plus a backfill;
each full identity keeps its own vectors, so switching **back** to a
previously-registered provider reactivates its vectors with no re-backfill (see *Switching or
removing providers*).

## Option 1 — Voyage AI (free tier)

The Voyage-4 generation (`voyage-4`, `voyage-4-large`, `voyage-4-lite`) carries **200 million free
tokens for that model group**, and signup does not require a credit card. For a personal LCM corpus
(thousands of summaries), that free allotment covers initial backfill and years of queries; past it,
embedding costs $0.02/M (`voyage-4-lite`), $0.06/M (`voyage-4`), or $0.12/M (`voyage-4-large`). A
query embeds exactly one short vector, so a semantic or hybrid query costs a fraction of a cent.
Voyage documents its allotments and rate tiers on its
[pricing](https://docs.voyageai.com/docs/pricing) and
[rate-limits](https://docs.voyageai.com/docs/rate-limits) pages — treat those as the source of
truth; the numbers above were verified 2026-07.

```bash
export VOYAGE_API_KEY=...           # from dash.voyageai.com
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=voyage
export LCM_EMBEDDING_MODEL=voyage-4-lite   # or voyage-4 / voyage-4-large
/lcm embed warmup                   # probes the API, registers model + dimensions
/lcm embed backfill                 # dry run: shows counts + estimated tokens, writes nothing
/lcm embed backfill --apply         # embeds your history in bounded batches
```

Notes: requests are batched under Voyage's caps — both the token budget and the 1000-item
per-request cap; over-length documents are skipped and reported, never silently truncated;
rate-limit responses are honored with bounded waits under one absolute per-operation deadline.
That deadline starts before document conversion/token counting and covers every split, retry, and
backoff, plus response decoding and validation. Automatic HTTP resend is deliberately narrower:
Voyage `429` is an authoritative rejection and may be retried within the remaining deadline, but a
timeout/network failure or `5xx` after transport starts may follow remote acceptance and is never
automatically resent. A deadline that expires after durable dispatch marking but before transport
is a typed `not started` outcome, so backfill can safely clear those exact rows. Individual
token-count calls run in bounded workers; an overrun returns at the deadline without dispatching a
later request (the timed-out worker may finish in the background while holding one of the fixed
worker slots).

## Option 2 — fastembed (local, no signup, recommended local default)

[fastembed](https://github.com/qdrant/fastembed) runs ONNX models on CPU with no PyTorch and no
external service — just a pip package.

```bash
pip install fastembed
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=fastembed
export LCM_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5   # 384-dim, compact and quick on CPU
/lcm embed warmup     # downloads the model ONCE, explicitly (a few hundred MB incl. onnxruntime)
/lcm embed backfill --apply
```

The model download happens **only** during `warmup` — never lazily during a query or an agent turn.
If you skip warmup, semantic search simply stays off and the tools tell you why. Queries use the
model's query-specific encoding (distinct from document encoding) so query/passage asymmetry is
preserved. When `LCM_EMBEDDINGS_ENABLED=false`, `warmup` is inert: it does not resolve a provider,
download a model, create embedding tables, or create the configured database.

## Option 3 — Ollama (local daemon)

If you already run [Ollama](https://ollama.com), use its embeddings endpoint:

```bash
ollama pull nomic-embed-text        # 768-dim; mxbai-embed-large and bge-m3 also work
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=ollama
export LCM_EMBEDDING_MODEL=nomic-embed-text
# LCM_OLLAMA_BASE_URL defaults to http://localhost:11434
/lcm embed warmup && /lcm embed backfill --apply
```

Ollama requests set `truncate: false`, so an input that exceeds the model's context fails loudly
rather than being silently truncated to a misleading embedding. As with Voyage, an Ollama
timeout/network failure after transport starts is acceptance-ambiguous and is not automatically
resent.

Bulk document embedding uses `LCM_EMBEDDING_BACKFILL_TIMEOUT_S` as its
per-provider-operation deadline (120 seconds by default) for Voyage, Ollama,
and fastembed. This is intentionally separate from the latency-sensitive
`LCM_EMBEDDING_QUERY_TIMEOUT_S` (3 seconds by default), so a normal document
batch or local model load is not aborted by the interactive query policy. The
optional `LCM_EMBEDDING_BACKFILL_BUDGET_S` still caps the whole apply run
between batches (`0`, the default, means no whole-run cap); the lease and
post-call ownership CAS remain authoritative independently of both timeouts.

## Option 4 — OpenAI-compatible API (reuse an existing embeddings service)

If you already run any OpenAI-format `/v1/embeddings` endpoint — SiliconFlow,
llama.cpp server, vLLM, or a local embedding model on a workstation — point
LCM at it instead of running a second embedding stack:

```bash
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=openai-compatible
export LCM_EMBEDDING_MODEL=BAAI/bge-m3     # any model id your endpoint serves
export LCM_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
export LCM_EMBEDDING_API_KEY=sk-...        # falls back to SILICONFLOW_API_KEY
/lcm embed warmup && /lcm embed backfill --apply
```

`LCM_EMBEDDING_BASE_URL` is the API root (a trailing slash is stripped) and
`/embeddings` is appended. The request body is the standard OpenAI shape —
`{"model": ..., "input": [...]}` with a `Bearer` authorization header — so any
compatible server works, including local llama.cpp/vLLM instances that cost
nothing per call.

The same deadline, spend-guard, and identity rules apply as for the other
remote providers: `LCM_EMBEDDING_QUERY_TIMEOUT_S` bounds interactive queries,
`LCM_EMBEDDING_BACKFILL_TIMEOUT_S` bounds each backfill batch, and vectors are
stored under the full `(provider, model, ...)` identity so switching back to
this provider reactivates existing vectors without re-backfilling.

## What you get

`lcm_grep` gains two modes on top of the existing ones:

- `semantic` — paraphrase-tolerant vector search; local providers make this $0
- `hybrid` — keyword ∪ semantic, fused with reciprocal-rank fusion (RRF); the best
  "have we discussed X?" mode. (Fusion is RRF only — there is no external reranker.)

Semantic and hybrid requests use one absolute deadline beginning at `lcm_grep` entry. It includes
provider resolution, query embedding, optional NumPy import, bounded KNN, result hydration, any FTS
fallback, both hybrid arms, and fusion. A semantic failure can degrade to full-text with
`degraded_to_fts`; the fallback uses separate read-only SQLite connections with progress
interruption. If hybrid already computed FTS results before its semantic arm times out, it may
return that existing payload without starting new I/O. If no usable result exists when time runs
out, the request returns an explicit `timeout` error and starts no later fallback/arm. Provider
authentication errors also remain operator-visible instead of degrading. Explicit `full_text`
mode itself is unchanged and byte-for-byte identical to prior behavior.

Role, time, conversation, and broader-session filters degrade to raw FTS before provider work,
because summaries cannot prove those raw-message dimensions. Source is different: SQL first selects
a bounded candidate window, then verifies descendant source lineage inside that window. The lineage
walk is itself capped; a missing legacy `source` column or an over-budget lineage graph fails closed
with `unverifiable_provenance`, so it never becomes an allow-all. A source-filtered semantic result
therefore reports bounded coverage rather than claiming universal pre-bound source coverage.

## Performance & footprint

- Vector math is dependency-free by default; installing **numpy** (optional) accelerates large
  corpora — the top-k scan is milliseconds warm once numpy is loaded.
- Metadata/id resolution uses a temp-table join rather than a giant `IN (...)` list, so it scales
  past the SQLite host-parameter limit that previously failed near ~32k ids (validated to 40k).
- Without numpy, search scans the most recent `LCM_EMBEDDING_BOUNDED_SCAN_ROWS` vectors (default
  2,000) and reports `coverage: bounded`. The candidate enumeration is bounded at the SQL layer
  (`ORDER BY` recency `+ LIMIT`), so a large corpus never materializes every id in host memory.
- With numpy, the cache is still only for that bounded candidate set and is keyed by canonical
  identity, transactional `data_version`, and candidate ids; it is not a corpus-sized matrix.

## Switching or removing providers

Change provider/model → run `/lcm embed warmup` (registers the new profile as the current identity)
→ `/lcm embed backfill --apply` (embeds under the new identity; the previous model's vectors are
kept separate and never mixed). Every vector is published under the exact identity that produced it
— the identity is captured at provider-resolution time and carried through the write, so switching
the active provider A→B mid-backfill can never rebind an A-vector onto B. If A becomes inactive
after its request was accepted but before publication, the exact A request is atomically moved to
`uncertain`, the still-owned backfill lease is released, and the run stops before another dispatch.
Because each identity
`(provider, model, revision, dim, dtype, byteorder, task)` owns its own vectors, switching **back**
to a previously-registered provider reactivates it with its existing vectors — no re-backfill
needed. The stored representation is currently restricted to `float32` / `little` / `summary`;
unsupported identity variants are rejected rather than normalized onto another profile.

Backfill records each actual remote dispatch durably. Every accepted sub-batch is published
immediately under the captured identity and current lease CAS. If remote acceptance is ambiguous or
local publication fails after acceptance, those rows become `uncertain` and normal discovery will
not bill them again. Recovery is deliberately operator-authorized:

```bash
/lcm embed backfill --apply --retry-uncertain --limit 32
```

The authorization is bound to the exact oldest uncertain rows selected by that invocation, up to
`--limit`; the risky recovery run does not mix in ordinary pending rows. Their durable uncertainty
markers are not cleared before discovery or dispatch, and any row not successfully published
(including a skipped row, definitive rejection, budget stop, or lease loss) remains `uncertain` for
another explicit decision. The command reports the uncertain count and warning because retrying may
rebill. Disable everything with `LCM_EMBEDDINGS_ENABLED=false` — data stays, behavior reverts to
FTS-only instantly.

## The chunk corpus — raw verbatim text, and the consent gate

`/lcm embed backfill` has two corpora, selected with `--corpus`:

- `summary` (default) — embeds the generated **summaries** of your history.
- `chunks` — embeds **raw, verbatim message text**, chunked by `--policy`
  (`conversational` | `heads` | `full`), for verbatim/chunk-KNN recall.
- `both` — runs the summary backfill, then the chunk backfill, in one command.

The distinction matters for privacy. The summary corpus sends only model-generated summaries to the
embedding provider. **The chunk corpus sends the raw message bytes** — including tool-result output
and error/traceback content (the `heads`/`full` policies specifically target error signatures) —
which is exactly the content most likely to carry secrets. When the provider is a **cloud** provider
(e.g. Voyage), that raw text leaves this machine.

Because of this, `--corpus chunks --apply` and `--corpus both --apply` **refuse** on a cloud provider
unless you pass an explicit acknowledgment:

```bash
/lcm embed backfill --corpus chunks --apply --confirm-raw-text
```

Local providers (**fastembed**, **ollama**) never transmit text off the machine, so the gate is
waived for them. Dry runs (no `--apply`) never send anything and never require the flag.

> **Redaction caveat.** `LCM_SENSITIVE_PATTERNS_ENABLED` redaction runs at **ingest** time, so it
> only affects text stored *after* it was enabled. Turning it on does **not** retro-redact history
> already in the store — that older raw text is still what gets sent to the provider during a chunk
> backfill. Prefer a local provider for the chunk corpus if the history may contain secrets.
