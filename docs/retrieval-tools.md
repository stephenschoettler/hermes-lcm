# Retrieval tools reference

Use this page when you need the exact LCM tool contract or archive-migration notes. For install, activation, and runtime configuration, start with [Operator guide](operator-guide.md).

## Agent Tools

Use these tools for current-session recall after compaction. Use `session_search`
for earlier separate sessions or broad cross-session history.

| Tool | Use |
|------|-----|
| `lcm_grep` | Search current-session raw messages and summaries. Opt into `session_scope='all'` or `session_scope='session'` (with `session_id`) for bounded archive recovery over rows already present in `lcm.db`, including externally backfilled rows that may carry source strings such as `openclaw-lcm:*`; broader scopes return raw-message hits only. Raw-message filters `role`, `time_from`, and `time_to` are pushed into the search query; when any of them is supplied, summary hits are omitted so the filter contract stays exact. Use `session_search` for earlier separate sessions or broad cross-session recall. |
| `lcm_load_session` | Load one ordered raw-message transcript page for an explicit `session_id`. This is not search: it returns raw rows in `store_id` order, bounded by `limit`, with per-message content bounded by `max_content_chars`, and continues with `after_store_id` from `next_cursor`. |
| `lcm_describe` | Inspect the current-session DAG or preview an `externalized_ref` without loading full content. |
| `lcm_expand` | Recover source messages, child summaries, or externalized payloads with pagination. Use `store_id` to fetch a single raw message regardless of session, suitable for drilling into a cross-session `lcm_grep` result. |
| `lcm_expand_query` | Answer a question using expanded current-session LCM context while returning a bounded answer. |
| `lcm_status` | Show runtime health, context pressure, config, source lineage, and lifecycle stats. |
| `lcm_doctor` | Run database, FTS, lifecycle, config, and context-pressure diagnostics. |

### Retrieval contract

LCM retrieval tools default to current-session scope. `lcm_grep` accepts
`session_scope='all'` or `session_scope='session'` as an explicit opt-in for
bounded archive search over rows already present in `lcm.db` (raw-message hits
only). Once a session id is known, `lcm_load_session` can enumerate that session's
raw transcript in chronological `store_id` pages without a search query. Use
Hermes `session_search` for broad cross-session history outside the LCM database.

Within the current session, `source` filters raw rows directly and filters
summary nodes by descendant raw-message source lineage. `unknown` is a real
source value, not a wildcard. Legacy blank-source rows are treated as `unknown`.
`role`, `time_from`, and `time_to` are raw-message filters and are applied in the
message search query before result limiting. `time_from` and `time_to` accept Unix
seconds or timezone-aware ISO 8601 strings; naive ISO strings are rejected so the
same query means the same thing across machines. When a raw-message filter is
active, `lcm_grep` returns raw rows only and reports `summary_results_omitted`.

Carried-over summary nodes can become current-session content after `/new`, but
their source eligibility still comes from the descendant raw messages. Expanding
a carried-over current-session node recovers those original raw message sources
even when the sources still belong to the previous session.

### Lossless raw recovery contract

Tool responses are bounded so one retrieval call cannot flood the main context.
Lossless recovery means raw content is stored with stable source lineage and can
be recovered in deterministic pages.

- `lcm_expand(node_id=...)` pages immediate sources with `source_offset` and `source_limit`
- `lcm_load_session(session_id=...)` pages ordered raw session rows with `after_store_id` and `next_cursor`; each row includes bounded content plus truncation metadata, and large individual rows can be recovered with `lcm_expand(store_id=...)` using `content_offset`
- oversized raw messages continue with `content_offset`
- `lcm_expand(externalized_ref=...)` pages payload content with `content_offset`
- `lcm_expand_query` uses `context_max_tokens` for auxiliary context and reports truncation/pagination hints when needed

### lossless-claw/OpenClaw import utility

`hermes-lcm` includes an opt-in operator script for backfilling raw message rows from a lossless-claw/OpenClaw LCM SQLite database into the local hermes-lcm SQLite store:

```bash
python scripts/import_lossless_claw.py \
  --source-db ~/.openclaw/path/to/lcm.db \
  --target-db ~/.hermes/lcm.db \
  --agent sammy
```

The script is intentionally conservative:

- dry-run is the default; pass `--apply` to write
- run it against an explicit target DB path, preferably while Hermes is stopped for that profile
- writes create a timestamped target DB backup first when the target already exists
- only raw messages are imported; summary DAG import is out of scope
- imported rows keep explicit provenance in `session_id` and `source`, for example `openclaw-lcm:agent:sammy:<source-session>`
- the default provenance identity is the concrete source `conversations.session_id`, preserving source session boundaries even when many conversations share one `session_key`
- pass `--session-identity session_key` only when you intentionally want conversations with the same source session key grouped into one imported LCM session
- reruns are idempotent for the same `--import-id`; the default `import_id` is path-derived, so pass a stable `--import-id` if you may import the same copied DB from different paths
- changing `--agent`, `--namespace`, or `--session-identity` under the same `--import-id` is treated as the same import and will skip already-tracked source messages; use a new `--import-id` for a different mapping
- no OpenClaw config or separate secret tables are imported, but raw transcripts and tool payloads are imported and may contain sensitive user data

This is a local archive migration path. It does not make LCM a general memory provider, and it does not change the current-session retrieval contract for agent tools.

## Related references

- [Operator guide](operator-guide.md)
- [Architecture notes](architecture.md)
- [Benchmarking and stress checks](../benchmarks/README.md)
