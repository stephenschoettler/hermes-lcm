# Agent configuration profiles

Copy-paste starting points for common agent shapes. Every profile below is
**additive to a stock install** — with none of these variables set, LCM runs
exactly as before. Mix and match: the feature families are independent.

For the full variable table see
[Operator guide → Configuration](operator-guide.md#configuration); for what
each feature does and why, see [Feature overview](features-overview.md).

## Where configuration lives

1. **Environment variables (`LCM_*`) are the primary surface** and always win.
   Set them in the environment that launches Hermes.
2. **`~/.hermes/config.yaml`** participates in three narrow, deliberate ways:
   - `plugins.enabled: [hermes-lcm]` + `context.engine: lcm` activate the
     plugin (see [Operator guide → Activate](operator-guide.md#activate));
   - `lcm.context_threshold` controls the legacy LCM threshold, and
     `lcm.trigger_mode: hermes_config` opts into the plugin-side adapter that
     reads Hermes' existing `compression.threshold`, `compression.threshold_tokens`,
     and `compression.model_thresholds` (other keys under `lcm:` are ignored
     and reported by `/lcm doctor`);
   - when neither is set, LCM inherits the Hermes global
     `compression.threshold`.
3. **Summarization inherits Hermes auxiliary routing.** Rollup builds and
   compaction summaries go through the auxiliary model unless you override
   `LCM_SUMMARY_MODEL` — so a fully-local Hermes (local auxiliary model) makes
   every LCM feature below local too, including temporal rollups.
4. Secrets stay in the environment: the only one LCM ever reads is
   `VOYAGE_API_KEY`, and only when `LCM_EMBEDDING_PROVIDER=voyage`.

Check what actually took effect at runtime with `/lcm status` (reports config
sources) and `/lcm doctor` (flags ignored YAML keys and misconfiguration).

## Profile: default chat assistant

Nothing to configure. Enable the plugin and stop:

```yaml
# ~/.hermes/config.yaml
plugins:
  enabled:
    - hermes-lcm
context:
  engine: lcm
```

You get bounded active context, the summary DAG, lossless recovery, and the
full `lcm_*` tool set at their tested defaults.

## Profile: heavy tool-use coding agent

For agents that run builds, tests, linters, and searches all day. The goal is
to stop giant tool outputs from monopolizing the prompt while keeping every
byte recoverable.

```bash
# Externalize oversized payloads out of lcm.db into recoverable files
export LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED=true

# Replace token-heavy tool results in the provider-visible prompt with refs
export LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED=true
# Default stub threshold is 25000 tokens; lower it for chattier tools
# export LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS=25000

# Cap the protected fresh tail by tokens (0 = off). Prevents one giant recent
# tool result from pinning the whole budget; newest message and complete
# assistant/tool groups are always retained.
export LCM_FRESH_TAIL_MAX_TOKENS=24000

# Optional: at threshold, drain the whole raw backlog in one bounded sweep
# (fewer, larger compactions — useful after long unattended runs)
export LCM_THRESHOLD_FULL_SWEEP_ENABLED=true
```

Recovery stays first-class: `lcm_describe`/`lcm_expand` read externalized
refs, and `lcm_grep(content_scope='both')` searches externalized payloads when
the agent needs "which run printed that error?".

## Profile: long-horizon personal / companion agent

For agents that live for weeks and get asked "what did we do last Tuesday?"
and "have we talked about this before?".

```bash
# Time-indexed memory: day/week/month rollups + the lcm_recent tool
export LCM_TEMPORAL_ROLLUPS_ENABLED=true

# Meaning-based recall: semantic + hybrid lcm_grep over summaries
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=voyage          # or: fastembed / ollama (below)
export LCM_EMBEDDING_MODEL=voyage-4-lite
export VOYAGE_API_KEY=...                     # free key from dash.voyageai.com

# The /lcm operator commands used below are an opt-in surface
export LCM_ENABLE_SLASH_COMMAND=1
```

Then, once:

```
/lcm embed warmup            # resolve + dimension-lock the profile
/lcm embed backfill          # dry-run: shows cost/coverage estimate
/lcm embed backfill --apply  # embed existing summaries
```

`lcm_recent` works immediately even before rollups are built (transparent
leaf-summary fallback), and `lcm_grep` keeps `mode='full_text'` as the
byte-compatible default — semantic is per-call opt-in.

## Profile: fully local / air-gapped

No bytes leave the machine. Pair with a local Hermes auxiliary model so
summarization is local too.

```bash
export LCM_TEMPORAL_ROLLUPS_ENABLED=true

export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=fastembed       # in-process ONNX, CPU-friendly
# Default model downloads ~90–130 MB once at warmup (bge-small / MiniLM class)

# Or, if you already run Ollama:
# export LCM_EMBEDDING_PROVIDER=ollama
# export LCM_EMBEDDING_MODEL=nomic-embed-text
# export LCM_OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Optional: installing `numpy` accelerates KNN on large vector sets; without it
LCM uses a dependency-free bounded scan that stays correct, just smaller-scale
(see [Embeddings setup](embeddings-setup.md)).

## Profile: cost-guarded cloud embeddings

Voyage's free tier (200M tokens on the voyage-4 family) is far more than this
workload typically needs — LCM embeds bounded summaries, not raw transcripts.
To keep hard boundaries anyway:

```bash
export LCM_EMBEDDINGS_ENABLED=true
export LCM_EMBEDDING_PROVIDER=voyage
export LCM_EMBEDDING_MODEL=voyage-4-lite      # cheapest tier, $0.02/M after free
export VOYAGE_API_KEY=...

# Interactive queries: hard per-call wall-clock budget (default 3s)
export LCM_EMBEDDING_QUERY_TIMEOUT_S=3

# Bulk backfill: separate per-request deadline (default 120s) — bulk work
# never inherits the interactive query budget
export LCM_EMBEDDING_BACKFILL_TIMEOUT_S=120
```

Always dry-run `/lcm embed backfill` first: it reports document counts and
token estimates with the same eligibility rules apply-mode uses, so the
estimate is the bill.

## Verifying a profile

```bash
hermes plugins        # plugin + engine loaded
```

The `/lcm` operator commands below require the opt-in command surface:

```bash
export LCM_ENABLE_SLASH_COMMAND=1
```

(Without it, the `lcm_status` / `lcm_doctor` *agent tools* still cover the
read-only checks — ask the agent to call them.) Then in a session:

```
/lcm status           # effective config + sources, context pressure
/lcm doctor           # DB health, ignored YAML keys, misconfig warnings
/lcm rollups          # temporal rollup readiness (when enabled)
/lcm embed backfill   # dry-run: embedding coverage + cost estimate (when enabled)
```
