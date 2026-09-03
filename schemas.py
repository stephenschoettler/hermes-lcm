"""Tool schemas for LCM — what the LLM sees."""

LCM_GREP = {
    "name": "lcm_grep",
    "description": (
        "Search the plugin-local LCM database for past conversation content using full-text, semantic, or hybrid retrieval. "
        "Default scope is the active session and returns both raw messages and summary nodes across all depths. "
        "Broader scopes ('all' or 'session') must be requested explicitly and exist for bounded archive recovery "
        "over rows already present in lcm.db, including externally backfilled rows that may carry source strings "
        "such as openclaw-lcm:* . In broader scopes only raw-message hits are returned; cross-session summary "
        "node expansion is intentionally deferred. Use lcm_expand(store_id=...) on a cross-session message hit "
        "to drill into its full content. Set content_scope='externalized' or 'both' to opt into bounded, active-session "
        "search over recoverable payload sidecars. For Hermes-tracked session history outside the LCM database, use session_search. "
        "For open-ended cross-conversation recall by meaning ('have we ever discussed…'), prefer lcm_recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["full_text", "semantic", "hybrid"],
                "description": (
                    "Retrieval mode. 'full_text' preserves the historical FTS behavior byte-for-byte. "
                    "'semantic' searches embedded summaries and degrades to full-text on provider timeout or transient unavailability. "
                    "'hybrid' fuses full-text and semantic ranks with reciprocal-rank fusion (RRF)."
                ),
                "default": "full_text",
            },
            "query": {
                "type": "string",
                "description": (
                    "Search query (FTS5 syntax: keywords, phrases, OR/NOT). "
                    "FTS5 defaults to AND matching, so prefer 1-3 distinctive terms or one quoted multi-word phrase. "
                    "Wrap exact phrases in quotes. Short CJK fragments and emoji-heavy queries may use substring fallback instead of plain FTS token matching."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Max results to return (default 10, hard upper bound 200). "
                    "Values above the cap are clamped and reported via limit_clamped_from in the response."
                ),
                "default": 10,
            },
            "sort": {
                "type": "string",
                "enum": ["recency", "relevance", "hybrid"],
                "description": (
                    "How to order matches. 'recency' favors newer hits, 'relevance' favors strongest FTS matches, "
                    "and 'hybrid' keeps strong older matches competitive while still boosting newer context."
                ),
                "default": "recency",
            },
            "content_scope": {
                "type": "string",
                "enum": ["history", "externalized", "both"],
                "description": (
                    "Content stores to search. 'history' (default) preserves current raw-message and summary behavior. "
                    "'externalized' searches only bounded externalized-payload prefixes owned by the active session. "
                    "'both' searches history and those payloads. Externalized search supports session_scope='current' only."
                ),
                "default": "history",
            },
            "externalized_refs": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 256,
                "description": (
                    "Optional externalized ref filenames to search. Valid only with content_scope='externalized' or 'both'. "
                    "Every ref must be a regular payload owned by the active session."
                ),
            },
            "session_scope": {
                "type": "string",
                "enum": ["current", "all", "session"],
                "description": (
                    "Scope of the search across the plugin-local LCM database. "
                    "'current' (default) restricts to the active session and preserves historical behavior. "
                    "'all' searches every session in the local LCM database. "
                    "'session' restricts to the session_id supplied via the session_id parameter. "
                    "Cross-session search returns snippets and message store_ids; cross-session summary node expansion is deferred. "
                    "For Hermes-tracked session history outside the LCM database, use session_search."
                ),
                "default": "current",
            },
            "session_id": {
                "type": "string",
                "description": (
                    "When session_scope='session', the explicit session id to restrict the search to. "
                    "Must not be supplied with session_scope='current' or session_scope='all'."
                ),
            },
            "source": {
                "type": "string",
                "description": (
                    "Optional source/platform filter (for example cli, discord, telegram). "
                    "Applies directly to raw messages and to summaries via descendant source lineage. "
                    "Use 'unknown' for explicit unknown-source content."
                ),
            },
            "conversation_id": {
                "type": "string",
                "description": (
                    "Optional gateway conversation/session key filter for lane-scoped retrieval. "
                    "Use this to restrict Discord searches to one channel/thread/forum topic lane when rows carry metadata."
                ),
            },
            "role": {
                "type": "string",
                "enum": ["system", "user", "assistant", "tool", "unknown"],
                "description": "Optional raw-message role filter. When supplied, lcm_grep returns raw message hits only.",
            },
            "time_from": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional inclusive minimum raw-message timestamp. Accepts Unix seconds or timezone-aware ISO 8601; "
                    "naive ISO timestamps are rejected. When supplied, lcm_grep returns raw message hits only."
                ),
            },
            "time_to": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional inclusive maximum raw-message timestamp. Accepts Unix seconds or timezone-aware ISO 8601; "
                    "naive ISO timestamps are rejected. When supplied, lcm_grep returns raw message hits only."
                ),
            },
        },
        "required": ["query"],
    },
}

LCM_RECALL = {
    "name": "lcm_recall",
    "description": (
        "Search the agent's entire memory across ALL conversations and all time by meaning. "
        "Returns the most relevant memories — summaries and verbatim excerpts — ranked by relevance, "
        "recency, and relatedness to the current conversation, each with an expand_hint handle to the "
        "original content: verbatim/current-session hits get lcm_expand(...), while cross-session summary "
        "hits get lcm_load_session(...) (lcm_expand's node_id mode is current-session only, so it cannot "
        "expand a cross-session summary). Not for retrieving exact/verbatim text within a known time range — "
        "use lcm_grep(mode='full_text') for that. Not for full transcripts — after locating the right "
        "conversation, use lcm_load_session(session_id). Recency and current-conversation preference are soft "
        "ranking boosts, not filters; for hard time bounds use lcm_grep time_from/time_to. "
        "Set detail='answer_ready' to apply bounded per-session diversity and hydrate exact "
        "refs into answer-ready evidence windows without running another search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language description of what to recall. Searched by meaning across every "
                    "conversation and keyword-matched over raw history; distinctive phrasing recalls best."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max memories to return (default 8, hard upper bound 25).",
                "default": 8,
            },
            "scope_bias": {
                "type": "number",
                "description": (
                    "Soft preference for the current conversation, 0..1 (default 0.5). 0 = global-neutral, "
                    "1 = strongly prefer memories from the current conversation. This is a ranking boost, "
                    "never a hard filter — cross-conversation memories always remain eligible."
                ),
                "default": 0.5,
            },
            "include": {
                "type": "string",
                "enum": ["all", "summaries", "verbatim"],
                "description": (
                    "Which memory kinds to return. 'all' (default) mixes summaries and verbatim excerpts, "
                    "'summaries' returns compacted summary memories only, 'verbatim' returns raw message "
                    "excerpts only."
                ),
                "default": "all",
            },
            "detail": {
                "type": "string",
                "enum": ["snippets", "answer_ready"],
                "description": (
                    "Response detail. 'snippets' (default) preserves the existing compact "
                    "hit response byte-for-byte. 'answer_ready' keeps at most five hits per "
                    "session and hydrates the first eight selected exact refs into bounded "
                    "windows of at most 2400 characters each; the complete response remains "
                    "capped at 64000 characters."
                ),
                "default": "snippets",
            },
            "seen_refs": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 100,
                "description": (
                    "Optional stateless answer-ready delta cursor. When supplied with "
                    "detail='answer_ready', only evidence whose exact ref is not in this "
                    "list is returned. Omitting it preserves the baseline response bytes."
                ),
            },
            "include_occurrence_time": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Opt in to source-backed occurrence-time fields on exact message "
                    "evidence. Unknown remains valid and observation time is never "
                    "silently used as event time."
                ),
            },
        },
        "required": ["query"],
    },
}

LCM_QUERY_STATE = {
    "name": "lcm_query_state",
    "description": (
        "Query the opt-in V4 assertion sidecar for bounded, typed, source-cited state. "
        "Use this when a question needs attributable current or historical facts, preferences, "
        "recommendations, commitments, actions, or status. Every returned assertion includes an "
        "exact message store_id, character span, and quote. Conflicting state is preserved rather "
        "than resolved by recency. Returns status=disabled when assertions are not enabled."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject_key": {
                "type": "string",
                "description": (
                    "Required canonical assertion subject, for example user:self, "
                    "assistant:self, person:alex, or project:apollo."
                ),
            },
            "predicate_key": {
                "type": "string",
                "description": "Optional canonical lowercase dotted predicate filter.",
            },
            "kinds": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "fact",
                        "event",
                        "preference",
                        "recommendation",
                        "commitment",
                        "action",
                        "status",
                        "quotation",
                    ],
                },
                "maxItems": 8,
                "description": "Optional assertion-kind filter.",
            },
            "scope_key": {
                "type": "string",
                "description": "Optional exact canonical scope filter; empty means unscoped.",
            },
            "speaker_role": {
                "type": "string",
                "enum": ["system", "user", "assistant", "tool", "unknown"],
                "description": "Optional exact source-message role filter.",
            },
            "as_of": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional historical knowledge/validity boundary as Unix seconds or a "
                    "timezone-aware ISO 8601 timestamp."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum assertion rows (default 25, hard upper bound 50).",
                "default": 25,
            },
        },
        "required": ["subject_key"],
    },
}

LCM_COMPUTE = {
    "name": "lcm_compute",
    "description": (
        "Execute a supported date, count, compatible-unit sum, directed/absolute "
        "difference, ordering, or latest-state operation over exact cited LCM evidence. "
        "The operation is inferred from the question; this tool never calls a model. "
        "Supply only exact message spans (or assertion IDs) whose values, units, labels, "
        "keys, and dates are explicit in the cited evidence. Unsupported, incomplete, "
        "mixed-unit, conflicting, or unverifiable inputs fail closed to the ordinary "
        "evidence-only answer path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The user's question, unchanged.",
            },
            "question_date": {
                "type": "string",
                "description": (
                    "Optional ISO date anchoring relative-time and historical questions."
                ),
            },
            "evidence_complete": {
                "type": "boolean",
                "description": (
                    "Set true only after the current retrieval turn has closed all "
                    "evidence slots for open-cardinality operations."
                ),
                "default": False,
            },
            "operands": {
                "type": "array",
                "minItems": 1,
                "maxItems": 50,
                "items": {
                    "type": "object",
                    "properties": {
                        "assertion_id": {
                            "type": "string",
                            "description": "Optional exact V4 assertion SHA-256 ID.",
                        },
                        "store_id": {"type": "integer"},
                        "span_start": {"type": "integer"},
                        "span_end": {"type": "integer"},
                        "quote": {"type": "string"},
                        "value": {
                            "anyOf": [
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "null"},
                            ],
                            "description": "Explicit numeric or string operand value.",
                        },
                        "unit": {"type": "string"},
                        "key": {
                            "type": "string",
                            "description": (
                                "Canonical distinct-event/entity key whose tokens occur "
                                "in the exact quote."
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": (
                                "Exact source-grounded label. Directed differences must "
                                "supply labels in question mention order."
                            ),
                        },
                        "date": {
                            "type": "string",
                            "description": (
                                "Optional evidence date supported by source metadata or "
                                "the exact quote."
                            ),
                        },
                        "occurrence_time": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "observed_at": {"type": "number"},
                                "stored_at": {"type": "number"},
                                "event_at": {
                                    "anyOf": [{"type": "number"}, {"type": "null"}]
                                },
                                "event_date": {"type": "string"},
                                "event_time_source": {
                                    "type": "string",
                                    "enum": [
                                        "explicit",
                                        "relative_to_session",
                                        "derived_assertion",
                                        "unknown",
                                    ],
                                },
                                "session_date": {"type": "string"},
                                "precision": {"type": "string"},
                                "policy_version": {"type": "string"},
                                "reason": {"type": "string"},
                                "support": {"type": "object"},
                            },
                            "required": ["event_time_source"],
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "candidate_answer": {
                "type": "string",
                "description": (
                    "Optional narrative answer to verify against the immutable trace. "
                    "A failed verification is discarded in favor of canonical output."
                ),
            },
        },
        "required": ["question", "operands"],
        "additionalProperties": False,
    },
}

LCM_EVIDENCE_PACK = {
    "name": "lcm_evidence_pack",
    "description": (
        "Build a bounded, same-database evidence packet from baseline exact LCM refs. "
        "It returns no prose answer: the tool normalizes the question-date anchor, "
        "hydrates exact source spans, validates proposed facets, keeps occurrence time "
        "distinct from observation time, deduplicates refs, and emits a canonical "
        "lcm_compute trace only when product-verified grounding and cardinality close. "
        "A quote may narrow a declared ref only when it occurs exactly once inside it; "
        "open cardinality never closes from a caller assertion alone."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question": {"type": "string"},
            "question_date": {
                "type": "string",
                "description": (
                    "Optional calendar-date anchor. A local ISO timestamp is reduced "
                    "to its validated date component."
                ),
            },
            "baseline_refs": {
                "type": "array",
                "minItems": 1,
                "maxItems": 50,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "exact_ref": {"type": "string"},
                        "store_id": {"type": "integer"},
                        "span_start": {"type": "integer"},
                        "span_end": {"type": "integer"},
                        "quote": {"type": "string"},
                        "value": {
                            "anyOf": [
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "null"},
                            ],
                        },
                        "unit": {"type": "string"},
                        "key": {"type": "string"},
                        "label": {"type": "string"},
                    },
                },
            },
            "budgets": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "max_refs": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 25,
                    },
                    "max_quote_chars": {
                        "type": "integer",
                        "minimum": 64,
                        "maximum": 2400,
                        "default": 2400,
                    },
                    "max_retrieval_calls": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0,
                        "description": (
                            "Optional single product recall probe for novel exact refs. "
                            "It never turns no-novel into open-world completeness."
                        ),
                    },
                    "max_novel_refs": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 4,
                    },
                    "max_per_session": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 5,
                    },
                    "max_per_date": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 5,
                    },
                },
            },
        },
        "required": ["question", "baseline_refs"],
    },
}

LCM_COMPILE_EVIDENCE = {
    "name": "lcm_compile_evidence",
    "description": (
        "Compile a bounded, source-grounded evidence brief from baseline exact LCM "
        "refs. Proposal mode validates one provider-neutral semantic proposal; "
        "auto mode deterministically compiles answer requirements and may retrieve "
        "only for named missing slots. Product code validates "
        "every quote, span, entity, date, value, unit, distinct key, role, and source "
        "against immutable stored evidence. The result distinguishes partial, "
        "answer-sufficient, finite-coverage, computation-sufficient, conflicted, and "
        "unknown states and never returns final prose. Finite coverage cannot be "
        "asserted by the proposal or inferred from linguistic singularity."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["proposal", "auto"],
                "default": "proposal",
                "description": (
                    "Default proposal mode preserves the legacy wire contract. "
                    "Auto mode uses the provider-free requirements compiler."
                ),
            },
            "question": {"type": "string", "minLength": 1, "maxLength": 4000},
            "question_date": {
                "type": "string",
                "description": "Optional explicit calendar-date/as-of anchor.",
            },
            "baseline_refs": {
                "type": "array",
                "minItems": 1,
                "maxItems": 50,
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "additionalProperties": True,
                            "required": ["exact_ref"],
                            "properties": {
                                "exact_ref": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                        },
                    ]
                },
            },
            "proposal": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "version",
                    "selections",
                    "missing_facets",
                ],
                "properties": {
                    "version": {"type": "string", "enum": ["evidence-selector-v1"]},
                    "requested_facets": {
                        "type": "array",
                        "maxItems": 12,
                        "description": (
                            "Optional generic named facets required by the question. "
                            "Product code keeps deterministic facets, replaces only the "
                            "catch-all answer facet, and requires exact evidence for each "
                            "facet before answer_sufficient."
                        ),
                        "items": {"type": "string"},
                    },
                    "selections": {
                        "type": "array",
                        "maxItems": 50,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "claim_id",
                                "facet",
                                "exact_ref",
                                "quote",
                            ],
                            "properties": {
                                "claim_id": {"type": "string"},
                                "facet": {"type": "string"},
                                "exact_ref": {"type": "string"},
                                "quote": {"type": "string"},
                                "entity": {"type": "string"},
                                "date": {"type": "string"},
                                "value": {
                                    "anyOf": [
                                        {"type": "number"},
                                        {"type": "string"},
                                        {"type": "null"},
                                    ]
                                },
                                "unit": {"type": "string"},
                                "distinct_key": {"type": "string"},
                                "label": {"type": "string"},
                                "role": {"type": "string"},
                                "source": {"type": "string"},
                            },
                        },
                    },
                    "missing_facets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "usage": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "provider": {"type": "string"},
                            "model": {"type": "string"},
                            "effort": {"type": "string"},
                            "calls": {"type": "integer", "minimum": 0},
                            "input_tokens": {"type": "integer", "minimum": 0},
                            "output_tokens": {"type": "integer", "minimum": 0},
                            "latency_ms": {"type": "number", "minimum": 0},
                            "cost_usd": {"type": "number", "minimum": 0},
                        },
                    },
                },
            },
            "persist_view": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Default-off. Persist only validated high-value operational evidence "
                    "as an exact-dependency query view in the same lcm.db."
                ),
            },
            "budgets": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "max_input_refs": {"type": "integer", "minimum": 1, "maximum": 50},
                    "max_selections": {"type": "integer", "minimum": 1, "maximum": 50},
                    "max_selector_chars": {
                        "type": "integer",
                        "minimum": 1024,
                        "maximum": 64000,
                    },
                    "max_quote_chars": {
                        "type": "integer",
                        "minimum": 64,
                        "maximum": 2400,
                    },
                    "max_retrieval_calls": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                    },
                    "max_novel_refs": {"type": "integer", "minimum": 1, "maximum": 8},
                    "max_candidates": {"type": "integer", "minimum": 1, "maximum": 48},
                    "max_hydrated_candidates": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 12,
                    },
                    "max_added_context_tokens": {
                        "type": "integer",
                        "minimum": 64,
                        "maximum": 850,
                    },
                    "max_scan_rows": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 4096,
                        "description": (
                            "Auto mode only. Whole-corpus finite-coverage scan bound; "
                            "truncation always falls back."
                        ),
                    },
                    "response_char_cap": {
                        "type": "integer",
                        "minimum": 4096,
                        "maximum": 64000,
                    },
                },
            },
        },
        "required": ["question", "baseline_refs"],
        "allOf": [
            {
                "if": {
                    "properties": {"mode": {"const": "proposal"}},
                },
                "then": {"required": ["proposal"]},
            }
        ],
    },
}

LCM_RETRIEVE = {
    "name": "lcm_retrieve",
    "description": (
        "Coordinate a bounded evidence-retrieval episode inside the current "
        "answerer's existing tool turn. Start with a typed intent and named "
        "evidence requirements, make at most three targeted calls to existing "
        "LCM retrieval tools, then finish with exact selected refs and an optional "
        "provider-neutral lcm_compute trace. The controller contains no model "
        "client; dispatched retrieval tools retain their own provider behavior "
        "and provenance. It never accepts benchmark metadata or persists final prose. It is "
        "available only when LCM_ADAPTIVE_RETRIEVAL_ENABLED is true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "search", "finish", "status", "abandon"],
            },
            "retrieval_id": {
                "type": "string",
                "description": "Controller ID returned by the start action.",
            },
            "question": {
                "type": "string",
                "description": "The user's question, unchanged.",
            },
            "question_date": {
                "type": "string",
                "description": "Optional ISO date anchoring relative-time intent.",
            },
            "identity": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "intent_type": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": [
                            "evidence_only",
                            "date_interval",
                            "date_filter",
                            "count_distinct",
                            "sum",
                            "difference",
                            "order",
                            "latest_fact",
                        ],
                    },
                    "subject_key": {"type": "string"},
                    "predicate_key": {"type": "string"},
                    "role_key": {"type": "string"},
                    "scope_key": {"type": "string"},
                    "conversation_id": {"type": "string"},
                    "unit": {"type": "string"},
                    "distinct_policy": {"type": "string"},
                    "time_mode": {
                        "type": "string",
                        "enum": ["none", "absolute", "relative"],
                    },
                    "question_anchor": {"type": "string"},
                    "window_start": {"type": "string"},
                    "window_end": {"type": "string"},
                    "policy_version": {"type": "string"},
                },
                "required": ["intent_type"],
            },
            "requirements": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "description": (
                    "Required only for action=start. Use this exact top-level key; "
                    "each item names one evidence slot and its exact-reference cardinality."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "slot_id": {"type": "string"},
                        "description": {"type": "string"},
                        "minimum_refs": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 40,
                            "default": 1,
                        },
                    },
                    "required": ["slot_id", "description"],
                },
            },
            "missing_slot": {"type": "string"},
            "tool": {
                "type": "string",
                "enum": [
                    "lcm_recall",
                    "lcm_recent",
                    "lcm_query_state",
                    "lcm_load_session",
                    "lcm_expand",
                ],
            },
            "tool_args": {"type": "object"},
            "resolved_slots": {
                "type": "array",
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "slot_id": {"type": "string"},
                        "evidence_refs": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 40,
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["slot_id", "evidence_refs"],
                },
            },
            "selected_refs": {
                "type": "array",
                "minItems": 1,
                "maxItems": 40,
                "items": {"type": "string"},
            },
            "computation": {
                "type": "object",
                "additionalProperties": False,
                "description": (
                    "Optional only for action=finish. Every operand must copy the exact "
                    "quote and either its assertion_id or its raw store_id/span offsets "
                    "from evidence returned by this retrieval episode."
                ),
                "properties": {
                    "operands": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 50,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "assertion_id": {"type": "string"},
                                "store_id": {"type": "integer"},
                                "span_start": {"type": "integer"},
                                "span_end": {"type": "integer"},
                                "quote": {"type": "string"},
                                "value": {
                                    "anyOf": [
                                        {"type": "number"},
                                        {"type": "string"},
                                        {"type": "null"},
                                    ],
                                },
                                "unit": {"type": "string"},
                                "key": {"type": "string"},
                                "label": {"type": "string"},
                                "date": {"type": "string"},
                                "occurrence_time": {"type": "object"},
                            },
                            "required": ["quote"],
                            "anyOf": [
                                {"required": ["assertion_id"]},
                                {
                                    "required": [
                                        "store_id",
                                        "span_start",
                                        "span_end",
                                    ]
                                },
                            ],
                        },
                    },
                    "candidate_answer": {"type": "string"},
                },
                "required": ["operands"],
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}

LCM_RECENT = {
    "name": "lcm_recent",
    "description": (
        "Retrieve recent conversation summaries by a natural UTC time period. "
        "Ready temporal rollups are served when available; otherwise the tool "
        "transparently falls back to existing leaf summaries in the same window."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "period": {
                "type": "string",
                "description": (
                    "UTC period: today, yesterday, Nd (for example 7d), week, "
                    "month, date:YYYY-MM-DD, or last Nh (for example last 6h)."
                ),
            },
            "scope": {
                "type": "string",
                "enum": ["conversation"],
                "description": "Use the active conversation. Cross-session rollups are future work.",
                "default": "conversation",
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum sections to return (default 10, hard upper bound 200). "
                    "Values above the cap are clamped and reported."
                ),
                "default": 10,
            },
        },
        "required": ["period"],
    },
}

LCM_LOAD_SESSION = {
    "name": "lcm_load_session",
    "description": (
        "Load an ordered raw-message transcript page for one explicit session_id from the plugin-local LCM database. "
        "This is enumeration, not search: it does not require a query, returns raw message content rather than snippets, "
        "and orders rows chronologically by store_id. Use this after session_search or lcm_grep has identified a session_id "
        "that already exists in lcm.db. Output is bounded by limit, per-row content is bounded by max_content_chars, "
        "and row pagination uses after_store_id/next_cursor. "
        "It returns raw rows only; cross-session summary/DAG expansion remains out of scope."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Explicit LCM session id to load. Required; no implicit current/all fallback is applied.",
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum raw messages to return (default 100, hard upper bound 200). "
                    "Values above the cap are clamped and reported via limit_clamped_from."
                ),
                "default": 100,
            },
            "max_content_chars": {
                "type": "integer",
                "description": (
                    "Maximum content characters to include per message (default 4000, hard upper bound 20000). "
                    "Longer rows include content_truncated=true and can be recovered fully with lcm_expand(store_id=...)."
                ),
                "default": 4000,
            },
            "after_store_id": {
                "type": "integer",
                "description": (
                    "Exclusive cursor for pagination. Pass the previous response's next_cursor "
                    "to continue with rows whose store_id is greater than this value."
                ),
                "default": 0,
            },
            "roles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional role filter, for example ['user', 'assistant', 'tool', 'system'].",
            },
            "time_from": {
                "type": "number",
                "description": "Optional inclusive minimum message timestamp (Unix seconds).",
            },
            "time_to": {
                "type": "number",
                "description": "Optional inclusive maximum message timestamp (Unix seconds).",
            },
            "include_exact_ref": {
                "type": "boolean",
                "description": (
                    "Opt in to an exact_ref for each returned content slice. "
                    "Omitting it preserves the legacy response bytes."
                ),
                "default": False,
            },
        },
        "required": ["session_id"],
    },
}

LCM_DESCRIBE = {
    "name": "lcm_describe",
    "description": (
        "Inspect a current-session summary node's subtree metadata WITHOUT loading full "
        "content, or inspect an externalized payload ref without opening the "
        "full payload. Returns token counts, child manifest, expand hints, "
        "or externalized payload metadata/preview. Use this to plan retrieval "
        "strategy before spending tokens on lcm_expand inside the active conversation. "
        "For cross-session recall, use session_search first. If called with no "
        "node_id or externalized_ref, returns the top-level DAG overview for "
        "the current session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "integer",
                "description": "Summary node ID to inspect. Omit for session overview.",
            },
            "externalized_ref": {
                "type": "string",
                "description": "Optional externalized payload ref filename to inspect instead of a summary node.",
            },
        },
        "required": [],
    },
}

LCM_EXPAND = {
    "name": "lcm_expand",
    "description": (
        "Recover the original detail behind a summary node, externalized payload, or raw message. "
        "Mode selection (exactly one): node_id (current session only) returns the source messages "
        "or lower-depth summaries that were compacted into a summary node; externalized_ref "
        "(current session only) returns a stored externalized payload's content; store_id returns "
        "a single raw message by store_id and works across sessions, suitable for drilling into "
        "cross-session lcm_grep results. Output is bounded by max_tokens; raw recovery is pageable "
        "via content_offset (and source_offset/source_limit for node_id mode). For Hermes-tracked "
        "session history outside the LCM database, prefer session_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "integer",
                "description": (
                    "Summary node ID to expand. Current-session only — cross-session DAG expansion "
                    "is not supported in this version."
                ),
            },
            "externalized_ref": {
                "type": "string",
                "description": "Externalized payload ref filename to expand instead of a summary node. Current-session only.",
            },
            "store_id": {
                "type": "integer",
                "description": (
                    "Raw message store_id to fetch. Works across sessions, so a store_id surfaced by "
                    "a cross-session lcm_grep result can be expanded directly. Returns the message's "
                    "content paged by content_offset. If the row references an externalized payload, "
                    "the ref is surfaced via 'externalized_ref'; payload metadata and content are "
                    "session-scoped, so a cross-session row also includes 'externalized_note' "
                    "explaining that the ref is for traceability only and cannot be expanded in this version."
                ),
            },
            "max_tokens": {
                "type": "integer",
                "description": "Token budget for returned content (default 4000)",
                "default": 4000,
            },
            "source_offset": {
                "type": "integer",
                "description": "Zero-based pagination offset into the node's immediate source list (node_id mode only).",
                "default": 0,
            },
            "source_limit": {
                "type": "integer",
                "description": "Maximum number of immediate sources to return from source_offset (node_id mode only). Output still respects max_tokens.",
            },
            "content_offset": {
                "type": "integer",
                "description": "Character offset used to continue an oversized raw message, externalized payload, or store_id-mode message. Use next_content_offset from the previous response.",
                "default": 0,
            },
            "include_exact_ref": {
                "type": "boolean",
                "description": (
                    "In store_id mode, opt in to an exact_ref for the returned content slice. "
                    "Omitting it preserves the legacy response bytes."
                ),
                "default": False,
            },
        },
        "required": [],
    },
}

LCM_STATUS = {
    "name": "lcm_status",
    "description": (
        "Get a quick health overview of the LCM engine for the current session. "
        "Shows compression count, store size, DAG depth distribution, context usage, "
        "active configuration, session/message filter state, and rotate snapshot "
        "state (last_rotate_at, rotate_backup_path, rotate_backup_size when a "
        "/lcm rotate apply has been run). Use this to understand how much history "
        "has been compacted, how the engine is performing, whether the current "
        "session is matched by ignore or stateless session patterns, which message "
        "noise-suppression patterns are loaded, and when the rolling rotate "
        "backup was last written."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

LCM_INSPECT = {
    "name": "lcm_inspect",
    "description": (
        "Inspect read-only LCM metadata for the current session: session/conversation "
        "lineage, message frontier and fresh tail, DAG compaction frontier, latest "
        "compaction skip/no-op reason, externalized payload refs and readability, "
        "and matched ignore/stateless patterns. This is an operator inventory tool; "
        "use lcm_grep/lcm_load_session/lcm_expand when you need actual content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of rows/items to return for bounded sections. Defaults to 20 and is capped at 200.",
                "default": 20,
            },
        },
        "required": [],
    },
}

LCM_DOCTOR = {
    "name": "lcm_doctor",
    "description": (
        "Run diagnostics on the LCM database and configuration. Checks database "
        "integrity, detects orphaned DAG nodes, validates configuration, and "
        "reports potential issues. Use this to troubleshoot problems or verify "
        "a healthy setup."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

LCM_EXPAND_QUERY = {
    "name": "lcm_expand_query",
    "description": (
        "Answer a natural-language question using expanded LCM context from the current session. Provide a prompt, and either "
        "query matching summaries/raw messages to expand or explicit node_ids to inspect. Uses the expansion path "
        "instead of the summarization path so retrieval/synthesis can use a different model or timeout. "
        "When expanding parent summary nodes, it recursively descends the DAG under the context budget to include leaf evidence where possible. "
        "The response includes a nested, default-JSON-bounded, tool-extracted evidence provenance object with locator coverage, while marking locator replay and semantic entailment as unverified and identifiers as non-authoritative. This adds no authorization: node_id and externalized_ref retain current-session checks, store_id remains an intentional cross-session locator, and hosts must authorize expansion before invoking it. "
        "Prefer this for questions about the active conversation after compaction; for cross-session recall, use session_search first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The question or task to answer from expanded LCM context",
            },
            "query": {
                "type": "string",
                "description": "Optional search query used to find candidate summaries before expansion",
            },
            "node_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional explicit summary node IDs to expand instead of searching",
            },
            "max_results": {
                "type": "integer",
                "description": "Max candidate summaries to expand when using query (default 5)",
                "default": 5,
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max answer tokens for bounded synthesis returned to the main agent (default 2000)",
                "default": 2000,
            },
            "context_max_tokens": {
                "type": "integer",
                "description": "Expanded serialized summary/raw/child-source/externalized fresh context budget for the auxiliary LLM before it returns the bounded answer (default max(answer max_tokens, 32000 or LCM_EXPANSION_CONTEXT_TOKENS))",
                "default": 32000,
            },
        },
        "required": ["prompt"],
    },
}
