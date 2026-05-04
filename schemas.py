"""Tool schemas for LCM — what the LLM sees."""

LCM_GREP = {
    "name": "lcm_grep",
    "description": (
        "Search the LCM database for past conversation content (raw messages AND summaries across all depths) "
        "to recover details from the active session or, when explicitly scoped, from other sessions in the "
        "plugin-local LCM database (including history imported from OpenClaw or lossless-claw). "
        "Default scope is current-session only; broader scopes must be requested explicitly. "
        "Cross-session summary hits are returned as snippets but cannot be expanded by node_id in this version "
        "(they carry cross_session_expand_supported=false); use lcm_expand with the result's store_id for raw-message "
        "expansion across sessions. For Hermes-tracked session history outside the LCM database, use session_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
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
            "role": {
                "type": "string",
                "enum": ["user", "assistant", "tool"],
                "description": (
                    "Optional role filter. Applies to message hits only; summary hits are returned unfiltered "
                    "(role does not exist on summary nodes). The response echoes role_filter_applies='messages_only' when this is set."
                ),
            },
            "time_from": {
                "type": "string",
                "description": (
                    "Optional inclusive ISO 8601 lower bound for message timestamps and summary latest_at. "
                    "Examples: '2026-01-01T00:00:00Z' or '2026-01-01T00:00:00+00:00'."
                ),
            },
            "time_to": {
                "type": "string",
                "description": (
                    "Optional inclusive ISO 8601 upper bound for message timestamps and summary latest_at."
                ),
            },
        },
        "required": ["query"],
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
        "Recover the original detail behind a current session summary node, or open an "
        "externalized payload ref directly. Given a node_id, returns the "
        "source messages or lower-depth summaries that were compacted into "
        "that node. Output is bounded by default, but raw recovery is pageable: "
        "use source_offset/source_limit to page immediate sources and content_offset "
        "to continue an oversized message or externalized payload. Given externalized_ref, "
        "returns the stored payload content plus cursor metadata. Use after lcm_describe "
        "to drill into specific parts of the active conversation or large externalized "
        "tool output. For cross-session recall, prefer session_search first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "integer",
                "description": "Summary node ID to expand",
            },
            "externalized_ref": {
                "type": "string",
                "description": "Optional externalized payload ref filename to expand instead of a summary node.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Token budget for returned content (default 4000)",
                "default": 4000,
            },
            "source_offset": {
                "type": "integer",
                "description": "Zero-based pagination offset into the node's immediate source list (messages or child nodes). Use pagination.next_source_offset to continue.",
                "default": 0,
            },
            "source_limit": {
                "type": "integer",
                "description": "Maximum number of immediate sources to return from source_offset. Output still respects max_tokens.",
            },
            "content_offset": {
                "type": "integer",
                "description": "Character offset used to continue an oversized raw message or externalized payload. Use next_content_offset from the previous response.",
                "default": 0,
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
        "and active configuration. Use this to understand how much history has been "
        "compacted and how the engine is performing."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
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
        "query matching summaries to expand or explicit node_ids to inspect. Uses the expansion path "
        "instead of the summarization path so retrieval/synthesis can use a different model or timeout. "
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
