"""Tool schemas for LCM — what the LLM sees."""

LCM_GREP = {
    "name": "lcm_grep",
    "description": (
        "Search the full conversation history — raw messages AND summaries "
        "across all depths. Use this to find specific topics, decisions, "
        "file paths, or error messages from earlier in the conversation, "
        "even if those turns have been compacted. Returns matches with "
        "depth labels showing where in the hierarchy each result lives."
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
                "description": "Max results to return (default 10)",
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
                "enum": ["current", "all"],
                "description": "Whether to search only the current session or all stored sessions.",
                "default": "current",
            },
            "source": {
                "type": "string",
                "description": (
                    "Optional source/platform filter (for example cli, discord, telegram). "
                    "Applies directly to raw messages and to summaries via descendant source lineage. "
                    "Use 'unknown' for explicit unknown-source content."
                ),
            },
        },
        "required": ["query"],
    },
}

LCM_DESCRIBE = {
    "name": "lcm_describe",
    "description": (
        "Inspect a summary node's subtree metadata WITHOUT loading full "
        "content, or inspect an externalized payload ref without opening the "
        "full payload. Returns token counts, child manifest, expand hints, "
        "or externalized payload metadata/preview. Use this to plan retrieval "
        "strategy before spending tokens on lcm_expand. If called with no "
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
        "Recover the original detail behind a summary node, or open an "
        "externalized payload ref directly. Given a node_id, returns the "
        "source messages or lower-depth summaries that were compacted into "
        "that node. Given externalized_ref, returns the stored payload "
        "content plus metadata. Use after lcm_describe to drill into "
        "specific parts of the conversation history or large externalized "
        "tool output."
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
        "Answer a natural-language question using expanded LCM context. Provide a prompt, and either "
        "query matching summaries to expand or explicit node_ids to inspect. Uses the expansion path "
        "instead of the summarization path so retrieval/synthesis can use a different model or timeout."
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
                "description": "Max answer tokens for synthesis (default 2000)",
                "default": 2000,
            },
        },
        "required": ["prompt"],
    },
}
