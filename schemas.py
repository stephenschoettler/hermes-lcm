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
                "description": "Search query (FTS5 syntax: keywords, phrases, OR/NOT)",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 10)",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}

LCM_DESCRIBE = {
    "name": "lcm_describe",
    "description": (
        "Inspect a summary node's subtree metadata WITHOUT loading full "
        "content. Returns token counts, child manifest, and expand hints. "
        "Use this to plan retrieval strategy before spending tokens on "
        "lcm_expand. If called with no node_id, returns the top-level "
        "DAG overview for the current session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "integer",
                "description": "Summary node ID to inspect. Omit for session overview.",
            },
        },
        "required": [],
    },
}

LCM_EXPAND = {
    "name": "lcm_expand",
    "description": (
        "Recover the original detail behind a summary node. Given a node_id, "
        "returns the source messages or lower-depth summaries that were "
        "compacted into that node. Use after lcm_describe to drill into "
        "specific parts of the conversation history."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "integer",
                "description": "Summary node ID to expand",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Token budget for returned content (default 4000)",
                "default": 4000,
            },
        },
        "required": ["node_id"],
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
