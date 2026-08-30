#!/usr/bin/env python3
"""Emit deterministic flag-off recall responses for byte comparison.

The original probe covered only an ASCII prose question, so it could not catch
a flag-independent Unicode-symbol fallback. The symbol case below distinguishes
the historical flag-off FTS route from the opt-in LIKE preservation route.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


repo = Path(sys.argv[1]).resolve()
database = Path(sys.argv[2]).resolve()
raw_output = Path(sys.argv[3]).resolve()
if database.exists():
    raise SystemExit(f"probe database must not exist: {database}")

spec = importlib.util.spec_from_file_location(
    "hermes_lcm",
    repo / "__init__.py",
    submodule_search_locations=[str(repo)],
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules["hermes_lcm"] = module
spec.loader.exec_module(module)

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.store import MessageStore
import hermes_lcm.store as lcm_store
import hermes_lcm.tools as lcm_tools


config = LCMConfig(database_path=str(database), embeddings_enabled=False)
if hasattr(config, "fts_prose_mode"):
    config.fts_prose_mode = False
lcm_store.time.time = lambda: 1_800_000_000.0
lcm_tools.time.time = lambda: 1_800_000_000.0
store = MessageStore(str(database), ingest_protection_config=config)
dag = SummaryDAG(str(database))
try:
    store.append(
        "session-a",
        {"role": "user", "content": "the dog vet appointment was Friday"},
    )
    store.append(
        "session-b",
        {"role": "user", "content": "dog grooming supplies are in the hall"},
    )
    store.append(
        "session-a",
        {"role": "user", "content": "licensed © archive"},
    )
    store.append(
        "session-a",
        {"role": "user", "content": "licensed material without the mark"},
    )
    engine = SimpleNamespace(
        _config=config,
        _store=store,
        _dag=dag,
        _hermes_home=str(database.parent),
        current_session_id="session-a",
        _session_occurrence_dates={},
    )
    raw = json.dumps(
        [
            lcm_tools.lcm_recall(
                {
                    "query": "What did I say about my dog's vet appointment?",
                    "limit": 8,
                    "include": "verbatim",
                    "detail": "snippets",
                    "scope_bias": 0.0,
                },
                engine=engine,
            ),
            lcm_tools.lcm_recall(
                {
                    "query": "licensed ©",
                    "limit": 8,
                    "include": "verbatim",
                    "detail": "snippets",
                    "scope_bias": 0.0,
                },
                engine=engine,
            ),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
finally:
    dag.close()
    store.close()

raw_output.write_text(raw, encoding="utf-8")
print(
    json.dumps(
        {
            "bytes": len(raw.encode("utf-8")),
            "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        },
        sort_keys=True,
    )
)
