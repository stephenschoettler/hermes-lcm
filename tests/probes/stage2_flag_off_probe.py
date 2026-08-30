#!/usr/bin/env python3
"""Emit one deterministic flag-off answer_ready response for byte comparison."""
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
import hermes_lcm.tools as lcm_tools


config = LCMConfig(database_path=str(database), embeddings_enabled=False)
if hasattr(config, "session_expand_v1"):
    config.session_expand_v1 = False
store = MessageStore(str(database), ingest_protection_config=config)
dag = SummaryDAG(str(database))
try:
    engine = SimpleNamespace(
        _config=config,
        _store=store,
        _dag=dag,
        _hermes_home=str(database.parent),
        current_session_id="__stage2_flag_off_probe__",
        _session_occurrence_dates={},
    )
    lcm_tools.time.time = lambda: 1_800_000_000.0
    raw = lcm_tools.lcm_recall(
        {
            "query": "kanban dashboard sprint exact evidence",
            "limit": 12,
            "include": "verbatim",
            "detail": "answer_ready",
            "scope_bias": 0.0,
        },
        engine=engine,
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
