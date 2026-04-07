"""Hermes LCM Plugin — Lossless Context Management.

Replaces the built-in ContextCompressor with a DAG-based context engine
that persists every message and provides structured retrieval tools.

Based on the LCM paper by Ehrlich & Blackman (Voltropy PBC, Feb 2026).
"""

import logging

logger = logging.getLogger(__name__)


def register(ctx):
    """Plugin entry point — register the LCM context engine."""
    from .config import LCMConfig
    from .engine import LCMEngine

    config = LCMConfig.from_env()

    # Resolve hermes_home for profile-scoped storage
    hermes_home = ""
    try:
        from hermes_cli.config import get_hermes_home
        hermes_home = str(get_hermes_home())
    except Exception:
        import os
        hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))

    engine = LCMEngine(config=config, hermes_home=hermes_home)

    # Register as the context engine (replaces ContextCompressor)
    ctx.register_context_engine(engine)

    logger.info("LCM plugin loaded — lossless context management active")
