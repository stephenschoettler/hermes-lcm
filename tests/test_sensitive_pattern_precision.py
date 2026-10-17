"""Regression tests for sensitive-pattern value-shape precision.

The ``api_key`` / ``bearer_token`` / ``password_assignment`` patterns anchor on
a credential-ish name plus a separator. Without a shape test on the captured
value they fire on the code that proves a secret was NOT hardcoded (environment
lookups, config templating, plain identifier references). Because redaction runs
on the ingest write path, every such match destroys non-secret text
irreversibly.

These tests lock both halves of the contract: non-credential values survive
intact, and real hardcoded credentials are still redacted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.ingest_protection import redact_sensitive_text


def _sensitive_config(tmp_path: Path, **overrides) -> LCMConfig:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    config.sensitive_patterns_enabled = True
    config.sensitive_patterns = [
        "api_key",
        "bearer_token",
        "password_assignment",
        "private_key",
    ]
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _sensitive_engine(tmp_path: Path, **overrides) -> LCMEngine:
    engine = LCMEngine(
        config=_sensitive_config(tmp_path, **overrides),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "precision-session",
        platform="telegram",
        conversation_id="precision-conversation",
        context_length=200_000,
    )
    return engine


# --- 1. Precision: values that must NOT be redacted ------------------------

ENV_INDIRECTION = [
    'client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))',
    'client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])',
    'api_key = os.getenv("ANTHROPIC_API_KEY")',
    "const apiKey = process.env.OPENAI_API_KEY;",
    "api_key: process.env.STRIPE_SECRET_KEY,",
    'access_token = os.environ.get("GITHUB_ACCESS_TOKEN")',
    'secret_key = os.environ.get("DJANGO_SECRET_KEY")',
    'client_secret = os.getenv("OAUTH_CLIENT_SECRET")',
    'password = os.environ["POSTGRES_PASSWORD"]',
    'passwd = os.getenv("MYSQL_PASSWORD")',
    "password: process.env.REDIS_PASSWORD,",
]

TEMPLATE_REFERENCES = [
    "export API_KEY=$OPENAI_API_KEY",
    "export API_KEY=${OPENAI_API_KEY}",
    "api_key: ${{ secrets.OPENAI_API_KEY }}",
    "password: ${{ secrets.DB_PASSWORD }}",
    "api_key: <your-api-key-here>",
    "password=<your-password>",
    'api_key: "YOUR_API_KEY_HERE"',
    'password = "CHANGEME"',
    "api_key: {{ vault_openai_api_key }}",
    "api_key: !secret openai_api_key",
    "password: !secret postgres_password",
]

PLAIN_REFERENCES = [
    "client = OpenAI(api_key=settings.openai_api_key)",
    "api_key = config.api_key",
    "api_key=self._api_key",
    "api_key = credentials.api_key",
    "password=user_password,",
    "password=hashed_password",
    'password = request.json.get("password")',
    'access_token = response.json()["access_token"]',
    "client_secret=oauth_config.client_secret",
    "api_key: apiKeyFromContext,",
    "password: passwordInput.value,",
]

PROSE_AND_SIGNATURES = [
    "Set your api_key in the configuration file before running the server.",
    "The api_key parameter is required and must be a non-empty string.",
    'raise ValueError("api_key must be provided or set OPENAI_API_KEY")',
    "def __init__(self, api_key: str | None = None) -> None:",
    "password must be at least twelve characters long",
    "Authorization: Bearer <token>",
    "The password field is write-only and never returned by the API.",
    "api_key: string;",
]

NON_SECRETS = ENV_INDIRECTION + TEMPLATE_REFERENCES + PLAIN_REFERENCES + PROSE_AND_SIGNATURES


# --- 1. Precision: values that MUST still be redacted ----------------------
# All credential values below are synthetic and were generated for this file.

HARDCODED_SECRETS = [
    'client = OpenAI(api_key="sk-proj-9vQ2mTf4LpXw7RaB3nKdY8HcJ1sZ6eUgW0oI5tMv")',
    'api_key = "sk-ant-api03-7Kq2Zx9Lm4Rb8Tn1Vp6Ws3Yd0Hf5Jg2Ce7Ak4Nu9Bi"',
    'api_key="AIzaSyD3kL9mQ2vX7bN4pR8tW1cF6hJ0sY5uZ2a"',
    'secret_key = "8f3a91c04be27d65a0f1e8c72b94d6503fa7e21c9b8d40576"',
    'access_token = "ghp_A9kZ2mQ7xW4pL1vN8bR3tY6cF0hJ5sD2uE7g"',
    'api_token = "1a2B3c4D5e6F7g8H9i0J1k2L3m4N5o6P7q8R"',
    'password = "hT7#kQ2mZ9xW4pL1"',
    'passphrase = "Zq7Wm2Kx9Lp4Nv8B"',
    'auth_header = "Bearer 4f8Kd92Mz7Qp1Xn6Bv3Lw0Ae5Rt8Yu2Io"',
]


@pytest.mark.parametrize("text", NON_SECRETS)
def test_sensitive_patterns_do_not_redact_non_credential_values(tmp_path, text):
    """Environment lookups, templates, references and prose survive intact."""
    config = _sensitive_config(tmp_path)

    assert redact_sensitive_text(text, config) == text


@pytest.mark.parametrize("text", HARDCODED_SECRETS)
def test_sensitive_patterns_still_redact_hardcoded_credentials(tmp_path, text):
    """The value-shape gate must not cost detection of real secrets."""
    config = _sensitive_config(tmp_path)

    redacted = redact_sensitive_text(text, config)

    assert redacted != text
    assert "[LCM sensitive redaction:" in redacted


def test_sensitive_precision_holds_on_the_ingest_write_path(tmp_path):
    """The gate applies where it matters: before the SQLite INSERT."""
    engine = _sensitive_engine(tmp_path)
    non_secret = 'client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))'
    secret = "sk-proj-9vQ2mTf4LpXw7RaB3nKdY8HcJ1sZ6eUgW0oI5tMv"

    engine._ingest_messages([
        {"role": "user", "content": non_secret},
        {"role": "user", "content": f'api_key = "{secret}"'},
    ])

    rows = engine._store._conn.execute(
        "SELECT content FROM messages ORDER BY store_id"
    ).fetchall()
    assert rows[0][0] == non_secret
    assert secret not in rows[1][0]
    assert "[LCM sensitive redaction:" in rows[1][0]


def test_private_key_pattern_is_not_value_shape_gated(tmp_path):
    """private_key is PEM-anchored and self-validating; leave it alone."""
    config = _sensitive_config(tmp_path)
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAxyz\n"
        "-----END RSA PRIVATE KEY-----"
    )

    redacted = redact_sensitive_text(pem, config)

    assert "MIIEowIBAAKCAQEAxyz" not in redacted
    assert "name=private_key" in redacted
