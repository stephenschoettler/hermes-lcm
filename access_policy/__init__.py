"""Policy layer for the LCM authorization seam.

No longer inert: a valid Teams context now resolves to :class:`TeamsPolicy`,
which scopes to the acting principal. ``TrustedOwnerPolicy`` remains the
default-off path and is unchanged.
"""

from .errors import AuthorizationRequiredError
from .fail_closed import FailClosedPolicy
from .resolution import (
    ACCESS_CONTEXT_ACCESSOR,
    TEAMS_ENABLED_ATTR,
    policy_access_context,
    policy_for_engine,
    resolve_policy,
)
from .teams_policy import TeamsPolicy
from .trusted_owner import TrustedOwnerPolicy

__all__ = [
    "AuthorizationRequiredError",
    "ACCESS_CONTEXT_ACCESSOR",
    "FailClosedPolicy",
    "TEAMS_ENABLED_ATTR",
    "policy_access_context",
    "policy_for_engine",
    "TeamsPolicy",
    "TrustedOwnerPolicy",
    "resolve_policy",
]
