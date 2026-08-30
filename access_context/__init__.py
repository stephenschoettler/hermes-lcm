"""Opt-in AccessContextV1 contract; importing this package wires nothing."""
from .denials import (
    PUBLIC_DENIAL_PROJECTION,
    Decision,
    DenialReason,
    InternalDecision,
    PublicDecision,
    project_public,
    public_decision,
)
from .model import (
    ACCESS_CONTEXT_CONTRACT_REVISION,
    AccessContextV1,
    ActorType,
    ScopeMismatchError,
    is_subset_of,
)
from .protocols import HostContextCarrier, LcmAuthorizationConsumer
from .validation import (
    ResolutionMode,
    VALIDATION_ORDER,
    ValidationStage,
    derive_child,
    resolve_mode,
    validate,
)

__all__ = [
    "ACCESS_CONTEXT_CONTRACT_REVISION",
    "AccessContextV1",
    "ActorType",
    "Decision",
    "DenialReason",
    "HostContextCarrier",
    "InternalDecision",
    "LcmAuthorizationConsumer",
    "PUBLIC_DENIAL_PROJECTION",
    "PublicDecision",
    "ResolutionMode",
    "ScopeMismatchError",
    "VALIDATION_ORDER",
    "ValidationStage",
    "derive_child",
    "is_subset_of",
    "project_public",
    "public_decision",
    "resolve_mode",
    "validate",
]
