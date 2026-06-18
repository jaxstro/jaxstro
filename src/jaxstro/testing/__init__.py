"""Shared testing utilities for the jaxstro ecosystem.

Currently hosts the deduplicated AD-vs-FD gradient-audit engine consumed by sibling
packages' differentiability gates (fluxax, progenax). Dependency-light (jax + stdlib);
no pytest at import time so it ships with the installed package.
"""

from jaxstro.testing.grad_audit import (
    AuditResult,
    Case,
    Direction,
    EdgeConfig,
    Expect,
    audit_entry_point,
)

__all__ = [
    "audit_entry_point",
    "Case",
    "AuditResult",
    "EdgeConfig",
    "Direction",
    "Expect",
]
