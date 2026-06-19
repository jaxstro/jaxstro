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
from jaxstro.testing.ratchet import (
    ASSERT_HELPERS,
    assert_no_stale,
    assert_partition,
    has_nearby_citation,
    resolve_node_ids,
    scan_module_numeric_literals,
    test_body_has_assert,
)

__all__ = [
    "audit_entry_point",
    "Case",
    "AuditResult",
    "EdgeConfig",
    "Direction",
    "Expect",
    "ASSERT_HELPERS",
    "assert_no_stale",
    "assert_partition",
    "has_nearby_citation",
    "resolve_node_ids",
    "scan_module_numeric_literals",
    "test_body_has_assert",
]
