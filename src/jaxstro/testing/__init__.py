"""Shared testing utilities for the jaxstro ecosystem.

Currently hosts the deduplicated AD-vs-FD gradient-audit engine consumed by sibling
packages' differentiability gates (fluxax, progenax), plus contract labels for safe
downstream interpretation. Dependency-light (jax + stdlib); no pytest at import time so
it ships with the installed package.
"""

from jaxstro.testing.contracts import (
    LIVE_GRAD_CONTRACTS,
    RESERVED_GRAD_CONTRACTS,
    GradContract,
    contract_requires_fd,
    default_contract_for_expect,
    is_grad_contract,
    is_inference_ready,
)
from jaxstro.testing.diagnostics import (
    DifferenceReport,
    check_directional_derivative,
    compare_gradients,
    compare_jacobians,
    directional_derivative,
    finite_difference_grad,
    finite_difference_jacobian,
)
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
    DEFAULT_CITE_RE,
    assert_no_stale,
    assert_partition,
    has_nearby_citation,
    resolve_node_ids,
    scan_module_numeric_literals,
    test_body_has_assert,
)
from jaxstro.testing.reports import (
    EvidenceAnchor,
    MethodEvidence,
    NumericalTrustReport,
    default_numerics_trust_report,
    trust_report_to_dict,
    trust_report_to_json,
    trust_report_to_markdown,
)

__all__ = [
    "audit_entry_point",
    "Case",
    "AuditResult",
    "EdgeConfig",
    "Direction",
    "Expect",
    "GradContract",
    "LIVE_GRAD_CONTRACTS",
    "RESERVED_GRAD_CONTRACTS",
    "contract_requires_fd",
    "default_contract_for_expect",
    "is_grad_contract",
    "is_inference_ready",
    "DifferenceReport",
    "finite_difference_grad",
    "finite_difference_jacobian",
    "directional_derivative",
    "compare_gradients",
    "compare_jacobians",
    "check_directional_derivative",
    "ASSERT_HELPERS",
    "DEFAULT_CITE_RE",
    "assert_no_stale",
    "assert_partition",
    "has_nearby_citation",
    "resolve_node_ids",
    "scan_module_numeric_literals",
    "test_body_has_assert",
    "EvidenceAnchor",
    "MethodEvidence",
    "NumericalTrustReport",
    "trust_report_to_dict",
    "trust_report_to_json",
    "trust_report_to_markdown",
    "default_numerics_trust_report",
]
