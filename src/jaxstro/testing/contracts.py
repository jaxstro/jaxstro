"""Gradient-contract vocabulary and fail-closed consumer predicates.

This module is intentionally smaller than the finite-difference audit engine. Downstream
consumers can inspect contract labels without importing the AD-vs-FD machinery, while the
engine remains the place that computes numerical status.
"""

from __future__ import annotations

import math
from typing import Any, Literal, TypeGuard, cast, get_args

GradContract = Literal[
    "smooth_pathwise",
    "known_zero",
    "known_blocked",
    "surrogate",
    "validation_only",
]

LIVE_GRAD_CONTRACTS: tuple[str, ...] = get_args(GradContract)
RESERVED_GRAD_CONTRACTS: tuple[str, ...] = (
    "fixed_subsystem",
    "implicit_event",
    "fixed_trace",
    "distributional",
)

_EXPECT_TO_CONTRACT: dict[str, GradContract] = {
    "consistent": "smooth_pathwise",
    "known_zero": "known_zero",
    "known_blocked": "known_blocked",
}


def is_grad_contract(contract: object) -> TypeGuard[GradContract]:
    """Return True only for the five live v1 contract labels."""

    return isinstance(contract, str) and contract in LIVE_GRAD_CONTRACTS


def default_contract_for_expect(expect: str) -> GradContract:
    """Map legacy numerical expectation labels to their default contract."""

    try:
        return _EXPECT_TO_CONTRACT[expect]
    except KeyError as exc:
        raise ValueError(
            f"unknown expect class for gradient contract: {expect!r}"
        ) from exc


def contract_requires_fd(contract: str) -> bool:
    """Whether v1 interprets AD-vs-FD agreement as part of this contract's gate."""

    return contract in {"smooth_pathwise", "known_zero"}


def _finite_float(value: object) -> float | None:
    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def is_inference_ready(result: object) -> bool:
    """Conservative v1 whitelist for gradients that may feed inference/Fisher/OED.

    Only clean ``smooth_pathwise`` results that still satisfy the existing AD-vs-FD audit
    evidence are accepted. Everything else, including unknown/reserved labels and legacy
    results with missing contract metadata, fails closed.
    """

    if getattr(result, "grad_contract", None) != "smooth_pathwise":
        return False
    if getattr(result, "expect", None) != "consistent":
        return False
    if getattr(result, "status", None) != "clean":
        return False
    if getattr(result, "finite", None) is not True:
        return False

    ratio = getattr(result, "ratio", None)
    tol = getattr(result, "tol", None)
    abs_ad = getattr(result, "abs_ad", None)
    ad = getattr(result, "ad", None)
    fd = getattr(result, "fd", None)
    ratio_float = _finite_float(ratio)
    tol_float = _finite_float(tol)
    abs_ad_float = _finite_float(abs_ad)
    ad_float = _finite_float(ad)
    fd_float = _finite_float(fd)
    if ratio_float is None:
        return False
    if tol_float is None:
        return False
    if abs_ad_float is None:
        return False
    if ad_float is None:
        return False
    if fd_float is None:
        return False
    if abs_ad_float <= 0.0:
        return False
    return abs(ratio_float - 1.0) < tol_float


__all__ = [
    "GradContract",
    "LIVE_GRAD_CONTRACTS",
    "RESERVED_GRAD_CONTRACTS",
    "contract_requires_fd",
    "default_contract_for_expect",
    "is_grad_contract",
    "is_inference_ready",
]
