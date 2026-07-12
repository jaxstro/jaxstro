"""Contracts for strictly gated implicit scalar roots."""

import jax.numpy as jnp
import pytest

from jaxstro.numerics import rootfinding


def _primal():
    return rootfinding.safeguarded_bracketed_root(
        lambda x: x - 1.0,
        0.0,
        2.0,
        max_steps=8,
        atol=1.0e-14,
        rtol=1.0e-14,
    )


def test_implicit_certificate_all_pass_truth_table() -> None:
    certificate, status = rootfinding._build_implicit_certificate(
        _primal(),
        jnp.asarray(1.0),
        rootfinding.ImplicitRootAssumptions(True, True),
        residual_atol=1.0e-13,
        residual_rtol=0.0,
        width_atol=1.0e-13,
        width_rtol=0.0,
        slope_floor=1.0e-8,
    )

    assert certificate.certified
    assert status == rootfinding.DERIVATIVE_STATUS_CERTIFIED


@pytest.mark.parametrize(
    ("assumptions", "slope", "residual_atol", "width_atol", "expected"),
    [
        (
            rootfinding.ImplicitRootAssumptions(False, True),
            1.0,
            1.0e-13,
            1.0e-13,
            rootfinding.DERIVATIVE_STATUS_ASSUMPTIONS_REJECTED,
        ),
        (
            rootfinding.ImplicitRootAssumptions(True, True),
            jnp.nan,
            1.0e-13,
            1.0e-13,
            rootfinding.DERIVATIVE_STATUS_NONFINITE,
        ),
        (
            rootfinding.ImplicitRootAssumptions(True, True),
            0.0,
            1.0e-13,
            1.0e-13,
            rootfinding.DERIVATIVE_STATUS_SLOPE_ILL_CONDITIONED,
        ),
        (
            rootfinding.ImplicitRootAssumptions(True, True),
            1.0,
            1.0e-13,
            -1.0,
            rootfinding.DERIVATIVE_STATUS_BRACKET_TOO_WIDE,
        ),
    ],
)
def test_implicit_certificate_status_precedence(
    assumptions, slope, residual_atol, width_atol, expected
) -> None:
    _, status = rootfinding._build_implicit_certificate(
        _primal(),
        jnp.asarray(slope),
        assumptions,
        residual_atol=residual_atol,
        residual_rtol=0.0,
        width_atol=width_atol,
        width_rtol=0.0,
        slope_floor=1.0e-8,
    )

    assert status == expected


def test_implicit_contract_field_order_is_stable() -> None:
    assert rootfinding.ImplicitRootAssumptions._fields == (
        "unique_root",
        "smooth_branch",
    )
    assert rootfinding.ImplicitRootResult._fields == (
        "root",
        "residual",
        "slope",
        "status",
        "certified",
        "certificate",
        "primal",
    )
