"""Independent references for fixed quadrature construction."""

import math

import numpy as np
import pytest

from jaxstro.quad import (
    GaussianRule,
    JacobiMeasure,
    LaguerreMeasure,
    LebesgueMeasure,
    PhysicistsHermiteMeasure,
    StandardNormalMeasure,
)
from jaxstro.quad._recurrence import gaussian_rule_data

scipy_special = pytest.importorskip("scipy.special")


@pytest.mark.parametrize(
    ("measure", "reference"),
    [
        (LebesgueMeasure(), lambda n: scipy_special.roots_legendre(n)),
        (
            JacobiMeasure(0.25, 0.5),
            lambda n: scipy_special.roots_jacobi(n, 0.25, 0.5),
        ),
        (
            LaguerreMeasure(0.5),
            lambda n: scipy_special.roots_genlaguerre(n, 0.5),
        ),
        (
            PhysicistsHermiteMeasure(),
            lambda n: scipy_special.roots_hermite(n),
        ),
    ],
)
def test_gaussian_rule_matches_scipy_reference(measure, reference) -> None:
    order = 12
    data = gaussian_rule_data(GaussianRule(order), measure)
    nodes, weights = reference(order)
    np.testing.assert_allclose(np.asarray(data.nodes), nodes, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        np.asarray(data.weights), weights, rtol=2e-12, atol=2e-12
    )


def test_standard_normal_matches_normalized_hermitenorm_reference() -> None:
    order = 12
    data = gaussian_rule_data(GaussianRule(order), StandardNormalMeasure())
    nodes, weights = scipy_special.roots_hermitenorm(order)
    weights = weights / math.sqrt(2.0 * math.pi)
    np.testing.assert_allclose(np.asarray(data.nodes), nodes, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        np.asarray(data.weights), weights, rtol=2e-12, atol=2e-12
    )
