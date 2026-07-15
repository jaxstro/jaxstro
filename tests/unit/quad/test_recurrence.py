"""Classical Gaussian recurrence and exactness contracts."""

import math

import jax.numpy as jnp
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


@pytest.mark.parametrize(
    ("measure", "mass"),
    [
        (LebesgueMeasure(), 2.0),
        (
            JacobiMeasure(0.25, 0.5),
            2.0**1.75 * math.gamma(1.25) * math.gamma(1.5) / math.gamma(2.75),
        ),
        (LaguerreMeasure(0.5), math.gamma(1.5)),
        (PhysicistsHermiteMeasure(), math.sqrt(math.pi)),
        (StandardNormalMeasure(), 1.0),
    ],
)
def test_gaussian_weights_reproduce_measure_mass(measure, mass) -> None:
    data = gaussian_rule_data(GaussianRule(8), measure)
    assert jnp.all(data.weights > 0.0)
    assert jnp.all(jnp.diff(data.nodes) > 0.0)
    assert jnp.allclose(jnp.sum(data.weights), mass, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    "measure",
    [
        JacobiMeasure(0.25, 0.5, normalized=True),
        LaguerreMeasure(0.5, normalized=True),
        PhysicistsHermiteMeasure(normalized=True),
    ],
)
def test_normalized_classical_measure_has_unit_mass(measure) -> None:
    data = gaussian_rule_data(GaussianRule(8), measure)
    assert jnp.allclose(jnp.sum(data.weights), 1.0, rtol=2e-12, atol=2e-12)


def test_legendre_exact_through_degree_two_n_minus_one() -> None:
    order = 6
    data = gaussian_rule_data(GaussianRule(order), LebesgueMeasure())
    assert data.degree == 2 * order - 1
    assert data.nested is False
    for degree in range(2 * order):
        expected = 0.0 if degree % 2 else 2.0 / (degree + 1)
        got = jnp.sum(data.weights * data.nodes**degree)
        assert jnp.allclose(got, expected, rtol=2e-11, atol=2e-11)


def test_symmetric_jacobi_removable_diagonal_singularity() -> None:
    data = gaussian_rule_data(GaussianRule(5), JacobiMeasure(0.25, -0.25))
    assert jnp.all(jnp.isfinite(data.nodes))
    assert jnp.all(jnp.isfinite(data.weights))


def test_gaussian_rule_rejects_nonclassical_weight() -> None:
    from jaxstro import quantity
    from jaxstro.quad import WeightedMeasure

    measure = WeightedMeasure(
        lambda x, args: jnp.ones_like(x),
        density_unit=quantity.dimensionless,
    )
    with pytest.raises(TypeError, match="classical measure"):
        gaussian_rule_data(GaussianRule(4), measure)
