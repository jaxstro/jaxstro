"""``jaxstro.composition.Composition``: a record that checks it is a composition."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.composition import SUM_TOLERANCE, Composition

pytestmark = pytest.mark.unit


def test_a_solar_composition_is_accepted():
    c = Composition(X=0.70, Y=0.28, Z=0.02, deuterium_to_hydrogen=2.5e-5)
    assert abs(c.X + c.Y + c.Z - 1.0) <= SUM_TOLERANCE


def test_a_primordial_composition_is_accepted():
    Composition(X=0.76, Y=0.24, Z=0.0, deuterium_to_hydrogen=2.5e-5)


def test_every_field_is_required():
    with pytest.raises(TypeError):
        Composition(X=0.70, Y=0.28, Z=0.02)


@pytest.mark.parametrize(
    "fields",
    [
        dict(X=0.70, Y=0.28, Z=0.03, deuterium_to_hydrogen=0.0),  # sums to 1.01
        dict(
            X=1.20, Y=-0.22, Z=0.02, deuterium_to_hydrogen=0.0
        ),  # sums to 1, out of range
        dict(X=0.70, Y=0.28, Z=0.02, deuterium_to_hydrogen=-1e-5),  # negative D/H
    ],
)
def test_a_non_composition_is_refused(fields):
    with pytest.raises(ValueError):
        Composition(**fields)


def test_a_traced_metallicity_is_differentiable():
    def metals(z):
        c = Composition(X=0.70, Y=0.30 - z, Z=z, deuterium_to_hydrogen=2.5e-5)
        return c.Z * c.X

    assert float(jax.grad(metals)(jnp.asarray(0.02))) == pytest.approx(0.70, rel=1e-15)
