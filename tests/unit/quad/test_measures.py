import inspect

import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quantity import dimensionless


def _density(x, args):
    return jnp.exp(-args[0] * x)


def test_weighted_measure_metadata_is_static() -> None:
    measure = quad.WeightedMeasure(
        _density,
        density_unit=dimensionless,
        normalized=False,
    )
    leaves, structure = jax.tree.flatten(measure)
    assert leaves == []
    assert jax.tree.unflatten(structure, leaves) == measure


def test_measure_constructor_signatures_match_the_approved_contract() -> None:
    weighted = inspect.signature(quad.WeightedMeasure).parameters
    assert weighted["density"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert weighted["density_unit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert weighted["density_unit"].default is inspect.Parameter.empty
    assert weighted["normalized"].kind is inspect.Parameter.KEYWORD_ONLY
    assert weighted["normalized"].default is False

    defaults = {
        quad.JacobiMeasure: False,
        quad.LaguerreMeasure: False,
        quad.PhysicistsHermiteMeasure: False,
    }
    for measure_type, expected_default in defaults.items():
        normalized = inspect.signature(measure_type).parameters["normalized"]
        assert normalized.kind is inspect.Parameter.KEYWORD_ONLY
        assert normalized.default is expected_default

    assert not inspect.signature(quad.StandardNormalMeasure).parameters


@pytest.mark.parametrize(
    "measure",
    (
        quad.LebesgueMeasure(),
        quad.WeightedMeasure(_density, density_unit=dimensionless),
        quad.JacobiMeasure(0.25, 0.5),
        quad.LaguerreMeasure(0.25),
        quad.PhysicistsHermiteMeasure(),
        quad.StandardNormalMeasure(),
    ),
)
def test_every_measure_round_trips_through_its_static_pytree(measure) -> None:
    leaves, structure = jax.tree.flatten(measure)
    assert leaves == []
    assert jax.tree.unflatten(structure, leaves) == measure


def test_normalized_is_a_declaration_not_a_numerical_action() -> None:
    raw = quad.WeightedMeasure(
        _density,
        density_unit=dimensionless,
        normalized=False,
    )
    declared = quad.WeightedMeasure(
        _density,
        density_unit=dimensionless,
        normalized=True,
    )
    assert raw.density is declared.density
    assert not raw.normalized
    assert declared.normalized


@pytest.mark.parametrize(
    "factory",
    (
        lambda: quad.JacobiMeasure(-1.0, 0.0),
        lambda: quad.JacobiMeasure(0.0, -1.0),
        lambda: quad.LaguerreMeasure(-1.0),
    ),
)
def test_nonintegrable_classical_parameters_raise_eagerly(factory) -> None:
    with pytest.raises(ValueError, match="greater than -1"):
        factory()


def test_standard_normal_is_explicitly_normalized() -> None:
    assert quad.StandardNormalMeasure().normalized
    assert not quad.PhysicistsHermiteMeasure().normalized
