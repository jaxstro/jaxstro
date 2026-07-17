import jax.numpy as jnp

from jaxstro import quad
from jaxstro.quad._multidim import evaluate_multidim, map_hyperrectangle
from jaxstro.quantity import units as q_units


def test_map_hyperrectangle_preserves_coordinate_last_and_orientation():
    domain = quad.Hyperrectangle(
        jnp.array([1.0, 5.0]),
        jnp.array([3.0, 1.0]),
    )
    reference = jnp.array([[0.0, 0.25], [1.0, 0.75]])
    mapped = map_hyperrectangle(domain, reference)

    assert mapped.x.shape == (2, 2)
    assert jnp.allclose(mapped.x, jnp.array([[1.0, 4.0], [3.0, 2.0]]))
    assert mapped.jacobian == 8.0
    assert mapped.orientation == -1.0
    assert mapped.valid


def test_evaluator_keeps_point_axis_separate_from_payload_axes():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    reference = jnp.array([[0.25, 0.5], [0.75, 1.0]])

    evaluated = evaluate_multidim(
        lambda x, scale: scale * jnp.stack((x[:, 0], x[:, 1]), axis=-1),
        domain,
        reference,
        args=jnp.asarray(2.0),
        measure=quad.LebesgueMeasure(),
    )

    assert evaluated.values.shape == (2, 2)
    assert jnp.allclose(
        evaluated.values,
        jnp.array([[0.5, 1.0], [1.5, 2.0]]),
    )
    assert evaluated.weights.shape == (2,)
    assert not evaluated.nonfinite


def test_weighted_density_receives_physical_coordinate_last_points():
    domain = quad.Hyperrectangle(jnp.zeros(2), 2.0 * jnp.ones(2))
    measure = quad.WeightedMeasure(
        lambda x, args: args + x[:, 0] * x[:, 1],
        density_unit=q_units.dimensionless,
    )
    evaluated = evaluate_multidim(
        lambda x, _args: jnp.ones(x.shape[0]),
        domain,
        jnp.array([[0.5, 0.25]]),
        args=jnp.asarray(1.0),
        measure=measure,
    )
    assert jnp.allclose(evaluated.weights, jnp.array([6.0]))
