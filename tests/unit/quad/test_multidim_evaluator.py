import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quad._multidim import (
    evaluate_multidim,
    infer_multidim_payload_zero,
    map_hyperrectangle,
)
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


def test_infer_multidim_scalar_payload_zero_preserves_shape_and_dtype():
    zero = infer_multidim_payload_zero(
        lambda x, scale: (scale * jnp.sum(x, axis=-1)).astype(jnp.float32),
        args=jnp.asarray(2.0),
        dimension=3,
        dtype=jnp.float64,
    )

    assert zero.shape == ()
    assert zero.dtype == jnp.dtype(jnp.float32)
    assert zero == 0.0


def test_infer_multidim_array_payload_zero_preserves_shape_and_dtype():
    zero = infer_multidim_payload_zero(
        lambda x: jnp.stack((x, x**2), axis=-1).astype(jnp.complex64),
        args=(),
        dimension=2,
        dtype=jnp.float64,
    )

    assert zero.shape == (2, 2)
    assert zero.dtype == jnp.dtype(jnp.complex64)
    assert jnp.array_equal(zero, jnp.zeros((2, 2), dtype=jnp.complex64))


def test_infer_multidim_payload_zero_rejects_missing_point_axis():
    with pytest.raises(ValueError, match="leading point axis"):
        infer_multidim_payload_zero(
            lambda x: jnp.ones((x.shape[-1],), dtype=x.dtype),
            args=(),
            dimension=3,
            dtype=jnp.float64,
        )


@pytest.mark.parametrize(
    "fun",
    [
        lambda x: jnp.asarray(1.0),
        lambda x: jnp.ones((x.shape[0] + 1,), dtype=x.dtype),
    ],
)
def test_evaluator_rejects_malformed_integrand_point_axis(fun):
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    reference = jnp.array([[0.25, 0.5], [0.75, 1.0]])

    with pytest.raises(ValueError, match="leading point axis"):
        evaluate_multidim(
            fun,
            domain,
            reference,
            args=(),
            measure=quad.LebesgueMeasure(),
        )


@pytest.mark.parametrize(
    "density",
    [
        lambda _x, _args: jnp.asarray(1.0),
        lambda x, _args: jnp.ones((x.shape[0], 1), dtype=x.dtype),
    ],
)
def test_evaluator_rejects_malformed_density_point_axis(density):
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    reference = jnp.array([[0.25, 0.5], [0.75, 1.0]])
    measure = quad.WeightedMeasure(
        density,
        density_unit=q_units.dimensionless,
    )

    with pytest.raises(ValueError, match=r"shape \(point_count,\)"):
        evaluate_multidim(
            lambda x: jnp.ones(x.shape[0]),
            domain,
            reference,
            args=(),
            measure=measure,
        )


def test_evaluator_propagates_nonfinite_integrand_state():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    evaluated = evaluate_multidim(
        lambda x: jnp.where(jnp.arange(x.shape[0]) == 0, jnp.nan, 1.0),
        domain,
        jnp.array([[0.25, 0.5], [0.75, 1.0]]),
        args=(),
        measure=quad.LebesgueMeasure(),
    )

    assert evaluated.nonfinite
    assert jnp.isnan(evaluated.values[0])
    assert jnp.all(jnp.isfinite(evaluated.weights))


def test_evaluator_propagates_nonfinite_density_state():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    measure = quad.WeightedMeasure(
        lambda x, _args: jnp.where(x[:, 0] < 0.5, jnp.inf, 1.0),
        density_unit=q_units.dimensionless,
    )
    evaluated = evaluate_multidim(
        lambda x: jnp.ones(x.shape[0]),
        domain,
        jnp.array([[0.25, 0.5], [0.75, 1.0]]),
        args=(),
        measure=measure,
    )

    assert evaluated.nonfinite
    assert jnp.isinf(evaluated.weights[0])
    assert jnp.all(jnp.isfinite(evaluated.values))


def test_evaluator_accepts_array_like_reference_once_at_boundary():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    evaluated = evaluate_multidim(
        lambda x: jnp.sum(x, axis=-1),
        domain,
        [[0.25, 0.5], [0.75, 1.0]],
        args=(),
        measure=quad.LebesgueMeasure(),
    )

    assert evaluated.values.shape == (2,)
    assert jnp.allclose(evaluated.values, jnp.array([0.75, 1.75]))


def test_evaluator_preserves_reversed_orientation_in_signed_weights():
    domain = quad.Hyperrectangle(
        jnp.array([1.0, 0.0]),
        jnp.array([-1.0, 3.0]),
    )
    evaluated = evaluate_multidim(
        lambda x: jnp.ones(x.shape[0]),
        domain,
        jnp.array([[0.25, 0.5], [0.75, 1.0]]),
        args=(),
        measure=quad.LebesgueMeasure(),
    )

    assert jnp.array_equal(evaluated.weights, jnp.array([-6.0, -6.0]))
    assert not evaluated.nonfinite
    assert evaluated.valid


def test_evaluator_preserves_exact_zero_volume_and_orientation():
    domain = quad.Hyperrectangle(
        jnp.array([0.0, 1.0]),
        jnp.array([2.0, 1.0]),
    )
    reference = jnp.array([[0.25, 0.5], [0.75, 1.0]])
    mapped = map_hyperrectangle(domain, reference)
    evaluated = evaluate_multidim(
        lambda x: jnp.ones(x.shape[0]),
        domain,
        reference,
        args=(),
        measure=quad.LebesgueMeasure(),
    )

    assert mapped.jacobian == 0.0
    assert mapped.orientation == 0.0
    assert jnp.array_equal(evaluated.weights, jnp.zeros(2))
    assert not evaluated.nonfinite
    assert evaluated.valid


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
