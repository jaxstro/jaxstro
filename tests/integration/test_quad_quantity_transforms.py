import jax
import jax.numpy as jnp

from jaxstro import quad
from jaxstro import quantity as q


def _length_integral(bound_value, bound_unit):
    bound = q.Quantity(bound_value, bound_unit)
    return quad.integrate(
        lambda x: x,
        quad.Interval(q.Quantity(0.0, bound_unit), bound),
        method=quad.GaussKronrod(21),
        epsabs=q.Quantity(1e-10, bound_unit**2),
        epsrel=1e-10,
        max_evaluations=63,
        max_regions=2,
        gradient="replay",
    ).value


def test_quantity_replay_preserves_physical_value_across_length_units():
    metres = _length_integral(2.0, q.m).to_value(q.cm**2)
    centimetres = _length_integral(200.0, q.cm).to_value(q.cm**2)
    assert jnp.allclose(metres, centimetres, rtol=2e-12)


def test_raw_value_jacobians_rescale_with_declared_derivative_units():
    d_metres = jax.grad(lambda value: _length_integral(value, q.m).to_value(q.m**2))(
        2.0
    )
    d_centimetres = jax.grad(
        lambda value: _length_integral(value, q.cm).to_value(q.cm**2)
    )(200.0)
    assert jnp.allclose(d_metres, 2.0, rtol=2e-10)
    assert jnp.allclose(d_centimetres, 200.0, rtol=2e-10)
    assert jnp.allclose(d_metres * 100.0, d_centimetres, rtol=2e-10)


def test_quantity_result_jvp_keeps_static_integral_unit():
    primal, tangent = jax.jvp(
        lambda value: _length_integral(value, q.cm),
        (200.0,),
        (1.0,),
    )
    assert primal.unit == q.cm**2
    assert tangent.unit == q.cm**2


def _weighted_expectation(scale_value, length_unit):
    scale = q.Quantity(scale_value, length_unit)
    measure = quad.WeightedMeasure(
        lambda x, args: q.math.exp(-1.0 * (x / args)),
        density_unit=q.dimensionless,
    )
    return quad.integrate(
        lambda x, _args: x,
        quad.Interval(q.Quantity(0.0, length_unit), 2.0 * scale),
        args=scale,
        measure=measure,
        method=quad.GaussKronrod(21),
        epsabs=q.Quantity(1e-9, length_unit**2),
        epsrel=1e-9,
        max_evaluations=147,
        max_regions=4,
        gradient="replay",
    ).value


def test_quantity_weighted_density_sees_physical_coordinates():
    left = _weighted_expectation(1.0, q.m).to_value(q.cm**2)
    right = _weighted_expectation(100.0, q.cm).to_value(q.cm**2)
    assert jnp.allclose(left, right, rtol=2e-8)


def _normalized_weighted_integral(scale_value, length_unit):
    scale = q.Quantity(scale_value, length_unit)
    measure = quad.WeightedMeasure(
        lambda x, args: q.math.exp(-1.0 * (x / args)) / args,
        density_unit=q.dimensionless / length_unit,
    )
    return quad.integrate(
        lambda x, _args: x,
        quad.Interval(q.Quantity(0.0, length_unit), 2.0 * scale),
        args=scale,
        measure=measure,
        method=quad.GaussKronrod(21),
        epsabs=q.Quantity(1e-9, length_unit),
        epsrel=1e-9,
        max_evaluations=147,
        max_regions=4,
        gradient="replay",
    ).value


def test_inverse_length_density_converts_with_coordinate_representation():
    metres = _normalized_weighted_integral(1.0, q.m).to_value(q.cm)
    centimetres = _normalized_weighted_integral(100.0, q.cm).to_value(q.cm)
    assert jnp.allclose(metres, centimetres, rtol=2e-8)
