import jax
import jax.numpy as jnp
import pytest

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


def test_quantity_replay_composes_with_jit_and_vmap():
    derivative = jax.jit(
        jax.grad(lambda value: _length_integral(value, q.cm).to_value(q.cm**2))
    )
    bounds = jnp.asarray([50.0, 100.0, 200.0])
    assert jnp.allclose(jax.vmap(derivative)(bounds), bounds, rtol=2e-10)


def _improper_quantity_integral(kind, unit, physical_scale):
    zero = q.Quantity(0.0, unit)
    scale = physical_scale.to(unit)
    if kind == "right":
        domain = quad.RightInfinite(zero, scale=scale)
    elif kind == "left":
        domain = quad.LeftInfinite(zero, scale=scale)
    else:
        domain = quad.Infinite(unit=unit, scale=scale)

    def fun(x, args):
        if kind == "right":
            return q.math.exp((-1.0 * x) / args)
        if kind == "left":
            return q.math.exp(x / args)
        return q.math.exp(-1.0 * (x / args) ** 2)

    return quad.integrate(
        fun,
        domain,
        args=scale,
        method=quad.AdaptiveTanhSinh(3),
        epsabs=q.Quantity(1e-9, q.m).to(unit),
        epsrel=1e-9,
        max_evaluations=1800,
        max_regions=24,
        gradient="replay",
    )


@pytest.mark.parametrize("kind", ["right", "left", "full"])
def test_improper_quantity_scale_is_invariant_to_coordinate_representation(kind):
    physical_scale = q.Quantity(1.0, q.m)
    metres = _improper_quantity_integral(kind, q.m, physical_scale)
    centimetres = _improper_quantity_integral(kind, q.cm, physical_scale)

    assert jnp.allclose(
        metres.value.to_value(q.cm),
        centimetres.value.to_value(q.cm),
        rtol=2e-10,
        atol=2e-10,
    )
    assert jnp.allclose(
        metres.error.estimate.to_value(q.cm),
        centimetres.error.estimate.to_value(q.cm),
        rtol=2e-10,
        atol=2e-10,
    )
    assert metres.status == centimetres.status
    assert metres.work == centimetres.work


@pytest.mark.parametrize(
    "domain",
    [
        quad.RightInfinite(0.0 * q.cm),
        quad.LeftInfinite(0.0 * q.cm),
        quad.Infinite(unit=q.cm),
    ],
)
def test_dimensional_improper_quantity_domain_requires_explicit_scale(domain):
    with pytest.raises(TypeError, match="explicit Quantity scale"):
        quad.integrate(
            lambda x: q.math.exp(-((x / (1.0 * q.cm)) ** 2)),
            domain,
            method=quad.AdaptiveTanhSinh(3),
            epsabs=1e-8 * q.cm,
            epsrel=1e-8,
            max_evaluations=600,
            max_regions=12,
        )


def test_dimensional_improper_quantity_scale_must_match_coordinate_unit():
    with pytest.raises(q.DimensionError, match="scale must match"):
        quad.integrate(
            lambda x: x,
            quad.RightInfinite(0.0 * q.cm, scale=1.0 * q.s),
            method=quad.AdaptiveTanhSinh(3),
            epsabs=1e-8 * q.cm**2,
            epsrel=1e-8,
            max_evaluations=600,
            max_regions=12,
        )


@pytest.mark.parametrize("scale_value", [0.0, -1.0, jnp.inf, jnp.nan])
def test_invalid_quantity_improper_scale_returns_invalid_input(scale_value):
    result = quad.integrate(
        lambda x: q.math.exp(-1.0 * (x / (1.0 * q.cm)) ** 2),
        quad.Infinite(unit=q.cm, scale=scale_value * q.cm),
        method=quad.AdaptiveTanhSinh(3),
        epsabs=1e-8 * q.cm,
        epsrel=1e-8,
        max_evaluations=600,
        max_regions=12,
    )
    assert result.status == quad.QuadStatus.INVALID_INPUT
    assert not jnp.isfinite(result.value.value)


def test_array_valued_quantity_improper_scale_fails_with_scalar_contract():
    with pytest.raises(ValueError, match="improper-domain scale must be scalar"):
        quad.integrate(
            lambda x: q.math.exp(-1.0 * (x / (1.0 * q.cm)) ** 2),
            quad.Infinite(unit=q.cm, scale=jnp.asarray([1.0, 2.0]) * q.cm),
            method=quad.AdaptiveTanhSinh(3),
            epsabs=1e-8 * q.cm,
            epsrel=1e-8,
            max_evaluations=600,
            max_regions=12,
        )


def test_complex_quantity_improper_scale_fails_with_real_contract():
    with pytest.raises(TypeError, match="improper-domain scale must be real"):
        quad.integrate(
            lambda x: q.math.exp(-1.0 * (x / (1.0 * q.cm)) ** 2),
            quad.Infinite(unit=q.cm, scale=(1.0 + 1.0j) * q.cm),
            method=quad.AdaptiveTanhSinh(3),
            epsabs=1e-8 * q.cm,
            epsrel=1e-8,
            max_evaluations=600,
            max_regions=12,
        )
