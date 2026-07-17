import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quad import _tensor
from jaxstro.quad._tensor import tensor_rule_data
from jaxstro.quantity import units as q_units


def _tensor_kwargs(method, *, max_evaluations):
    return dict(
        method=method,
        epsabs=1e-12,
        epsrel=1e-12,
        max_evaluations=max_evaluations,
        gradient="stop",
    )


def test_heterogeneous_tensor_integrates_bivariate_polynomial():
    result = quad.integrate(
        lambda x: x[:, 0] ** 3 * x[:, 1] ** 2,
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_tensor_kwargs(
            quad.TensorProduct((quad.GaussianRule(3), quad.ClenshawCurtisRule(5))),
            max_evaluations=15,
        ),
    )

    assert jnp.allclose(result.value, 1.0 / 12.0, rtol=1e-12, atol=1e-12)
    assert result.error.kind == quad.ErrorKind.UNAVAILABLE
    assert jnp.isnan(result.error.estimate)
    assert jnp.isnan(result.error.norm)
    assert jnp.isnan(result.error.confidence_level)
    assert result.tolerance == 1e-12
    assert result.status == quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE
    assert result.work.evaluations == 15
    assert result.work.refinements == 0
    assert result.work.active_regions == 0
    assert result.work.levels == 0
    assert result.work.replicates == 0


def test_replicated_rule_and_reversed_axis_preserve_orientation():
    result = quad.integrate(
        lambda x: jnp.ones(x.shape[0]),
        quad.Hyperrectangle(jnp.array([0.0, 2.0]), jnp.array([1.0, -1.0])),
        **_tensor_kwargs(
            quad.TensorProduct(quad.GaussianRule(2)),
            max_evaluations=4,
        ),
    )

    assert result.value == -3.0


def test_zero_volume_returns_exact_zero_without_point_work():
    result = quad.integrate(
        lambda x: jnp.stack((x[:, 0], x[:, 1]), axis=-1),
        quad.Hyperrectangle(jnp.array([0.0, 1.0]), jnp.array([2.0, 1.0])),
        **_tensor_kwargs(
            quad.TensorProduct(quad.GaussianRule(3)),
            max_evaluations=9,
        ),
    )

    assert jnp.array_equal(result.value, jnp.zeros(2))
    assert result.error.kind == quad.ErrorKind.UNAVAILABLE
    assert jnp.all(jnp.isnan(result.error.estimate))
    assert jnp.isnan(result.error.norm)
    assert result.status == quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE
    assert result.work.evaluations == 0


def test_tensor_capacity_fails_before_materialization(monkeypatch):
    def fail_mesh_materialization(*_args, **_kwargs):
        raise AssertionError("tensor mesh must not be materialized")

    monkeypatch.setattr(_tensor.jnp, "meshgrid", fail_mesh_materialization)
    with pytest.raises(ValueError, match="requires 81 evaluations"):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(4), jnp.ones(4)),
            method=quad.TensorProduct(quad.GaussianRule(3)),
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=80,
            gradient="stop",
        )


@pytest.mark.parametrize(
    "rule, tolerance",
    [
        (quad.GaussianRule(3), 2e-12),
        (quad.ClenshawCurtisRule(5), 2e-12),
        (quad.FejerIRule(4), 2e-12),
        (quad.FejerIIRule(4), 2e-12),
        (quad.TanhSinhRule(1), 1e-5),
    ],
)
def test_tensor_rule_data_maps_every_supported_rule_to_unit_interval(rule, tolerance):
    data = tensor_rule_data(quad.TensorProduct(rule), 2, jnp.float64)

    assert data.points.shape == (data.point_count, 2)
    assert data.weights.shape == (data.point_count,)
    assert jnp.all((data.points >= 0.0) & (data.points <= 1.0))
    assert jnp.allclose(
        jnp.sum(data.weights),
        1.0,
        rtol=tolerance,
        atol=tolerance,
    )


def test_tensor_product_requires_exactly_one_rule_per_axis():
    with pytest.raises(ValueError, match="one rule or one rule per axis"):
        tensor_rule_data(
            quad.TensorProduct((quad.GaussianRule(2),)),
            2,
            jnp.float64,
        )


def test_tensor_product_rules_are_static_pytree_metadata():
    method = quad.TensorProduct((quad.GaussianRule(2), quad.ClenshawCurtisRule(3)))
    leaves, structure = jax.tree.flatten(method)
    rebuilt = jax.tree.unflatten(structure, leaves)

    assert leaves == []
    assert rebuilt == method


def test_tensor_product_rejects_unsupported_rule():
    with pytest.raises(TypeError, match="unsupported tensor rule: object"):
        tensor_rule_data(quad.TensorProduct(object()), 2, jnp.float64)


@pytest.mark.parametrize("dimension", [1, 9])
def test_tensor_product_rejects_dimensions_outside_b1_envelope(dimension):
    with pytest.raises(
        ValueError,
        match="Phase B1 deterministic methods require dimension 2 through 8",
    ):
        tensor_rule_data(
            quad.TensorProduct(quad.GaussianRule(1)),
            dimension,
            jnp.float64,
        )


@pytest.mark.parametrize("dimension", [2, 8])
def test_tensor_product_accepts_b1_dimension_endpoints(dimension):
    data = tensor_rule_data(
        quad.TensorProduct(quad.GaussianRule(1)),
        dimension,
        jnp.float64,
    )

    assert data.points.shape == (1, dimension)
    assert data.point_count == 1


def test_traced_invalid_bounds_return_fail_closed_result():
    @jax.jit
    def solve(upper):
        return quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), upper),
            **_tensor_kwargs(
                quad.TensorProduct(quad.GaussianRule(2)),
                max_evaluations=4,
            ),
        )

    result = solve(jnp.array([1.0, jnp.inf]))

    assert result.status == quad.QuadStatus.INVALID_INPUT
    assert jnp.isnan(result.value)
    assert result.work.evaluations == 0


def test_invalid_bound_takes_precedence_over_coincident_axis():
    @jax.jit
    def solve(upper):
        return quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), upper),
            **_tensor_kwargs(
                quad.TensorProduct(quad.GaussianRule(2)),
                max_evaluations=4,
            ),
        )

    result = solve(jnp.array([0.0, jnp.inf]))

    assert result.status == quad.QuadStatus.INVALID_INPUT
    assert jnp.isnan(result.value)
    assert result.work.evaluations == 0


def test_nonfinite_integrand_returns_fail_closed_result():
    result = quad.integrate(
        lambda x: jnp.where(x[:, 0] < 0.5, jnp.nan, 1.0),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_tensor_kwargs(
            quad.TensorProduct(quad.GaussianRule(2)),
            max_evaluations=4,
        ),
    )

    assert result.status == quad.QuadStatus.NONFINITE_INTEGRAND
    assert jnp.isnan(result.value)
    assert result.work.evaluations == 4


def test_nonfinite_density_returns_fail_closed_result():
    measure = quad.WeightedMeasure(
        lambda x, _args: jnp.where(x[:, 0] < 0.5, jnp.inf, 1.0),
        density_unit=q_units.dimensionless,
    )
    result = quad.integrate(
        lambda x: jnp.ones(x.shape[0]),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        measure=measure,
        **_tensor_kwargs(
            quad.TensorProduct(quad.GaussianRule(2)),
            max_evaluations=4,
        ),
    )

    assert result.status == quad.QuadStatus.NONFINITE_INTEGRAND
    assert jnp.isnan(result.value)
    assert result.work.evaluations == 4


def test_tensor_stop_mode_has_exact_zero_grad_and_jvp_tangents():
    method = quad.TensorProduct(quad.GaussianRule(2))
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))

    def parameterized(scale):
        return quad.integrate(
            lambda x, factor: factor * jnp.sum(x, axis=-1),
            domain,
            args=scale,
            **_tensor_kwargs(method, max_evaluations=4),
        ).value

    derivative = jax.grad(parameterized)(jnp.asarray(2.0))
    _, tangent = jax.jvp(
        parameterized,
        (jnp.asarray(2.0),),
        (jnp.asarray(1.0),),
    )

    assert jnp.array_equal(derivative, 0.0)
    assert jnp.array_equal(tangent, 0.0)


@pytest.mark.parametrize("gradient", ["replay", "through", "invalid"])
def test_tensor_rejects_every_gradient_mode_except_stop(gradient):
    with pytest.raises(ValueError, match='requires gradient="stop"'):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            method=quad.TensorProduct(quad.GaussianRule(2)),
            epsabs=1e-12,
            epsrel=1e-12,
            max_evaluations=4,
            gradient=gradient,
        )


def test_tensor_composes_with_jit_and_vmap_in_stop_mode():
    method = quad.TensorProduct((quad.GaussianRule(2), quad.ClenshawCurtisRule(3)))
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))

    @jax.jit
    def solve(scale):
        return quad.integrate(
            lambda x, factor: factor * jnp.sum(x, axis=-1),
            domain,
            args=scale,
            **_tensor_kwargs(method, max_evaluations=6),
        ).value

    values = jax.vmap(solve)(jnp.array([1.0, 2.0, 3.0]))

    assert jnp.allclose(values, jnp.array([1.0, 2.0, 3.0]))
