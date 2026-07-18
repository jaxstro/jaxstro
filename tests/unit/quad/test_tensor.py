import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quad import _tanh_sinh, _tensor
from jaxstro.quad._chebyshev import chebyshev_rule_data
from jaxstro.quad._recurrence import gaussian_rule_data
from jaxstro.quad._tanh_sinh import _tanh_sinh_lattice_data
from jaxstro.quad._tensor import tensor_rule_data
from jaxstro.quad.measures import LebesgueMeasure
from jaxstro.quad.rules import FixedRuleData
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


def test_tanh_sinh_capacity_fails_before_rule_array_materialization(monkeypatch):
    def fail_rule_materialization(*_args, **_kwargs):
        raise AssertionError("tanh-sinh rule arrays must not be materialized")

    monkeypatch.setattr(
        _tensor,
        "tanh_sinh_rule_data",
        fail_rule_materialization,
    )
    monkeypatch.setattr(
        _tanh_sinh,
        "_host_lattice",
        fail_rule_materialization,
    )
    with pytest.raises(ValueError, match="requires 361 evaluations"):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(
                jnp.zeros(2, dtype=jnp.float32),
                jnp.ones(2, dtype=jnp.float32),
            ),
            method=quad.TensorProduct(quad.TanhSinhRule(2)),
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=360,
            gradient="stop",
        )


def test_high_level_tanh_sinh_capacity_rejects_before_materialization(monkeypatch):
    def fail_materialization(*_args, **_kwargs):
        raise AssertionError("tanh-sinh rule data and mesh must not be materialized")

    monkeypatch.setattr(_tensor, "tanh_sinh_rule_data", fail_materialization)
    monkeypatch.setattr(_tanh_sinh, "_host_lattice", fail_materialization)
    monkeypatch.setattr(_tensor.jnp, "meshgrid", fail_materialization)

    with pytest.raises(ValueError, match="requires 149 evaluations"):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(
                jnp.zeros(2, dtype=jnp.float32),
                jnp.ones(2, dtype=jnp.float32),
            ),
            method=quad.TensorProduct((quad.TanhSinhRule(5), quad.GaussianRule(1))),
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=148,
            gradient="stop",
        )


@pytest.mark.parametrize(
    ("dtype", "expected_axis_count"),
    [(jnp.float32, 19), (jnp.float64, 25)],
)
def test_tanh_sinh_unit_rule_is_representable_in_target_dtype(
    dtype,
    expected_axis_count,
):
    points, weights = _tensor._unit_rule_data(quad.TanhSinhRule(2), dtype)

    assert points.dtype == dtype
    assert weights.dtype == dtype
    assert points.shape == (expected_axis_count,)
    assert jnp.all((points > 0.0) & (points < 1.0))
    assert jnp.all(jnp.diff(points) > 0.0)
    assert jnp.allclose(
        points + points[::-1],
        jnp.ones_like(points),
        rtol=0.0,
        atol=2.0 * jnp.finfo(dtype).eps,
    )


@pytest.mark.parametrize(
    ("dtype", "axis_count", "rtol"),
    [(jnp.float32, 19, 5e-4), (jnp.float64, 25, 5e-8)],
)
def test_tanh_sinh_integrates_endpoint_singular_product(dtype, axis_count, rtol):
    result = quad.integrate(
        lambda x: 1.0 / jnp.sqrt(x[:, 0] * (1.0 - x[:, 0]) * x[:, 1] * (1.0 - x[:, 1])),
        quad.Hyperrectangle(
            jnp.zeros(2, dtype=dtype),
            jnp.ones(2, dtype=dtype),
        ),
        method=quad.TensorProduct(quad.TanhSinhRule(2)),
        epsabs=0.0,
        epsrel=0.0,
        max_evaluations=axis_count**2,
        gradient="stop",
    )

    assert jnp.isfinite(result.value)
    assert jnp.allclose(result.value, jnp.pi**2, rtol=rtol, atol=0.0)


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


@pytest.mark.parametrize(
    "rule",
    [
        quad.GaussianRule(11),
        quad.GaussianRule(32),
        quad.ClenshawCurtisRule(17),
        quad.FejerIRule(3),
        quad.FejerIRule(512),
        quad.FejerIIRule(5),
    ],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_exact_rule_unit_mapping_uses_only_symmetric_global_scaling(rule, dtype):
    if isinstance(rule, quad.GaussianRule):
        source = gaussian_rule_data(rule, LebesgueMeasure())
    else:
        source = chebyshev_rule_data(rule, dtype=dtype)
    raw_weights = 0.5 * jnp.asarray(source.weights, dtype=dtype)
    expected_weights = raw_weights / jnp.sum(raw_weights)

    _points, weights = _tensor._unit_rule_data(rule, dtype)

    assert jnp.array_equal(weights, expected_weights)


def test_exact_rule_rejects_unit_mass_residual_outside_roundoff(monkeypatch):
    corrupted = FixedRuleData(
        nodes=jnp.array([-0.5, 0.5], dtype=jnp.float64),
        weights=jnp.array([0.75, 0.75], dtype=jnp.float64),
        degree=3,
        nested=False,
    )
    monkeypatch.setattr(
        _tensor,
        "gaussian_rule_data",
        lambda _rule, _measure: corrupted,
    )

    with pytest.raises(ValueError, match="unit-mass residual exceeds roundoff"):
        _tensor._unit_rule_data(quad.GaussianRule(2), jnp.float64)


@pytest.mark.parametrize(
    "rule",
    [
        quad.GaussianRule(1),
        quad.GaussianRule(2),
        quad.GaussianRule(5),
        quad.GaussianRule(11),
        quad.GaussianRule(32),
        quad.ClenshawCurtisRule(1),
        quad.ClenshawCurtisRule(2),
        quad.ClenshawCurtisRule(5),
        quad.ClenshawCurtisRule(17),
        quad.FejerIRule(1),
        quad.FejerIRule(2),
        quad.FejerIRule(5),
        quad.FejerIRule(11),
        quad.FejerIRule(512),
        quad.FejerIIRule(1),
        quad.FejerIIRule(2),
        quad.FejerIIRule(5),
        quad.FejerIIRule(11),
    ],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_exact_unit_rules_preserve_positive_symmetric_moments(rule, dtype):
    points, weights = _tensor._unit_rule_data(rule, dtype)
    eps = jnp.finfo(dtype).eps
    reduction_bound = 8.0 * weights.size * eps
    mass = jnp.sum(weights)
    first_moment = jnp.sum(weights * points)

    assert jnp.all(weights > 0.0)
    assert jnp.allclose(
        weights,
        weights[::-1],
        rtol=0.0,
        atol=reduction_bound,
    )
    assert jnp.allclose(
        points + points[::-1],
        jnp.ones_like(points),
        rtol=0.0,
        atol=reduction_bound,
    )
    assert jnp.abs(mass - 1.0) <= reduction_bound
    assert jnp.abs(first_moment - 0.5) <= reduction_bound
    assert jnp.abs(first_moment - 0.5 * mass) <= reduction_bound


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_tanh_sinh_unit_rule_is_not_mass_normalized(level, dtype):
    lattice = _tanh_sinh_lattice_data(level, dtype=dtype)
    mapped = 0.5 * (lattice.compact_nodes + 1.0)
    representable = (mapped > 0.0) & (mapped < 1.0)
    symmetric_representable = representable & representable[::-1]
    expected_weights = 0.5 * lattice.compact_weights[symmetric_representable]

    _points, weights = _tensor._unit_rule_data(quad.TanhSinhRule(level), dtype)

    assert jnp.array_equal(weights, expected_weights)
    if level == 0:
        assert jnp.abs(jnp.sum(weights) - 1.0) > 1e-3


@pytest.mark.parametrize("level", range(8))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_tanh_sinh_scalar_point_count_matches_materialized_rules(level, dtype):
    rule = quad.TanhSinhRule(level)
    data = _tanh_sinh.tanh_sinh_rule_data(
        rule,
        dtype=dtype,
        open_unit_interval=True,
    )
    mapped = 0.5 * (data.nodes + 1.0)

    assert (
        _tanh_sinh.tanh_sinh_rule_point_count(
            rule,
            dtype=dtype,
            open_unit_interval=True,
        )
        == data.nodes.size
    )
    assert jnp.all((mapped > 0.0) & (mapped < 1.0))
    assert jnp.all(jnp.diff(mapped) > 0.0)
    assert jnp.all(data.weights > 0.0)
    assert jnp.array_equal(data.nodes, -data.nodes[::-1])
    assert jnp.array_equal(data.weights, data.weights[::-1])


@pytest.mark.parametrize(
    ("dtype", "level", "axis_count"),
    [(jnp.float32, 5, 149), (jnp.float64, 7, 803)],
)
def test_high_level_tanh_sinh_public_exact_capacity(dtype, level, axis_count):
    result = quad.integrate(
        lambda x: jnp.ones(x.shape[0], dtype=dtype),
        quad.Hyperrectangle(
            jnp.zeros(2, dtype=dtype),
            jnp.ones(2, dtype=dtype),
        ),
        method=quad.TensorProduct((quad.TanhSinhRule(level), quad.GaussianRule(1))),
        epsabs=0.0,
        epsrel=0.0,
        max_evaluations=axis_count,
        gradient="stop",
    )

    assert jnp.isfinite(result.value)
    assert result.status == quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE
    assert result.work.evaluations == axis_count


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
    with pytest.raises(ValueError) as exc_info:
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            method=quad.TensorProduct(quad.GaussianRule(2)),
            epsabs=1e-12,
            epsrel=1e-12,
            max_evaluations=4,
            gradient=gradient,
        )
    assert str(exc_info.value) == (
        'TensorProduct supports only gradient="stop" in Phase B1; '
        'gradient="replay" is introduced in Phase B4'
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
