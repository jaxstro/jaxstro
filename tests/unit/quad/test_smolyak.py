import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quad import _sparse, sparse
from jaxstro.quad._sparse import fixed_index_set, smolyak_rule_data


def _options(method, **overrides):
    options = dict(
        method=method,
        epsabs=1.0e-10,
        epsrel=1.0e-10,
        max_evaluations=2_000,
        max_indices=128,
        max_frontier=256,
        max_nodes=2_000,
        gradient="stop",
    )
    options.update(overrides)
    return options


def test_isotropic_index_set_is_downward_closed():
    indices = fixed_index_set(quad.Smolyak(level=3), dimension=3)
    accepted = set(indices)
    for index in accepted:
        for axis, value in enumerate(index):
            if value > 1:
                backward = list(index)
                backward[axis] -= 1
                assert tuple(backward) in accepted


def test_anisotropy_restricts_expensive_axis():
    indices = fixed_index_set(
        quad.Smolyak(level=4, anisotropy=(1.0, 3.0)),
        dimension=2,
    )
    assert max(index[1] for index in indices) < max(index[0] for index in indices)


def test_fixed_index_set_uses_deterministic_total_degree_then_lexicographic_order():
    indices = fixed_index_set(quad.Smolyak(level=3), dimension=3)
    assert indices == tuple(sorted(indices, key=lambda index: (sum(index), index)))


@pytest.mark.parametrize("level", (0, -1, True, 2.5))
def test_smolyak_rejects_invalid_level(level):
    with pytest.raises(ValueError, match="positive integer"):
        quad.Smolyak(level=level)


@pytest.mark.parametrize(
    "anisotropy",
    ((1.0, 0.0), (1.0, -1.0), (1.0, jnp.inf), (1.0, jnp.nan)),
)
def test_smolyak_rejects_invalid_anisotropy(anisotropy):
    with pytest.raises(ValueError, match="finite and positive"):
        quad.Smolyak(level=2, anisotropy=anisotropy)


def test_anisotropy_length_must_match_dimension():
    with pytest.raises(ValueError, match="one weight per dimension"):
        fixed_index_set(
            quad.Smolyak(level=2, anisotropy=(1.0, 2.0)),
            dimension=3,
        )


def test_smolyak_integrates_product_moment_with_unique_work_count():
    method = quad.Smolyak(level=4)
    result = quad.integrate(
        lambda x: jnp.prod(x**2, axis=-1),
        quad.Hyperrectangle(jnp.zeros(3), jnp.ones(3)),
        **_options(method),
    )
    data = smolyak_rule_data(method, dimension=3, dtype=jnp.float64)

    assert jnp.allclose(result.value, 1.0 / 27.0, atol=1e-10)
    assert result.error.kind == quad.ErrorKind.SPARSE_GRID_SURPLUS
    assert result.work.evaluations == data.points.shape[0]
    assert result.work.evaluations == len(set(data.identities))
    assert result.work.refinements == data.index_count - 1


def test_fixed_sparse_frontier_evidence_is_derived_from_outer_increments():
    method = quad.Smolyak(level=2)
    data = smolyak_rule_data(method, dimension=2, dtype=jnp.float64)
    result = quad.integrate(
        lambda x: jnp.exp(jnp.sum(x, axis=-1)),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(method, epsabs=0.0, epsrel=0.0),
    )

    values = jnp.exp(jnp.sum(data.points, axis=-1))
    increments = data.increment_weights @ values
    expected_error = jnp.sum(jnp.abs(increments[data.frontier_mask]))

    assert jnp.allclose(result.value, jnp.sum(increments), rtol=2e-14, atol=2e-14)
    assert jnp.allclose(result.error.estimate, expected_error)
    assert jnp.allclose(result.error.norm, expected_error)
    assert result.status == quad.QuadStatus.MAX_INDICES


def test_fixed_capacity_rejects_before_payload_inference(monkeypatch):
    def fail_payload(*_args, **_kwargs):
        raise AssertionError("payload inference must follow sparse capacity checks")

    monkeypatch.setattr(sparse, "infer_multidim_payload_zero", fail_payload)
    method = quad.Smolyak(level=4)
    data = smolyak_rule_data(method, dimension=3, dtype=jnp.float64)

    with pytest.raises(ValueError, match="max_nodes"):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(3), jnp.ones(3)),
            **_options(method, max_nodes=data.point_count - 1),
        )


def test_node_capacity_rejects_before_hierarchical_rule_materialization(monkeypatch):
    def fail_rule(*_args, **_kwargs):
        raise AssertionError("sparse rule arrays must follow exact node preflight")

    monkeypatch.setattr(_sparse, "hierarchical_rule", fail_rule)
    with pytest.raises(ValueError, match="max_nodes"):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(3), jnp.ones(3)),
            **_options(
                quad.Smolyak(level=2),
                max_nodes=1,
                max_evaluations=2_000,
            ),
        )


def test_zero_volume_returns_exact_zero_without_integrand_work():
    calls = 0

    def integrand(x):
        def record(_value):
            nonlocal calls
            calls += 1

        jax.debug.callback(record, x[0, 0])
        return jnp.sum(x, axis=-1)

    result = quad.integrate(
        integrand,
        quad.Hyperrectangle(jnp.array([0.0, 1.0]), jnp.array([2.0, 1.0])),
        **_options(quad.Smolyak(level=2)),
    )

    assert calls == 0
    assert result.value == 0.0
    assert result.work.evaluations == 0
    assert result.status == quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE


def test_traced_invalid_bound_precedes_zero_volume():
    @jax.jit
    def solve(upper):
        return quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), upper),
            **_options(quad.Smolyak(level=2)),
        )

    result = solve(jnp.array([0.0, jnp.inf]))
    assert result.status == quad.QuadStatus.INVALID_INPUT
    assert jnp.isnan(result.value)
    assert result.work.evaluations == 0


def test_nonfinite_sparse_integrand_fails_closed():
    result = quad.integrate(
        lambda x: jnp.where(x[:, 0] < 0.5, jnp.nan, 1.0),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(quad.Smolyak(level=2)),
    )

    assert result.status == quad.QuadStatus.NONFINITE_INTEGRAND
    assert jnp.isnan(result.value)
    assert jnp.isnan(result.error.estimate)
