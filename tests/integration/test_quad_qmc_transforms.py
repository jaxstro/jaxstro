import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad


def _fixed_method():
    return quad.ScrambledSobol(level=4, replicates=8)


def _adaptive_method():
    return quad.AdaptiveScrambledSobol(
        schedule=((3, 8), (4, 16)),
        estimate_bounds=(0.0, 10.0),
    )


def _solve(method, key, upper):
    return quad.integrate(
        lambda x: jnp.exp(jnp.sum(x, axis=-1)),
        quad.Hyperrectangle(jnp.zeros(2, dtype=upper.dtype), upper),
        method=method,
        key=key,
        epsabs=0.5,
        epsrel=0.0,
        max_evaluations=256,
        gradient="stop",
    )


@pytest.mark.parametrize("method", (_fixed_method(), _adaptive_method()))
def test_randomized_qmc_eager_and_jit_agree_to_roundoff(method):
    key = jax.random.key(3)
    upper = jnp.ones(2, dtype=jnp.float64)
    eager = _solve(method, key, upper)
    compiled = jax.jit(lambda runtime_key: _solve(method, runtime_key, upper))(key)
    assert jnp.array_equal(eager.value, compiled.value)
    assert jnp.allclose(
        eager.error.estimate,
        compiled.error.estimate,
        rtol=8.0 * jnp.finfo(eager.error.estimate.dtype).eps,
        atol=0.0,
    )
    assert jnp.array_equal(eager.status, compiled.status)
    assert jnp.array_equal(eager.work, compiled.work)


@pytest.mark.parametrize("method", (_fixed_method(), _adaptive_method()))
def test_randomized_qmc_vmaps_over_keys(method):
    keys = jax.random.split(jax.random.key(5), 4)
    upper = jnp.ones(2, dtype=jnp.float64)
    results = jax.jit(jax.vmap(lambda key: _solve(method, key, upper)))(keys)
    assert results.value.shape == (4,)
    assert jnp.all(jnp.isfinite(results.value))
    assert jnp.unique(results.value).shape[0] > 1


@pytest.mark.parametrize("method", (_fixed_method(), _adaptive_method()))
def test_randomized_qmc_vmaps_over_dynamic_domains(method):
    key = jax.random.key(7)
    uppers = jnp.asarray(((1.0, 1.0), (0.8, 1.2), (1.1, 0.7)))
    results = jax.jit(jax.vmap(lambda upper: _solve(method, key, upper)))(uppers)
    assert results.value.shape == (3,)
    assert jnp.all(jnp.isfinite(results.value))


def test_replicate_fold_in_identity_is_capacity_stable():
    key = jax.random.key(11)
    small = jax.vmap(lambda index: jax.random.fold_in(key, index))(
        jnp.arange(8, dtype=jnp.uint32)
    )
    grown = jax.vmap(lambda index: jax.random.fold_in(key, index))(
        jnp.arange(16, dtype=jnp.uint32)
    )
    assert jnp.array_equal(jax.random.key_data(small), jax.random.key_data(grown[:8]))


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_randomized_qmc_preserves_declared_domain_dtype(dtype):
    result = _solve(
        _fixed_method(),
        jax.random.key(13),
        jnp.ones(2, dtype=dtype),
    )
    assert result.value.dtype == dtype
    assert result.error.estimate.dtype == dtype


@pytest.mark.parametrize("method", (_fixed_method(), _adaptive_method()))
@pytest.mark.parametrize(
    "integrand",
    (
        lambda x: x,
        lambda x: (1.0 + 1.0j) * jnp.sum(x, axis=-1),
    ),
)
def test_randomized_qmc_rejects_array_and_complex_payloads(method, integrand):
    with pytest.raises(ValueError, match="scalar real"):
        quad.integrate(
            integrand,
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            method=method,
            key=jax.random.key(17),
            epsabs=0.1,
            epsrel=0.0,
            max_evaluations=256,
            gradient="stop",
        )


@pytest.mark.parametrize("method", (_fixed_method(), _adaptive_method()))
def test_randomized_qmc_stop_mode_has_zero_derivative(method):
    derivative = jax.grad(
        lambda upper: (
            _solve(
                method,
                jax.random.key(19),
                jnp.asarray((upper, 1.0)),
            ).value
        )
    )(1.0)
    assert derivative == 0.0


@pytest.mark.parametrize("method", (_fixed_method(), _adaptive_method()))
def test_randomized_method_configuration_is_static(method):
    leaves, structure = jax.tree.flatten(method)
    assert leaves == []
    assert jax.tree.unflatten(structure, leaves) == method
