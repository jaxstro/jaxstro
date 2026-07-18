import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro import quad

METHODS = (
    quad.TensorProduct(quad.GaussianRule(3)),
    quad.AdaptiveTensorClenshawCurtis(initial_level=2),
    quad.AdaptiveCubature(),
    quad.Smolyak(level=3),
    quad.AdaptiveSmolyak(initial_level=1),
    quad.Sobol(level=7),
    quad.ScrambledSobol(level=7, replicates=8),
    quad.AdaptiveScrambledSobol(
        schedule=((5, 8), (6, 16), (7, 16)),
        estimate_bounds=(-10.0, 10.0),
    ),
)


def controls(method):
    selected = {
        "epsabs": 1e-7,
        "epsrel": 1e-7,
        "max_evaluations": 4096,
        "gradient": "replay",
    }
    if isinstance(method, quad.AdaptiveCubature):
        selected["max_regions"] = 64
    if isinstance(method, (quad.Smolyak, quad.AdaptiveSmolyak)):
        selected.update(
            max_indices=64,
            max_frontier=256,
            max_nodes=4096,
        )
    if isinstance(method, (quad.ScrambledSobol, quad.AdaptiveScrambledSobol)):
        selected["key"] = jax.random.key(5)
    return selected


@pytest.mark.parametrize("method", METHODS)
def test_replay_gradient_matches_exact_accepted_formula_derivative(method):
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))

    def objective(scale):
        return quad.integrate(
            lambda x, live_scale: live_scale * jnp.sum(x, axis=-1),
            domain,
            args=scale,
            method=method,
            **controls(method),
        ).value

    assert jnp.allclose(
        jax.grad(objective)(2.0),
        objective(1.0),
        rtol=2e-12,
        atol=2e-12,
    )


def test_coincident_bound_replay_fails_closed():
    domain = quad.Hyperrectangle(
        jnp.array([0.0, 1.0]),
        jnp.array([2.0, 1.0]),
    )

    def objective(upper):
        return quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(domain.lower, upper),
            method=quad.TensorProduct(quad.GaussianRule(3)),
            epsabs=1e-8,
            epsrel=1e-8,
            max_evaluations=9,
            gradient="replay",
        )

    result, tangent = jax.jvp(
        objective,
        (domain.upper,),
        (jnp.ones(2),),
    )

    assert result.status == quad.QuadStatus.INVALID_INPUT
    assert not jnp.isfinite(tangent.value)
    assert np.asarray(tangent.status).dtype == jax.dtypes.float0


def test_inactive_padding_cannot_evaluate_a_singular_point():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))

    def objective(scale):
        return quad.integrate(
            lambda x, live_scale: live_scale
            * jnp.where(
                jnp.all(x == 0.0, axis=-1),
                jnp.asarray(jnp.nan),
                jnp.sum(x, axis=-1),
            ),
            domain,
            args=scale,
            method=quad.AdaptiveCubature(),
            epsabs=1e-5,
            epsrel=1e-5,
            max_evaluations=1024,
            max_regions=16,
            gradient="replay",
        ).value

    value, tangent = jax.jvp(objective, (2.0,), (1.0,))

    assert jnp.isfinite(value)
    assert jnp.isfinite(tangent)
