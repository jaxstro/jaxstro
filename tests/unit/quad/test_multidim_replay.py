import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quad._multidim_replay import replay_formula_value
from jaxstro.quad._sparse import identities_to_points
from jaxstro.quad.integrate import _prepare_multidim_solve

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
        estimate_bounds=(0.0, 10.0),
    ),
)


@pytest.mark.parametrize("method", METHODS)
def test_stopped_formula_reproduces_primal_value(method):
    key = (
        jax.random.key(3)
        if isinstance(
            method,
            (quad.ScrambledSobol, quad.AdaptiveScrambledSobol),
        )
        else None
    )
    solve = _prepare_multidim_solve(
        lambda x: jnp.exp(jnp.sum(x, axis=-1)),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        method=method,
        key=key,
        args=(),
        measure=quad.LebesgueMeasure(),
        epsabs=1e-8,
        epsrel=1e-8,
        max_evaluations=4096,
        max_regions=64,
        max_indices=64,
        max_frontier=256,
        max_nodes=4096,
        error_norm=quad.MaxNorm(),
    )

    replayed = replay_formula_value(
        solve.config,
        solve.domain,
        solve.args,
        solve.formula,
    )

    assert solve.formula.reference_points.ndim == 2
    assert solve.formula.reference_points.shape[1] == 2
    assert solve.formula.reference_weights.shape == solve.formula.active_mask.shape
    assert solve.formula.reference_points.shape[0] == len(
        solve.formula.reference_weights
    )
    assert jnp.allclose(
        jnp.sum(
            jnp.where(
                solve.formula.active_mask,
                solve.formula.reference_weights,
                0.0,
            )
        ),
        1.0,
        rtol=2e-12,
        atol=2e-12,
    )
    assert jnp.allclose(replayed, solve.result.value, rtol=2e-12, atol=2e-12)


def test_replay_formula_uses_normalized_reference_weights():
    solve = _prepare_multidim_solve(
        lambda x: jnp.ones(x.shape[0]),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        method=quad.TensorProduct(quad.GaussianRule(3)),
        key=None,
        args=(),
        measure=quad.LebesgueMeasure(),
        epsabs=1e-8,
        epsrel=1e-8,
        max_evaluations=64,
        max_regions=8,
        max_indices=8,
        max_frontier=16,
        max_nodes=64,
        error_norm=quad.MaxNorm(),
    )

    assert jnp.allclose(
        jnp.sum(
            jnp.where(
                solve.formula.active_mask,
                solve.formula.reference_weights,
                0.0,
            )
        ),
        1.0,
        rtol=2e-14,
        atol=2e-14,
    )


def test_sparse_replay_uses_bit_exact_canonical_special_points():
    identities = jnp.asarray(
        [
            [[0, 0], [1, 1]],
            [[1, 0], [1, 1]],
        ],
        dtype=jnp.int32,
    )

    points = identities_to_points(identities, jnp.float64)

    assert jnp.array_equal(
        points,
        jnp.asarray([[0.0, 0.5], [1.0, 0.5]], dtype=jnp.float64),
    )
