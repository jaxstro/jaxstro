from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad


def _controls(method):
    if isinstance(method, quad.AdaptiveSmolyak):
        return dict(
            max_evaluations=512,
            max_indices=12,
            max_frontier=25,
            max_nodes=512,
        )
    return dict(
        max_evaluations=128,
        max_indices=16,
        max_frontier=16,
        max_nodes=128,
    )


def _integrate(fun, lower, upper, method, *, gradient="stop"):
    return quad.integrate(
        fun,
        quad.Hyperrectangle(lower, upper),
        method=method,
        epsabs=1.0e-6,
        epsrel=1.0e-6,
        gradient=gradient,
        **_controls(method),
    )


PAYLOADS: tuple[tuple[str, Callable], ...] = (
    ("scalar", lambda x: jnp.exp(-jnp.sum(x, axis=-1))),
    (
        "array",
        lambda x: jnp.stack(
            (
                jnp.exp(-jnp.sum(x, axis=-1)),
                jnp.prod(1.0 + x**2, axis=-1),
            ),
            axis=-1,
        ),
    ),
    (
        "complex",
        lambda x: (1.0 + 2.0j) * jnp.exp(-jnp.sum(x, axis=-1)),
    ),
)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
@pytest.mark.parametrize(("payload_id", "fun"), PAYLOADS, ids=lambda value: str(value))
@pytest.mark.parametrize(
    "method",
    (quad.Smolyak(3), quad.AdaptiveSmolyak()),
    ids=("fixed", "adaptive"),
)
def test_sparse_stop_mode_eager_and_jit_payload_matrix(
    dtype,
    payload_id,
    fun,
    method,
):
    del payload_id
    lower = jnp.zeros(2, dtype=dtype)
    upper = jnp.ones(2, dtype=dtype)
    eager = _integrate(fun, lower, upper, method)
    compiled = jax.jit(lambda lo, hi: _integrate(fun, lo, hi, method))(
        lower,
        upper,
    )
    assert jnp.allclose(compiled.value, eager.value, rtol=2.0e-5, atol=2.0e-6)
    assert jnp.allclose(
        compiled.error.estimate,
        eager.error.estimate,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert compiled.status == eager.status
    assert compiled.work == eager.work
    assert compiled.error.kind == quad.ErrorKind.SPARSE_GRID_SURPLUS


@pytest.mark.parametrize(
    "method",
    (quad.Smolyak(3), quad.AdaptiveSmolyak()),
    ids=("fixed", "adaptive"),
)
def test_sparse_vmap_matches_scalar_lanes_with_heterogeneous_work(method):
    upper = jnp.asarray(((1.0, 1.0), (4.0, 0.2), (0.3, 3.0)))
    lower = jnp.zeros_like(upper)

    def solve(lo, hi):
        return _integrate(
            lambda x: jnp.exp(-x[:, 0]) * jnp.cos(3.0 * x[:, 1]),
            lo,
            hi,
            method,
        )

    batched = jax.jit(jax.vmap(solve))(lower, upper)
    scalar = [solve(lower[row], upper[row]) for row in range(upper.shape[0])]
    assert jnp.allclose(
        batched.value,
        jnp.stack([result.value for result in scalar]),
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert jnp.array_equal(
        batched.status,
        jnp.stack([result.status for result in scalar]),
    )
    assert jnp.array_equal(
        batched.work.evaluations,
        jnp.stack([result.work.evaluations for result in scalar]),
    )
    if isinstance(method, quad.AdaptiveSmolyak):
        assert jnp.unique(batched.work.evaluations).size > 1


@pytest.mark.parametrize(
    "method",
    (quad.Smolyak(3), quad.AdaptiveSmolyak()),
    ids=("fixed", "adaptive"),
)
def test_sparse_orientation_and_stop_gradient_contract(method):
    lower = jnp.asarray((2.0, -1.0))
    upper = jnp.asarray((-1.0, 3.0))
    forward = _integrate(
        lambda x: jnp.ones(x.shape[0]),
        jnp.minimum(lower, upper),
        jnp.maximum(lower, upper),
        method,
    )
    reversed_axis = _integrate(
        lambda x: jnp.ones(x.shape[0]),
        lower,
        upper,
        method,
    )
    assert reversed_axis.value == pytest.approx(-forward.value)

    gradient = jax.grad(
        lambda hi: jnp.real(
            _integrate(
                lambda x: jnp.exp(-jnp.sum(x, axis=-1)),
                jnp.zeros(2),
                hi,
                method,
            ).value
        )
    )(jnp.ones(2))
    assert jnp.array_equal(gradient, jnp.zeros(2))


@pytest.mark.parametrize(
    "method",
    (quad.Smolyak(3), quad.AdaptiveSmolyak()),
    ids=("fixed", "adaptive"),
)
def test_sparse_replay_fails_closed_at_the_phase_b4_boundary(method):
    with pytest.raises(
        ValueError,
        match='supports only gradient="stop" in Phase B2',
    ):
        _integrate(
            lambda x: jnp.sum(x, axis=-1),
            jnp.zeros(2),
            jnp.ones(2),
            method,
            gradient="replay",
        )
