"""JIT, VMAP, payload, and stopped-derivative adaptive contracts."""

import jax
import jax.numpy as jnp

from jaxstro.quad import GaussKronrod, Interval, MaxNorm, QuadStatus, integrate

OPTIONS = dict(
    method=GaussKronrod(pair=21),
    max_evaluations=1024,
    max_regions=32,
    error_norm=MaxNorm(),
)


def test_integrate_jit_compiles_dynamic_bounds_args_and_tolerances() -> None:
    evaluate = jax.jit(
        lambda lower, upper, scale, epsabs: integrate(
            lambda x, args: jnp.exp(args * x),
            Interval(lower, upper),
            args=scale,
            epsabs=epsabs,
            epsrel=1e-10,
            **OPTIONS,
        )
    )
    result = evaluate(0.0, 1.0, 0.2, 1e-10)
    assert result.status == QuadStatus.CONVERGED
    assert jnp.allclose(result.value, jnp.expm1(0.2) / 0.2, rtol=2e-11)


def test_integrate_vmap_tracks_logical_per_lane_work() -> None:
    evaluate = jax.vmap(
        lambda scale: integrate(
            lambda x, args: jnp.exp(args * x),
            Interval(0.0, 1.0),
            args=scale,
            epsabs=1e-10,
            epsrel=1e-10,
            **OPTIONS,
        )
    )
    result = evaluate(jnp.asarray([0.1, 0.2, 0.3]))
    assert result.value.shape == (3,)
    assert jnp.all(result.status == QuadStatus.CONVERGED)
    assert jnp.all(result.work.evaluations == 21)


def test_integrate_jit_reports_dynamic_invalid_input() -> None:
    evaluate = jax.jit(
        lambda breakpoint: integrate(
            lambda x: x,
            Interval(0.0, 1.0, breakpoints=(breakpoint,)),
            epsabs=1e-10,
            epsrel=1e-10,
            **OPTIONS,
        )
    )
    assert evaluate(0.5).status == QuadStatus.CONVERGED
    assert evaluate(2.0).status == QuadStatus.INVALID_INPUT


def test_integrate_stop_policy_zeroes_parameter_and_bound_derivatives() -> None:
    parameter_derivative = jax.grad(
        lambda scale: (
            integrate(
                lambda x, args: jnp.exp(args * x),
                Interval(0.0, 1.0),
                args=scale,
                epsabs=1e-10,
                epsrel=1e-10,
                **OPTIONS,
            ).value
        )
    )(0.2)
    bound_derivative = jax.grad(
        lambda upper: (
            integrate(
                lambda x: x**2,
                Interval(0.0, upper),
                epsabs=1e-10,
                epsrel=1e-10,
                **OPTIONS,
            ).value
        )
    )(1.0)
    assert jnp.array_equal(parameter_derivative, 0.0)
    assert jnp.array_equal(bound_derivative, 0.0)


def test_integrate_stop_policy_zeroes_every_result_leaf_in_both_ad_modes() -> None:
    def evaluate(scale):
        return integrate(
            lambda x, args: jnp.stack((jnp.exp(args * x), x**2), axis=-1),
            Interval(0.0, 1.0),
            args=scale,
            epsabs=1e-10,
            epsrel=1e-10,
            **OPTIONS,
        )

    _, tangent = jax.jvp(evaluate, (0.2,), (1.0,))
    for leaf in jax.tree.leaves(tangent):
        if leaf.dtype == jax.dtypes.float0:
            continue
        assert jnp.all(leaf == 0.0)

    def all_inexact_evidence(scale):
        terms = []
        for leaf in jax.tree.leaves(evaluate(scale)):
            if jnp.issubdtype(leaf.dtype, jnp.inexact):
                terms.append(jnp.sum(jnp.nan_to_num(jnp.real(leaf))))
        return sum(terms)

    assert jnp.array_equal(jax.grad(all_inexact_evidence)(0.2), 0.0)
