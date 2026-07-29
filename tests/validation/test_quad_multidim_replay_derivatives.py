import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro import quad
from jaxstro.quantity import dimensionless

METHODS = (
    quad.TensorProduct(quad.GaussianRule(3)),
    quad.AdaptiveTensorClenshawCurtis(initial_level=2),
    quad.AdaptiveCubature(),
    quad.Smolyak(level=3),
    quad.AdaptiveSmolyak(initial_level=1),
    quad.Sobol(level=5),
    quad.ScrambledSobol(level=5, replicates=8),
    quad.AdaptiveScrambledSobol(
        schedule=((3, 8), (4, 16)),
        estimate_bounds=(-10.0, 10.0),
    ),
)

DETERMINISTIC_METHODS = METHODS[:6]


def _controls(method, *, gradient="replay"):
    selected = {
        "epsabs": 1e-6,
        "epsrel": 1e-6,
        "max_evaluations": 512,
        "gradient": gradient,
    }
    if isinstance(method, quad.AdaptiveCubature):
        selected["max_regions"] = 16
    if isinstance(method, (quad.Smolyak, quad.AdaptiveSmolyak)):
        selected.update(
            max_indices=16,
            max_frontier=33,
            max_nodes=256,
        )
    if isinstance(method, (quad.ScrambledSobol, quad.AdaptiveScrambledSobol)):
        selected["key"] = jax.random.key(17)
    return selected


def _scalar_objective(method, scale):
    return quad.integrate(
        lambda x, live_scale: live_scale * (1.0 + 0.1 * jnp.sum(x, axis=-1)),
        quad.Hyperrectangle(jnp.asarray([-0.2, 0.1]), jnp.asarray([1.1, 1.3])),
        args=scale,
        method=method,
        **_controls(method),
    ).value


@pytest.mark.parametrize("method", METHODS)
def test_complete_first_order_transform_matrix(method):
    def objective(scale):
        return _scalar_objective(method, scale)

    scale = jnp.asarray(1.7)
    expected = objective(1.0)

    eager = objective(scale)
    _, jvp = jax.jvp(objective, (scale,), (jnp.asarray(1.0),))
    primal, pullback = jax.vjp(objective, scale)
    vjp = pullback(jnp.asarray(1.0))[0]
    grad = jax.grad(objective)(scale)
    jacfwd = jax.jacfwd(objective)(scale)
    jacrev = jax.jacrev(objective)(scale)
    jit_grad = jax.jit(jax.grad(objective))(scale)
    scales = jnp.asarray([0.7, 1.1, 1.9])
    vmap_grad = jax.vmap(jax.grad(objective))(scales)
    jit_vmap_grad = jax.jit(jax.vmap(jax.grad(objective)))(scales)

    assert jnp.allclose(eager, scale * expected, rtol=2e-11, atol=2e-11)
    assert jnp.allclose(primal, eager, rtol=2e-12, atol=2e-12)
    for derivative in (jvp, vjp, grad, jacfwd, jacrev, jit_grad):
        assert jnp.allclose(derivative, expected, rtol=2e-11, atol=2e-11)
    assert jnp.allclose(vmap_grad, expected, rtol=2e-11, atol=2e-11)
    assert jnp.allclose(jit_vmap_grad, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("method", METHODS)
def test_moving_upper_bound_gradient_matches_adaptive_rerun_finite_difference(
    method,
):
    lower = jnp.asarray([-0.3, 0.2])

    def objective(first_upper):
        upper = jnp.stack((first_upper, jnp.asarray(1.4)))
        return quad.integrate(
            lambda x: 1.0 + 0.2 * jnp.sum(x, axis=-1),
            quad.Hyperrectangle(lower, upper),
            method=method,
            **_controls(method),
        ).value

    location = jnp.asarray(1.2)
    step = jnp.asarray(2e-5)
    derivative = jax.grad(objective)(location)
    finite_difference = (objective(location + step) - objective(location - step)) / (
        2.0 * step
    )

    assert jnp.allclose(
        derivative,
        finite_difference,
        rtol=3e-5,
        atol=3e-5,
    )


def test_explicit_parameter_pytree_replay_derivative():
    method = quad.TensorProduct(quad.GaussianRule(4))
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))

    def objective(parameters):
        return quad.integrate(
            lambda x, live: live["amplitude"] * (live["offset"] + x @ live["slope"]),
            domain,
            args=parameters,
            method=method,
            **_controls(method),
        ).value

    parameters = {
        "amplitude": jnp.asarray(1.5),
        "offset": jnp.asarray(0.7),
        "slope": jnp.asarray([0.2, -0.4]),
    }
    derivative = jax.grad(objective)(parameters)

    assert derivative["amplitude"] == pytest.approx(0.6)
    assert derivative["offset"] == pytest.approx(1.5)
    assert jnp.allclose(derivative["slope"], jnp.asarray([0.75, 0.75]))


def test_weighted_measure_density_and_integrand_share_live_args():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    measure = quad.WeightedMeasure(
        lambda x, live: 1.0 + live["density_slope"] * x[:, 0],
        density_unit=dimensionless,
    )

    def objective(parameters):
        return quad.integrate(
            lambda x, live: live["scale"] * jnp.sum(x, axis=-1),
            domain,
            args=parameters,
            measure=measure,
            method=quad.TensorProduct(quad.GaussianRule(4)),
            **_controls(quad.TensorProduct(quad.GaussianRule(4))),
        ).value

    parameters = {
        "scale": jnp.asarray(2.0),
        "density_slope": jnp.asarray(0.3),
    }
    derivative = jax.grad(objective)(parameters)

    assert derivative["scale"] == pytest.approx(1.0 + 0.3 * 7.0 / 12.0)
    assert derivative["density_slope"] == pytest.approx(2.0 * 7.0 / 12.0)


@pytest.mark.parametrize("method", DETERMINISTIC_METHODS)
def test_deterministic_replay_supports_array_and_complex_payloads(method):
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))

    def array_objective(scale):
        return quad.integrate(
            lambda x, live_scale: (
                live_scale
                * jnp.stack(
                    (
                        jnp.sum(x, axis=-1),
                        jnp.prod(x, axis=-1),
                    ),
                    axis=-1,
                )
            ),
            domain,
            args=scale,
            method=method,
            **_controls(method),
        ).value

    _, array_tangent = jax.jvp(
        array_objective,
        (jnp.asarray(2.0),),
        (jnp.asarray(1.0),),
    )
    expected_array = array_objective(1.0)
    assert jnp.allclose(
        array_tangent,
        expected_array,
        rtol=2e-11,
        atol=2e-11,
    )

    def complex_objective(scale):
        return quad.integrate(
            lambda x, live_scale: (
                (live_scale + 1j * live_scale**2) * jnp.sum(x, axis=-1)
            ),
            domain,
            args=scale,
            method=method,
            **_controls(method),
        ).value

    _, complex_tangent = jax.jvp(
        complex_objective,
        (jnp.asarray(2.0),),
        (jnp.asarray(1.0),),
    )
    base = quad.integrate(
        lambda x: jnp.sum(x, axis=-1),
        domain,
        method=method,
        **_controls(method, gradient="stop"),
    ).value
    assert jnp.allclose(
        complex_tangent,
        (1.0 + 4.0j) * base,
        rtol=2e-11,
        atol=2e-11,
    )


def test_diagnostic_tangents_are_exact_zero_or_float0():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))

    def solve(scale):
        return quad.integrate(
            lambda x, live_scale: live_scale * jnp.sum(x, axis=-1),
            domain,
            args=scale,
            method=quad.TensorProduct(quad.GaussianRule(3)),
            **_controls(quad.TensorProduct(quad.GaussianRule(3))),
        )

    _result, tangent = jax.jvp(
        solve,
        (jnp.asarray(2.0),),
        (jnp.asarray(1.0),),
    )
    diagnostic = tangent._replace(value=jnp.asarray(0.0))

    for leaf in jax.tree.leaves(diagnostic):
        array = np.asarray(leaf)
        if array.dtype == jax.dtypes.float0:
            continue
        assert np.array_equal(array, np.zeros_like(array))


def test_nested_forward_and_reverse_derivatives_fail_explicitly():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))

    def objective(scale):
        return quad.integrate(
            lambda x, live_scale: live_scale**2 * jnp.sum(x, axis=-1),
            domain,
            args=scale,
            method=quad.TensorProduct(quad.GaussianRule(3)),
            **_controls(quad.TensorProduct(quad.GaussianRule(3))),
        ).value

    with pytest.raises(
        ValueError,
        match="multidimensional replay supports first derivatives only",
    ):
        jax.grad(jax.grad(objective))(jnp.asarray(2.0))

    for higher_derivative in (
        jax.hessian(objective),
        jax.jacfwd(jax.jacrev(objective)),
        jax.jacrev(jax.jacfwd(objective)),
    ):
        with pytest.raises(
            ValueError,
            match="multidimensional replay supports first derivatives only",
        ):
            higher_derivative(jnp.asarray(2.0))

    def first_jvp(scale):
        return jax.jvp(
            objective,
            (scale,),
            (jnp.asarray(1.0),),
        )[1]

    with pytest.raises(
        ValueError,
        match="multidimensional replay supports first derivatives only",
    ):
        jax.jvp(
            first_jvp,
            (jnp.asarray(2.0),),
            (jnp.asarray(1.0),),
        )
