"""Global Romberg and Romberg-tanh-sinh refinement contracts."""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from jaxstro.quad import (
    ErrorKind,
    Infinite,
    Interval,
    MaxNorm,
    QuadStatus,
    RightInfinite,
    Romberg,
    RombergTanhSinh,
    integrate,
)
from jaxstro.quad._romberg import (
    _gamma,
    _richardson_error,
    _tanh_sinh_tables,
)
from jaxstro.quad._tanh_sinh import _tanh_sinh_lattice_data


def _options(method, **overrides):
    options = dict(
        method=method,
        epsabs=1e-10,
        epsrel=1e-10,
        max_evaluations=1025,
        max_regions=1,
        error_norm=MaxNorm(),
    )
    options.update(overrides)
    return options


def test_romberg_richardson_exactness_and_zero_based_work_semantics() -> None:
    result = integrate(
        lambda x: x**2,
        Interval(-1.0, 1.0),
        **_options(Romberg(initial_level=1)),
    )
    assert result.status == QuadStatus.CONVERGED
    assert jnp.allclose(result.value, 2.0 / 3.0, atol=2e-13)
    assert result.error.kind == ErrorKind.REFINEMENT_DIFFERENCE
    assert result.work.evaluations == 5
    assert result.work.levels == 3
    assert result.work.refinements == 2
    assert result.work.active_regions == 1


@pytest.mark.parametrize(
    ("degree", "evaluations", "level"), [(2, 5, 2), (4, 9, 3), (6, 17, 4)]
)
def test_romberg_exact_work_advances_one_complete_grid_at_a_time(
    degree, evaluations, level
) -> None:
    result = integrate(
        lambda x: x**degree,
        Interval(-1.0, 1.0),
        **_options(Romberg(initial_level=1)),
    )
    assert result.status == QuadStatus.CONVERGED
    assert result.work.evaluations == evaluations == 2**level + 1
    assert result.work.refinements == level
    assert result.work.levels == level + 1


def test_romberg_supports_vector_and_complex_payloads() -> None:
    result = integrate(
        lambda x: jnp.stack((x**2 + 1j * x, x**4), axis=-1),
        Interval(-1.0, 1.0),
        **_options(Romberg(initial_level=2)),
    )
    assert result.status == QuadStatus.CONVERGED
    assert jnp.allclose(result.value, jnp.asarray([2.0 / 3.0, 2.0 / 5.0]))
    assert result.error.estimate.shape == (2,)


def test_romberg_capacity_nonfinite_and_structure_statuses() -> None:
    limited = integrate(
        lambda x: jnp.abs(x - 0.123),
        Interval(-1.0, 1.0),
        **_options(
            Romberg(initial_level=1),
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=5,
        ),
    )
    nonfinite = integrate(
        lambda x: jnp.where(x == 1.0, jnp.nan, x),
        Interval(-1.0, 1.0),
        **_options(Romberg(initial_level=1)),
    )
    assert limited.status == QuadStatus.MAX_EVALUATIONS
    assert limited.work.evaluations == 5
    assert nonfinite.status == QuadStatus.NONFINITE_INTEGRAND
    with pytest.raises(ValueError, match="breakpoints"):
        integrate(
            lambda x: x,
            Interval(-1.0, 1.0, breakpoints=(0.0,)),
            **_options(Romberg()),
        )
    with pytest.raises(TypeError, match="finite Interval"):
        integrate(lambda x: x, Infinite(), **_options(Romberg()))


@pytest.mark.parametrize("method", [Romberg(), RombergTanhSinh()])
def test_global_methods_validate_before_tracing_and_use_zero_fast_path(method) -> None:
    def must_not_trace(_x):
        raise AssertionError("invalid capacity must not trace user code")

    with pytest.raises(ValueError, match="initial"):
        integrate(
            must_not_trace,
            Interval(0.0, 1.0),
            **_options(method, max_evaluations=1),
        )
    with pytest.raises(ValueError, match="positive integer"):
        integrate(
            must_not_trace,
            Interval(0.0, 1.0),
            **_options(method, max_regions=0),
        )
    zero = integrate(
        lambda x: jnp.full_like(x, jnp.nan),
        Interval(2.0, 2.0),
        **_options(method),
    )
    assert zero.status == QuadStatus.CONVERGED
    assert zero.value == 0.0
    assert zero.work.evaluations == 0
    assert zero.work.refinements == 0
    assert zero.work.levels == 0
    assert zero.work.active_regions == 0


@pytest.mark.parametrize("method", [Romberg(), RombergTanhSinh()])
def test_global_methods_report_dynamic_invalid_tolerance_and_domain(method) -> None:
    evaluate_tolerance = jax.jit(
        lambda tolerance: (
            integrate(
                lambda x: x**2,
                Interval(0.0, 1.0),
                **_options(method, epsabs=tolerance, epsrel=tolerance),
            ).status
        )
    )
    assert evaluate_tolerance(-1.0) == QuadStatus.INVALID_INPUT
    assert evaluate_tolerance(jnp.nan) == QuadStatus.INVALID_INPUT
    assert evaluate_tolerance(jnp.inf) == QuadStatus.INVALID_INPUT

    if isinstance(method, Romberg):
        evaluate_domain = jax.jit(
            lambda upper: (
                integrate(
                    lambda x: x,
                    Interval(0.0, upper),
                    **_options(method),
                ).status
            )
        )
    else:
        evaluate_domain = jax.jit(
            lambda lower: (
                integrate(
                    lambda x: jnp.exp(-x),
                    RightInfinite(lower),
                    **_options(method),
                ).status
            )
        )
    assert evaluate_domain(jnp.inf) == QuadStatus.INVALID_INPUT


def test_romberg_tanh_sinh_supports_finite_and_improper_domains() -> None:
    finite = integrate(
        jnp.exp,
        Interval(-1.0, 1.0),
        **_options(RombergTanhSinh(initial_level=1), epsabs=1e-8, epsrel=1e-8),
    )
    improper = integrate(
        lambda x: jnp.exp(-x),
        RightInfinite(0.0),
        **_options(RombergTanhSinh(initial_level=1), epsabs=1e-8, epsrel=1e-8),
    )
    assert finite.status in (QuadStatus.CONVERGED, QuadStatus.ROUNDOFF_LIMITED)
    assert improper.status in (QuadStatus.CONVERGED, QuadStatus.ROUNDOFF_LIMITED)
    assert jnp.allclose(finite.value, jnp.e - 1.0 / jnp.e, rtol=2e-7)
    assert jnp.allclose(improper.value, 1.0, rtol=2e-7)
    assert finite.work.evaluations > 0
    assert finite.work.levels == finite.work.refinements + 1
    expected_work = _tanh_sinh_lattice_data(
        int(finite.work.refinements), dtype=jnp.float64
    ).compact_nodes.shape[0]
    assert finite.work.evaluations == expected_work
    if finite.status == QuadStatus.ROUNDOFF_LIMITED:
        assert finite.error.norm > finite.tolerance


def test_romberg_tanh_sinh_capacity_nonfinite_and_active_work() -> None:
    initial_level = 1
    initial_work = _tanh_sinh_lattice_data(
        initial_level, dtype=jnp.float64
    ).compact_nodes.shape[0]
    limited = integrate(
        lambda x: jnp.abs(x - 0.123),
        Interval(-1.0, 1.0),
        **_options(
            RombergTanhSinh(initial_level=initial_level),
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=initial_work,
        ),
    )
    nonfinite = integrate(
        lambda x: jnp.where(x > 0.9, jnp.nan, 1.0),
        Interval(-1.0, 1.0),
        **_options(RombergTanhSinh(initial_level=initial_level)),
    )
    assert limited.status == QuadStatus.MAX_EVALUATIONS
    assert limited.work.evaluations == initial_work
    assert nonfinite.status == QuadStatus.NONFINITE_INTEGRAND


def test_romberg_propagated_floor_prevents_cancellation_false_success() -> None:
    synthetic = _richardson_error(
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        jnp.asarray(0.25),
        jnp.asarray(0.5),
    )
    assert synthetic == 0.75
    result = integrate(
        lambda x: 1e16 * x + 1.0,
        Interval(-1.0, 1.0),
        **_options(Romberg(), epsabs=1e-12, epsrel=1e-12, max_evaluations=65),
    )
    assert result.status != QuadStatus.CONVERGED
    assert result.error.norm > result.tolerance


def test_romberg_tanh_sinh_exhaustion_and_error_decomposition() -> None:
    result = integrate(
        lambda x: (1.0 - x**2) ** (-0.5),
        Interval(-1.0, 1.0),
        **_options(
            RombergTanhSinh(initial_level=1),
            epsabs=1e-8,
            epsrel=0.0,
        ),
    )
    assert result.status == QuadStatus.ROUNDOFF_LIMITED
    assert result.error.norm > result.tolerance
    assert result.work.refinements == 6
    assert result.work.evaluations == 407

    tables = _tanh_sinh_tables(1, 1025, "float64")
    _, nodes, weights, densities, _, terminal, counts, exhausted = tables
    level = int(result.work.refinements)
    values = (1.0 - jnp.asarray(nodes) ** 2) ** (-0.5)
    high_weights = jnp.asarray(weights[level])
    low_weights = jnp.asarray(weights[level - 1])
    high = jnp.sum(values * high_weights)
    low = jnp.sum(values * low_weights)
    summation = _gamma(counts[level], jnp.float64) * jnp.sum(
        jnp.abs(values) * high_weights
    ) + _gamma(counts[level - 1], jnp.float64) * jnp.sum(jnp.abs(values) * low_weights)
    core = jnp.abs(high - low) + summation
    tail = jnp.sum(
        jnp.where(
            jnp.asarray(terminal[level]),
            jnp.abs(values * jnp.asarray(densities[level])),
            0.0,
        )
    )
    assert exhausted[level]
    assert core <= result.tolerance < tail
    assert jnp.allclose(result.error.estimate, core + tail)


def test_romberg_tanh_sinh_supports_complex_vector_payload() -> None:
    result = integrate(
        lambda x: jnp.stack((jnp.exp(1j * x), x**2), axis=-1),
        Interval(-1.0, 1.0),
        **_options(RombergTanhSinh(), epsabs=1e-8, epsrel=1e-8),
    )
    assert result.status in (QuadStatus.CONVERGED, QuadStatus.ROUNDOFF_LIMITED)
    assert jnp.allclose(result.value, jnp.asarray([2.0 * jnp.sin(1.0), 2.0 / 3.0]))
    assert result.error.estimate.shape == (2,)


def test_romberg_tanh_sinh_float32_status_and_work_in_subprocess() -> None:
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    program = """
import jax.numpy as jnp
from jaxstro.quad import Interval, MaxNorm, QuadStatus, RombergTanhSinh, integrate
from jaxstro.quad._tanh_sinh import _tanh_sinh_lattice_data

result = integrate(
    jnp.exp,
    Interval(-1.0, 1.0),
    method=RombergTanhSinh(initial_level=1),
    epsabs=1e-4,
    epsrel=1e-4,
    max_evaluations=257,
    max_regions=1,
    error_norm=MaxNorm(),
)
assert result.status == QuadStatus.CONVERGED
assert jnp.allclose(result.value, jnp.e - 1.0 / jnp.e, rtol=2e-5)
expected = _tanh_sinh_lattice_data(int(result.work.refinements), dtype=jnp.float32)
assert result.work.evaluations == expected.compact_nodes.shape[0]
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_romberg_jaxpr_loop_structure_is_budget_invariant() -> None:
    def trace(method, budget):
        return str(
            jax.make_jaxpr(
                lambda: integrate(
                    lambda x: x**2,
                    Interval(-1.0, 1.0),
                    **_options(method, max_evaluations=budget),
                )
            )()
        )

    for method in (Romberg(), RombergTanhSinh()):
        small = trace(method, 65)
        large = trace(method, 1025)
        expected_scans = 5 if isinstance(method, Romberg) else 2
        expected_conds = 5 if isinstance(method, Romberg) else 3
        assert small.count("while[") == large.count("while[") == 1
        assert small.count("scan[") == large.count("scan[") == expected_scans
        assert small.count("cond[") == large.count("cond[") == expected_conds
        assert small.count("integer_pow") == large.count("integer_pow") == 1


def test_romberg_tanh_sinh_rejects_breakpoints() -> None:
    with pytest.raises(ValueError, match="breakpoints"):
        integrate(
            lambda x: x,
            Interval(-1.0, 1.0, breakpoints=(0.0,)),
            **_options(RombergTanhSinh()),
        )


def test_romberg_families_compile_vmap_and_stop_gradients() -> None:
    for method in (Romberg(initial_level=1), RombergTanhSinh(initial_level=1)):
        evaluate = jax.jit(
            jax.vmap(
                lambda scale: integrate(
                    lambda x, args: jnp.exp(args * x),
                    Interval(0.0, 1.0),
                    args=scale,
                    **_options(method, epsabs=1e-7, epsrel=1e-7),
                )
            )
        )
        result = evaluate(jnp.asarray([0.1, 0.2]))
        assert result.value.shape == (2,)
        assert jnp.all(result.status == QuadStatus.CONVERGED)
        derivative = jax.grad(
            lambda scale: (
                integrate(
                    lambda x, args: jnp.exp(args * x),
                    Interval(0.0, 1.0),
                    args=scale,
                    **_options(method, epsabs=1e-7, epsrel=1e-7),
                ).value
            )
        )(0.2)
        assert derivative == 0.0
