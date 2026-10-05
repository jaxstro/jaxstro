"""jaxstro.optics.mie: analytic limits, batching, derivatives (reference comparison:
tests/validation/test_optics_mie_reference.py)."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.optics.mie import mie_efficiencies, mie_term_counts

pytestmark = pytest.mark.unit

MS = [1.5 + 0.01j, 1.33 + 1e-8j, 3.0 + 4.0j]


def _run(m, x):
    nt, ns = mie_term_counts(jnp.asarray(m), jnp.asarray(x))
    return mie_efficiencies(m, jnp.asarray(x), n_terms=nt, n_start=ns)


@pytest.mark.parametrize("m", MS)
def test_rayleigh_limit(m):
    """x << 1: Q_abs = 4 x Im L, Q_sca = (8/3) x^4 |L|^2 with L = (m^2-1)/(m^2+2); the
    leading corrections are O(x^2) (measured 5.6e-7 at x = 1e-3 for m = 1.5 + 0.01 i)."""
    x = jnp.array([1e-3, 3e-3])
    r = _run(m, x)
    lorentz = (m**2 - 1) / (m**2 + 2)
    assert jnp.all(jnp.abs(r.q_abs / (4 * x * lorentz.imag) - 1) <= 20 * x**2)
    assert jnp.all(jnp.abs(r.q_sca / (8 / 3 * x**4 * abs(lorentz) ** 2) - 1) <= 20 * x**2)
    assert jnp.all(jnp.abs(r.g) <= x)


def test_extinction_paradox_at_large_x():
    """Q_ext -> 2 as x -> infinity; 2.0043 at x = 1e4 for m = 1.5 + 0.01 i."""
    r = _run(1.5 + 0.01j, jnp.array([1e4]))
    assert abs(float(r.q_ext[0]) - 2.0) <= 0.01
    assert float(r.q_abs[0]) > 0.0 and 0.0 < float(r.g[0]) < 1.0


def test_batched_result_equals_single_runs():
    """A batch shares the static series of its largest x; masked terms and frozen
    recurrences must not change small-x elements."""
    m, x = 1.7 + 0.3j, jnp.array([1e-3, 0.5, 30.0, 900.0])
    batch = _run(m, x)
    for i in range(x.shape[0]):
        single = _run(m, x[i : i + 1])
        for k in ("q_ext", "q_sca", "g"):
            assert float(jnp.abs(getattr(batch, k)[i] / getattr(single, k)[0] - 1)) <= 1e-12


def test_gradients_match_finite_differences_and_forward_mode():
    def f(p):
        r = mie_efficiencies(p[0] + 1j * p[1], p[2], n_terms=40, n_start=80)
        return jnp.stack([r.q_abs, r.q_sca, r.g])

    p0 = jnp.array([1.7, 0.3, 7.3])
    rev, fwd = jax.jacrev(f)(p0), jax.jacfwd(f)(p0)
    h = 1e-6
    fd = jnp.stack([(f(p0 + h * e) - f(p0 - h * e)) / (2 * h) for e in jnp.eye(3)], axis=1)
    assert float(jnp.max(jnp.abs(rev - fwd))) <= 1e-14
    assert float(jnp.max(jnp.abs(rev / fd - 1))) <= 1e-6
    assert bool(jnp.array_equal(jax.jit(f)(p0), f(p0)))
