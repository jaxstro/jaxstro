"""neumaier_add returns the exact rounding error of the sum when compiled (jit, scan, vmap).

stellax carries its Newton iterate's luminosity as hi + lo built from this step
(stellax docs/plans/2026-10-03-compensated-luminosity.md); a compiler that reassociated or
contracted the error expression would make lo silently wrong.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np

from jaxstro.numerics.compensated import neumaier_add

jax.config.update("jax_enable_x64", True)


def _exact(s, c, y, t, c_new):
    """s + c + y == t + c_new in exact rational arithmetic."""
    return Fraction(float(s)) + Fraction(float(c)) + Fraction(float(y)) == \
        Fraction(float(t)) + Fraction(float(c_new))


def test_cancellation_case_is_exact_under_jit():
    t, c = jax.jit(neumaier_add)(jnp.float64(1e16), jnp.float64(0.0), jnp.float64(1.0))
    assert float(t) == 1e16 and float(c) == 1.0      # 1e16 + 1 is not a float64; c holds the 1


def test_error_is_exact_for_random_pairs_under_jit_vmap_and_scan():
    rng = np.random.default_rng(0)
    s = jnp.asarray(rng.normal(size=400) * 10.0 ** rng.uniform(-5, 33, 400))
    y = jnp.asarray(rng.normal(size=400) * 10.0 ** rng.uniform(-20, 20, 400))
    c0 = jnp.zeros(400)
    t, c = jax.jit(jax.vmap(neumaier_add))(s, c0, y)
    assert all(_exact(*v) for v in zip(np.asarray(s), np.asarray(c0), np.asarray(y),
                                       np.asarray(t), np.asarray(c)))

    # a running sum of 30 updates (a Newton iteration's worth), compiled as a scan
    steps = jnp.asarray(rng.normal(size=30) * 1e-6)

    def body(carry, d):
        return neumaier_add(*carry, d), None

    (hi, lo), _ = jax.jit(lambda: jax.lax.scan(body, (jnp.float64(3.7e33 / 3.828e33), jnp.float64(0.0)),
                                               steps))()
    exact = Fraction(3.7e33 / 3.828e33) + sum(Fraction(float(d)) for d in np.asarray(steps))
    assert abs(Fraction(float(hi)) + Fraction(float(lo)) - exact) <= abs(exact) * Fraction(1, 2 ** 100)
