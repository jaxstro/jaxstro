"""Smooth (van Albada) slopes and linear face values for the split/merge transfer."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxstro.numerics import lagrangian_remap as lr

jax.config.update("jax_enable_x64", True)


def _ave(a, b, eps2):
    """van Albada, van Leer & Roberts 1982, A&A 108, 76, eq (37), as printed."""
    return ((b * b + eps2) * a + (a * a + eps2) * b) / (a * a + b * b + 2 * eps2)


def _one_sided(q, dm):
    """Changes across one cell width from the two one-sided centre-to-centre slopes."""
    h = 0.5 * (dm[1:] + dm[:-1])
    d = (q[1:] - q[:-1]) / h
    w = dm[1:-1]
    return d[:-1] * w, d[1:] * w


def test_slopes_are_eq_37_on_rms_scaled_differences():
    rng = np.random.default_rng(0)
    n, k = 12, 0.7
    dm = jnp.asarray(rng.uniform(0.3, 2.0, n))
    q = jnp.asarray(rng.normal(size=n))
    s = lr.smooth_slopes(q, dm, k)
    a, b = _one_sided(q, dm)
    rms = jnp.sqrt((q[:-2] ** 2 + q[1:-1] ** 2 + q[2:] ** 2) / 3)
    eps2 = (k * dm[1:-1] / jnp.sum(dm)) ** 3
    want = rms * _ave(a / rms, b / rms, eps2) / dm[1:-1]
    np.testing.assert_allclose(s[1:-1], want, rtol=1e-13)
    assert float(s[0]) == 0.0 and float(s[-1]) == 0.0


def test_relative_slopes_are_eq_37_on_eq_31_3_differences():
    rng = np.random.default_rng(1)
    n, k = 12, 2.0
    dm = jnp.asarray(rng.uniform(0.3, 2.0, n))
    q = jnp.asarray(rng.uniform(0.1, 5.0, n))
    s = lr.smooth_slopes(q, dm, k, relative=True)
    h = 0.5 * (dm[1:] + dm[:-1])
    rel = 2 * (q[1:] - q[:-1]) / (q[1:] + q[:-1])          # eq (31.3)
    w = dm[1:-1]
    a, b = rel[:-1] * w / h[:-1], rel[1:] * w / h[1:]
    eps2 = (k * w / jnp.sum(dm)) ** 3
    np.testing.assert_allclose(s[1:-1], q[1:-1] * _ave(a, b, eps2) / w, rtol=1e-13)


def test_slope_is_a_convex_combination_and_exact_for_equal_differences():
    dm = jnp.ones(5)
    q = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])                # linear, uniform: a = b
    np.testing.assert_allclose(lr.smooth_slopes(q, dm, 1.0)[1:-1], 1.0, rtol=1e-14)
    rng = np.random.default_rng(2)
    for _ in range(20):
        dm = jnp.asarray(rng.uniform(0.2, 3.0, 9))
        q = jnp.asarray(rng.normal(size=9))
        a, b = _one_sided(q, dm)
        cell = lr.smooth_slopes(q, dm, 0.5)[1:-1] * dm[1:-1]
        assert bool(jnp.all(jnp.abs(cell) <= jnp.maximum(jnp.abs(a), jnp.abs(b)) * (1 + 1e-12)))


def test_relative_slopes_keep_half_cell_children_positive():
    """|ave| < 4 for any neighbour mass ratio, so q (1 -/+ ave / 4) > 0."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = 16
        dm = jnp.asarray(10.0 ** rng.uniform(-2, 2, n))     # neighbour ratios up to 1e4
        q = jnp.asarray(10.0 ** rng.uniform(-30, 0, n))     # trace species to dominant
        s = lr.smooth_slopes(q, dm, 1.0, relative=True)
        half = s * dm / 4
        assert bool(jnp.all(q - jnp.abs(half) > 0))


def test_sum_to_zero_keeps_fractions_summing_to_one_and_conserves_each():
    rng = np.random.default_rng(4)
    n = 10
    dm = jnp.asarray(rng.uniform(0.5, 1.5, n))
    xa = jnp.asarray(rng.dirichlet(np.ones(4), n))
    s = lr.smooth_slopes(xa, dm, 1.0, relative=True, sum_to_zero=True)
    np.testing.assert_allclose(jnp.sum(s, axis=1), 0.0, atol=1e-15)
    plan = lr.plan_from_actions(jnp.full(n, lr.SPLIT), capacity=2 * n)
    xi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dm)])
    _, dm_new, new = lr.apply_plan(plan, xi, dm, {"xa": xa * dm[:, None]}, {"xa": s})
    child = new["xa"] / dm_new[:, None]
    np.testing.assert_allclose(jnp.sum(child, axis=1), 1.0, rtol=1e-14)
    np.testing.assert_allclose(new["xa"].reshape(n, 2, 4).sum(axis=1), xa * dm[:, None], rtol=1e-14)


def test_slopes_are_smooth_through_a_near_tie():
    """The case that made MESA's selection rough: two one-sided differences crossing at
    round-off size. The slope's derivative must be continuous across the crossing."""
    dm = jnp.ones(5)

    def slope(t):
        # b crosses a at t = 0 (the outer neighbour moves; a is fixed)
        q = jnp.array([1.0, 1.0 + 1e-14, 1.0 + 2e-14, 1.0 + 3e-14 + t, 1.0 + 4e-14])
        return lr.smooth_slopes(q, dm, 1.0, relative=True)[2]

    g = jax.vmap(jax.grad(slope))(jnp.linspace(-3e-14, 3e-14, 61))
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.min(g)) > 0.4                      # d slope / d q_out ~ 1/2 (central)
    assert float(jnp.max(jnp.abs(jnp.diff(g)))) < 1e-6


def test_face_values_are_linear_in_mass_and_exact_on_old_faces():
    n = 4
    dm = jnp.array([1.0, 2.0, 1.0, 3.0])
    f = jnp.array([0.0, 2.0, 5.0, 6.0, 9.5])                 # n + 1 faces, centre first
    # [cell 0], [inner half of 1], [outer half of 1 + cell 2], [cell 3]
    plan = lr.RemapPlan(start=jnp.array([0, 2, 3, 6]), length=jnp.array([2, 1, 3, 2]),
                        n_active=jnp.array(4))
    got = lr.face_values(plan, f)
    np.testing.assert_array_equal(got, [0.0, 2.0, 3.5, 6.0, 9.5])
    # quarter sub-cells: a face a quarter of the way through cell 1
    plan4 = lr.RemapPlan(start=jnp.array([0, 4, 5, 12]), length=jnp.array([4, 1, 7, 4]),
                         n_active=jnp.array(4), max_length=8, subcells=4)
    np.testing.assert_allclose(lr.face_values(plan4, f), [0.0, 2.0, 2.75, 6.0, 9.5])
