"""apply_cuts: a Lagrangian remap onto faces at arbitrary mass fractions of the old cells."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxstro.numerics import lagrangian_remap as lr

jax.config.update("jax_enable_x64", True)


def _mesh(n=12, seed=0):
    rng = np.random.default_rng(seed)
    dm = jnp.asarray(rng.uniform(0.5, 2.0, n))
    xi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.asarray(rng.uniform(0.3, 1.5, n)))])
    q = jnp.asarray(rng.uniform(1.0, 3.0, (n, 3)))              # specific values, 3 components
    return dm, xi, q


def _cuts(points, cap):
    """Faces from a sorted list of (cell, frac), padded to cap + 1 with the last face."""
    pts = list(points) + [points[-1]] * (cap + 1 - len(points))
    return (jnp.asarray([p[0] for p in pts], jnp.int32), jnp.asarray([p[1] for p in pts], jnp.float64))


def test_cuts_on_every_old_face_copy_every_cell_bitwise():
    dm, xi, q = _mesh()
    n = dm.shape[0]
    cell, frac = _cuts([(i, 0.0) for i in range(n + 1)], n)
    slope = lr.smooth_slopes(q, dm, 1e-4)
    xi_n, dm_n, tot, fits = lr.apply_cuts(cell, frac, n, xi, dm, q * dm[:, None], slope * 0 + slope,
                                          lr.smooth_slopes(jnp.diff(xi) / dm, dm, 1e-4))
    assert bool(fits)
    assert jnp.array_equal(xi_n, xi) and jnp.array_equal(dm_n, dm)
    assert jnp.array_equal(tot, q * dm[:, None])


def test_half_cuts_match_the_lattice_remap():
    """Every cell split at its half-mass point: the same contents as apply_plan with subcells 2."""
    dm, xi, q = _mesh()
    n = dm.shape[0]
    pts = [(0, 0.0)]
    for i in range(n):
        pts += [(i, 0.5), (i + 1, 0.0)]
    cell, frac = _cuts(pts, 2 * n)
    slope = lr.smooth_slopes(q, dm, 1e-4)
    vs = lr.smooth_slopes(jnp.diff(xi) / dm, dm, 1e-4)
    xc, dc, tc, fits = lr.apply_cuts(cell, frac, 2 * n, xi, dm, q * dm[:, None], slope, vs)
    plan = lr.plan_from_actions(jnp.full(n, lr.SPLIT), 2 * n)
    xp, dp, tp = lr.apply_plan(plan, xi, dm, q * dm[:, None], slope, vs)
    assert bool(fits)
    np.testing.assert_allclose(np.asarray(dc), np.asarray(dp), rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.asarray(tc), np.asarray(tp), rtol=1e-13, atol=0)
    np.testing.assert_allclose(np.asarray(xc), np.asarray(xp), rtol=1e-14, atol=0)


def test_arbitrary_cuts_conserve_and_reproduce_a_linear_profile():
    """Cuts at random fractions, including 15 halvings of the outer cell: mass, volume and contents
    conserved; a specific value linear in mass with its exact slope is reproduced exactly."""
    n = 10
    dm = jnp.full(n, 1.0)
    xi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.full(n, 2.0))])
    m_mid = jnp.cumsum(dm) - 0.5 * dm
    q = (3.0 + 0.2 * m_mid)[:, None]                             # linear in mass, slope 0.2
    slope = jnp.full((n, 1), 0.2)
    rng = np.random.default_rng(1)
    pts = [(0, 0.0)]
    for i in range(n - 1):
        f = float(rng.uniform(0.1, 0.9))
        pts += [(i, f), (i + 1, 0.0)] if rng.uniform() < 0.5 else [(i + 1, 0.0)]
    outer = [(n - 1, 1.0 - 2.0 ** -j) for j in range(1, 16)] + [(n, 0.0)]
    pts += outer
    cap = len(pts) - 1
    cell, frac = _cuts(pts, cap)
    xn, dn, tn, fits = lr.apply_cuts(cell, frac, cap, xi, dm, q * dm[:, None], slope, jnp.zeros(n))
    assert bool(fits) and bool(jnp.all(dn > 0)) and bool(jnp.all(jnp.diff(xn) > 0))
    assert abs(float(jnp.sum(dn)) - n) < 1e-13 and float(xn[-1]) == float(xi[-1])
    assert abs(float(jnp.sum(tn)) / float(jnp.sum(q * dm[:, None])) - 1.0) < 1e-14
    m_new = jnp.cumsum(dn) - 0.5 * dn
    np.testing.assert_allclose(np.asarray(tn[:, 0] / dn), np.asarray(3.0 + 0.2 * m_new), rtol=1e-13, atol=0)
    assert float(dn[-1]) == 2.0 ** -15


def test_face_values_are_linear_in_mass_and_exact_on_old_faces():
    face = jnp.asarray([0.0, 1.0, 4.0, 9.0])
    cell = jnp.asarray([0, 1, 1, 3], jnp.int32)
    frac = jnp.asarray([0.0, 0.0, 0.25, 0.0])
    out = lr.cut_face_values(cell, frac, face)
    assert [float(x) for x in out] == [0.0, 1.0, 1.75, 9.0]
