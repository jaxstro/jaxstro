"""The half-mass split against its exact child means, and the transfer's derivative.

Prediction (derived in ``docs/20-methods/discrete-space/lagrangian-remap.md`` before the
run): on a smooth monotone profile the inner child's mean is wrong by ``q' dm / 4`` under
piecewise-constant reconstruction (order 1) and by ``O(q''' dm^3)`` under the limited-linear
rule, because the ``q''`` terms of the cell mean and of the half-cell mean are equal
(order 3). Measured 2026-10-03 on N = 32..512: orders 1.001 and 3.001 at the finest pair.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jaxstro.numerics import lagrangian_remap as lr


def _split_errors(n):
    m = jnp.linspace(0.0, 1.0, n + 1)
    dm = jnp.diff(m)
    total = 0.5 * jnp.diff(jnp.exp(2.0 * m))  # exact totals of exp(2m)
    extent = 0.5 * (
        jnp.exp(-2.0 * m[:-1]) - jnp.exp(-2.0 * m[1:])
    )  # specific extent exp(-2m)
    xi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(extent)])
    mid = 0.5 * (m[1:] + m[:-1])
    exact_inner = 0.5 * (jnp.exp(2.0 * mid) - jnp.exp(2.0 * m[:-1]))
    exact_face = xi[:-1] + 0.5 * (jnp.exp(-2.0 * m[:-1]) - jnp.exp(-2.0 * mid))
    plan = lr.plan_from_actions(jnp.full(n, lr.SPLIT), capacity=2 * n)
    out = []
    for slope, vslope in (
        (None, None),
        (lr.limited_slopes(total / dm, dm), lr.limited_slopes(extent / dm, dm)),
    ):
        xi_new, _, new = lr.apply_plan(plan, xi, dm, total, slope, vslope)
        interior = slice(2, n - 2)  # the end cells take a zero slope by construction
        field = jnp.abs(new[0::2] - exact_inner)[interior] / exact_inner[interior]
        face = jnp.abs(xi_new[1::2] - exact_face)[interior] / extent[interior]
        out.append((float(jnp.max(field)), float(jnp.max(face))))
    return np.asarray(out)  # [reconstruction, (field, face)]


def test_split_converges_at_the_derived_orders():
    ladder = np.stack([_split_errors(n) for n in (32, 64, 128, 256, 512)])
    orders = np.log2(ladder[:-1] / ladder[1:])  # [pair, reconstruction, quantity]
    np.testing.assert_allclose(orders[-1, 0], 1.0, atol=0.03)
    np.testing.assert_allclose(orders[-1, 1], 3.0, atol=0.03)
    assert ladder[-1, 1, 0] < 1e-8  # the linear rule is excited, not trivially exact


def test_transfer_gradient_matches_central_differences():
    n = 24
    m = jnp.linspace(0.0, 1.0, n + 1)
    dm = jnp.diff(m)
    total = 0.5 * jnp.diff(jnp.exp(2.0 * m))
    xi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.exp(-m[:-1]) * dm)])
    score = jnp.abs(jnp.gradient(total / dm))
    cut = float(jnp.median(score))
    plan = lr.fixed_count_plan(score, score[:-1] + score[1:], cut, cut, max_changes=4)
    assert int(jnp.sum(plan.is_split)) > 0 and int(jnp.sum(plan.is_merge)) > 0

    def objective(total, xi):
        # Not a conserved quantity (agreement: a conserved observable hides an AD seam).
        xi_new, dm_new, new = lr.apply_plan(
            plan,
            xi,
            dm,
            total,
            lr.limited_slopes(total / dm, dm),
            lr.limited_slopes(jnp.diff(xi) / dm, dm),
        )
        return jnp.sum(new**2 / dm_new) + jnp.sum(jnp.diff(xi_new) ** 2 / dm_new)

    rng = np.random.default_rng(0)
    d_total = jnp.asarray(rng.normal(size=n))
    d_xi = 1e-3 * jnp.asarray(rng.normal(size=n + 1)).at[0].set(0.0)
    g_total, g_xi = jax.grad(objective, argnums=(0, 1))(total, xi)
    ad = float(g_total @ d_total + g_xi @ d_xi)
    h = 1e-4
    fd = float(
        (
            objective(total + h * d_total, xi + h * d_xi)
            - objective(total - h * d_total, xi - h * d_xi)
        )
        / (2.0 * h)
    )
    assert abs(fd - ad) / abs(ad) < 1e-11  # measured 3.3e-13
