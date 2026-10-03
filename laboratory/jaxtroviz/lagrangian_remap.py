"""Split/merge figure generated from the public ``jaxstro.numerics.lagrangian_remap`` API."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from jaxstro.jaxconfig import enable_high_precision
from jaxstro.numerics import lagrangian_remap as lr

from .style import NEGATIVE, NEUTRAL, POSITIVE, polish_axes, setup_style

FRONT_WIDTH = 0.02
REGRID_PASSES = 4
LADDER = (32, 64, 128, 256, 512)


def _front(m):
    return 1.0 + np.tanh((m - 0.5) / FRONT_WIDTH)


def _front_totals(edges):
    """Exact cell totals of ``1 + tanh((m - 0.5)/w)`` (its antiderivative is
    ``m + w ln cosh((m - 0.5)/w)``)."""
    big = FRONT_WIDTH * jnp.log(jnp.cosh((edges - 0.5) / FRONT_WIDTH))
    return jnp.diff(edges + big)


def regrid_front(n: int = 32):
    """Regrid a uniform-in-mass slab mesh around a steep front, ``REGRID_PASSES`` times.

    Returns the initial and final faces (slab: ``xi = m`` at unit density) and the
    transferred cell totals after each pass's limited-linear split.
    """
    enable_high_precision()
    edges = jnp.linspace(0.0, 1.0, n + 1)
    dm = jnp.diff(edges)
    total = _front_totals(edges)
    first = (np.asarray(edges), np.asarray(total / dm))
    xi = edges
    for _ in range(
        REGRID_PASSES
    ):  # host-loop: 4 sequential regrids, each a whole-array call
        q = total / dm
        jump = jnp.abs(jnp.diff(q))
        split_score = jnp.concatenate(
            [jump[:1], jnp.maximum(jump[:-1], jump[1:]), jump[-1:]]
        )
        merge_score = jump + jnp.concatenate([jump[1:], jnp.zeros(1)])
        plan = lr.fixed_count_plan(
            split_score, merge_score, 0.1, 0.02, max_changes=4, size=dm, max_ratio=2.5
        )
        xi, dm, total = lr.apply_plan(
            plan,
            xi,
            dm,
            total,
            lr.limited_slopes(q, dm),
            lr.limited_slopes(jnp.diff(xi) / dm, dm),
        )
    if not np.isclose(
        float(jnp.sum(total)),
        float(jnp.sum(_front_totals(edges))),
        rtol=1e-14,
        atol=0.0,
    ):
        raise RuntimeError("lagrangian-remap figure: transfer no longer conserves")
    return first, (np.asarray(xi), np.asarray(total / dm))


def split_error_ladder():
    """Max relative error of the inner-half mean after splitting every cell, for a smooth
    monotone ``q(m) = exp(2m)``, piecewise-constant against limited-linear."""
    enable_high_precision()
    rows = []
    for n in LADDER:  # host-loop: resolution ladder, one shape per N
        m = jnp.linspace(0.0, 1.0, n + 1)
        dm = jnp.diff(m)
        total = 0.5 * jnp.diff(jnp.exp(2.0 * m))
        mid = 0.5 * (m[1:] + m[:-1])
        exact = 0.5 * (jnp.exp(2.0 * mid) - jnp.exp(2.0 * m[:-1]))
        plan = lr.plan_from_actions(jnp.full(n, lr.SPLIT), capacity=2 * n)
        errs = []
        for slope in (
            None,
            lr.limited_slopes(total / dm, dm),
        ):  # host-loop: two reconstructions compared
            _, _, new = lr.apply_plan(plan, m, dm, total, slope)
            rel = jnp.abs(new[0::2] - exact) / exact
            errs.append(float(jnp.max(rel[2:-2])))
        rows.append((n, *errs))
    table = np.asarray(rows)
    orders = np.log2(table[:-1, 1:] / table[1:, 1:])
    if not (
        np.allclose(orders[-1, 0], 1.0, atol=0.02)
        and np.allclose(orders[-1, 1], 3.0, atol=0.02)
    ):
        raise RuntimeError(
            f"lagrangian-remap figure: split orders drifted to {orders[-1]}"
        )
    return table


def build_lagrangian_remap_contracts() -> Figure:
    """A regridded front and the measured order of the split."""
    setup_style(font_scale=1.0)
    (e0, q0), (e1, q1) = regrid_front()
    table = split_error_ladder()
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(9.4, 4.3), constrained_layout=True
    )

    fine = np.linspace(0.0, 1.0, 2001)
    left.plot(fine, _front(fine), color=NEUTRAL, linewidth=0.9, label="exact profile")
    left.stairs(
        q0, e0, color=NEGATIVE, linewidth=1.1, label=f"initial, {q0.size} cells"
    )
    left.stairs(
        q1,
        e1,
        color=POSITIVE,
        linewidth=1.4,
        label=f"after {REGRID_PASSES} regrids, {q1.size} cells",
    )
    left.plot(e1, np.full(e1.size, -0.12), "|", color=POSITIVE, markersize=7)
    left.plot(e0, np.full(e0.size, -0.24), "|", color=NEGATIVE, markersize=7)
    left.set_xlim(0.3, 0.7)
    left.set_xlabel("enclosed mass m")
    left.set_ylabel("specific value q = Q / dm")
    left.set_title("Fixed-count split/merge at a front", weight="bold")
    left.legend(frameon=False, fontsize=7.5, loc="upper left")
    polish_axes(left, grid_axis="y")

    n = table[:, 0]
    right.loglog(n, table[:, 1], "o-", color=NEGATIVE, label="piecewise constant")
    right.loglog(n, table[:, 2], "s-", color=POSITIVE, label="limited linear (MC)")
    right.loglog(
        n, table[0, 1] * (n / n[0]) ** -1.0, ":", color=NEUTRAL, label="order 1"
    )
    right.loglog(
        n, table[0, 2] * (n / n[0]) ** -3.0, "--", color=NEUTRAL, label="order 3"
    )
    right.set_xlabel("cells N")
    right.set_ylabel("max relative error, inner-half mean")
    right.set_title("Split error, q = exp(2m)", weight="bold")
    right.legend(frameon=False, fontsize=7.5)
    polish_axes(right, grid_axis="both")
    return figure
