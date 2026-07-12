"""Regular-grid pedagogy figures generated from the public jaxstro API."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from jaxstro.jaxconfig import enable_high_precision
from jaxstro.numerics.regular_grid import bilinear_interp

from .style import NEGATIVE, NEUTRAL, PALETTE, POSITIVE, polish_axes, setup_style


def regular_grid_results() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return the public-API corner weights and boundary-policy scan."""
    enable_high_precision()
    axis = jnp.array([0.0, 1.0])
    query = jnp.array([0.3, 0.65])

    corner_tables = jnp.eye(4).reshape(4, 2, 2)
    weights = jnp.stack(
        [
            bilinear_interp(
                axis,
                axis,
                table,
                query[0],
                query[1],
            )
            for table in corner_tables
        ]
    )

    xx, yy = jnp.meshgrid(axis, axis, indexing="ij")
    values = xx + yy
    scan = jnp.linspace(-0.5, 1.5, 301)
    fixed_y = jnp.full_like(scan, 0.5)
    clamped = bilinear_interp(axis, axis, values, scan, fixed_y, boundary="clamp")
    filled = bilinear_interp(
        axis,
        axis,
        values,
        scan,
        fixed_y,
        boundary="fill",
        fill_value=-1.0,
    )

    results = tuple(
        np.asarray(array) for array in (query, weights, scan, clamped, filled)
    )
    query_np, weights_np, scan_np, clamped_np, filled_np = results
    if not np.allclose(weights_np.sum(), 1.0, atol=1e-12):
        raise RuntimeError("regular-grid figure corner weights no longer sum to one")
    expected = np.array([0.245, 0.455, 0.105, 0.195])
    if not np.allclose(weights_np, expected, atol=1e-12):
        raise RuntimeError("regular-grid figure corner weights drifted")
    outside = (scan_np < 0.0) | (scan_np > 1.0)
    if not np.all(filled_np[outside] == -1.0):
        raise RuntimeError("regular-grid figure fill boundary drifted")
    if not np.all(np.isfinite(clamped_np)):
        raise RuntimeError("regular-grid figure clamp boundary became non-finite")
    return results


def build_regular_grid_contracts() -> Figure:
    """Show measured bilinear weights and explicit boundary policies."""
    setup_style(font_scale=1.0)
    query, weights, scan, clamped, filled = regular_grid_results()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.3),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.35)},
    )

    left, right = axes
    left.set_title("Bilinear corner weights", weight="bold")
    corners = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = (r"$w_{00}$", r"$w_{01}$", r"$w_{10}$", r"$w_{11}$")
    for corner, weight, label, color in zip(
        corners, weights, labels, PALETTE[:4], strict=True
    ):
        left.plot(
            [query[0], corner[0]],
            [query[1], corner[1]],
            color=color,
            linewidth=0.8 + 5.0 * weight,
            alpha=0.8,
        )
        x_offset = -0.08 if corner[0] == 0.0 else 0.08
        y_offset = -0.08 if corner[1] == 0.0 else 0.08
        left.text(
            corner[0] + x_offset,
            corner[1] + y_offset,
            f"{label} = {weight:.3f}",
            ha="center",
            va="center",
            fontsize=7.2,
            color=color,
        )
    left.scatter(corners[:, 0], corners[:, 1], color=NEUTRAL, s=28, zorder=4)
    left.scatter(query[0], query[1], color=POSITIVE, marker="*", s=85, zorder=5)
    left.text(
        query[0] + 0.05,
        query[1] - 0.02,
        "query",
        ha="left",
        va="center",
        fontsize=7.5,
        color=NEUTRAL,
    )
    left.text(
        0.5,
        -0.18,
        f"measured sum = {weights.sum():.3f}",
        transform=left.transAxes,
        ha="center",
        fontsize=7.5,
        color=NEUTRAL,
    )
    left.set_xlabel("grid coordinate $x$")
    left.set_ylabel("grid coordinate $y$")
    left.set_xlim(-0.18, 1.18)
    left.set_ylim(-0.18, 1.18)
    left.set_aspect("equal")
    polish_axes(left)

    right.set_title("Boundary policies", weight="bold")
    interior = (scan >= 0.0) & (scan <= 1.0)
    right.plot(
        scan,
        clamped,
        color=POSITIVE,
        linewidth=1.9,
        label="clamp",
    )
    right.plot(
        scan,
        filled,
        color=NEGATIVE,
        linewidth=1.5,
        linestyle="--",
        label="fill = -1",
    )
    right.plot(
        scan[interior],
        scan[interior] + 0.5,
        color=NEUTRAL,
        linewidth=0.8,
    )
    right.axvline(0.0, color=NEUTRAL, linewidth=0.8, linestyle=":")
    right.axvline(1.0, color=NEUTRAL, linewidth=0.8, linestyle=":")
    right.axvspan(-0.5, 0.0, color=PALETTE[3], alpha=0.12)
    right.axvspan(1.0, 1.5, color=PALETTE[3], alpha=0.12)
    right.text(-0.25, 1.65, "outside", ha="center", fontsize=7.2, color=NEUTRAL)
    right.text(1.25, 1.65, "outside", ha="center", fontsize=7.2, color=NEUTRAL)
    right.set_xlabel("query $x$ at fixed $y=0.5$")
    right.set_ylabel("interpolated value")
    right.set_xlim(-0.5, 1.5)
    right.set_ylim(-1.15, 1.8)
    right.legend(loc="upper left", bbox_to_anchor=(0.04, 0.82), frameon=False)
    polish_axes(right, grid_axis="y")
    return figure
