"""Interpolation pedagogy figures generated from the public jaxstro API."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from jaxstro.jaxconfig import enable_high_precision
from jaxstro.numerics import interpolation

from .style import NEGATIVE, NEUTRAL, PALETTE, POSITIVE, polish_axes, setup_style


def interpolation_results() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return the fixed public-API interpolation results shown in the figure."""
    enable_high_precision()
    x_grid = jnp.arange(5.0)
    values = jnp.array([0.0, 0.01, 0.9, 0.91, 1.0])
    x_query = jnp.linspace(0.0, 4.0, 801)
    natural = interpolation.eval_cubic_spline(
        x_grid,
        interpolation.natural_cubic_spline_coeffs(x_grid, values),
        x_query,
    )
    monotone = interpolation.monotone_cubic_interp(x_grid, values, x_query)

    results = tuple(
        np.asarray(array) for array in (x_grid, values, x_query, natural, monotone)
    )
    x_np, values_np, query_np, natural_np, monotone_np = results
    if not np.all(np.diff(values_np) >= 0.0):
        raise RuntimeError("interpolation figure samples must be monotone")
    if not natural_np.min() < -0.1:
        raise RuntimeError("interpolation figure natural-spline undershoot drifted")
    if monotone_np.min() < -1e-12 or monotone_np.max() > 1.0 + 1e-12:
        raise RuntimeError("interpolation figure PCHIP bounds drifted")
    if np.diff(monotone_np).min() < -1e-12:
        raise RuntimeError("interpolation figure PCHIP monotonicity drifted")
    return results


def build_interpolation_shape_contracts() -> Figure:
    """Compare natural-spline undershoot with PCHIP shape preservation."""
    setup_style(font_scale=1.0)
    x_grid, values, x_query, natural, monotone = interpolation_results()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.3),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.35, 1.0)},
    )

    left, right = axes
    left.set_title("Same monotone samples", weight="bold")
    left.plot(
        x_query,
        natural,
        color=NEGATIVE,
        linewidth=1.7,
        label="natural cubic",
    )
    left.plot(
        x_query,
        monotone,
        color=POSITIVE,
        linewidth=1.9,
        label="PCHIP",
    )
    left.scatter(
        x_grid,
        values,
        color=NEUTRAL,
        s=28,
        zorder=4,
        label="samples",
    )
    left.axhline(0.0, color=NEUTRAL, linestyle="--", linewidth=0.8)
    left.fill_between(
        x_query,
        natural,
        0.0,
        where=natural < 0.0,
        color=PALETTE[5],
        alpha=0.16,
        label="natural undershoot",
    )
    left.set_xlabel("query coordinate $x$")
    left.set_ylabel("interpolated value")
    left.set_xlim(0.0, 4.0)
    left.legend(loc="upper left", frameon=False)
    left.text(
        0.98,
        0.05,
        f"natural minimum = {natural.min():.3f}",
        transform=left.transAxes,
        ha="right",
        fontsize=7.5,
        color=NEGATIVE,
    )
    polish_axes(left, grid_axis="y")

    right.set_title("Step-by-step monotonicity", weight="bold")
    query_mid = 0.5 * (x_query[:-1] + x_query[1:])
    natural_steps = np.diff(natural)
    monotone_steps = np.diff(monotone)
    right.plot(
        query_mid,
        natural_steps,
        color=NEGATIVE,
        linewidth=1.4,
        label=r"natural $\Delta y$",
    )
    right.plot(
        query_mid,
        monotone_steps,
        color=POSITIVE,
        linewidth=1.6,
        label=r"PCHIP $\Delta y$",
    )
    right.axhline(0.0, color=NEUTRAL, linestyle="--", linewidth=0.8)
    right.fill_between(
        query_mid,
        natural_steps,
        0.0,
        where=natural_steps < 0.0,
        color=PALETTE[5],
        alpha=0.16,
    )
    right.set_xlabel("query coordinate $x$")
    right.set_ylabel(r"successive $\Delta$ value")
    right.set_xlim(0.0, 4.0)
    right.legend(loc="upper left", frameon=False)
    polish_axes(right, grid_axis="y")
    return figure
