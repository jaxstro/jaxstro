"""B-spline pedagogy figures generated from the public jaxstro API."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from jaxstro.jaxconfig import enable_high_precision
from jaxstro.numerics import bspline_basis, open_uniform_knots

from .style import NEUTRAL, PALETTE, POSITIVE, polish_axes, setup_style


def basis_results() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed public-API results displayed by the figure."""
    enable_high_precision()
    x = jnp.linspace(0.0, 1.0, 401)
    knots = open_uniform_knots(0.0, 1.0, n_basis=6, degree=3)
    basis = bspline_basis(knots, x, degree=3)
    basis_sum = jnp.sum(basis, axis=-1)
    x_np = np.asarray(x)
    basis_np = np.asarray(basis)
    sum_np = np.asarray(basis_sum)
    if basis_np.shape != (401, 6):
        raise RuntimeError(f"B-spline figure basis shape drifted: {basis_np.shape}")
    if np.any(basis_np < -1e-7):
        raise RuntimeError("B-spline figure requires a nonnegative basis")
    if not np.allclose(sum_np, 1.0, atol=1e-6):
        raise RuntimeError("B-spline figure partition-of-unity contract drifted")
    return x_np, basis_np, sum_np


def build_bspline_local_support() -> Figure:
    """Show cubic basis locality and the measured partition-of-unity sum."""
    setup_style(font_scale=1.0)
    x, basis, basis_sum = basis_results()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.1),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.45, 1.0)},
    )

    left, right = axes
    left.set_title("Local cubic basis functions", weight="bold")
    for index in range(basis.shape[1]):
        left.plot(x, basis[:, index], color=PALETTE[index], label=f"$B_{{{index},3}}$")
    left.set_xlabel("query coordinate $x$")
    left.set_ylabel("basis value")
    left.set_xlim(0.0, 1.0)
    left.set_ylim(0.0, 1.05)
    left.legend(ncol=2, loc="upper center", frameon=False)
    polish_axes(left, grid_axis="y")

    right.set_title("Partition of unity", weight="bold")
    right.plot(
        x, basis_sum, color=POSITIVE, linewidth=2.0, label=r"$\sum_i B_{i,3}(x)$"
    )
    right.axhline(1.0, color=NEUTRAL, linestyle="--", linewidth=0.9, label="unity")
    right.fill_between(x, basis_sum, 1.0, color=PALETTE[2], alpha=0.18)
    right.set_xlabel("query coordinate $x$")
    right.set_ylabel("sum of basis values")
    right.set_xlim(0.0, 1.0)
    right.set_ylim(0.985, 1.015)
    right.legend(loc="upper center", frameon=False)
    right.text(
        0.5,
        0.12,
        "measured from public API",
        transform=right.transAxes,
        ha="center",
        fontsize=7.5,
        color=NEUTRAL,
    )
    polish_axes(right, grid_axis="y")
    return figure
