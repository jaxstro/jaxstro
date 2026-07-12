"""Linear-algebra pedagogy figures generated from the public jaxstro API."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from jaxstro.jaxconfig import enable_high_precision
from jaxstro.numerics.linear_algebra import (
    positive_definite_jitter,
    weighted_lstsq,
)

from .style import NEGATIVE, NEUTRAL, PALETTE, POSITIVE, polish_axes, setup_style


def linear_algebra_results() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    bool,
]:
    """Return public-API regression and diagonal-jitter results."""
    enable_high_precision()
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([1.0, 3.0, 5.0, 20.0])
    design = jnp.stack([jnp.ones_like(x), x], axis=1)
    unweighted = weighted_lstsq(design, y)
    weighted = weighted_lstsq(design, y, jnp.array([1.0, 1.0, 1.0, 0.0]))

    matrix = jnp.diag(jnp.array([-0.03, 2.0]))
    shifted, jitter, success = positive_definite_jitter(
        matrix,
        initial_jitter=1.0e-3,
        growth=10.0,
        max_steps=4,
    )
    eigenvalues_before = jnp.linalg.eigvalsh(matrix)
    eigenvalues_after = jnp.linalg.eigvalsh(shifted)

    arrays = tuple(
        np.asarray(array)
        for array in (
            x,
            y,
            unweighted,
            weighted,
            eigenvalues_before,
            eigenvalues_after,
        )
    )
    jitter_value = float(jitter)
    success_value = bool(success)
    if not np.allclose(arrays[2], [-1.6, 5.9], atol=1e-12):
        raise RuntimeError("linear-algebra figure unweighted fit drifted")
    if not np.allclose(arrays[3], [1.0, 2.0], atol=1e-12):
        raise RuntimeError("linear-algebra figure weighted fit drifted")
    if not np.isclose(jitter_value, 0.1, atol=1e-12) or not success_value:
        raise RuntimeError("linear-algebra figure jitter selection drifted")
    if not arrays[4].min() < 0.0 or not arrays[5].min() > 0.0:
        raise RuntimeError("linear-algebra figure eigenvalue crossing drifted")
    return (*arrays, jitter_value, success_value)


def build_linear_algebra_contracts() -> Figure:
    """Show a weighted fit and the selected positive-definite shift."""
    setup_style(font_scale=1.0)
    (
        x,
        y,
        unweighted,
        weighted,
        eigenvalues_before,
        eigenvalues_after,
        jitter,
        _,
    ) = linear_algebra_results()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.3),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.25, 1.0)},
    )

    left, right = axes
    left.set_title("Weight changes the fit", weight="bold")
    x_line = np.linspace(-0.1, 3.1, 201)
    left.scatter(x[:3], y[:3], color=NEUTRAL, s=31, zorder=4, label="unit-weight data")
    left.scatter(
        x[3],
        y[3],
        color=NEGATIVE,
        marker="X",
        s=48,
        zorder=5,
        label="zero-weight outlier",
    )
    left.plot(
        x_line,
        unweighted[0] + unweighted[1] * x_line,
        color=NEGATIVE,
        linestyle="--",
        linewidth=1.6,
        label="unweighted",
    )
    left.plot(
        x_line,
        weighted[0] + weighted[1] * x_line,
        color=POSITIVE,
        linewidth=1.9,
        label="weighted",
    )
    left.text(
        0.03,
        0.96,
        r"weighted: $\beta=(1,2)$",
        transform=left.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=POSITIVE,
    )
    left.set_xlabel("predictor $x$")
    left.set_ylabel("observation $y$")
    left.set_xlim(-0.1, 3.1)
    left.set_ylim(0.0, 21.0)
    left.legend(loc="upper left", bbox_to_anchor=(0.0, 0.83), frameon=False)
    polish_axes(left, grid_axis="y")

    right.set_title("Jitter crosses the PD boundary", weight="bold")
    positions = np.arange(2)
    width = 0.34
    right.bar(
        positions - width / 2,
        eigenvalues_before,
        width,
        color=PALETTE[0],
        label="before",
    )
    right.bar(
        positions + width / 2,
        eigenvalues_after,
        width,
        color=POSITIVE,
        label="after",
    )
    right.axhline(0.0, color=NEUTRAL, linestyle="--", linewidth=0.9)
    right.text(
        0.5,
        0.93,
        f"first successful tested jitter = {jitter:.3g}",
        transform=right.transAxes,
        ha="center",
        fontsize=7.5,
        color=NEUTRAL,
    )
    right.set_xticks(positions, (r"$\lambda_1$", r"$\lambda_2$"))
    right.set_ylabel("eigenvalue")
    right.set_ylim(-0.16, 2.25)
    right.legend(loc="upper left", frameon=False)
    polish_axes(right, grid_axis="y")
    return figure
