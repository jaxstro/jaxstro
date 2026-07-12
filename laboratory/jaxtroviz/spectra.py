"""Spectra host-to-JAX boundary figure from the public atmosphere API."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from jaxstro.atmospheres import AtmosphereParams, PreparedSpectralGrid

from .style import NEUTRAL, PALETTE, POSITIVE, setup_style


def _portable_grid() -> PreparedSpectralGrid:
    return PreparedSpectralGrid(
        teff=jnp.array([5000.0, 6000.0]),
        logg=jnp.array([4.0, 5.0]),
        wavelength=jnp.array([100.0, 101.0, 102.0]),
        flux=jnp.array(
            [
                [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
            ]
        ),
    )


def spectra_runtime_results() -> tuple[np.ndarray, int, int, int, float]:
    """Return portable interpolation, status, and derivative evidence."""
    prepared = _portable_grid()
    midpoint = prepared.spectrum(AtmosphereParams(teff=5500.0, logg=4.5))
    outside = prepared.spectrum(AtmosphereParams(teff=4500.0, logg=4.5))
    wrong_plane = prepared.spectrum(AtmosphereParams(teff=5500.0, logg=4.5, m_h=0.5))

    @jax.jit
    def first_flux(teff):
        return prepared.spectrum(
            AtmosphereParams(teff=teff, logg=4.0)
        ).spectrum.flux_lambda[0]

    result = (
        np.asarray(midpoint.spectrum.flux_lambda),
        int(midpoint.status.code),
        int(outside.status.code),
        int(wrong_plane.status.code),
        float(jax.grad(first_flux)(5500.0)),
    )
    if not np.allclose(result[0], [2.5, 3.5, 4.5]):
        raise RuntimeError("spectra figure midpoint interpolation drifted")
    if result[1:4] != (0, 1, 2):
        raise RuntimeError("spectra figure status contract drifted")
    if not np.isclose(result[4], 0.002):
        raise RuntimeError("spectra figure local derivative drifted")
    return result


def _box(ax, x: float, title: str, subtitle: str, *, fill: str, edge: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, 0.37),
            0.25,
            0.34,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.25,
        )
    )
    ax.text(x + 0.125, 0.62, title, ha="center", va="center", weight="bold")
    ax.text(
        x + 0.125,
        0.49,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.0,
        linespacing=1.4,
        color=NEUTRAL,
    )


def build_spectra_runtime_boundary() -> Figure:
    """Show the filesystem boundary and measured prepared-grid contract."""
    setup_style(font_scale=1.0)
    midpoint, midpoint_code, outside_code, wrong_plane_code, local_slope = (
        spectra_runtime_results()
    )
    figure, ax = plt.subplots(figsize=(10.2, 4.4))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    _box(
        ax,
        0.04,
        "Host preparation",
        "catalog ranking\nartifact opening\nlocal-cell selection",
        fill="#FFF7E5",
        edge=PALETTE[3],
    )
    _box(
        ax,
        0.375,
        "Prepared JAX grid",
        "array-only PyTree\nbilinear interpolation\nstatus + raw spectrum",
        fill="#EDF6F4",
        edge=POSITIVE,
    )
    _box(
        ax,
        0.71,
        "Downstream package",
        "filters + zero points\nmagnitudes + surveys\nphysical interpretation",
        fill="#F2F0F6",
        edge=PALETTE[1],
    )

    for start, end, label in (
        (0.295, 0.37, "local files → arrays"),
        (0.63, 0.705, "arrays → raw spectrum"),
    ):
        ax.add_patch(
            FancyArrowPatch(
                (start, 0.54),
                (end, 0.54),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.3,
                color=PALETTE[0],
            )
        )
        ax.text(
            (start + end) / 2,
            0.60,
            label,
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=PALETTE[0],
        )
    ax.text(
        0.835,
        0.77,
        "spectrum → observable",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=PALETTE[1],
    )

    evidence = (
        f"portable fixture: midpoint flux = {midpoint.tolist()}   •   "
        f"status codes = {midpoint_code}/{outside_code}/{wrong_plane_code}   •   "
        f"local dF0/dT = {local_slope:.3f}"
    )
    ax.text(
        0.5,
        0.18,
        evidence,
        ha="center",
        va="center",
        fontsize=8.3,
        color=NEUTRAL,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F7F8FA",
            "edgecolor": "#C8CDD3",
        },
    )
    ax.text(
        0.5,
        0.075,
        "Only the prepared array object crosses into JAX transforms",
        ha="center",
        va="center",
        fontsize=8.8,
        weight="bold",
        color=PALETTE[0],
    )
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    return figure
