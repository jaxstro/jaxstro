"""Spectra host-to-JAX boundary figure from the canonical public API."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from jaxstro.spectra import (
    FluxInterpolation,
    PreparedRectilinearStencil,
    SpectralAxis,
    SpectralCoordinate,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
)

from .style import NEUTRAL, PALETTE, POSITIVE, setup_style


def _portable_stencil() -> PreparedRectilinearStencil:
    axis = SpectralAxis.points(
        jnp.array([500.0, 600.0, 700.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )
    template = Spectrum(
        axis=axis,
        values=jnp.array([1.0, 2.0, 3.0]),
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=SpectrumProvenance(
            source_id="figure-fixture",
            product_id="figure-fixture",
            native_coordinate="wavelength_nm",
            native_density="F_lambda",
            native_unit="erg s^-1 cm^-2 nm^-1",
            canonical_conversion="identity",
            citations=("fixture:jaxtroviz",),
        ),
    )
    return PreparedRectilinearStencil(
        parameter_axes=(jnp.array([5000.0, 6000.0]), jnp.array([4.0, 5.0])),
        vertex_values=jnp.array(
            [
                [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
            ]
        ),
        template=template,
        interpolation=FluxInterpolation.LINEAR,
    )


def spectra_runtime_results() -> tuple[np.ndarray, int, int, tuple[int, ...], float]:
    """Return portable interpolation, status, and derivative evidence."""
    prepared = _portable_stencil()
    midpoint = prepared.evaluate(jnp.array([5500.0, 4.5]))
    outside = prepared.evaluate(jnp.array([4500.0, 4.5]))

    def values(point):
        return prepared.evaluate(point).spectrum.values

    batched = jax.vmap(values)(jnp.array([[5250.0, 4.5], [5750.0, 4.5]]))

    @jax.jit
    def first_flux(teff):
        return prepared.evaluate(jnp.array([teff, 4.0])).spectrum.values[0]

    result = (
        np.asarray(midpoint.spectrum.values),
        int(midpoint.status.code),
        int(outside.status.code),
        tuple(batched.shape),
        float(jax.grad(first_flux)(5500.0)),
    )
    if not np.allclose(result[0], [2.5, 3.5, 4.5]):
        raise RuntimeError("spectra figure midpoint interpolation drifted")
    if result[1:3] != (0, 4) or result[3] != (2, 3):
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
    midpoint, midpoint_code, outside_code, batch_shape, local_slope = (
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
        "Prepared JAX stencil",
        "array-only PyTree\nfixed complete cell\nstatus + surface spectrum",
        fill="#EDF6F4",
        edge=POSITIVE,
    )
    _box(
        ax,
        0.71,
        "Fluxax + domain package",
        "distance + extinction\nfilters + instruments\nlikelihood observables",
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
        f"status codes = {midpoint_code}/{outside_code}   •   "
        f"vmap shape = {batch_shape}   •   "
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
