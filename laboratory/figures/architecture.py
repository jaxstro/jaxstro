"""Architecture and ownership figures for the documentation site."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import PALETTE, setup_style


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = "#355C7D",
    linewidth: float = 1.1,
    radius: float = 0.018,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    )


def build_jaxstro_foundation() -> Figure:
    """Show jaxstro's one-way dependency and module-ownership boundaries."""
    setup_style(font_scale=1.0)
    figure, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    _box(ax, (0.12, 0.80), 0.76, 0.13, facecolor="#EEF3F7")
    ax.text(
        0.50,
        0.888,
        "Domain packages",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    ax.text(
        0.50,
        0.836,
        "gravax  •  progenax  •  fluxax  •  startrax  •  stellax (planned)",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#344454",
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.50, 0.79),
            (0.50, 0.705),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.4,
            color=PALETTE[0],
        )
    )
    ax.text(
        0.515,
        0.748,
        "depends on",
        ha="left",
        va="center",
        fontsize=9.5,
        color=PALETTE[0],
    )

    _box(ax, (0.055, 0.235), 0.89, 0.455, facecolor="#FAFBFC", linewidth=1.5)
    ax.text(
        0.50,
        0.655,
        "jaxstro foundation",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        0.50,
        0.620,
        "generic, differentiable, dependency-light infrastructure",
        ha="center",
        va="center",
        fontsize=10,
        color="#4B5563",
    )

    column_x = (0.085, 0.375, 0.665)
    column_titles = (
        "Physical contracts",
        "Transforms + compute",
        "Evidence + boundaries",
    )
    column_lines = (
        "units  (current contract)\nquantity  (evaluation; adoption deferred)\nconstants",
        "astrometry  •  coords  •  geometry\nnumerics  •  params",
        "provenance  •  testing  •  jaxconfig\natmospheres  (in progress)",
    )
    fills = ("#EDF6F4", "#F2F0F6", "#FFF7E5")
    edges = (PALETTE[2], PALETTE[1], PALETTE[3])
    for x, title, lines, fill, edge in zip(
        column_x, column_titles, column_lines, fills, edges, strict=True
    ):
        _box(ax, (x, 0.385), 0.25, 0.185, facecolor=fill, edgecolor=edge)
        ax.text(
            x + 0.125,
            0.535,
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            weight="bold",
        )
        ax.text(
            x + 0.125,
            0.455,
            lines,
            ha="center",
            va="center",
            fontsize=8.8,
            linespacing=1.55,
        )

    _box(
        ax,
        (0.26, 0.275),
        0.48,
        0.063,
        facecolor="#FCEFEA",
        edgecolor=PALETTE[5],
        radius=0.012,
    )
    ax.text(
        0.50,
        0.306,
        "spatial + atmosphere selection: host-side / discrete boundary",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#7A3E2D",
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.50, 0.225),
            (0.50, 0.155),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.4,
            color=PALETTE[0],
        )
    )
    ax.text(
        0.515, 0.190, "built on", ha="left", va="center", fontsize=9.5, color=PALETTE[0]
    )
    _box(ax, (0.25, 0.055), 0.50, 0.085, facecolor="#F2F4F7")
    ax.text(
        0.50,
        0.098,
        "JAX + Equinox + jaxtyping",
        ha="center",
        va="center",
        fontsize=11.5,
        weight="bold",
    )
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    return figure
