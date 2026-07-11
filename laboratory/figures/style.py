"""Shared StarViz-derived style and deterministic export helpers."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from PIL import Image

PALETTE = [
    "#355C7D",
    "#6C5B7B",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#E76F51",
    "#457B9D",
    "#8AB17D",
]


@dataclass(frozen=True)
class ExportSpec:
    """Output settings shared by paper masters and website images."""

    dpi: int = 350
    webp_quality: int = 92
    bbox_inches: str = "tight"
    pad_inches: float = 0.02
    facecolor: str = "white"


def setup_style(font_scale: float = 0.95) -> list[str]:
    """Apply the ecosystem's StarViz-derived publication theme."""
    sns.set_theme(
        context="paper",
        style="ticks",
        palette=PALETTE,
        font_scale=font_scale,
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 350,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2B2B2B",
            "axes.linewidth": 0.75,
            "axes.titleweight": "normal",
            "axes.labelcolor": "#252525",
            "xtick.color": "#303030",
            "ytick.color": "#303030",
            "grid.color": "#EAEAEA",
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.35,
            "patch.linewidth": 0.6,
            "mathtext.fontset": "dejavuserif",
        },
    )
    return list(PALETTE)


def polish_axes(
    ax: plt.Axes, *, grid_axis: Literal["x", "y", "both"] | None = None
) -> None:
    """Apply the shared lightweight grid, tick, and spine treatment."""
    if grid_axis is None:
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, color="#EAEAEA", linewidth=0.45)
    ax.tick_params(length=3.0, width=0.7, pad=1.5)
    sns.despine(ax=ax, trim=False)


def render_webp_bytes(fig: Figure, *, spec: ExportSpec | None = None) -> bytes:
    """Render a figure to deterministic WebP bytes and close it."""
    export = spec or ExportSpec()
    common = {
        "bbox_inches": export.bbox_inches,
        "pad_inches": export.pad_inches,
        "facecolor": export.facecolor,
    }
    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=export.dpi, **common)
    plt.close(fig)
    png_buffer.seek(0)
    webp_buffer = io.BytesIO()
    with Image.open(png_buffer) as image:
        image.convert("RGB").save(
            webp_buffer,
            "WEBP",
            quality=export.webp_quality,
            method=6,
        )
    return webp_buffer.getvalue()


def save_figure_formats(
    fig: Figure,
    output_stem: Path,
    *,
    spec: ExportSpec | None = None,
) -> tuple[Path, ...]:
    """Export PDF and PNG masters plus a deterministic WebP site image."""
    export = spec or ExportSpec()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "bbox_inches": export.bbox_inches,
        "pad_inches": export.pad_inches,
        "facecolor": export.facecolor,
    }
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(pdf, **common)
    png = output_stem.with_suffix(".png")
    fig.savefig(png, dpi=export.dpi, **common)
    webp_bytes = render_webp_bytes(fig, spec=export)
    webp = output_stem.with_suffix(".webp")
    webp.write_bytes(webp_bytes)
    return pdf, png, webp
