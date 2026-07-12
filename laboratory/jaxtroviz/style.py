"""StarViz-derived visual identity and deterministic export helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from PIL import Image

from .specs import ExportSpec

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
NEGATIVE = "#8E5A7F"
POSITIVE = "#2A9D8F"
NEUTRAL = "#3A3A3A"


def setup_style(font_scale: float = 0.95) -> list[str]:
    """Apply the ecosystem's compact StarViz publication theme."""
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
            "axes.titlesize": 9.5,
            "axes.titleweight": "normal",
            "axes.labelsize": 8.8,
            "axes.labelcolor": "#252525",
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "xtick.color": "#303030",
            "ytick.color": "#303030",
            "grid.color": "#EAEAEA",
            "grid.linewidth": 0.45,
            "legend.fontsize": 7.2,
            "legend.title_fontsize": 7.2,
            "lines.linewidth": 1.35,
            "patch.linewidth": 0.35,
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
    sns.despine(ax=ax, trim=True)


def _save_kwargs(export: ExportSpec) -> dict[str, str | float]:
    return {
        "bbox_inches": export.bbox_inches,
        "pad_inches": export.pad_inches,
        "facecolor": export.facecolor,
    }


def render_webp_bytes(fig: Figure, *, spec: ExportSpec | None = None) -> bytes:
    """Render deterministic WebP bytes from one live figure and close it."""
    export = spec or ExportSpec()
    png_buffer = io.BytesIO()
    fig.savefig(
        png_buffer,
        format="png",
        dpi=export.dpi,
        **_save_kwargs(export),
    )
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


def save_figure(
    fig: Figure,
    path: Path,
    *,
    spec: ExportSpec | None = None,
) -> Path:
    """Save one figure using the shared export contract and close it."""
    export = spec or ExportSpec()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=export.dpi, **_save_kwargs(export))
    plt.close(fig)
    return path


def save_figure_formats(
    fig: Figure,
    output_stem: Path,
    *,
    spec: ExportSpec | None = None,
) -> tuple[Path, ...]:
    """Export PDF and PNG masters plus the deterministic WebP site image."""
    export = spec or ExportSpec()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    common = _save_kwargs(export)
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(pdf, **common)
    png = output_stem.with_suffix(".png")
    fig.savefig(png, dpi=export.dpi, **common)
    webp_bytes = render_webp_bytes(fig, spec=export)
    webp = output_stem.with_suffix(".webp")
    webp.write_bytes(webp_bytes)
    return pdf, png, webp
