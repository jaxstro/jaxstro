"""JaxtroViz: registered, evidence-backed figures for jaxstro documentation."""

from .registry import FIGURES
from .specs import EncodingSpec, ExportSpec, FigureSpec, PanelSpec
from .style import (
    NEGATIVE,
    NEUTRAL,
    PALETTE,
    POSITIVE,
    polish_axes,
    render_webp_bytes,
    save_figure,
    save_figure_formats,
    setup_style,
)

__all__ = [
    "FIGURES",
    "EncodingSpec",
    "ExportSpec",
    "FigureSpec",
    "NEGATIVE",
    "NEUTRAL",
    "PALETTE",
    "POSITIVE",
    "PanelSpec",
    "polish_axes",
    "render_webp_bytes",
    "save_figure",
    "save_figure_formats",
    "setup_style",
]
