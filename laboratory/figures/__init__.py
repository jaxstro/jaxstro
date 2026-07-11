"""Registered, reproducible figures for the jaxstro documentation site."""

from .registry import FIGURES
from .specs import PLOTS_DIR, SITE_FIGURE_DIR, FigureSpec
from .style import PALETTE, render_webp_bytes, save_figure_formats, setup_style

__all__ = [
    "FIGURES",
    "PALETTE",
    "PLOTS_DIR",
    "SITE_FIGURE_DIR",
    "FigureSpec",
    "render_webp_bytes",
    "save_figure_formats",
    "setup_style",
]
