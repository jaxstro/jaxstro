"""Declarative composition and registry specifications for JaxtroViz."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "laboratory" / "jaxtroviz" / "plots"

GridAxis = Literal["x", "y", "both"] | None


@dataclass(frozen=True)
class PanelSpec:
    """Per-axis text, limits, orientation, and grid controls."""

    title: str | None = None
    subtitle: str | None = None
    panel_label: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    invert_x: bool | None = None
    invert_y: bool | None = None
    grid_axis: GridAxis = None


@dataclass(frozen=True)
class EncodingSpec:
    """Reusable visual encoding controls."""

    color_by: str | None = None
    linewidth: float = 1.45
    alpha: float = 0.93
    endpoint_markers: bool = True
    max_groups: int | None = None


@dataclass(frozen=True)
class ExportSpec:
    """Paper-master and website export controls."""

    width: float = 7.2
    height: float = 4.8
    dpi: int = 350
    webp_quality: int = 92
    bbox_inches: str = "tight"
    pad_inches: float = 0.02
    facecolor: str = "white"

    @property
    def figsize(self) -> tuple[float, float]:
        return self.width, self.height


@dataclass(frozen=True)
class FigureSpec:
    """One registered figure and its reproducibility contract."""

    name: str
    builder: Callable[[], Figure]
    stem: str
    page: str
    site_path: str
    seed: int
    caption: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    export: ExportSpec = field(default_factory=ExportSpec)

    @property
    def output_stem(self) -> Path:
        return PLOTS_DIR / self.stem

    @property
    def site_webp(self) -> Path:
        return REPO_ROOT / self.site_path
