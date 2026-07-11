"""Declarative metadata for registered jaxstro documentation figures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "laboratory" / "figures" / "plots"
SITE_FIGURE_DIR = REPO_ROOT / "docs" / "20-architecture" / "figures"


@dataclass(frozen=True)
class FigureSpec:
    """A reproducible figure registered to one documentation page."""

    name: str
    builder: Callable[[], Figure]
    stem: str
    page: str
    seed: int
    caption: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def output_stem(self) -> Path:
        return PLOTS_DIR / self.stem

    @property
    def site_webp(self) -> Path:
        return SITE_FIGURE_DIR / f"{self.stem}.webp"
