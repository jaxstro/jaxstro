"""Canonical registry for all JaxtroViz documentation figures."""

from __future__ import annotations

from .architecture import build_jaxstro_foundation
from .spatial import build_spatial_neighbor_contracts
from .specs import ExportSpec, FigureSpec

FIGURES: dict[str, FigureSpec] = {
    spec.name: spec
    for spec in [
        FigureSpec(
            name="jaxstro-foundation",
            builder=build_jaxstro_foundation,
            stem="jaxstro-foundation",
            page="20-architecture/index.md",
            site_path="docs/20-architecture/figures/jaxstro-foundation.webp",
            seed=0,
            caption=(
                "One-way package dependencies and the current jaxstro ownership, "
                "evidence, and host-side preprocessing boundaries."
            ),
            tags=("architecture", "ownership"),
            export=ExportSpec(width=11.5, height=7.2),
        ),
        FigureSpec(
            name="spatial-neighbor-contracts",
            builder=build_spatial_neighbor_contracts,
            stem="spatial-neighbor-contracts",
            page="10-theory/spatial.md",
            site_path="docs/10-theory/figures/spatial-neighbor-contracts.webp",
            seed=0,
            caption=(
                "A real grid candidate pool compared with exact fixed-radius "
                "filtering for the same focal particle."
            ),
            tags=("spatial", "neighbors", "contracts"),
            export=ExportSpec(width=9.4, height=4.5),
        ),
    ]
}
