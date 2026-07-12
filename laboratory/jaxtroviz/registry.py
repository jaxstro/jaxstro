"""Canonical registry for all JaxtroViz documentation figures."""

from __future__ import annotations

from .architecture import build_jaxstro_foundation
from .bsplines import build_bspline_local_support
from .interpolation import build_interpolation_shape_contracts
from .linear_algebra import build_linear_algebra_contracts
from .regular_grid import build_regular_grid_contracts
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
        FigureSpec(
            name="bspline-local-support",
            builder=build_bspline_local_support,
            stem="bspline-local-support",
            page="10-theory/bsplines.md",
            site_path="docs/10-theory/figures/bspline-local-support.webp",
            seed=0,
            caption=(
                "Six cubic basis functions computed by the public API and their "
                "measured partition-of-unity sum."
            ),
            tags=("bsplines", "basis", "pedagogy"),
            export=ExportSpec(width=9.4, height=4.1),
        ),
        FigureSpec(
            name="interpolation-shape-contracts",
            builder=build_interpolation_shape_contracts,
            stem="interpolation-shape-contracts",
            page="10-theory/interpolation.md",
            site_path="docs/10-theory/figures/interpolation-shape-contracts.webp",
            seed=0,
            caption=(
                "Natural cubic and PCHIP interpolation of the same monotone "
                "samples, with stepwise monotonicity measured from public APIs."
            ),
            tags=("interpolation", "pchip", "pedagogy"),
            export=ExportSpec(width=9.4, height=4.3),
        ),
        FigureSpec(
            name="regular-grid-contracts",
            builder=build_regular_grid_contracts,
            stem="regular-grid-contracts",
            page="10-theory/regular-grid.md",
            site_path="docs/10-theory/figures/regular-grid-contracts.webp",
            seed=0,
            caption=(
                "Bilinear corner weights and clamp/fill boundary behavior "
                "measured from public interpolation APIs."
            ),
            tags=("regular-grid", "interpolation", "pedagogy"),
            export=ExportSpec(width=9.4, height=4.3),
        ),
        FigureSpec(
            name="linear-algebra-contracts",
            builder=build_linear_algebra_contracts,
            stem="linear-algebra-contracts",
            page="10-theory/linear-algebra.md",
            site_path="docs/10-theory/figures/linear-algebra-contracts.webp",
            seed=0,
            caption=(
                "Weighted and unweighted regression plus the eigenvalue effect "
                "of the first successful tested diagonal jitter."
            ),
            tags=("linear-algebra", "least-squares", "positive-definite"),
            export=ExportSpec(width=9.4, height=4.3),
        ),
    ]
}
