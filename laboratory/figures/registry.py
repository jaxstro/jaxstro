"""The canonical registry for jaxstro documentation figures."""

from __future__ import annotations

from .architecture import build_jaxstro_foundation
from .specs import FigureSpec

FIGURES: dict[str, FigureSpec] = {
    spec.name: spec
    for spec in [
        FigureSpec(
            name="jaxstro-foundation",
            builder=build_jaxstro_foundation,
            stem="jaxstro-foundation",
            page="20-architecture/index.md",
            seed=0,
            caption=(
                "One-way package dependencies and the current jaxstro ownership, "
                "evidence, and host-side preprocessing boundaries."
            ),
            tags=("architecture", "ownership"),
        )
    ]
}
