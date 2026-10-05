"""Optics of small particles: Mie scattering by homogeneous spheres (jaxstro.optics.mie)
and dust optics of size distributions of spheres (jaxstro.optics.dust_mixture).

Design and decisions: docs/plans/2026-10-04-mie-scattering-design.md; validation:
docs/60-validation/numerical/mie-scattering.md.
"""

from jaxstro.optics.dust_mixture import (
    MixtureOptics,
    SpeciesCrossSections,
    mixture_optics,
    size_distribution_term_counts,
    species_cross_sections,
)
from jaxstro.optics.mie import MieEfficiencies, mie_efficiencies, mie_term_counts, n_stop

__all__ = [
    "MieEfficiencies",
    "MixtureOptics",
    "SpeciesCrossSections",
    "mie_efficiencies",
    "mie_term_counts",
    "mixture_optics",
    "n_stop",
    "size_distribution_term_counts",
    "species_cross_sections",
]
