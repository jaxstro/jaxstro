"""Optics of small particles: Mie scattering by homogeneous spheres (jaxstro.optics.mie).

Design and decisions: docs/plans/2026-10-04-mie-scattering-design.md.
"""

from jaxstro.optics.mie import MieEfficiencies, mie_efficiencies, mie_term_counts, n_stop

__all__ = ["MieEfficiencies", "mie_efficiencies", "mie_term_counts", "n_stop"]
