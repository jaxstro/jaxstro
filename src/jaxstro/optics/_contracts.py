from jaxstro.contracts import ExecutionBoundary
from jaxstro.contracts.schema import MaturityLevel
from jaxstro.contracts._core import module_contract

MODULE_CONTRACT = module_contract(
    "optics",
    "Optics of small particles: Mie scattering by homogeneous spheres.",
    "Dust-model selection, radiative transfer, or non-spherical grains.",
    "Computing extinction, scattering and absorption efficiencies and the asymmetry parameter.",
    "Size parameters are dimensionless; refractive indices are complex with a non-negative imaginary part.",
    boundary=ExecutionBoundary.MIXED,
    # Validated 2026-10-04 against a 50-digit mpmath evaluation (Q_ext, Q_sca <= 1e-10, g bound)
    # and miepython: docs/60-validation/numerical/mie-scattering.md.
    maturity=MaturityLevel.VALIDATED,
)
