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
    # Implemented; validation against miepython and a 50-digit mpmath reference is in progress.
    maturity=MaturityLevel.IMPLEMENTED,
)
