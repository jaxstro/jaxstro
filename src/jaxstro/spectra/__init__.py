"""Domain-neutral spectral representations and numerical operations."""

from .plan import CoveragePolicy, PointResamplingMethod, SpectralPlan
from .resampling import resample_spectrum
from .stencils import (
    FluxInterpolation,
    PreparedRectilinearStencil,
    PreparedSimplexStencil,
)
from .transforms import (
    surface_flux_to_luminosity,
    surface_flux_to_observer_flux,
    to_flux_lambda,
    to_flux_nu,
    to_frequency,
    to_wavelength,
)
from .types import (
    SpectralAxis,
    SpectralCoordinate,
    SpectralSampling,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
    SpectrumResult,
    SpectrumStatus,
    SpectrumStatusCode,
)

__all__ = [
    "CoveragePolicy",
    "FluxInterpolation",
    "PointResamplingMethod",
    "PreparedRectilinearStencil",
    "PreparedSimplexStencil",
    "SpectralAxis",
    "SpectralCoordinate",
    "SpectralSampling",
    "SpectralSemantic",
    "SpectralPlan",
    "Spectrum",
    "SpectrumProvenance",
    "SpectrumResult",
    "SpectrumStatus",
    "SpectrumStatusCode",
    "resample_spectrum",
    "surface_flux_to_luminosity",
    "surface_flux_to_observer_flux",
    "to_flux_lambda",
    "to_flux_nu",
    "to_frequency",
    "to_wavelength",
]
