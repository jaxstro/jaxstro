"""JAX-native spectra containers and prepared atmosphere interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp

STATUS_OK = 0
STATUS_OUT_OF_GRID = 1
STATUS_MISSING_ABUNDANCE = 2


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AtmosphereParams:
    """Coordinates for an atmosphere-grid query."""

    teff: Any
    logg: Any
    m_h: Any = 0.0
    alpha_m: Any = 0.0

    def tree_flatten(self):
        return (self.teff, self.logg, self.m_h, self.alpha_m), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        teff, logg, m_h, alpha_m = children
        return cls(teff=teff, logg=logg, m_h=m_h, alpha_m=alpha_m)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Spectrum:
    """A raw spectral array with explicit static unit metadata."""

    wavelength: Any
    flux_lambda: Any
    wavelength_unit: str = "nm"
    flux_unit: str = "source_flux_lambda"

    def tree_flatten(self):
        children = (self.wavelength, self.flux_lambda)
        aux_data = (self.wavelength_unit, self.flux_unit)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        wavelength_unit, flux_unit = aux_data
        wavelength, flux_lambda = children
        return cls(
            wavelength=wavelength,
            flux_lambda=flux_lambda,
            wavelength_unit=wavelength_unit,
            flux_unit=flux_unit,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpectrumStatus:
    """Structured status for an atmosphere-grid query."""

    code: Any
    in_grid: Any
    clamped: Any
    unmodeled: Any

    def tree_flatten(self):
        return (self.code, self.in_grid, self.clamped, self.unmodeled), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        code, in_grid, clamped, unmodeled = children
        return cls(code=code, in_grid=in_grid, clamped=clamped, unmodeled=unmodeled)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpectrumResult:
    """Raw spectrum plus the coverage status for the query."""

    spectrum: Spectrum
    status: SpectrumStatus

    def tree_flatten(self):
        return (self.spectrum, self.status), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        spectrum, status = children
        return cls(spectrum=spectrum, status=status)


class AtmosphereBackend(Protocol):
    """Protocol for host-side atmosphere backends."""

    def prepare(self, params: AtmosphereParams) -> "PreparedSpectralGrid":
        """Load a local interpolation grid around ``params``."""

    def spectrum(self, params: AtmosphereParams) -> SpectrumResult:
        """Return a spectrum for ``params`` through the backend's convenience path."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PreparedSpectralGrid:
    """A JAX-ready local grid for bilinear atmosphere interpolation.

    The v1 prepared grid covers one exact abundance plane (``m_h``, ``alpha_m``)
    and interpolates over ``teff`` and ``logg``. It never extrapolates silently:
    out-of-grid coordinates are clamped to the prepared axes and marked non-OK
    in the returned ``SpectrumStatus``.
    """

    teff: Any
    logg: Any
    wavelength: Any
    flux: Any
    m_h: float = 0.0
    alpha_m: float = 0.0
    wavelength_unit: str = "nm"
    flux_unit: str = "source_flux_lambda"

    def spectrum(self, params: AtmosphereParams) -> SpectrumResult:
        """Interpolate a raw spectrum at ``params``."""
        teff_axis = jnp.asarray(self.teff)
        logg_axis = jnp.asarray(self.logg)
        flux = jnp.asarray(self.flux)

        teff_idx, teff_weight, teff_in = _axis_index_weight(teff_axis, params.teff)
        logg_idx, logg_weight, logg_in = _axis_index_weight(logg_axis, params.logg)

        f00 = flux[teff_idx, logg_idx, :]
        f10 = flux[teff_idx + 1, logg_idx, :]
        f01 = flux[teff_idx, logg_idx + 1, :]
        f11 = flux[teff_idx + 1, logg_idx + 1, :]

        f0 = (1.0 - teff_weight) * f00 + teff_weight * f10
        f1 = (1.0 - teff_weight) * f01 + teff_weight * f11
        flux_lambda = (1.0 - logg_weight) * f0 + logg_weight * f1

        abundance_in = jnp.logical_and(
            jnp.isclose(params.m_h, self.m_h),
            jnp.isclose(params.alpha_m, self.alpha_m),
        )
        axis_in = jnp.logical_and(teff_in, logg_in)
        in_grid = jnp.logical_and(axis_in, abundance_in)
        clamped = jnp.logical_and(jnp.logical_not(axis_in), abundance_in)
        unmodeled = jnp.logical_not(abundance_in)
        code = jnp.where(
            in_grid,
            STATUS_OK,
            jnp.where(unmodeled, STATUS_MISSING_ABUNDANCE, STATUS_OUT_OF_GRID),
        )

        return SpectrumResult(
            spectrum=Spectrum(
                wavelength=jnp.asarray(self.wavelength),
                flux_lambda=flux_lambda,
                wavelength_unit=self.wavelength_unit,
                flux_unit=self.flux_unit,
            ),
            status=SpectrumStatus(
                code=code,
                in_grid=in_grid,
                clamped=clamped,
                unmodeled=unmodeled,
            ),
        )

    def tree_flatten(self):
        children = (self.teff, self.logg, self.wavelength, self.flux)
        aux_data = (self.m_h, self.alpha_m, self.wavelength_unit, self.flux_unit)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        m_h, alpha_m, wavelength_unit, flux_unit = aux_data
        teff, logg, wavelength, flux = children
        return cls(
            teff=teff,
            logg=logg,
            wavelength=wavelength,
            flux=flux,
            m_h=m_h,
            alpha_m=alpha_m,
            wavelength_unit=wavelength_unit,
            flux_unit=flux_unit,
        )


def _axis_index_weight(axis, value):
    axis = jnp.asarray(axis)
    value = jnp.asarray(value)
    idx = jnp.searchsorted(axis, value, side="right") - 1
    idx = jnp.clip(idx, 0, axis.shape[0] - 2)

    lo = axis[idx]
    hi = axis[idx + 1]
    denom = jnp.where(hi == lo, 1.0, hi - lo)
    weight = jnp.clip((value - lo) / denom, 0.0, 1.0)
    in_axis = jnp.logical_and(value >= axis[0], value <= axis[-1])
    return idx, weight, in_axis


__all__ = [
    "STATUS_MISSING_ABUNDANCE",
    "STATUS_OK",
    "STATUS_OUT_OF_GRID",
    "AtmosphereBackend",
    "AtmosphereParams",
    "PreparedSpectralGrid",
    "Spectrum",
    "SpectrumResult",
    "SpectrumStatus",
]
