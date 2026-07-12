"""Canonical atmosphere query parameters and fixed spectral requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from jaxstro.spectra import SpectralPlan


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AtmosphereParams:
    """Physical coordinates for an atmosphere-grid query."""

    teff: Any
    logg: Any
    m_h: Any = 0.0
    alpha_m: Any = 0.0
    c_m: Any = 0.0
    vturb_km_s: Any = 2.0
    c_o: Any = 0.55

    def tree_flatten(self):
        return (
            self.teff,
            self.logg,
            self.m_h,
            self.alpha_m,
            self.c_m,
            self.vturb_km_s,
            self.c_o,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AtmosphereQuery:
    """Exact product request with parameters and a fixed spectral output plan."""

    params: AtmosphereParams
    product_id: str
    spectral_plan: SpectralPlan
    family: str | None = None
    cloud_label: str | None = None
    requested_parameter_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.product_id.strip():
            raise ValueError("atmosphere query product_id must be non-empty")
        if self.family is not None and not self.family.strip():
            raise ValueError("atmosphere query family must be non-empty when provided")
        if len(set(self.requested_parameter_names)) != len(
            self.requested_parameter_names
        ):
            raise ValueError("requested parameter names must be unique")

    def tree_flatten(self):
        children = (self.params, self.spectral_plan)
        aux_data = (
            self.product_id,
            self.family,
            self.cloud_label,
            self.requested_parameter_names,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        product_id, family, cloud_label, requested_parameter_names = aux_data
        params, spectral_plan = children
        return cls(
            params=params,
            product_id=product_id,
            spectral_plan=spectral_plan,
            family=family,
            cloud_label=cloud_label,
            requested_parameter_names=requested_parameter_names,
        )


__all__ = ["AtmosphereParams", "AtmosphereQuery"]
