"""Explicit physical equivalencies for non-identical dimensions."""

from __future__ import annotations

from dataclasses import dataclass

from jaxstro import constants as C

from . import dimensions as d
from . import units
from .quantity import Quantity
from .unit import Unit


@dataclass(frozen=True)
class Equivalency:
    """Callable unit equivalency."""

    name: str

    def convert(self, quantity: Quantity, target: Unit) -> Quantity | None:
        raise NotImplementedError


class SpectralEquivalency(Equivalency):
    """Wavelength, frequency, and photon-energy equivalency."""

    def convert(self, quantity: Quantity, target: Unit) -> Quantity | None:
        freq = _as_frequency_hz(quantity)
        if freq is None:
            return None
        if target.dimensions == d.time**-1:
            return Quantity(freq, units.Hz).to(target)
        if target.dimensions == d.length:
            return Quantity(C.C_CGS / freq, units.cm).to(target)
        if target.dimensions == d.energy:
            return Quantity(C.H_CGS * freq, units.erg).to(target)
        return None


class TemperatureEnergyEquivalency(Equivalency):
    """Temperature-energy equivalency through k_B."""

    def convert(self, quantity: Quantity, target: Unit) -> Quantity | None:
        if quantity.unit.dimensions == d.temperature and target.dimensions == d.energy:
            return Quantity(C.K_B * quantity.to_value(units.K), units.erg).to(target)
        if quantity.unit.dimensions == d.energy and target.dimensions == d.temperature:
            return Quantity(quantity.to_value(units.erg) / C.K_B, units.K).to(target)
        return None


class MassEnergyEquivalency(Equivalency):
    """Mass-energy equivalency through E = m c^2."""

    def convert(self, quantity: Quantity, target: Unit) -> Quantity | None:
        if quantity.unit.dimensions == d.mass and target.dimensions == d.energy:
            return Quantity(quantity.to_value(units.g) * C.C_CGS**2, units.erg).to(
                target
            )
        if quantity.unit.dimensions == d.energy and target.dimensions == d.mass:
            return Quantity(quantity.to_value(units.erg) / C.C_CGS**2, units.g).to(
                target
            )
        return None


def _as_frequency_hz(quantity: Quantity):
    if quantity.unit.dimensions == d.time**-1:
        return quantity.to_value(units.Hz)
    if quantity.unit.dimensions == d.length:
        return C.C_CGS / quantity.to_value(units.cm)
    if quantity.unit.dimensions == d.energy:
        return quantity.to_value(units.erg) / C.H_CGS
    return None


def spectral() -> tuple[Equivalency, ...]:
    return (SpectralEquivalency("spectral"),)


def temperature_energy() -> tuple[Equivalency, ...]:
    return (TemperatureEnergyEquivalency("temperature_energy"),)


def mass_energy() -> tuple[Equivalency, ...]:
    return (MassEnergyEquivalency("mass_energy"),)


__all__ = [
    "Equivalency",
    "MassEnergyEquivalency",
    "SpectralEquivalency",
    "TemperatureEnergyEquivalency",
    "mass_energy",
    "spectral",
    "temperature_energy",
]
