"""Tests for explicit quantity equivalencies."""

import pytest

from jaxstro import constants as C
from jaxstro import quantity as q
from jaxstro.quantity.errors import DimensionError, EquivalencyError


def test_exact_conversion_rejects_wavelength_to_frequency_without_equivalency():
    with pytest.raises(DimensionError):
        (500.0 * q.nm).to(q.Hz)


def test_spectral_equivalency_converts_wavelength_frequency_and_energy():
    spectral = q.equivalencies.spectral()

    frequency = (500.0 * q.nm).to(q.Hz, equivalencies=spectral)
    wavelength = frequency.to(q.nm, equivalencies=spectral)
    energy = frequency.to(q.erg, equivalencies=spectral)
    frequency_back = energy.to(q.Hz, equivalencies=spectral)

    assert frequency.value == pytest.approx(C.C_CGS / (500.0 * q.nm.scale_to_cgs))
    assert wavelength.value == pytest.approx(500.0)
    assert energy.value == pytest.approx(C.H_CGS * frequency.value)
    assert frequency_back.value == pytest.approx(frequency.value)


def test_temperature_energy_equivalency_uses_boltzmann_constant():
    thermal = q.equivalencies.temperature_energy()

    energy = (1.0 * q.K).to(q.erg, equivalencies=thermal)
    temp = energy.to(q.K, equivalencies=thermal)

    assert energy.value == C.K_B
    assert temp.value == pytest.approx(1.0)


def test_mass_energy_equivalency_uses_c_squared():
    mass_energy = q.equivalencies.mass_energy()

    energy = (2.0 * q.g).to(q.erg, equivalencies=mass_energy)
    mass = energy.to(q.g, equivalencies=mass_energy)

    assert energy.value == pytest.approx(2.0 * C.C_CGS**2)
    assert mass.value == pytest.approx(2.0)


def test_impossible_equivalency_raises_structured_error():
    with pytest.raises(EquivalencyError) as exc:
        (1.0 * q.cm).to(q.K, equivalencies=q.equivalencies.spectral())

    assert exc.value.operation == "equivalency-convert"
    assert exc.value.expected == q.K.dimensions
