"""Source-semantic contracts for atmosphere spectral products."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO_ROOT / "docs" / "provenance" / "registry" / "atmospheres.yaml"


def _records() -> dict[str, dict[str, object]]:
    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    return {str(record["id"]): record for record in raw}


def _conventions(record: dict[str, object]) -> set[str]:
    conventions = record["conventions"]
    assert isinstance(conventions, list)
    return {str(value) for value in conventions}


def test_registry_covers_all_enabled_atmosphere_source_products() -> None:
    records = _records()

    assert set(records) == {
        "bosz-2025-recomputed",
        "newera-v3-lowres",
        "sonora-diamondback-2024",
        "tlusty-bstar2006",
        "tlusty-ostar2002",
    }


def test_newera_and_bosz_canonical_flux_conversions_are_explicit() -> None:
    records = _records()

    newera = _conventions(records["newera-v3-lowres"])
    bosz = _conventions(records["bosz-2025-recomputed"])
    assert "native_coordinate=wavelength_nm" in newera
    assert "native_density=F_lambda" in newera
    assert "native_unit=W m^-2 nm^-1" in newera
    assert "canonical_factor=1e3" in newera
    assert "canonical_unit=erg s^-1 cm^-2 nm^-1" in newera
    assert "native_coordinate=wavelength_angstrom" in bosz
    assert "native_density=F_lambda_resampled" in bosz
    assert "native_unit=erg s^-1 cm^-2 angstrom^-1" in bosz
    assert "canonical_factor=10" in bosz
    assert "canonical_unit=erg s^-1 cm^-2 nm^-1" in bosz


def test_sonora_source_inconsistency_and_conversion_are_explicit() -> None:
    sonora = _conventions(_records()["sonora-diamondback-2024"])

    assert "native_coordinate=wavelength_micron" in sonora
    assert "native_density=wavelength-density flux" in sonora
    assert "native_unit=W m^-2 m^-1" in sonora
    assert "canonical_factor=1e-6" in sonora
    assert "canonical_unit=erg s^-1 cm^-2 nm^-1" in sonora
    assert (
        "source_deviation=printed F_nu subscript conflicts with wavelength axis and per-metre density unit"
        in sonora
    )


def test_tlusty_eddington_flux_conversion_is_explicit() -> None:
    records = _records()

    for product_id in ("tlusty-ostar2002", "tlusty-bstar2006"):
        conventions = _conventions(records[product_id])
        assert "native_coordinate=frequency_hz" in conventions
        assert "native_density=H_nu" in conventions
        assert "native_unit=erg s^-1 cm^-2 Hz^-1" in conventions
        assert "canonical_factor=4*pi then F_nu-to-F_lambda" in conventions


def test_all_atmosphere_cards_name_the_new_owner_and_primary_locator() -> None:
    for record in _records().values():
        assert "owner=jaxstro.spectra" in _conventions(record)
        sources = record["sources"]
        assert isinstance(sources, list) and sources
        for source in sources:
            assert isinstance(source, dict)
            assert str(source["reference"]).startswith(("https://doi.org/", "https://"))
            assert str(source["locator"]).strip()
            assert str(source["supports"]).strip()
