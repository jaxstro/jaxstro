"""Tests for catalog-first atmosphere library selection."""

from __future__ import annotations

import pytest

from jaxstro.atmospheres import AtmosphereParams
from jaxstro.atmospheres.library import (
    AtmosphereCatalogCoverage,
    AtmosphereLibrary,
)


def test_library_selects_ranked_processed_backend_candidate():
    library = AtmosphereLibrary.from_coverages(
        [
            AtmosphereCatalogCoverage(
                dataset="newera_lowres_v3",
                state="processed",
                n_spectra=12,
                teff_min=2300.0,
                teff_max=12000.0,
                logg_min=0.0,
                logg_max=6.0,
                m_h_values=(-0.5, 0.0),
                alpha_m_values=(0.0,),
                wavelength_min=250.0,
                wavelength_max=2500.0,
                wavelength_unit="nm",
                backend_name="newera",
                backend_available=True,
            ),
            AtmosphereCatalogCoverage(
                dataset="sonora_2024",
                state="raw",
                n_spectra=4,
                teff_min=900.0,
                teff_max=2400.0,
                logg_min=3.5,
                logg_max=5.5,
                m_h_values=(0.0,),
                wavelength_min=0.3,
                wavelength_max=250.0,
                wavelength_unit="micron",
                backend_name=None,
                backend_available=False,
            ),
        ]
    )

    selection = library.select(
        AtmosphereParams(teff=2350.0, logg=4.5, m_h=0.0, alpha_m=0.0)
    )

    assert selection.status == "ok"
    assert selection.selected is not None
    assert selection.selected.dataset == "newera_lowres_v3"
    assert selection.selected.backend_name == "newera"
    assert selection.selected.rank_reason == "processed backend covers request"
    assert [candidate.dataset for candidate in selection.candidates] == [
        "newera_lowres_v3",
        "sonora_2024",
    ]


def test_library_reports_matching_raw_only_candidate_without_selecting_backend():
    library = AtmosphereLibrary.from_coverages(
        [
            AtmosphereCatalogCoverage(
                dataset="sonora_2024",
                state="raw",
                n_spectra=4,
                teff_min=900.0,
                teff_max=2400.0,
                logg_min=3.5,
                logg_max=5.5,
                m_h_values=(0.0,),
                cloud_labels=("f2",),
                wavelength_min=0.3,
                wavelength_max=250.0,
                wavelength_unit="micron",
                backend_name=None,
                backend_available=False,
            )
        ]
    )

    selection = library.select(
        AtmosphereParams(teff=1200.0, logg=4.0, m_h=0.0),
        family="sonora",
    )

    assert selection.status == "backend_unavailable"
    assert selection.selected is None
    assert len(selection.candidates) == 1
    assert selection.candidates[0].dataset == "sonora_2024"
    assert selection.candidates[0].backend_available is False
    assert selection.candidates[0].rank_reason == "coverage match without backend"


def test_library_returns_no_match_with_reasons():
    library = AtmosphereLibrary.from_coverages(
        [
            AtmosphereCatalogCoverage(
                dataset="bosz_2025_recomputed",
                state="processed",
                n_spectra=4,
                teff_min=2800.0,
                teff_max=16000.0,
                logg_min=-0.5,
                logg_max=5.5,
                m_h_values=(0.0,),
                alpha_m_values=(0.0,),
                c_m_values=(0.0,),
                vturb_values_km_s=(2.0,),
                resolution="r10000",
                wavelength_min=500.0,
                wavelength_max=319986.22,
                wavelength_unit="angstrom",
                backend_name="bosz",
                backend_available=True,
            )
        ]
    )

    selection = library.select(
        AtmosphereParams(teff=20000.0, logg=4.0, m_h=0.0),
        resolution="r10000",
    )

    assert selection.status == "no_match"
    assert selection.selected is None
    assert selection.candidates == ()
    assert "teff outside" in selection.message


def test_library_spectrum_refuses_when_no_backend_matches():
    library = AtmosphereLibrary.from_coverages(
        [
            AtmosphereCatalogCoverage(
                dataset="sonora_2024",
                state="raw",
                n_spectra=1,
                teff_min=900.0,
                teff_max=2400.0,
                logg_min=3.5,
                logg_max=5.5,
                wavelength_min=0.3,
                wavelength_max=250.0,
                wavelength_unit="micron",
            )
        ]
    )

    with pytest.raises(RuntimeError, match="No processed atmosphere backend"):
        library.spectrum(AtmosphereParams(teff=1000.0, logg=4.0))
