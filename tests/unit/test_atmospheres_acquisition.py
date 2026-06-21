"""Tests for no-download targeted acquisition planning."""

from __future__ import annotations

from jaxstro.atmospheres.acquisition import (
    acquisition_rows_to_markdown,
    plan_targeted_acquisition,
)
from jaxstro.atmospheres.library import AtmosphereCatalogCoverage


def test_acquisition_plan_lists_missing_categories_without_downloads():
    rows = plan_targeted_acquisition(
        [
            AtmosphereCatalogCoverage(
                dataset="sonora_2024",
                state="processed",
                n_spectra=10,
                teff_min=900.0,
                teff_max=2400.0,
            ),
            AtmosphereCatalogCoverage(
                dataset="tlusty_ostar_2002",
                state="processed",
                n_spectra=10,
                teff_min=27500.0,
                teff_max=55000.0,
            ),
        ]
    )

    categories = [row.category for row in rows]
    assert "white dwarfs" in categories
    assert "WR / wind-dominated very hot stars" in categories
    assert "very cool objects below 900 K" in categories
    assert all(row.action != "download" for row in rows)


def test_acquisition_markdown_is_decision_table():
    markdown = acquisition_rows_to_markdown(
        plan_targeted_acquisition(
            [
                AtmosphereCatalogCoverage(
                    dataset="bosz",
                    state="processed",
                    n_spectra=1,
                    teff_min=2800.0,
                    teff_max=16000.0,
                    alpha_m_values=(0.0,),
                    c_m_values=(0.0,),
                    resolution="r10000",
                )
            ]
        )
    )

    assert markdown.splitlines()[0].startswith("| category |")
    assert "non-solar BOSZ alpha/carbon" in markdown
    assert "defer" in markdown
