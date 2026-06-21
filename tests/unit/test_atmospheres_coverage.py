"""Tests for deterministic atmosphere coverage summaries."""

from __future__ import annotations

import json

from jaxstro.atmospheres.coverage import (
    coverage_rows_to_json,
    coverage_rows_to_markdown,
    summarize_catalog_rows,
)


def test_summarize_catalog_rows_preserves_axes_and_units():
    coverage = summarize_catalog_rows(
        dataset="synthetic_bosz",
        rows=[
            {
                "teff": 3000.0,
                "logg": 4.0,
                "m_h": 0.0,
                "alpha_m": 0.0,
                "c_m": 0.0,
                "vturb_km_s": 2.0,
                "resolution": "r10000",
                "wavelength_min": 500.0,
                "wavelength_max": 900.0,
                "wavelength_unit": "angstrom",
            },
            {
                "teff": 3200.0,
                "logg": 4.5,
                "m_h": -0.5,
                "alpha_m": 0.0,
                "c_m": 0.0,
                "vturb_km_s": 2.0,
                "resolution": "r10000",
                "wavelength_min": 500.0,
                "wavelength_max": 900.0,
                "wavelength_unit": "angstrom",
            },
        ],
        state="processed",
        backend_name="bosz",
    )

    assert coverage.dataset == "synthetic_bosz"
    assert coverage.n_spectra == 2
    assert coverage.teff_min == 3000.0
    assert coverage.teff_max == 3200.0
    assert coverage.logg_min == 4.0
    assert coverage.logg_max == 4.5
    assert coverage.m_h_values == (-0.5, 0.0)
    assert coverage.resolution == "r10000"
    assert coverage.wavelength_unit == "angstrom"
    assert coverage.backend_available is True


def test_coverage_outputs_are_deterministic_and_inspectable():
    rows = [
        summarize_catalog_rows(
            dataset="b",
            rows=[{"teff": 1.0, "logg": 2.0, "lambda_min": 10.0, "lambda_max": 20.0}],
            state="raw",
        ),
        summarize_catalog_rows(
            dataset="a",
            rows=[{"teff": 3.0, "logg": 4.0, "lambda_min": 30.0, "lambda_max": 40.0}],
            state="processed",
            backend_name="newera",
        ),
    ]

    markdown = coverage_rows_to_markdown(rows)
    data = json.loads(coverage_rows_to_json(rows))

    assert markdown.splitlines()[0].startswith("| dataset |")
    assert markdown.splitlines()[2].startswith("| a |")
    assert markdown.splitlines()[3].startswith("| b |")
    assert [entry["dataset"] for entry in data] == ["a", "b"]
    assert data[0]["state"] == "processed"
