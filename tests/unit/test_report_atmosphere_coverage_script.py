"""Tests for the atmosphere coverage reporting CLI helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from jaxstro.atmospheres.library import AtmosphereCatalogCoverage

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "report_atmosphere_coverage.py"
)


def _load_reporter():
    spec = importlib.util.spec_from_file_location(
        "report_atmosphere_coverage", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reporter_formats_coverage_and_optional_acquisition_table():
    reporter = _load_reporter()
    text = reporter.format_report(
        [
            AtmosphereCatalogCoverage(
                dataset="synthetic",
                state="processed",
                n_spectra=2,
                teff_min=1.0,
                teff_max=2.0,
                logg_min=3.0,
                logg_max=4.0,
                wavelength_min=10.0,
                wavelength_max=20.0,
                wavelength_unit="nm",
                backend_name="newera",
                backend_available=True,
            )
        ],
        include_acquisition=True,
    )

    assert "## Coverage" in text
    assert "| synthetic | processed |" in text
    assert "## Targeted Acquisition Decision Table" in text
