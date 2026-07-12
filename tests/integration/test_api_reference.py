"""Executable public-surface contracts for the API reference page."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import jaxstro

REPO_ROOT = Path(__file__).resolve().parents[2]
API_PAGE = REPO_ROOT / "docs" / "40-api" / "index.md"


def _api_text() -> str:
    return API_PAGE.read_text(encoding="utf-8")


def test_spatial_is_an_eager_top_level_public_module_in_a_clean_process() -> None:
    code = """
import jaxstro
assert hasattr(jaxstro, "spatial")
assert "spatial" in jaxstro.__all__
assert jaxstro.spatial.__name__ == "jaxstro.spatial"
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_documented_public_import_surface_is_executable() -> None:
    public_modules = (
        "astrometry",
        "atmospheres",
        "constants",
        "coords",
        "geometry",
        "numerics",
        "params",
        "provenance",
        "quantity",
        "spatial",
        "testing",
        "units",
    )

    for module in public_modules:
        imported = importlib.import_module(f"jaxstro.{module}")
        assert getattr(jaxstro, module) is imported
        assert module in jaxstro.__all__

    from jaxstro.jaxconfig import enable_high_precision

    assert callable(enable_high_precision)


def test_api_table_has_structured_ownership_boundary_and_evidence_fields() -> None:
    text = _api_text()

    assert "```{list-table} Public modules" in text
    assert "  - Ownership" in text
    assert "  - Runtime / preprocessing boundary" in text
    assert "  - Evidence and status" in text
    assert "`jaxstro.units` is the current ecosystem contract" in text
    assert "`jaxstro.quantity` is implemented" in text
    assert "ecosystem adoption and any replacement cutover remain deferred" in text
    assert "Atmosphere support is in progress" in text
    assert "../10-theory/spatial.md" in text


def test_api_reference_exposes_provenance_card_tooling_and_routes() -> None:
    text = _api_text()

    for symbol in ("ProvenanceCard", "validate_card", "render_card", "render_registry"):
        assert getattr(jaxstro.testing, symbol) is not None
        assert symbol in jaxstro.testing.__all__
        assert f"`{symbol}" in text

    assert "./provenance/index.md" in text
    assert "source-backed provenance cards" in text
    assert "runtime manifests" in text


def test_interpolation_reference_does_not_duplicate_symbol_descriptions() -> None:
    text = _api_text()

    assert text.count("`pchip_slopes(...)`") == 1
    assert text.count("`monotone_cubic_interp(...)`") == 1
