"""Executable currency contracts for the website landing page."""

from __future__ import annotations

import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LANDING_PAGE = REPO_ROOT / "docs" / "index.md"


def _landing_text() -> str:
    return LANDING_PAGE.read_text(encoding="utf-8")


def test_landing_lists_importable_public_modules() -> None:
    """Every module advertised on the landing page must import successfully."""
    text = _landing_text()
    public_modules = (
        "astrometry",
        "atmospheres",
        "constants",
        "coords",
        "geometry",
        "jaxconfig",
        "numerics",
        "params",
        "provenance",
        "quantity",
        "spatial",
        "testing",
        "units",
    )

    for module in public_modules:
        assert f"`{module}`" in text
        importlib.import_module(f"jaxstro.{module}")


def test_landing_distinguishes_current_units_from_deferred_quantity_adoption() -> None:
    text = _landing_text()

    assert "`jaxstro.units` is the current canonical ecosystem contract" in text
    assert "`jaxstro.quantity` is implemented" in text
    assert "ecosystem adoption and any replacement cutover are deferred" in text
    assert "planned `jaxstro.quantity`" not in text


def test_landing_routes_and_ecosystem_status_are_current() -> None:
    text = _landing_text()

    assert "(three-doors)=" in text
    assert "(two-doors)=" not in text
    assert "planned startrax" not in text.lower()
    assert "./40-api/provenance/index.md" in text
    assert "./60-validation/index.md" in text
    assert "./20-methods/discrete-space/spatial.md" in text
    assert "./30-representations/spectra-atmospheres/atmosphere-capabilities.md" in text
    assert "spatial" in text.lower()
