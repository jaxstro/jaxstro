"""Executable currency contracts for the architecture landing page."""

from __future__ import annotations

import importlib
from pathlib import Path

from jaxstro._public import PUBLIC_MODULES

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_PAGE = REPO_ROOT / "docs" / "70-project" / "direction" / "architecture.md"


def _architecture_text() -> str:
    return ARCHITECTURE_PAGE.read_text(encoding="utf-8")


def test_architecture_inventory_matches_importable_modules() -> None:
    text = _architecture_text()
    for module in PUBLIC_MODULES:
        assert f"`{module}`" in text
        importlib.import_module(f"jaxstro.{module}")


def test_architecture_states_current_ownership_boundaries() -> None:
    text = _architecture_text()

    assert "This section will tell" not in text
    assert "`jaxstro.units` remains the current canonical ecosystem contract" in text
    assert "ecosystem adoption and any replacement cutover remain deferred" in text
    assert "`jaxstro.quantity` is the planned" not in text
    assert "Atmosphere support remains in progress" in text
    assert "host-side, discrete preprocessing" in text


def test_architecture_distinguishes_two_provenance_surfaces() -> None:
    text = _architecture_text()

    assert "source-backed provenance cards" in text
    assert "runtime artifact manifests" in text
    assert (
        "../../50-api/research-infrastructure/source-provenance/source-provenance.md"
        in text
    )
    assert "../../50-api/research-infrastructure/provenance.md" in text


def test_architecture_embeds_accessible_registered_figure() -> None:
    text = _architecture_text()

    assert "./figures/jaxstro-foundation.webp" in text
    assert ":name: fig-jaxstro-foundation" in text
    assert (
        ":alt: One-way package dependency diagram with downstream astronomy "
        "packages depending on the jaxstro foundation"
    ) in text
