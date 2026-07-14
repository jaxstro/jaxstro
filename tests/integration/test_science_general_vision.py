"""Executable currency contracts for the science-general vision page."""

from __future__ import annotations

import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VISION_PAGE = (
    REPO_ROOT / "docs" / "70-project" / "direction" / "science-general-vision.md"
)


def _vision_text() -> str:
    return VISION_PAGE.read_text(encoding="utf-8")


def test_delivered_foundation_map_imports_every_former_candidate() -> None:
    text = _vision_text()
    delivered_modules = (
        "jaxstro.geometry",
        "jaxstro.provenance",
        "jaxstro.numerics.autodiff",
        "jaxstro.numerics.distributions",
        "jaxstro.numerics.meshes",
        "jaxstro.numerics.ode",
        "jaxstro.numerics.operators",
        "jaxstro.numerics.optimization",
        "jaxstro.numerics.random",
    )

    for module in delivered_modules:
        assert f"`{module}`" in text
        importlib.import_module(module)


def test_vision_separates_delivered_capabilities_from_future_admission() -> None:
    text = _vision_text()

    assert "## Delivered foundation map" in text
    assert "## Admission criteria for future work" in text
    assert "## Future core modules" not in text
    assert "natural extensions of the current foundation" not in text
    assert "should become the package" not in text


def test_vision_keeps_units_quantity_and_ecosystem_status_current() -> None:
    text = _vision_text()

    assert "`jaxstro.units` remains the current ecosystem contract" in text
    assert "`jaxstro.quantity` is implemented" in text
    assert "ecosystem adoption and any replacement cutover remain deferred" in text
    assert "most important missing user-facing abstraction" not in text
    assert "Startrax is active" in text
    assert "Stellax remains planned" in text


def test_vision_reuses_architecture_evidence_without_duplicate_figure() -> None:
    text = _vision_text()
    normalized = " ".join(text.split())

    assert "./architecture.md#fig-jaxstro-foundation" in text
    assert ":::{figure}" not in text
    assert "only where each method documents that transform" in normalized
