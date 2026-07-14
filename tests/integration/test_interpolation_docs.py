"""Executable pedagogy contracts for the cubic interpolation theory page."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERPOLATION_PAGE = (
    REPO_ROOT / "docs" / "20-methods" / "approximation-integration" / "interpolation.md"
)
BIBLIOGRAPHY = REPO_ROOT / "docs" / "99-bibliography" / "references.bib"
DESIGN_RECORD = (
    REPO_ROOT
    / "laboratory"
    / "jaxtroviz"
    / "design"
    / "2026-07-11-interpolation-shape-contracts.md"
)


def _page_text() -> str:
    return INTERPOLATION_PAGE.read_text(encoding="utf-8")


def _first_python_block() -> str:
    match = re.search(r"```python\n(?P<code>.*?)\n```", _page_text(), re.DOTALL)
    assert match is not None, "interpolation page needs a standalone Python example"
    return match.group("code")


def test_interpolation_example_is_standalone_and_executable() -> None:
    block = _first_python_block()
    for placeholder in ("x_grid", "values", "dydx", "x_new"):
        assert f"{placeholder} =" in block

    namespace: dict[str, object] = {}
    exec(compile(block, str(INTERPOLATION_PAGE), "exec"), namespace)

    natural = np.asarray(namespace["natural"])
    monotone = np.asarray(namespace["monotone"])
    hermite = np.asarray(namespace["hermite"])
    wrapped = np.asarray(namespace["wrapped_monotone"])
    assert natural.min() < -0.1
    assert monotone.min() >= -1e-12
    assert monotone.max() <= 1.0 + 1e-12
    assert np.diff(monotone).min() >= -1e-12
    np.testing.assert_allclose(hermite, monotone, atol=1e-12)
    np.testing.assert_allclose(wrapped, monotone, atol=1e-12)


def test_interpolation_page_names_live_gradient_boundaries() -> None:
    text = _page_text()

    assert "```{list-table} Interpolation gradient contracts" in text
    assert ":label: tbl-interpolation-gradient-contracts" in text
    for phrase in (
        "Hermite values and supplied derivatives",
        "Natural-spline values and interior query",
        "PCHIP inside a fixed limiter branch",
        "Clamped exterior query",
        "Knots and limiter transitions",
        "`smooth_pathwise`",
        "`known_zero`",
        "`validation_only`",
    ):
        assert phrase in text


def test_interpolation_page_states_validation_and_extrapolation_boundaries() -> None:
    text = _page_text()

    assert "eager validation is skipped while the grid is traced" in text
    assert "the caller must supply a strictly increasing grid under `jax.jit`" in text
    assert "numerical continuation, not a physical guarantee" in text
    assert "monotone data" in text


def test_interpolation_equations_have_verified_primary_provenance() -> None:
    text = _page_text()
    bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")

    assert "{cite:t}`deBoor2001`" in text
    assert "{cite:t}`FritschButland1984`" in text
    assert "@book{deBoor2001" in bibliography
    assert "10.1007/978-1-4612-6333-3" in bibliography
    assert "@article{FritschButland1984" in bibliography
    assert "10.1137/0905021" in bibliography


def test_interpolation_page_embeds_figure_and_evidence_routes() -> None:
    text = _page_text()

    assert "../../10-theory/figures/interpolation-shape-contracts.webp" in text
    assert ":name: fig-interpolation-shape-contracts" in text
    assert (
        ":alt: Two-panel comparison of natural cubic and PCHIP interpolation "
        "for the same monotone samples, showing natural-spline undershoot and "
        "nonnegative PCHIP increments"
    ) in text
    assert DESIGN_RECORD.is_file()
    assert "[](../../50-api/approximation-integration/interpolation.md)" in text
    assert "[](../../60-validation/index.md)" in text
    assert "[](../methods.md#gradient-contracts)" in text
