"""Executable pedagogy contracts for the regular-grid theory page."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REGULAR_GRID_PAGE = REPO_ROOT / "docs" / "10-theory" / "regular-grid.md"
BIBLIOGRAPHY = REPO_ROOT / "docs" / "99-bibliography" / "references.bib"
DESIGN_RECORD = (
    REPO_ROOT
    / "laboratory"
    / "jaxtroviz"
    / "design"
    / "2026-07-11-regular-grid-contracts.md"
)


def _page_text() -> str:
    return REGULAR_GRID_PAGE.read_text(encoding="utf-8")


def _first_python_block() -> str:
    match = re.search(r"```python\n(?P<code>.*?)\n```", _page_text(), re.DOTALL)
    assert match is not None, "regular-grid page needs a standalone Python example"
    return match.group("code")


def test_regular_grid_example_is_standalone_and_executable() -> None:
    block = _first_python_block()
    for definition in ("x_axis =", "y_axis =", "values =", "xi ="):
        assert definition in block

    namespace: dict[str, object] = {}
    exec(compile(block, str(REGULAR_GRID_PAGE), "exec"), namespace)

    interpolated = np.asarray(namespace["interpolated"])
    expected = np.asarray(namespace["expected"])
    bilinear = np.asarray(namespace["bilinear"])
    clamped = np.asarray(namespace["clamped"])
    filled = np.asarray(namespace["filled"])
    assert interpolated.shape == expected.shape == (2, 2)
    np.testing.assert_allclose(interpolated, expected, atol=1e-12)
    np.testing.assert_allclose(bilinear, expected[:, 0], atol=1e-12)
    np.testing.assert_allclose(clamped, [0.5, 10.0], atol=1e-12)
    np.testing.assert_allclose(filled, [-99.0, -99.0], atol=0.0)


def test_regular_grid_page_names_live_gradient_and_boundary_contracts() -> None:
    text = _page_text()

    assert "```{list-table} Regular-grid interpolation contracts" in text
    assert ":label: tbl-regular-grid-contracts" in text
    for phrase in (
        "Values at fixed axes and interior queries",
        "Interior query coordinates",
        "Clamped or filled exterior coordinates",
        "Cell boundaries and axis locations",
        "Reject validation",
        "`smooth_pathwise`",
        "`known_zero`",
        "`validation_only`",
        "eager validation is skipped while axes or queries are traced",
        "`fill_value` is static under `jax.jit`",
    ):
        assert phrase in text


def test_regular_grid_equation_has_primary_provenance() -> None:
    text = _page_text()
    bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")

    assert "{cite:t}`WeiserZarantonello1988`" in text
    assert "@article{WeiserZarantonello1988" in bibliography
    assert "10.1090/S0025-5718-1988-0917826-0" in bibliography


def test_regular_grid_page_embeds_registered_figure_and_evidence_routes() -> None:
    text = _page_text()

    assert "./figures/regular-grid-contracts.webp" in text
    assert ":name: fig-regular-grid-contracts" in text
    assert (
        ":alt: Unit-square interpolation query connected to four corners with "
        "measured bilinear weights, beside clamp and fill outputs across the "
        "grid boundary"
    ) in text
    assert DESIGN_RECORD.is_file()
    assert "[](../40-api/index.md#jaxstro-numerics-regular-grid)" in text
    assert "[](../60-validation/index.md)" in text
    assert "[](./index.md#gradient-contracts)" in text
