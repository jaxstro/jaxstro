"""Executable pedagogy contracts for the B-spline theory page."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BSPLINE_PAGE = (
    REPO_ROOT / "docs" / "20-methods" / "approximation-integration" / "bsplines.md"
)
BIBLIOGRAPHY = REPO_ROOT / "docs" / "99-bibliography" / "references.bib"
DESIGN_RECORD = (
    REPO_ROOT
    / "laboratory"
    / "jaxtroviz"
    / "design"
    / "2026-07-11-bspline-local-support.md"
)


def _page_text() -> str:
    return BSPLINE_PAGE.read_text(encoding="utf-8")


def _first_python_block() -> str:
    match = re.search(r"```python\n(?P<code>.*?)\n```", _page_text(), re.DOTALL)
    assert match is not None, "B-spline page needs a standalone Python example"
    return match.group("code")


def test_bspline_quick_example_is_standalone_and_executable() -> None:
    block = _first_python_block()
    assert "..." not in block

    namespace: dict[str, object] = {}
    exec(compile(block, str(BSPLINE_PAGE), "exec"), namespace)

    basis = np.asarray(namespace["basis"])
    values = np.asarray(namespace["values"])
    derivative = np.asarray(namespace["derivative"])
    wrapped_values = np.asarray(namespace["wrapped_values"])
    assert basis.shape == (9, 6)
    np.testing.assert_allclose(basis.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(values, wrapped_values, atol=1e-6)
    assert np.isfinite(derivative).all()


def test_bspline_page_names_the_live_gradient_boundaries() -> None:
    text = _page_text()

    assert "```{list-table} B-spline gradient contracts" in text
    assert ":label: tbl-bspline-gradient-contracts" in text
    for phrase in (
        "Coefficients at fixed knots",
        "Interior query coordinate",
        "Clamped exterior coordinate",
        "Knot boundaries",
        "Quantile knot construction",
        "`smooth_pathwise`",
        "`known_zero`",
        "`validation_only`",
    ):
        assert phrase in text


def test_bspline_equations_have_verified_primary_provenance() -> None:
    text = _page_text()
    bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")

    assert "{cite:t}`deBoor1972`" in text
    assert "@article{deBoor1972" in bibliography
    assert "10.1016/0021-9045(72)90080-9" in bibliography


def test_bspline_page_embeds_the_registered_accessible_figure() -> None:
    text = _page_text()

    assert "../../10-theory/figures/bspline-local-support.webp" in text
    assert ":name: fig-bspline-local-support" in text
    assert (
        ":alt: Six cubic B-spline basis curves with local support and their "
        "sum equal to one across the active domain"
    ) in text
    assert DESIGN_RECORD.is_file()


def test_bspline_page_routes_to_api_validation_and_contract_taxonomy() -> None:
    text = _page_text()

    assert "[](../../50-api/approximation-integration/splines.md)" in text
    assert "[](../../60-validation/index.md)" in text
    assert "[](../methods.md#gradient-contracts)" in text
