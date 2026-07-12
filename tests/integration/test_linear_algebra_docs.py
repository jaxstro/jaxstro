"""Executable pedagogy contracts for the linear-algebra theory page."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LINEAR_ALGEBRA_PAGE = REPO_ROOT / "docs" / "10-theory" / "linear-algebra.md"
BIBLIOGRAPHY = REPO_ROOT / "docs" / "99-bibliography" / "references.bib"
DESIGN_RECORD = (
    REPO_ROOT
    / "laboratory"
    / "jaxtroviz"
    / "design"
    / "2026-07-11-linear-algebra-contracts.md"
)


def _page_text() -> str:
    return LINEAR_ALGEBRA_PAGE.read_text(encoding="utf-8")


def _first_python_block() -> str:
    match = re.search(r"```python\n(?P<code>.*?)\n```", _page_text(), re.DOTALL)
    assert match is not None, "linear-algebra page needs a standalone Python example"
    return match.group("code")


def test_linear_algebra_example_is_standalone_and_executable() -> None:
    block = _first_python_block()
    for definition in ("design =", "observations =", "weights =", "matrix ="):
        assert definition in block

    namespace: dict[str, object] = {}
    exec(compile(block, str(LINEAR_ALGEBRA_PAGE), "exec"), namespace)

    np.testing.assert_allclose(namespace["weighted_coeffs"], [1.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(namespace["unweighted_coeffs"], [-1.6, 5.9], atol=1e-12)
    np.testing.assert_allclose(
        namespace["qr_coeffs"], namespace["svd_coeffs"], atol=1e-12
    )
    correlation = np.asarray(namespace["correlation"])
    assert correlation.shape == (3, 3)
    assert np.isfinite(correlation).all()
    np.testing.assert_allclose(correlation[2], 0.0, atol=0.0)
    assert bool(namespace["success"])
    assert float(namespace["jitter"]) == pytest.approx(0.1)
    shifted = np.asarray(namespace["shifted"])
    assert np.linalg.eigvalsh(shifted).min() > 0.0


def test_linear_algebra_page_names_live_gradient_and_boundary_contracts() -> None:
    text = _page_text()

    assert "```{list-table} Linear-algebra differentiation contracts" in text
    assert ":label: tbl-linear-algebra-contracts" in text
    for phrase in (
        "Norm and projection at regular points",
        "Weighted least squares with fixed full-rank design",
        "Zero-weight observation",
        "QR/SVD solve inside a fixed full-rank/cutoff regime",
        "Rank changes, SVD cutoff crossings, and condition numbers",
        "Zero variance and jitter selection",
        "`smooth_pathwise`",
        "`known_zero`",
        "`validation_only`",
        "value-dependent eager validation is skipped while inputs are traced",
    ):
        assert phrase in text


def test_linear_algebra_claims_have_verified_sources() -> None:
    text = _page_text()
    bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")

    assert "{cite:t}`GolubVanLoan2013`" in text
    assert "{cite:t}`ChengHigham1998`" in text
    assert "@book{GolubVanLoan2013" in bibliography
    assert "10.56021/9781421407944" in bibliography
    assert "@article{ChengHigham1998" in bibliography
    assert "10.1137/S0895479896302898" in bibliography


def test_linear_algebra_page_embeds_registered_figure_and_evidence_routes() -> None:
    text = _page_text()

    assert "./figures/linear-algebra-contracts.webp" in text
    assert ":name: fig-linear-algebra-contracts" in text
    assert (
        ":alt: Four regression observations with an outlier and measured "
        "weighted and unweighted fits, beside matrix eigenvalues before and "
        "after selected diagonal jitter"
    ) in text
    assert DESIGN_RECORD.is_file()
    assert "[](../40-api/index.md#jaxstro-numerics-linear-algebra)" in text
    assert "[](../60-validation/index.md)" in text
    assert "[](./index.md#gradient-contracts)" in text
