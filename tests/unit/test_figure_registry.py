"""Contracts for jaxstro's optional documentation-figure laboratory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")
pytest.importorskip("PIL")

from laboratory.jaxtroviz.cli import main  # noqa: E402
from laboratory.jaxtroviz.registry import FIGURES  # noqa: E402
from laboratory.jaxtroviz.style import render_webp_bytes  # noqa: E402


def test_architecture_figure_is_registered_for_the_approved_page() -> None:
    spec = FIGURES["jaxstro-foundation"]

    assert spec.page == "20-architecture/index.md"
    assert spec.seed == 0
    assert spec.stem == "jaxstro-foundation"
    assert spec.site_path == "docs/20-architecture/figures/jaxstro-foundation.webp"
    assert spec.site_webp == (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "20-architecture"
        / "figures"
        / "jaxstro-foundation.webp"
    )


def test_architecture_figure_contains_ownership_labels() -> None:
    figure = FIGURES["jaxstro-foundation"].builder()
    labels = {
        text.get_text()
        for text in figure.findobj(match=lambda item: hasattr(item, "get_text"))
    }

    assert {
        "Domain packages",
        "jaxstro foundation",
        "JAX + Equinox + jaxtyping",
        "spatial + atmosphere selection: host-side / discrete boundary",
        "depends on",
        "built on",
    } <= labels


def test_spatial_figure_is_registered_for_the_theory_page() -> None:
    spec = FIGURES["spatial-neighbor-contracts"]

    assert spec.page == "10-theory/spatial.md"
    assert spec.seed == 0
    assert spec.site_path == "docs/10-theory/figures/spatial-neighbor-contracts.webp"
    labels = {
        text.get_text()
        for text in spec.builder().findobj(match=lambda item: hasattr(item, "get_text"))
    }
    assert {"Candidate pool", "Exact radius filter"} <= labels
    assert any("did_overflow = False" in label for label in labels)


def test_bspline_figure_is_registered_and_computed_from_public_basis() -> None:
    spec = FIGURES["bspline-local-support"]

    assert spec.page == "10-theory/bsplines.md"
    assert spec.seed == 0
    assert spec.site_path == "docs/10-theory/figures/bspline-local-support.webp"

    from laboratory.jaxtroviz.bsplines import basis_results

    x, basis, basis_sum = basis_results()
    assert x.shape == (401,)
    assert basis.shape == (401, 6)
    assert basis_sum.shape == (401,)
    assert np.all(basis >= 0.0)
    np.testing.assert_allclose(basis_sum, 1.0, atol=1e-6)

    labels = {
        text.get_text()
        for text in spec.builder().findobj(match=lambda item: hasattr(item, "get_text"))
    }
    assert {"Local cubic basis functions", "Partition of unity"} <= labels


def test_bspline_figure_enables_x64_before_knot_allocation() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from laboratory.jaxtroviz.bsplines import basis_results; "
                "print(basis_results()[0].dtype)"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "float64"
    assert "will be truncated to dtype float32" not in result.stderr


def test_interpolation_figure_is_registered_and_uses_public_results() -> None:
    spec = FIGURES["interpolation-shape-contracts"]

    assert spec.page == "10-theory/interpolation.md"
    assert spec.seed == 0
    assert spec.site_path == "docs/10-theory/figures/interpolation-shape-contracts.webp"

    from laboratory.jaxtroviz.interpolation import interpolation_results

    x_grid, values, x_query, natural, monotone = interpolation_results()
    assert x_grid.shape == values.shape == (5,)
    assert x_query.shape == natural.shape == monotone.shape == (801,)
    assert x_query.dtype == np.float64
    assert np.all(np.diff(values) >= 0.0)
    assert natural.min() < -0.1
    assert monotone.min() >= -1e-12
    assert monotone.max() <= 1.0 + 1e-12
    assert np.diff(monotone).min() >= -1e-12

    labels = {
        text.get_text()
        for text in spec.builder().findobj(match=lambda item: hasattr(item, "get_text"))
    }
    assert {"Same monotone samples", "Step-by-step monotonicity"} <= labels


def test_regular_grid_figure_is_registered_and_uses_public_results() -> None:
    spec = FIGURES["regular-grid-contracts"]

    assert spec.page == "10-theory/regular-grid.md"
    assert spec.seed == 0
    assert spec.site_path == "docs/10-theory/figures/regular-grid-contracts.webp"

    from laboratory.jaxtroviz.regular_grid import regular_grid_results

    query, weights, scan, clamped, filled = regular_grid_results()
    assert query.shape == (2,)
    assert weights.shape == (4,)
    assert scan.shape == clamped.shape == filled.shape == (301,)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(weights, [0.245, 0.455, 0.105, 0.195], atol=1e-12)
    assert np.all(np.isfinite(clamped))
    assert np.all(filled[(scan < 0.0) | (scan > 1.0)] == -1.0)

    labels = {
        text.get_text()
        for text in spec.builder().findobj(match=lambda item: hasattr(item, "get_text"))
    }
    assert {"Bilinear corner weights", "Boundary policies"} <= labels


def test_linear_algebra_figure_is_registered_and_uses_public_results() -> None:
    spec = FIGURES["linear-algebra-contracts"]

    assert spec.page == "10-theory/linear-algebra.md"
    assert spec.seed == 0
    assert spec.site_path == "docs/10-theory/figures/linear-algebra-contracts.webp"

    from laboratory.jaxtroviz.linear_algebra import linear_algebra_results

    (
        x,
        y,
        unweighted,
        weighted,
        eigenvalues_before,
        eigenvalues_after,
        jitter,
        success,
    ) = linear_algebra_results()
    assert x.shape == y.shape == (4,)
    np.testing.assert_allclose(unweighted, [-1.6, 5.9], atol=1e-12)
    np.testing.assert_allclose(weighted, [1.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(eigenvalues_before, [-0.03, 2.0], atol=1e-12)
    np.testing.assert_allclose(eigenvalues_after, [0.07, 2.1], atol=1e-12)
    assert jitter == pytest.approx(0.1)
    assert success

    labels = {
        text.get_text()
        for text in spec.builder().findobj(match=lambda item: hasattr(item, "get_text"))
    }
    assert {"Weight changes the fit", "Jitter crosses the PD boundary"} <= labels


def test_architecture_webp_render_is_deterministic_and_committed() -> None:
    spec = FIGURES["jaxstro-foundation"]
    first = render_webp_bytes(spec.builder())
    second = render_webp_bytes(spec.builder())

    assert first == second
    assert spec.site_webp.is_file()
    assert spec.site_webp.stat().st_size > 10_000


def test_every_registered_webp_is_deterministic_and_committed() -> None:
    for spec in FIGURES.values():
        first = render_webp_bytes(spec.builder(), spec=spec.export)
        second = render_webp_bytes(spec.builder(), spec=spec.export)
        assert first == second
        assert spec.site_webp.is_file()


def test_cli_lists_and_checks_registered_figures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--list"]) == 0
    listed = capsys.readouterr().out
    assert "jaxstro-foundation" in listed
    assert "spatial-neighbor-contracts" in listed
    assert "bspline-local-support" in listed
    assert "interpolation-shape-contracts" in listed
    assert "regular-grid-contracts" in listed
    assert "linear-algebra-contracts" in listed

    assert main(["--check"]) == 0


def test_old_figure_namespace_is_removed() -> None:
    assert not (Path(__file__).resolve().parents[2] / "laboratory" / "figures").exists()
