"""Executable content contracts for scalar-root documentation."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADR = REPO_ROOT / "docs" / "30-decisions" / "0008-reject-ift-from-core.md"
ROOTFINDING = REPO_ROOT / "docs" / "10-theory" / "rootfinding.md"


def test_adr_distinguishes_information_field_theory_from_implicit_function_theorem():
    text = ADR.read_text(encoding="utf-8")

    assert "Information Field Theory" in text
    assert "implicit function theorem" in text
    assert "does not prohibit" in text


def test_rootfinding_docs_describe_actual_interpolation_order():
    text = ROOTFINDING.read_text(encoding="utf-8")

    assert "inverse-quadratic interpolation when three distinct" in text
    assert "otherwise the endpoint secant" in text
    assert "rejected selected interpolant" in text
    assert "Selected interpolant rejected" in text


def test_numerical_pages_expand_implicit_function_theorem_before_ift() -> None:
    paths = (
        ROOTFINDING,
        REPO_ROOT / "docs" / "40-api" / "index.md",
        REPO_ROOT / "docs" / "60-validation" / "index.md",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "implicit function theorem (IFT)" in text, path


def test_rootfinding_page_embeds_accessible_evidence_figures_and_activity() -> None:
    text = ROOTFINDING.read_text(encoding="utf-8")

    assert "./figures/rootfinding-safeguards.webp" in text
    assert "./figures/rootfinding-value-versus-ift.webp" in text
    assert text.count(":alt:") >= 2
    assert "Predict → compute → audit: which derivative are you asking for?" in text
    assert "Metric identity" in text
    assert "Central-FD root sensitivity" in text


def test_rootfinding_metric_table_matches_generated_quadratic_evidence() -> None:
    text = ROOTFINDING.read_text(encoding="utf-8")
    payload = json.loads(
        (REPO_ROOT / "docs" / "validation" / "implicit-root-gradients.json").read_text(
            encoding="utf-8"
        )
    )
    quadratic = next(case for case in payload["cases"] if case["name"] == "quadratic")
    for metric in (
        "root",
        "absolute_residual",
        "bracket_width",
        "ad_derivative",
        "fd_derivative",
    ):
        assert str(quadratic[metric]["value"]) in text


def test_rootfinding_derivative_guidance_separates_executed_and_implicit_maps() -> None:
    text = ROOTFINDING.read_text(encoding="utf-8")

    assert "sensitivity of its smooth finite executed iteration" in text
    assert "unique smooth mathematical root" in text
    assert "not a generic implicit-root" in text
    assert "opposite-sign endpoint" in text
    assert "invariant is checked" in text
    assert "finite executed-map sensitivity" in text
    assert "certified mathematical-root sensitivity" in text
    assert "smooth iterates, finite-map gradients" in text
