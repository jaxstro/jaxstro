"""Executable content contracts for scalar-root documentation."""

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
