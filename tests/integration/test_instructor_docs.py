"""Course-use, assessment, and navigation contracts for instructor materials."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def test_instructor_guide_supports_uneven_preparation_and_active_science() -> None:
    text = (DOCS / "80-instructor/teaching-with-jaxstro.md").read_text(encoding="utf-8")
    normalized = text.lower()
    for phrase in (
        "research students",
        "uneven preparation",
        "predict -> compute -> audit -> state the warranted claim",
        "common misconception",
        "accessibility",
        "astronomy extension",
        "computational-science extension",
        "root-values-and-sensitivities",
        "powerlaw-removable-limit",
        "interpolation-boundary-policies",
    ):
        assert phrase.lower() in normalized


def test_assessment_rubric_grades_reasoning_and_claims_not_only_numbers() -> None:
    text = (DOCS / "80-instructor/assessment-rubric.md").read_text(encoding="utf-8")
    for dimension in (
        "Prediction quality",
        "Model, units, and scale",
        "Method inspection",
        "Independent audit",
        "Derivative interpretation",
        "Provenance",
        "Failure analysis",
        "Warranted claim",
    ):
        assert dimension in text
    assert "A correct scalar with unsupported reasoning is not complete" in text


def test_instructor_pages_are_navigable_and_manifest_routes_are_resolved() -> None:
    myst = (DOCS / "myst.yml").read_text(encoding="utf-8")
    routes = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))
    for page, route in (
        ("80-instructor/instructor-resources.md", "/instructor-resources"),
        ("80-instructor/teaching-with-jaxstro.md", "/teaching-with-jaxstro"),
        ("80-instructor/assessment-rubric.md", "/assessment-rubric"),
    ):
        assert myst.count(f"file: {page}") == 1
        assert routes[page] == route
