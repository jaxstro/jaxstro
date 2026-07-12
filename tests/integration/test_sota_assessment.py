"""Coverage contracts for the package-wide SOTA assessment."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs" / "90-development-log" / "sota-assessment.md"


def test_sota_assessment_covers_package_and_website_dimensions() -> None:
    text = PAGE.read_text(encoding="utf-8")
    required = (
        "## Delivered strengths",
        "## High-confidence gaps",
        "## Now",
        "## Next",
        "## Later",
        "## Evidence required",
        "Scientific breadth",
        "Ownership discipline",
        "Numerical robustness",
        "Conditioning",
        "AD honesty",
        "JAX transform coverage",
        "Dimensional safety",
        "API cohesion",
        "Serialization",
        "Performance and compilation evidence",
        "Evidence freshness",
        "Provenance",
        "Curriculum quality",
        "Accessibility",
        "Discoverability",
        "Downstream reuse",
    )
    for phrase in required:
        assert phrase in text


def test_sota_assessment_uses_bounded_ranked_horizons() -> None:
    text = PAGE.read_text(encoding="utf-8")
    now = text.split("## Now", 1)[1].split("## Next", 1)[0]
    next_items = text.split("## Next", 1)[1].split("## Later", 1)[0]
    later = text.split("## Later", 1)[1].split("## Evidence required", 1)[0]

    assert now.count("### ") <= 5
    assert next_items.count("### ") <= 7
    assert later.count("### ") <= 5
    for section in (now, next_items, later):
        assert "**Impact.**" in section
        assert "**Evidence gate.**" in section


def test_sota_assessment_is_navigable() -> None:
    myst = (REPO_ROOT / "docs" / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads(
        (REPO_ROOT / "docs" / "route-manifest.json").read_text(encoding="utf-8")
    )

    assert myst.count("90-development-log/sota-assessment.md") == 1
    assert manifest["90-development-log/sota-assessment.md"] == "/sota-assessment"


def test_sota_assessment_calibrates_unfinished_infrastructure() -> None:
    text = PAGE.read_text(encoding="utf-8")

    assert "| Serialization | implemented |" in text
    assert "no public root round-trip/replay serializer is yet validated" in text
    assert "| Downstream reuse | implemented |" in text
    assert "pinned adoption and compatibility evidence is not yet" in text
    assert "First deliver the planned transform-contract or maturity\nregistry" in text
    assert "established specialized solvers as the default\nowner" in text
    assert "specialized libraries cannot own the solver" not in text
