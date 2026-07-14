"""Curriculum contracts for the predict-compute-audit learning page."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs" / "00-start-here" / "first-research-calculation.md"


def test_learning_page_explains_the_full_reasoning_cycle() -> None:
    text = PAGE.read_text(encoding="utf-8")
    required = (
        "# How to learn with Jaxstro: predict, compute, audit",
        "## Predict",
        "## Compute",
        "## Audit",
        "Prediction prevents post-hoc storytelling",
        "A finite output is not yet a scientific result",
        "The audit starts the next prediction",
        "safeguarded_bracketed_root",
        "powerlaw_cdf",
    )
    for phrase in required:
        assert phrase in text


def test_learning_page_is_wired_into_start_here_without_expanding_home() -> None:
    myst = (REPO_ROOT / "docs" / "myst.yml").read_text(encoding="utf-8")
    homepage = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    start_here = (REPO_ROOT / "docs" / "00-start-here" / "start-here.md").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(
        (REPO_ROOT / "docs" / "route-manifest.json").read_text(encoding="utf-8")
    )

    assert myst.count("00-start-here/first-research-calculation.md") == 1
    assert "./00-start-here/first-research-calculation.md" not in homepage
    assert "](./00-start-here/start-here.md)" in homepage
    assert start_here.count("./first-research-calculation.md") == 1
    assert manifest["00-start-here/first-research-calculation.md"] == (
        "/first-research-calculation"
    )
