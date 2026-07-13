"""Contracts for the future capabilities and module roadmap."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "90-development-log" / "future-capabilities-roadmap.md"


def test_future_capabilities_roadmap_preserves_scope_and_build_advice() -> None:
    text = PAGE.read_text(encoding="utf-8")

    required = (
        "# Future modules and capabilities roadmap",
        "## Existing methods and features",
        "## What Jaxstro should become",
        "## Recommended additions",
        "### Priority 1: a narrow `jaxstro.ml` scientific-ML substrate",
        "### Priority 2: matrix-free iterative linear solvers",
        "### Priority 3: vector nonlinear systems and fixed points",
        "### Priority 4: quasi-Monte Carlo and variance reduction",
        "### Priority 5: adaptive quadrature",
        "### Priority 6: signal and time-series primitives",
        "### Priority 7: generic uncertainty propagation",
        "### Priority 8: multidimensional field methods",
        "## What not to add",
        "## Build checklist",
        "predict → compute → audit → state the warranted claim",
        "Informax",
    )
    for phrase in required:
        assert phrase in text

    assert text.count("- [ ]") >= 10
    assert "No homegrown neural-network framework" in text
    assert "No general MCMC, VI, NPE, SBI" in text


def test_future_capabilities_roadmap_is_navigable() -> None:
    myst = (ROOT / "docs" / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads(
        (ROOT / "docs" / "route-manifest.json").read_text(encoding="utf-8")
    )

    path = "90-development-log/future-capabilities-roadmap.md"
    assert myst.count(path) == 1
    assert manifest[path] == "/future-capabilities-roadmap"


def test_development_log_links_future_capabilities_roadmap() -> None:
    index = (ROOT / "docs" / "90-development-log" / "index.md").read_text(
        encoding="utf-8"
    )
    assert "future-capabilities-roadmap.md" in index
