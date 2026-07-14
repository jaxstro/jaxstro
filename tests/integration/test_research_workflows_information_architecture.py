"""Information-architecture contracts for researcher-first workflows."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
MYST = DOCS / "myst.yml"
ROUTES = DOCS / "route-manifest.json"

FAMILIES = {
    "scientific-ml": (
        "preprocessing.md",
        "data-plans.md",
        "auditable-training.md",
        "ecosystem-boundaries.md",
    ),
    "data-pipelines": (
        "quantity-migration.md",
        "newera-data-processing.md",
        "bosz-data-processing.md",
        "sonora-data-processing.md",
        "tlusty-data-processing.md",
        "query-atmosphere-spectra.md",
    ),
    "differentiable-research": (
        "what-jax-differentiates.md",
        "auditing-derivatives.md",
        "branches-limits-implicit-sensitivities.md",
        "science-patterns.md",
    ),
    "reproducible-research": (
        "random-state-ownership.md",
        "provenance.md",
        "evidence-and-claim-boundaries.md",
    ),
    "investigations": (
        "investigations.md",
        "root-values-and-sensitivities.md",
        "powerlaw-removable-limit.md",
        "interpolation-boundary-policies.md",
    ),
}

PRESERVED_ROUTES = {
    "40-workflows/workflows.md": "/workflows",
    "40-workflows/data-pipelines/quantity-migration.md": "/quantity-migration",
    "40-workflows/data-pipelines/newera-data-processing.md": "/newera-data-processing",
    "40-workflows/data-pipelines/bosz-data-processing.md": "/bosz-data-processing",
    "40-workflows/data-pipelines/sonora-data-processing.md": "/sonora-data-processing",
    "40-workflows/data-pipelines/tlusty-data-processing.md": "/tlusty-data-processing",
    "40-workflows/data-pipelines/query-atmosphere-spectra.md": "/query-atmosphere-spectra",
    "40-workflows/differentiable-research/science-patterns.md": "/science-patterns",
    "40-workflows/reproducible-research/provenance.md": "/provenance",
    "40-workflows/investigations/investigations.md": "/investigations",
    "40-workflows/investigations/root-values-and-sensitivities.md": "/root-values-and-sensitivities",
    "40-workflows/investigations/powerlaw-removable-limit.md": "/powerlaw-removable-limit",
    "40-workflows/investigations/interpolation-boundary-policies.md": "/interpolation-boundary-policies",
}


def test_final_workflow_families_occur_once_in_toc_and_routes() -> None:
    myst = MYST.read_text(encoding="utf-8")
    routes = json.loads(ROUTES.read_text(encoding="utf-8"))
    assert (DOCS / "40-workflows/workflows.md").is_file()
    assert myst.count("40-workflows/workflows.md") == 1
    for family, names in FAMILIES.items():
        for name in names:
            relative = f"40-workflows/{family}/{name}"
            assert (DOCS / relative).is_file(), relative
            assert myst.count(relative) == 1, relative
            assert list(routes).count(relative) == 1, relative


def test_meaningful_workflow_routes_are_preserved() -> None:
    routes = json.loads(ROUTES.read_text(encoding="utf-8"))
    for page, route in PRESERVED_ROUTES.items():
        assert routes[page] == route


def test_methods_and_representations_keep_their_existing_routes() -> None:
    routes = json.loads(ROUTES.read_text(encoding="utf-8"))
    retained = {
        page: route
        for page, route in routes.items()
        if page.startswith(("20-methods/", "30-representations/"))
    }
    assert len(retained) == 50
    for page, route in retained.items():
        assert route == f"/{Path(page).stem}", page
        assert (DOCS / page).is_file(), page


def test_registry_script_has_sequence_annotation_and_no_legacy_validator() -> None:
    path = ROOT / "scripts/build_research_workflow_registry.py"
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "collections.abc"
        for alias in node.names
    }
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "Sequence" in imports
    assert {
        "load_and_validate_workflows",
        "validate_unique_references",
        "render_outputs",
    } <= set(functions)
    assert "validate_" + "instructor_route" not in functions


def test_scientific_ml_pages_are_planned_substantive_and_not_api_promises() -> None:
    required = {
        "preprocessing.md": (
            "eq-ml-standardization",
            "eq-ml-whitening",
            "data leakage",
        ),
        "data-plans.md": ("eq-ml-disjoint-splits", "fixed-shape", "padding"),
        "auditable-training.md": ("eq-ml-training-update", "exactly `K`", "optimizer"),
        "ecosystem-boundaries.md": ("Equinox", "Optax", "Informax"),
    }
    api_text = (DOCS / "40-api/index.md").read_text(encoding="utf-8")
    for name, phrases in required.items():
        path = DOCS / "40-workflows/scientific-ml" / name
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        assert text.lstrip().startswith("# ") or text.startswith("---")
        assert "Use this page when" in text
        assert ":::{important} Planned Jaxstro capability" in text
        assert "`jaxstro.ml` does not exist" in text
        assert "no implementation schedule" in text
        assert "## Evidence required before implementation" in text
        assert "## Where the claim stops" in text
        for phrase in phrases:
            assert phrase in text, (name, phrase)
        assert "jaxstro.ml." not in api_text


def test_differentiable_and_reproducible_pages_are_substantive_and_connected() -> None:
    pages = {
        "differentiable-research/what-jax-differentiates.md": (
            "JVP",
            "VJP",
            "JAXPR",
            ":label:",
        ),
        "differentiable-research/auditing-derivatives.md": (
            "directional finite difference",
            "step-size",
            "precision",
            ":label:",
        ),
        "differentiable-research/branches-limits-implicit-sensitivities.md": (
            "implicit function theorem",
            "conditioning",
            "fail closed",
            ":label:",
        ),
        "reproducible-research/random-state-ownership.md": (
            "key lineage",
            "statistical independence",
            "provenance",
        ),
        "reproducible-research/provenance.md": (
            "Runtime manifest",
            "Source-backed card",
            "SHA-256",
        ),
        "reproducible-research/evidence-and-claim-boundaries.md": (
            "Scientific contract",
            "Validation target",
            "Warranted scientific claim",
        ),
    }
    for relative, phrases in pages.items():
        path = DOCS / "40-workflows" / relative
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        assert "Use this page when" in text
        assert len(text.split()) >= 250, relative
        assert re.search(
            r"\.\./\.\./(10-foundations|20-methods|30-representations)/", text
        )
        for phrase in phrases:
            assert phrase in text, (relative, phrase)


def test_old_registry_and_workflow_sources_are_absent() -> None:
    old_paths = (
        ROOT / "scripts/build_" / "unused",
        ROOT / "docs/curriculum/units.json",
        ROOT / "docs/validation" / ("curric" + "ulum-coverage.json"),
        ROOT / "scripts" / ("build_curric" + "ulum_registry.py"),
        ROOT / "tests/unit" / ("test_curric" + "ulum_registry.py"),
        ROOT / "tests/integration" / ("test_curric" + "ulum_conventions.py"),
        ROOT / "docs/50-howto",
        ROOT / "docs/70-investigations",
        ROOT / "docs/10-theory/science-patterns.md",
        ROOT / "docs/20-architecture/provenance.md",
    )
    assert all(not path.exists() for path in old_paths)

    forbidden = ("build_curric" + "ulum_registry", "curric" + "ulum-coverage")
    for root in (ROOT / "scripts", ROOT / "tests"):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh"}:
                text = path.read_text(encoding="utf-8")
                assert all(term not in text for term in forbidden), path
