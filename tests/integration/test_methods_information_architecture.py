"""Contracts for the researcher-first numerical-method information architecture."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
METHODS = DOCS / "20-methods"

FAMILIES = {
    "change-constraints-evolution": (
        "autodiff",
        "rootfinding",
        "optimization",
        "ode",
    ),
    "approximation-integration": (
        "interpolation",
        "regular-grid",
        "bsplines",
        "cumulative-trapz",
        "quadrature",
        "sparse-grid-quadrature",
    ),
    "linear-structure": ("linear-algebra", "operators", "special-functions"),
    "probability-sampling": (
        "distributions",
        "random",
        "sampling",
    ),
    "discrete-space": ("grids", "meshes", "spatial"),
}

FAMILY_TITLES = {
    "change-constraints-evolution": "Change, constraints, and evolution",
    "approximation-integration": "Approximation from finite information",
    "linear-structure": "Linear structure and reusable operators",
    "probability-sampling": "Randomness as a computational object",
    "discrete-space": "Discrete worlds: grids, meshes, and neighborhoods",
}

TOC_FAMILIES = {
    "change-constraints-evolution": (
        "autodiff",
        "rootfinding",
        "nonlinear-systems",
        "optimization",
        "ode",
        "adaptive-differential-equations",
    ),
    "approximation-integration": (
        "interpolation",
        "regular-grid",
        "bsplines",
        "cumulative-trapz",
        "quadrature",
        "adaptive-quadrature",
        "sparse-grid-quadrature",
        "differentiating-an-integral",
    ),
    "linear-structure": (
        "linear-algebra",
        "operators",
        "iterative-linear-solvers",
        "special-functions",
    ),
    "probability-sampling": (
        "distributions",
        "random",
        "sampling",
        "quasi-monte-carlo",
    ),
    "discrete-space": ("grids", "meshes", "spatial"),
    "signals": (
        "signal-axes",
        "windows-spectral-leakage",
        "spectral-estimation",
        "phase-and-delay",
    ),
}

TOC_TITLES = {
    **FAMILY_TITLES,
    "signals": "Signals as sampled evidence",
}

ROUTES = {
    "methods.md": "/methods",
    "change-constraints-evolution/autodiff.md": "/autodiff",
    "change-constraints-evolution/rootfinding.md": "/rootfinding",
    "change-constraints-evolution/optimization.md": "/optimization",
    "change-constraints-evolution/ode.md": "/ode",
    "approximation-integration/interpolation.md": "/interpolation",
    "approximation-integration/regular-grid.md": "/regular-grid",
    "approximation-integration/bsplines.md": "/bsplines",
    "approximation-integration/cumulative-trapz.md": "/cumulative-trapz",
    "approximation-integration/quadrature.md": "/quadrature",
    "approximation-integration/sparse-grid-quadrature.md": "/sparse-grid-quadrature",
    "linear-structure/linear-algebra.md": "/linear-algebra",
    "linear-structure/operators.md": "/operators",
    "linear-structure/special-functions.md": "/special-functions",
    "probability-sampling/distributions.md": "/distributions",
    "probability-sampling/random.md": "/random",
    "probability-sampling/sampling.md": "/sampling",
    "discrete-space/grids.md": "/grids",
    "discrete-space/meshes.md": "/meshes",
    "discrete-space/spatial.md": "/spatial",
}


def test_current_method_pages_exist_once_in_the_toc_with_stable_routes() -> None:
    myst = (DOCS / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))

    assert sum(len(pages) for pages in FAMILIES.values()) == 19
    for family, pages in FAMILIES.items():
        for page in pages:
            relative = f"{family}/{page}.md"
            source = f"20-methods/{relative}"
            assert (METHODS / relative).is_file(), source
            assert myst.count(f"file: {source}") == 1, source
            assert manifest[source] == ROUTES[relative]


def test_methods_landing_uses_semantic_filename_and_family_titles() -> None:
    landing = METHODS / "methods.md"
    assert landing.is_file()
    assert not (METHODS / "index.md").exists()

    text = landing.read_text(encoding="utf-8")
    for title in (
        "Change, constraints, and evolution",
        "Approximation from finite information",
        "Linear structure and reusable operators",
        "Randomness as a computational object",
        "Discrete worlds: grids, meshes, and neighborhoods",
        "Signals as sampled evidence",
    ):
        assert title in text

    myst = (DOCS / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))
    assert myst.count("file: 20-methods/methods.md") == 1
    assert manifest["20-methods/methods.md"] == "/methods"
    assert "10-theory/index.md" not in manifest


def test_methods_toc_preserves_family_titles_and_page_order() -> None:
    config = yaml.safe_load((DOCS / "myst.yml").read_text(encoding="utf-8"))
    methods_toc = next(
        item
        for item in config["project"]["toc"]
        if item.get("title") == "Numerical methods"
    )

    expected_children = [{"file": "20-methods/methods.md"}]
    expected_children.extend(
        {
            "title": TOC_TITLES[family],
            "children": [{"file": f"20-methods/{family}/{page}.md"} for page in pages],
        }
        for family, pages in TOC_FAMILIES.items()
    )

    assert methods_toc["children"] == expected_children


def test_migration_retires_old_method_and_representation_sources() -> None:
    old_theory = DOCS / "10-theory"
    for pages in FAMILIES.values():
        for page in pages:
            assert not (old_theory / f"{page}.md").exists(), page

    assert (
        DOCS / "40-workflows" / "differentiable-research" / "science-patterns.md"
    ).is_file()
    assert not (old_theory / "quantities.md").exists()
    assert not (old_theory / "geometry.md").exists()
    assert (
        DOCS / "30-representations" / "units-quantities" / "quantities.md"
    ).is_file()
    assert (
        DOCS / "30-representations" / "geometry-coordinates" / "geometry.md"
    ).is_file()
    for representation in ("quantities.md", "geometry.md"):
        assert not any(METHODS.rglob(representation)), representation


def test_random_computation_and_sampling_have_distinct_scopes() -> None:
    random_text = (METHODS / "probability-sampling" / "random.md").read_text(
        encoding="utf-8"
    )
    sampling_text = (METHODS / "probability-sampling" / "sampling.md").read_text(
        encoding="utf-8"
    )

    assert "Explicit PRNG key ownership" in random_text
    assert "`key_stream(key, num)`" in random_text
    assert "`inverse_cdf_draw(weight, grid, unif" not in random_text

    assert "continuous inverse-CDF draw" in sampling_text
    assert "`inverse_cdf_draw(weight, grid, unif" in sampling_text
    assert "Systematic, stratified, and residual resamplers" in sampling_text


def test_new_landing_and_sampling_page_use_ascii_punctuation() -> None:
    for path in (
        METHODS / "methods.md",
        METHODS / "probability-sampling" / "sampling.md",
    ):
        text = path.read_text(encoding="utf-8")
        for punctuation in ("–", "—", "‘", "’", "“", "”", "→"):
            assert punctuation not in text, f"{path}: {punctuation!r}"
