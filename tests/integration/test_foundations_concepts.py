"""Concept and navigation contracts for the first-principles foundations spine."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FOUNDATIONS = DOCS / "10-foundations"

PAGES = {
    "mathematical-objects/functions-units-scales.md": (
        "Newtonian gravity",
        "Stefan-Boltzmann",
        "dimensionless",
        "order of magnitude",
    ),
    "models-and-computation/what-is-a-model.md": (
        "conceptual model",
        "mathematical model",
        "computational model",
        "statistical model",
        "generative model",
        "surrogate model",
        "information compression",
        "parameter-space dimension",
        "intrinsic dimension",
    ),
    "mathematical-objects/linear-algebra-language-of-change.md": (
        "vectors as perturbations",
        "linear map",
        "basis",
        "null space",
        "condition number",
        "Hessian",
    ),
    "mathematical-objects/what-is-a-derivative.md": (
        "local rate of change",
        "best local linear map",
        "scientific sensitivity",
        "JVP",
        "VJP",
        "likelihood score",
        "Fisher information",
        "implicit sensitivity",
        "executed program",
    ),
    "mathematical-objects/probability-and-distributions.md": (
        "probability mass",
        "probability density",
        "support",
        "normalization",
        "expectation",
        "covariance",
        "aleatoric",
        "epistemic",
    ),
    "models-and-computation/models-inference-information.md": (
        "measurement model",
        "likelihood",
        "prior",
        "posterior",
        "posterior predictive",
        "nuisance parameter",
        "Shannon information",
        "discarded information",
        "misspecified model",
    ),
    "models-and-computation/sensitivity-conditioning-identifiability.md": (
        "conditioning",
        "identifiability",
        "degeneracy",
        "null direction",
        "finite differences",
        "automatic differentiation",
    ),
    "models-and-computation/from-relations-to-differentiable-programs.md": (
        "mathematical relation",
        "executed program",
        "control flow",
        "fixed scan",
        "PyTree",
        "jit",
        "vmap",
        "value-first",
        "implicit derivative",
    ),
}


def test_each_foundation_page_has_required_concepts_and_learning_cycle() -> None:
    for filename, phrases in PAGES.items():
        path = FOUNDATIONS / filename
        assert path.is_file(), filename
        text = path.read_text(encoding="utf-8")
        plain = re.sub(r"\s+", " ", text.replace("**", "")).lower()
        for phrase in phrases:
            assert phrase.lower() in plain, f"{filename}: {phrase}"
        for heading in (
            "## Predict",
            "## Compute",
            "## Audit",
            "## State the warranted claim",
            "## Misconception check",
        ):
            assert heading in text, f"{filename}: {heading}"


def test_foundation_pages_are_navigable_without_replacing_module_pages() -> None:
    myst = (DOCS / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))
    for filename in PAGES:
        page = f"10-foundations/{filename}"
        assert myst.count(f"file: {page}") == 1
        assert manifest[page].startswith("/")
    assert "file: 10-theory/linear-algebra.md" in myst
    assert "file: 10-theory/autodiff.md" in myst
    assert "file: 10-theory/distributions.md" in myst


def test_method_pages_link_back_to_conceptual_foundations() -> None:
    pairs = (
        ("10-theory/linear-algebra.md", "linear-algebra-language-of-change.md"),
        ("10-theory/autodiff.md", "what-is-a-derivative.md"),
        ("10-theory/rootfinding.md", "what-is-a-derivative.md"),
    )
    for method_page, foundation_name in pairs:
        text = (DOCS / method_page).read_text(encoding="utf-8")
        assert foundation_name in text
