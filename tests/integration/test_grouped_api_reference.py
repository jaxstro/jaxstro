"""Ownership and navigation contracts for the grouped API reference."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
API_ROOT = DOCS / "50-api"

API_OWNERS = {
    "change-constraints/autodiff.md": "jaxstro.numerics.autodiff",
    "change-constraints/rootfinding.md": "jaxstro.numerics.rootfinding",
    "change-constraints/kepler.md": "jaxstro.numerics.kepler",
    "change-constraints/optimization.md": "jaxstro.numerics.optimization",
    "change-constraints/ode.md": "jaxstro.numerics.ode",
    "approximation-integration/interpolation.md": "jaxstro.numerics.interpolation",
    "approximation-integration/regular-grid.md": "jaxstro.numerics.regular_grid",
    "approximation-integration/splines.md": "jaxstro.numerics.splines",
    "approximation-integration/integration.md": "jaxstro.numerics.integration",
    "approximation-integration/quadrature.md": "jaxstro.numerics.quadrature",
    "linear-structure/linear-algebra.md": "jaxstro.numerics.linear_algebra",
    "linear-structure/compensated.md": "jaxstro.numerics.compensated",
    "linear-structure/operators.md": "jaxstro.numerics.operators",
    "linear-structure/special.md": "jaxstro.numerics.special",
    "randomness/distributions.md": "jaxstro.numerics.distributions",
    "randomness/rng.md": "jaxstro.numerics.rng",
    "randomness/random.md": "jaxstro.numerics.random",
    "randomness/sampling.md": "jaxstro.numerics.sampling",
    "randomness/stats.md": "jaxstro.numerics.stats",
    "discrete-space/grids.md": "jaxstro.numerics.grids",
    "discrete-space/meshes.md": "jaxstro.numerics.meshes",
    "discrete-space/spatial.md": "jaxstro.spatial",
    "physical-representations/constants.md": "jaxstro.constants",
    "physical-representations/units.md": "jaxstro.units",
    "physical-representations/quantity.md": "jaxstro.quantity",
    "physical-representations/coords.md": "jaxstro.coords",
    "physical-representations/geometry.md": "jaxstro.geometry",
    "physical-representations/astrometry.md": "jaxstro.astrometry",
    "physical-representations/params.md": "jaxstro.params",
    "scientific-data/spectra.md": "jaxstro.spectra",
    "scientific-data/atmospheres.md": "jaxstro.atmospheres",
    "research-infrastructure/checks.md": "jaxstro.numerics.checks",
    "research-infrastructure/jaxconfig.md": "jaxstro.jaxconfig",
    "research-infrastructure/contracts.md": "jaxstro.contracts",
    "research-infrastructure/evidence.md": "jaxstro.evidence",
    "research-infrastructure/provenance.md": "jaxstro.provenance",
    "research-infrastructure/testing.md": "jaxstro.testing",
}

REQUIRED_SECTIONS = (
    "## Owner import path",
    "## Purpose",
    "## Public records and callables",
    "## Shape and dtype expectations",
    "## JAX transforms and AD classification",
    "## Failure behavior",
    "## Contract and evidence links",
    "## Canonical import example",
)


def test_every_api_page_names_an_importable_owner_and_complete_contract() -> None:
    for relative, owner in API_OWNERS.items():
        importlib.import_module(owner)
        text = (API_ROOT / relative).read_text(encoding="utf-8")
        assert f"`{owner}`" in text, relative
        assert f"from {owner} import" in text, relative
        for section in REQUIRED_SECTIONS:
            assert section in text, f"{relative}: missing {section}"


def test_api_landing_teaches_route_first_owner_qualified_imports() -> None:
    text = (API_ROOT / "api.md").read_text(encoding="utf-8")
    assert "from jaxstro.numerics import rootfinding" in text
    assert "from jaxstro.numerics.rootfinding import safeguarded_bracketed_root" in text
    assert "legacy inventory awaiting Project 2" in text
    assert "not the canonical documentation path" in text
    assert "Planned Jaxstro capability" not in text


def test_grouped_api_pages_are_navigable_with_canonical_routes() -> None:
    myst = (DOCS / "myst.yml").read_text(encoding="utf-8")
    routes = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))

    assert myst.count("50-api/api.md") == 1
    assert routes["50-api/api.md"] == "/api"
    for relative in API_OWNERS:
        source = f"50-api/{relative}"
        assert myst.count(source) == 1, source
        assert source in routes, source

    assert not (DOCS / "40-api").exists()
    assert not any(
        route.startswith("/index-")
        for source, route in routes.items()
        if source.startswith("50-api/")
    )


def test_interpolation_symbol_descriptions_are_not_duplicated() -> None:
    texts = [
        path.read_text(encoding="utf-8")
        for path in API_ROOT.rglob("*.md")
        if path.name != "api.md"
    ]
    joined = "\n".join(texts)
    assert joined.count("`pchip_slopes(...)`") == 1
    assert joined.count("`monotone_cubic_interp(...)`") == 1


def test_api_sources_use_ascii_prose() -> None:
    non_ascii = [
        path.relative_to(API_ROOT).as_posix()
        for path in API_ROOT.rglob("*.md")
        if not path.read_text(encoding="utf-8").isascii()
    ]
    assert not non_ascii, f"non-ASCII API sources: {non_ascii}"
