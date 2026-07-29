"""Ownership and navigation contracts for the grouped API reference."""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from collections import Counter
from pathlib import Path

from jaxstro import numerics, quad
from jaxstro.numerics.sampling import inverse_cdf_draw, stratified_uniform

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
API_ROOT = DOCS / "50-api"

API_OWNERS = {
    "change-constraints/autodiff.md": "jaxstro.numerics.autodiff",
    "change-constraints/rootfinding.md": "jaxstro.numerics.rootfinding",
    "change-constraints/kepler.md": "jaxstro.numerics.kepler",
    "change-constraints/lane-emden.md": "jaxstro.numerics.lane_emden",
    "change-constraints/optimization.md": "jaxstro.numerics.optimization",
    "change-constraints/ode.md": "jaxstro.numerics.ode",
    "approximation-integration/interpolation.md": "jaxstro.numerics.interpolation",
    "approximation-integration/regular-grid.md": "jaxstro.numerics.regular_grid",
    "approximation-integration/splines.md": "jaxstro.numerics.splines",
    "approximation-integration/quad.md": "jaxstro.quad",
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
    "physical-representations/constants-api.md": "jaxstro.constants",
    "physical-representations/units.md": "jaxstro.units",
    "physical-representations/quantity.md": "jaxstro.quantity",
    "physical-representations/coords.md": "jaxstro.coords",
    "physical-representations/geometry.md": "jaxstro.geometry",
    "physical-representations/astrometry.md": "jaxstro.astrometry",
    "physical-representations/params.md": "jaxstro.params",
    "scientific-data/spectra.md": "jaxstro.spectra",
    "scientific-data/atmospheres-api.md": "jaxstro.atmospheres",
    "research-infrastructure/checks.md": "jaxstro.numerics.checks",
    "research-infrastructure/types.md": "jaxstro.numerics.types",
    "research-infrastructure/jaxconfig.md": "jaxstro.jaxconfig",
    "research-infrastructure/contracts.md": "jaxstro.contracts",
    "research-infrastructure/evidence.md": "jaxstro.evidence",
    "research-infrastructure/provenance.md": "jaxstro.provenance",
    "research-infrastructure/testing.md": "jaxstro.testing",
}

QUAD_FAMILY_PAGES = {
    "approximation-integration/quad-tensor-cubature.md": (
        "jaxstro.quad.tensor",
        "jaxstro.quad.cubature",
    ),
    "approximation-integration/quad-sparse.md": ("jaxstro.quad.sparse",),
    "approximation-integration/quad-qmc.md": ("jaxstro.quad.qmc",),
}

PRIVATE_NUMERICS_MODULE_EXCLUSIONS = {
    # Internal contract-registration helpers; public contracts are documented
    # through jaxstro.contracts and the generated contract registry.
    "_contracts": "private contract registration implementation",
    # Internal custom-root machinery owned by the public rootfinding module.
    "_implicit_root": "private rootfinding implementation",
}

COMPATIBILITY_NUMERICS_MODULE_EXCLUSIONS = {
    "integration": "temporary sampled-integration compatibility import",
    "quadrature": "temporary fixed-helper compatibility import",
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


def test_quad_family_pages_extend_one_canonical_owner_without_duplication() -> None:
    for relative, modules in QUAD_FAMILY_PAGES.items():
        text = (API_ROOT / relative).read_text(encoding="utf-8")
        for module in modules:
            importlib.import_module(module)
            assert f"`{module}`" in text, relative
        assert "exposed through `jaxstro.quad`" in text, relative
        assert "from jaxstro.quad import" in text, relative
        for section in REQUIRED_SECTIONS:
            assert section in text, f"{relative}: missing {section}"


def test_every_public_numerics_module_has_exactly_one_owner_page() -> None:
    discovered = {module.name for module in pkgutil.iter_modules(numerics.__path__)}
    private = {name for name in discovered if name.startswith("_")}
    assert private == set(PRIVATE_NUMERICS_MODULE_EXCLUSIONS)

    compatibility = set(COMPATIBILITY_NUMERICS_MODULE_EXCLUSIONS)
    assert compatibility < discovered
    public_owners = {
        f"jaxstro.numerics.{name}"
        for name in discovered
        if not name.startswith("_") and name not in compatibility
    }
    owner_counts = Counter(API_OWNERS.values())
    assert len(owner_counts) == len(API_OWNERS), "each owner must appear exactly once"
    assert {
        owner for owner in owner_counts if owner.startswith("jaxstro.numerics.")
    } == public_owners


def test_api_landing_teaches_route_first_owner_qualified_imports() -> None:
    text = (API_ROOT / "api.md").read_text(encoding="utf-8")
    assert "from jaxstro.numerics import rootfinding" in text
    assert "from jaxstro.numerics.rootfinding import safeguarded_bracketed_root" in text
    assert "legacy inventory awaiting Project 2" in text
    assert "not the canonical documentation path" in text
    assert "Planned Jaxstro capability" not in text


def test_quad_owner_page_teaches_canonical_and_legacy_boundaries() -> None:
    text = (API_ROOT / "approximation-integration/quad.md").read_text()
    assert "`jaxstro.quad`" in text
    assert "from jaxstro import quad" in text
    assert "jaxstro.numerics.integration" in text
    assert "jaxstro.numerics.quadrature" in text
    assert "temporary compatibility" in text
    assert "does not yet provide adaptive integration" not in text


def test_quad_owner_page_publishes_the_adaptive_contract() -> None:
    text = (API_ROOT / "approximation-integration/quad.md").read_text()
    normalized = " ".join(text.split())
    for required in (
        "quad.integrate",
        "GaussKronrod",
        "AdaptiveClenshawCurtis",
        "AdaptiveTanhSinh",
        "Romberg",
        "RombergTanhSinh",
        'gradient="stop"',
        "primal result",
        "status precedence",
        "integrand evaluations",
        "zero-based finest completed level",
        "roundoff-scale error floor alone",
        "all-zero `QuadWork` record",
        "not an exact error certificate",
        "quad-adaptive-envelope.json",
    ):
        assert required in normalized


def test_quad_owner_page_lists_the_complete_public_surface() -> None:
    text = (API_ROOT / "approximation-integration/quad.md").read_text()
    missing = {name for name in quad.__all__ if f"`{name}`" not in text}
    assert not missing


def test_quad_owner_page_claims_complete_a1_fixed_surface() -> None:
    text = (API_ROOT / "approximation-integration/quad.md").read_text()
    for required in (
        "`fixed`",
        "`GaussianRule`",
        "`ClenshawCurtisRule`",
        "`FejerIRule`",
        "`FejerIIRule`",
        "`TanhSinhRule`",
        "Gauss-Jacobi",
        "float64",
    ):
        assert required in text
    assert "Phase A0" not in text


def test_grouped_api_pages_are_navigable_with_canonical_routes() -> None:
    myst = (DOCS / "myst.yml").read_text(encoding="utf-8")
    routes = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))

    assert myst.count("50-api/api.md") == 1
    assert routes["50-api/api.md"] == "/api"
    for relative in API_OWNERS:
        source = f"50-api/{relative}"
        assert myst.count(source) == 1, source
        assert source in routes, source
    for relative in QUAD_FAMILY_PAGES:
        source = f"50-api/{relative}"
        assert myst.count(source) == 1, source
        assert source in routes, source

    assert not (DOCS / "40-api").exists()
    assert not any(
        route.startswith("/index-")
        for source, route in routes.items()
        if source.startswith("50-api/")
    )

    assert routes["50-api/research-infrastructure/source-provenance/constants.md"] == (
        "/constants"
    )
    assert (
        routes["50-api/research-infrastructure/source-provenance/atmospheres.md"]
        == "/atmospheres"
    )
    assert routes["50-api/physical-representations/constants-api.md"] == (
        "/constants-api"
    )
    assert routes["50-api/scientific-data/atmospheres-api.md"] == ("/atmospheres-api")


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


def test_sampling_reference_matches_runtime_signatures_and_contracts() -> None:
    inverse_signature = inspect.signature(inverse_cdf_draw)
    assert tuple(inverse_signature.parameters) == ("weight", "grid", "unif", "reg")
    assert inverse_signature.parameters["reg"].default == 1e-30

    stratified_signature = inspect.signature(stratified_uniform)
    assert tuple(stratified_signature.parameters) == (
        "key",
        "n",
        "minval",
        "maxval",
    )
    assert stratified_signature.parameters["minval"].default == 0.0
    assert stratified_signature.parameters["maxval"].default == 1.0

    text = (API_ROOT / "randomness/sampling.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    for phrase in (
        "`weight` and `grid` are one-dimensional floating arrays with matching shape",
        "`unif` is a scalar floating deviate",
        "uniformly spaced",
        "`cdf[-1] + reg`",
        "zero-total",
        "`grid[-1]`",
        "differentiable with respect to `weight` and `unif`",
        "`n` is a positive static integer",
        "caller owns the key",
    ):
        assert phrase in normalized

    assert "PPF callback" not in text
    assert "does not normalize" not in text


def test_random_reference_documents_zero_weight_and_tracing_boundaries() -> None:
    text = (API_ROOT / "randomness/random.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    for phrase in (
        "All-zero weights are valid",
        "uniform distribution",
        "Concrete eager inputs reject",
        "Value-dependent finite and nonnegative checks are skipped under tracing",
    ):
        assert phrase in normalized
    assert "non-normalizable weights raise" not in text


def test_generated_manifest_counts_the_current_api_surface() -> None:
    routes = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))

    assert len(API_OWNERS) == 38
    assert "jaxstro.quad" in API_OWNERS.values()
    assert len(routes) == 181
