"""Contracts for the current scientific-representation information architecture."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
REPRESENTATIONS = DOCS / "30-representations"

PAGES = {
    "units-quantities/constants-and-conventions.md": (
        "jaxstro.constants",
        "CGS",
    ),
    "units-quantities/quantity-system.md": (
        "jaxstro.units` and `jaxstro.quantity",
        "canonical ecosystem unit systems",
    ),
    "units-quantities/quantities.md": (
        "jaxstro.quantity",
        "exact rational exponents",
    ),
    "units-quantities/equivalencies.md": (
        "jaxstro.quantity",
        "explicit equivalencies",
    ),
    "geometry-coordinates/coordinate-transformations.md": (
        "jaxstro.coords",
        "ICRS",
    ),
    "geometry-coordinates/geometry.md": (
        "jaxstro.geometry",
        "right-handed",
    ),
    "geometry-coordinates/astrometry.md": (
        "jaxstro.astrometry",
        "IAU",
    ),
    "spectra-atmospheres/spectra-data-architecture.md": (
        "jaxstro.spectra",
        "spectral coordinate",
    ),
    "spectra-atmospheres/conservative-spectral-resampling.md": (
        "jaxstro.spectra",
        "bin integrals",
    ),
    "spectra-atmospheres/atmosphere-capabilities.md": (
        "jaxstro.atmospheres",
        "surface flux",
    ),
    "spectra-atmospheres/source-artifacts-and-adapters.md": (
        "jaxstro.atmospheres",
        "exact product identity",
    ),
    "parameters-state/parameters-and-transforms.md": (
        "jaxstro.params",
        "unconstrained parameter vector",
    ),
    "parameters-state/pytrees-as-scientific-state.md": (
        "jaxstro.params",
        "PyTree",
    ),
    "parameters-state/serialization-and-provenance.md": (
        "jaxstro.provenance",
        "deterministic",
    ),
}

IMPORT_OWNERS = (
    "jaxstro.constants",
    "jaxstro.units",
    "jaxstro.quantity",
    "jaxstro.coords",
    "jaxstro.geometry",
    "jaxstro.astrometry",
    "jaxstro.spectra",
    "jaxstro.atmospheres",
    "jaxstro.params",
    "jaxstro.provenance",
)

ROUTES = {
    "representations.md": "/representations",
    "units-quantities/quantity-system.md": "/quantity-system",
    "units-quantities/quantities.md": "/quantities",
    "geometry-coordinates/geometry.md": "/geometry",
    "spectra-atmospheres/spectra-data-architecture.md": ("/spectra-data-architecture"),
    "spectra-atmospheres/atmosphere-capabilities.md": ("/atmosphere-capabilities"),
}

CONTRACT_ROWS = (
    "Mathematical object",
    "Physical convention",
    "Runtime owner",
    "Shape and unit policy",
    "Transform boundary",
    "Evidence",
    "Downstream interpretation boundary",
)

OLD_SOURCES = (
    "10-theory/quantities.md",
    "10-theory/geometry.md",
    "20-architecture/quantity-system.md",
    "20-architecture/spectra-data-architecture.md",
    "20-architecture/atmosphere-capabilities.md",
)

RETAINED_RATIONALE = {
    "units-quantities/quantity-system.md": (
        "The system is additive",
        "Global registration is reserved for interactive convenience",
        "one `Quantity` array has one unit",
    ),
    "units-quantities/quantities.md": (
        "The value is the dynamic PyTree child",
        "convert once",
        "Array-valued quantity serialization is deliberately deferred",
    ),
    "geometry-coordinates/geometry.md": (
        "[w, x, y, z]",
        "outer(inner(point))",
        "zero-vector",
    ),
    "spectra-atmospheres/spectra-data-architecture.md": (
        "Product lookup and topology selection",
        "Sonora Diamondback and both BSTAR modes remain `POLICY_NOT_VALIDATED`",
        "Fluxax keeps ownership",
    ),
    "spectra-atmospheres/atmosphere-capabilities.md": (
        "TLUSTY has 27 composition-scoped products",
        "mean-coalesces samples",
        "no nearest-neighbor or arbitrary triangulation fallback",
    ),
}

CURRENT_METHOD_ROUTES = {
    "20-methods/change-constraints-evolution/rootfinding.md": "/rootfinding",
    "20-methods/approximation-integration/interpolation.md": "/interpolation",
    "20-methods/approximation-integration/cumulative-trapz.md": ("/cumulative-trapz"),
    "20-methods/probability-sampling/random.md": "/random",
    "20-methods/probability-sampling/sampling.md": "/sampling",
    "20-methods/discrete-space/spatial.md": "/spatial",
}


def _read(relative: str) -> str:
    return (REPRESENTATIONS / relative).read_text(encoding="utf-8")


def _toc_files(node: object) -> list[str]:
    if isinstance(node, list):
        return [path for item in node for path in _toc_files(item)]
    if isinstance(node, dict):
        paths = [str(node["file"])] if "file" in node else []
        return paths + _toc_files(node.get("children", []))
    return []


def test_semantic_landing_and_all_fourteen_current_pages_exist() -> None:
    assert (REPRESENTATIONS / "representations.md").is_file()
    assert not (REPRESENTATIONS / "index.md").exists()
    assert len(PAGES) == 14
    for relative in PAGES:
        assert (REPRESENTATIONS / relative).is_file(), relative


def test_every_page_occurs_once_in_toc_and_route_manifest() -> None:
    config = yaml.safe_load((DOCS / "myst.yml").read_text(encoding="utf-8"))
    toc_files = _toc_files(config["project"]["toc"])
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))

    for relative in ("representations.md", *PAGES):
        source = f"30-representations/{relative}"
        assert toc_files.count(source) == 1, source
        assert list(manifest).count(source) == 1, source

    for relative, route in ROUTES.items():
        assert manifest[f"30-representations/{relative}"] == route


def test_every_named_runtime_owner_imports() -> None:
    for owner in IMPORT_OWNERS:
        importlib.import_module(owner)


def test_every_page_opens_with_status_and_has_a_substantive_contract() -> None:
    for relative, (owner_text, convention_text) in PAGES.items():
        text = _read(relative)
        body = text.split("---", 2)[-1].lstrip()
        assert body.startswith("Use this page when"), relative
        assert text.count(":::{important} Implemented Jaxstro capability") == 1
        assert "## Representation contract" in text
        for row in CONTRACT_ROWS:
            assert len(re.findall(rf"(?m)^\| {re.escape(row)} \| .+ \|$", text)) == 1, (
                relative,
                row,
            )
        assert owner_text in text, relative
        assert convention_text in text, relative


def test_landing_explains_representation_choice_and_links_four_families() -> None:
    text = _read("representations.md")
    for phrase in (
        "between mathematical methods and research workflows",
        "Choosing an array dtype and shape is not the same as choosing a scientific representation",
        "Units and quantities",
        "Geometry and coordinates",
        "Spectra and atmospheres",
        "Parameters and scientific state",
        ":link: ./units-quantities/constants-and-conventions.md",
        ":link: ./geometry-coordinates/coordinate-transformations.md",
        ":link: ./spectra-atmospheres/spectra-data-architecture.md",
        ":link: ./parameters-state/parameters-and-transforms.md",
    ):
        assert phrase in text


def test_old_sources_are_absent_and_moved_pages_retain_unique_rationale() -> None:
    for source in OLD_SOURCES:
        assert not (DOCS / source).exists(), source

    for relative, phrases in RETAINED_RATIONALE.items():
        text = _read(relative)
        assert "## Learning objectives" not in text, relative
        assert "### Concept check" not in text, relative
        for phrase in phrases:
            assert phrase in text, (relative, phrase)


def test_new_and_refocused_sources_use_ascii_prose_and_labeled_latex() -> None:
    equation_pattern = re.compile(
        r"```\{math\}\s*\n:label:\s*eq-[a-z0-9-]+\s*\n", re.MULTILINE
    )
    for relative in PAGES:
        text = _read(relative)
        assert text.isascii(), relative
        assert equation_pattern.search(text), relative


def test_existing_method_routes_remain_stable() -> None:
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))
    for source, route in CURRENT_METHOD_ROUTES.items():
        assert manifest[source] == route
