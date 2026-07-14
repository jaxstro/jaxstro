"""Contracts for planned uncertainty and deferred field representation guides."""

from __future__ import annotations

import json
import re
from pathlib import Path

import jax.numpy as jnp
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
REPRESENTATIONS = DOCS / "30-representations"

UNCERTAINTY_GUIDES = (
    "uncertainty/what-uncertainty-represents.md",
    "uncertainty/linearized-propagation.md",
    "uncertainty/sigma-point-propagation.md",
    "uncertainty/ensemble-propagation.md",
)
FIELD_GUIDES = (
    "fields/fields-and-domains.md",
    "fields/topology-and-discretization.md",
    "fields/field-operators.md",
)
GUIDES = (*UNCERTAINTY_GUIDES, *FIELD_GUIDES)

COMMON_SECTIONS = (
    "## The scientific question",
    "## Mathematical objects",
    "## Core derivation",
    "## Failure modes and interpretation limits",
    "## What Jaxstro may add",
    "## Evidence required before implementation",
    "## Claim boundary",
    "## Connected representations, foundations, and methods",
)

CURRENT_REPRESENTATION_ROUTES = {
    "units-quantities/constants-and-conventions.md": "/constants-and-conventions",
    "units-quantities/quantity-system.md": "/quantity-system",
    "units-quantities/quantities.md": "/quantities",
    "units-quantities/equivalencies.md": "/equivalencies",
    "geometry-coordinates/coordinate-transformations.md": "/coordinate-transformations",
    "geometry-coordinates/geometry.md": "/geometry",
    "geometry-coordinates/astrometry.md": "/astrometry",
    "spectra-atmospheres/spectra-data-architecture.md": "/spectra-data-architecture",
    "spectra-atmospheres/conservative-spectral-resampling.md": (
        "/conservative-spectral-resampling"
    ),
    "spectra-atmospheres/atmosphere-capabilities.md": "/atmosphere-capabilities",
    "spectra-atmospheres/source-artifacts-and-adapters.md": (
        "/source-artifacts-and-adapters"
    ),
    "parameters-state/parameters-and-transforms.md": "/parameters-and-transforms",
    "parameters-state/pytrees-as-scientific-state.md": "/pytrees-as-scientific-state",
    "parameters-state/serialization-and-provenance.md": (
        "/serialization-and-provenance"
    ),
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


def _compact_math(source: str) -> str:
    return re.sub(r"\s+", "", source)


def _normalized_prose(source: str) -> str:
    return " ".join(source.split())


def test_all_guides_exist_once_in_toc_and_manifest_with_native_routes() -> None:
    config = yaml.safe_load((DOCS / "myst.yml").read_text(encoding="utf-8"))
    toc_files = _toc_files(config["project"]["toc"])
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))

    assert len(GUIDES) == 7
    for relative in GUIDES:
        source = f"30-representations/{relative}"
        assert (REPRESENTATIONS / relative).is_file(), source
        assert toc_files.count(source) == 1, source
        assert list(manifest).count(source) == 1, source
        assert manifest[source] == f"/{Path(relative).stem}"


def test_exact_status_admonitions_and_common_section_order() -> None:
    statuses = {
        **{relative: "Planned Jaxstro capability" for relative in UNCERTAINTY_GUIDES},
        **{relative: "Deferred abstraction" for relative in FIELD_GUIDES},
    }
    for relative, status in statuses.items():
        text = _read(relative)
        markers = (
            "Use this page when",
            f":::{'{'}important{'}'} {status}",
            *COMMON_SECTIONS,
        )
        positions = [text.index(marker) for marker in markers]
        assert positions == sorted(positions), relative
        assert text.count(f":::{'{'}important{'}'} {status}") == 1, relative


def test_each_guide_has_labeled_math_and_matching_prose_cross_reference() -> None:
    label_pattern = re.compile(
        r"```\{math\}\s*\n:label:\s*(eq-[a-z0-9-]+)\s*\n", re.MULTILINE
    )
    for relative in GUIDES:
        text = _read(relative)
        labels = label_pattern.findall(text)
        assert labels, relative
        assert any(f"[]({f'#{label}'})" in text for label in labels), relative


def test_linearized_formula_and_analytic_covariance_pushforward() -> None:
    text = _compact_math(_read("uncertainty/linearized-propagation.md"))
    assert ":label:eq-linearized-covariance" in text
    assert (
        r"\mathbf{C}_{y}\approx\mathbf{J}\mathbf{C}_{x}\mathbf{J}^{\mathsf{T}}" in text
    )

    jacobian = jnp.array([[2.0, -1.0], [0.5, 3.0]])
    input_covariance = jnp.array([[4.0, 1.5], [1.5, 9.0]])
    propagated = jacobian @ input_covariance @ jacobian.T
    expected = jnp.array([[19.0, -14.75], [-14.75, 86.5]])
    assert jnp.allclose(propagated, expected)


def test_sigma_and_ensemble_formula_contracts() -> None:
    sigma = _read("uncertainty/sigma-point-propagation.md")
    compact_sigma = _compact_math(sigma)
    assert ":label:eq-sigma-point-propagation" in compact_sigma
    assert r"w_i^{(m)}f(\boldsymbol{\chi}_i)" in compact_sigma
    assert (
        r"w_i^{(c)}\boldsymbol{\delta}_i\boldsymbol{\delta}_i^{\mathsf{T}}"
        in compact_sigma
    )
    assert "mean and covariance weights may differ" in sigma
    assert "negative weights" in sigma
    assert "positive-semidefinite output is not automatic" in sigma

    ensemble = _read("uncertainty/ensemble-propagation.md")
    compact_ensemble = _compact_math(ensemble)
    assert ":label:eq-ensemble-propagation" in compact_ensemble
    assert "N >= 2" in ensemble
    assert r"\frac{1}{N-1}\sum_{n=1}^{N}" in compact_ensemble

    outputs = jnp.array([[1.0, 2.0], [3.0, 1.0], [5.0, 6.0]])
    centered = outputs - jnp.mean(outputs, axis=0)
    unbiased = centered.T @ centered / (outputs.shape[0] - 1)
    assert jnp.allclose(unbiased, jnp.array([[4.0, 4.0], [4.0, 7.0]]))


def test_uncertainty_categories_and_inference_ecosystem_boundary() -> None:
    overview = _normalized_prose(
        _read("uncertainty/what-uncertainty-represents.md").replace("**", "")
    ).lower()
    for phrase in (
        "variation represented by a probability model",
        "uncertainty about parameters or latent state",
        "measurement and noise models",
        "numerical approximation error",
        "model discrepancy",
        "covariance matrix is not a complete probability distribution",
        "units and support",
    ):
        assert phrase in overview

    for relative in UNCERTAINTY_GUIDES:
        text = _normalized_prose(_read(relative))
        for phrase in (
            "JAX owns transformations and random primitives",
            "NumPyro and BlackJAX own probabilistic inference and sampling mechanics",
            "Informax owns inference-aware scientific workflows",
            "domain-agnostic propagation representations",
            "deterministic key policy",
            "does not perform inference",
        ):
            assert phrase in text, (relative, phrase)


def test_field_domain_topology_operator_and_deferral_contracts() -> None:
    fields = _compact_math(_read("fields/fields-and-domains.md")).lower()
    assert ":label:eq-field-map" in fields
    assert r"\phi:\omega\rightarrowv" in fields
    assert "sampledarrayisnotbyitselfafieldcontract" in fields

    topology = _normalized_prose(_read("fields/topology-and-discretization.md")).lower()
    for phrase in (
        "coordinates do not define topology",
        "nodes, cells, and faces",
        "orientation",
        "structured and unstructured",
        "topology changes are structural and nondifferentiable",
        "dynamic leaves",
    ):
        assert phrase in topology

    operators = _normalized_prose(_read("fields/field-operators.md"))
    for phrase in (
        "discrete gradient",
        "discrete divergence",
        "orientation",
        "spacing or metric",
        "boundary conditions",
        "units",
        "algebraic adjoint",
        "summation-by-parts",
        "conservation",
    ):
        assert phrase in operators

    for relative in FIELD_GUIDES:
        text = _normalized_prose(_read(relative))
        for phrase in (
            "`jaxstro.spatial`",
            "grid",
            "mesh",
            "geometry",
            "operator utilities",
            "two real consumers",
        ):
            assert phrase in text, (relative, phrase)


def test_guides_deny_runtime_existence_and_any_schedule_promise() -> None:
    for relative in UNCERTAINTY_GUIDES:
        text = _normalized_prose(_read(relative))
        assert "`jaxstro.uncertainty` does not exist" in text
        assert "No implementation schedule is promised" in text
    for relative in FIELD_GUIDES:
        text = _normalized_prose(_read(relative))
        assert "`jaxstro.fields` does not exist" in text
        assert "No implementation schedule is promised" in text


def test_guides_use_ascii_prose_and_latex_math() -> None:
    for relative in GUIDES:
        text = _read(relative)
        assert text.isascii(), relative
        assert len(text.split()) >= 450, relative


def test_landing_links_status_labeled_families() -> None:
    text = _read("representations.md")
    for phrase in (
        "Uncertainty propagation",
        "Planned Jaxstro capability",
        ":link: ./uncertainty/what-uncertainty-represents.md",
        "Fields and discretized domains",
        "Deferred abstraction",
        ":link: ./fields/fields-and-domains.md",
    ):
        assert phrase in text


def test_all_current_representation_routes_remain_stable() -> None:
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))
    assert len(CURRENT_REPRESENTATION_ROUTES) == 14
    for relative, route in CURRENT_REPRESENTATION_ROUTES.items():
        source = f"30-representations/{relative}"
        assert manifest[source] == route
