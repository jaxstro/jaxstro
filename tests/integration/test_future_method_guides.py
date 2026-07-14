"""Contracts for future method background and delegated ecosystem guides."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
METHODS = DOCS / "20-methods"

GUIDES = {
    "change-constraints-evolution/nonlinear-systems.md": "Ecosystem guide",
    "change-constraints-evolution/adaptive-differential-equations.md": (
        "Ecosystem guide"
    ),
    "approximation-integration/adaptive-quadrature.md": "Ecosystem guide",
    "linear-structure/iterative-linear-solvers.md": "Ecosystem guide",
    "probability-sampling/quasi-monte-carlo.md": "Planned Jaxstro capability",
    "signals/signal-axes.md": "Planned Jaxstro capability",
    "signals/windows-spectral-leakage.md": "Planned Jaxstro capability",
    "signals/spectral-estimation.md": "Planned Jaxstro capability",
    "signals/phase-and-delay.md": "Planned Jaxstro capability",
}

ROUTES = {relative: f"/{Path(relative).stem}" for relative in GUIDES}

COMMON_SECTIONS = (
    "## The scientific question",
    "## Mathematical objects",
    "## Core derivation",
    "## What the ecosystem already owns",
    "## What Jaxstro may add",
    "## Evidence required before implementation",
    "## Claim boundary",
    "## Connected foundations and methods",
)

DELEGATED_OWNERS = {
    "change-constraints-evolution/nonlinear-systems.md": (
        "Optimistix",
        "https://docs.kidger.site/optimistix/",
    ),
    "change-constraints-evolution/adaptive-differential-equations.md": (
        "Diffrax",
        "https://docs.kidger.site/diffrax/",
    ),
    "approximation-integration/adaptive-quadrature.md": (
        "Quadax",
        "https://quadax.readthedocs.io/en/",
    ),
    "linear-structure/iterative-linear-solvers.md": (
        "Lineax",
        "https://docs.kidger.site/lineax/",
    ),
}

CURRENT_METHOD_ROUTES = {
    "20-methods/change-constraints-evolution/autodiff.md": "/autodiff",
    "20-methods/change-constraints-evolution/rootfinding.md": "/rootfinding",
    "20-methods/change-constraints-evolution/optimization.md": "/optimization",
    "20-methods/change-constraints-evolution/ode.md": "/ode",
    "20-methods/approximation-integration/interpolation.md": "/interpolation",
    "20-methods/approximation-integration/regular-grid.md": "/regular-grid",
    "20-methods/approximation-integration/bsplines.md": "/bsplines",
    "20-methods/approximation-integration/cumulative-trapz.md": ("/cumulative-trapz"),
    "20-methods/approximation-integration/quadrature.md": "/quadrature",
    "20-methods/linear-structure/linear-algebra.md": "/linear-algebra",
    "20-methods/linear-structure/operators.md": "/operators",
    "20-methods/linear-structure/special-functions.md": "/special-functions",
    "20-methods/probability-sampling/distributions.md": "/distributions",
    "20-methods/probability-sampling/random.md": "/random",
    "20-methods/probability-sampling/sampling.md": "/sampling",
    "20-methods/discrete-space/grids.md": "/grids",
    "20-methods/discrete-space/meshes.md": "/meshes",
    "20-methods/discrete-space/spatial.md": "/spatial",
}


def _read(relative: str) -> str:
    return (METHODS / relative).read_text(encoding="utf-8")


def test_all_guides_exist_once_in_toc_and_manifest_with_native_routes() -> None:
    myst = (DOCS / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))

    assert len(GUIDES) == 9
    for relative in GUIDES:
        source = f"20-methods/{relative}"
        assert (METHODS / relative).is_file(), source
        assert myst.count(f"file: {source}") == 1, source
        assert list(manifest).count(source) == 1, source
        assert manifest[source] == ROUTES[relative]


def test_exact_status_admonitions_and_common_section_order() -> None:
    for relative, status in GUIDES.items():
        text = _read(relative)
        markers = (
            "Use this page when",
            f":::{'{'}important{'}'} {status}",
            *COMMON_SECTIONS,
        )
        positions = [text.index(marker) for marker in markers]
        assert positions == sorted(positions), relative
        assert text.count(f":::{'{'}important{'}'} {status}") == 1, relative


def test_each_guide_has_labeled_math_and_matching_cross_reference() -> None:
    label_pattern = re.compile(
        r"```\{math\}\s*\n:label:\s*(eq-[a-z0-9-]+)\s*\n", re.MULTILINE
    )
    for relative in GUIDES:
        text = _read(relative)
        labels = label_pattern.findall(text)
        assert labels, relative
        assert any(f"[]({f'#{label}'})" in text for label in labels), relative


def test_delegated_guides_name_and_link_official_owners() -> None:
    for relative, (owner, url) in DELEGATED_OWNERS.items():
        text = _read(relative)
        assert f"[{owner}]({url})" in text, relative

    iterative = _read("linear-structure/iterative-linear-solvers.md")
    assert "[JAX](https://docs.jax.dev/" in iterative


def test_qmc_distinguishes_three_point_constructions_and_error_claims() -> None:
    text = _read("probability-sampling/quasi-monte-carlo.md").lower()
    for phrase in (
        "deterministic low-discrepancy points",
        "independent random samples",
        "replicated randomized scrambles",
        "does not provide an uncertainty estimate",
    ):
        assert phrase in text


def test_signal_family_contains_required_scientific_contracts() -> None:
    signal = "\n".join(
        _read(relative) for relative in GUIDES if relative.startswith("signals/")
    )
    for token in (
        "X_k",
        "f_{\\mathrm{Nyq}}",
        "equivalent noise bandwidth",
        "coherent gain",
        "one-sided",
        "two-sided",
        "cross spectrum",
        "phase wrapping",
        r"\tau(f) = -\frac{\phi(f)}{2\pi f}",
    ):
        assert token in signal

    for relative in (
        "signals/signal-axes.md",
        "signals/windows-spectral-leakage.md",
        "signals/spectral-estimation.md",
        "signals/phase-and-delay.md",
    ):
        text = _read(relative)
        assert "[JAX FFT](https://docs.jax.dev/" in text, relative
        assert "`jaxstro.signal` does not exist" in text, relative


def test_planned_pages_do_not_claim_unimplemented_runtime_modules() -> None:
    planned = [
        relative for relative, status in GUIDES.items() if status.startswith("Planned")
    ]
    forbidden = (
        "from jaxstro.signal import",
        "from jaxstro.numerics.qmc import",
        "`jaxstro.signal` provides",
        "`jaxstro.numerics.qmc` provides",
        "implemented in `jaxstro.signal`",
        "implemented in `jaxstro.numerics.qmc`",
    )
    for relative in planned:
        text = _read(relative)
        assert "does not exist" in text, relative
        assert not any(phrase in text for phrase in forbidden), relative


def test_new_sources_are_ascii_and_use_latex_for_math() -> None:
    for relative in GUIDES:
        text = _read(relative)
        assert text.isascii(), relative
        assert "```{math}" in text, relative


def test_new_routes_preserve_all_eighteen_current_method_routes() -> None:
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))
    assert len(CURRENT_METHOD_ROUTES) == 18
    for source, route in CURRENT_METHOD_ROUTES.items():
        assert manifest[source] == route


def test_methods_landing_links_the_signal_family() -> None:
    text = (METHODS / "methods.md").read_text(encoding="utf-8")
    assert ":link: ./signals/signal-axes.md" in text
