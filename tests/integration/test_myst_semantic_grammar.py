"""Source-aware contracts for the final researcher-first MyST site."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

TOP_LEVEL = (
    "Start here",
    "Foundations",
    "Numerical methods",
    "Scientific representations",
    "Research workflows",
    "API reference",
    "Validation and evidence",
    "Project",
)

LANDING_PAGES = {
    "index.md",
    "00-start-here/start-here.md",
    "10-foundations/foundations.md",
    "20-methods/methods.md",
    "30-representations/representations.md",
    "40-workflows/workflows.md",
    "50-api/api.md",
    "60-validation/validation.md",
    "70-project/project.md",
}

SECTION_LANDINGS = LANDING_PAGES - {"index.md"}

STATUS_PAGES = {
    "20-methods/change-constraints-evolution/nonlinear-systems.md": "Ecosystem guide",
    "20-methods/change-constraints-evolution/adaptive-differential-equations.md": (
        "Ecosystem guide"
    ),
    "20-methods/approximation-integration/adaptive-quadrature.md": "Ecosystem guide",
    "20-methods/linear-structure/iterative-linear-solvers.md": "Ecosystem guide",
    "20-methods/probability-sampling/quasi-monte-carlo.md": (
        "Planned Jaxstro capability"
    ),
    "20-methods/signals/signal-axes.md": "Planned Jaxstro capability",
    "20-methods/signals/windows-spectral-leakage.md": ("Planned Jaxstro capability"),
    "20-methods/signals/spectral-estimation.md": "Planned Jaxstro capability",
    "20-methods/signals/phase-and-delay.md": "Planned Jaxstro capability",
    "30-representations/uncertainty/what-uncertainty-represents.md": (
        "Planned Jaxstro capability"
    ),
    "30-representations/uncertainty/linearized-propagation.md": (
        "Planned Jaxstro capability"
    ),
    "30-representations/uncertainty/sigma-point-propagation.md": (
        "Planned Jaxstro capability"
    ),
    "30-representations/uncertainty/ensemble-propagation.md": (
        "Planned Jaxstro capability"
    ),
    "30-representations/fields/fields-and-domains.md": "Deferred abstraction",
    "30-representations/fields/topology-and-discretization.md": (
        "Deferred abstraction"
    ),
    "30-representations/fields/field-operators.md": "Deferred abstraction",
    "40-workflows/scientific-ml/preprocessing.md": "Planned Jaxstro capability",
    "40-workflows/scientific-ml/data-plans.md": "Planned Jaxstro capability",
    "40-workflows/scientific-ml/auditable-training.md": ("Planned Jaxstro capability"),
    "40-workflows/scientific-ml/ecosystem-boundaries.md": (
        "Planned Jaxstro capability"
    ),
}

CRITICAL_VISIBLE_HEADINGS = (
    "Core derivation",
    "Assumptions",
    "What JAX differentiates",
    "Where the claim stops",
)


def _manifest() -> dict[str, str]:
    return json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))


def _routed_markdown() -> dict[str, str]:
    return {
        relative: (DOCS / relative).read_text(encoding="utf-8")
        for relative in _manifest()
        if relative.endswith(".md")
    }


def _directive_blocks(text: str, directive: str) -> list[str]:
    """Return complete colon-fenced directive blocks from one MyST source."""
    lines = text.splitlines()
    blocks: list[str] = []
    opener = re.compile(rf"^(:{{3,}})\{{{re.escape(directive)}\}}(?:\s.*)?$")
    for index, line in enumerate(lines):
        match = opener.match(line)
        if match is None:
            continue
        fence = match.group(1)
        for end in range(index + 1, len(lines)):
            if lines[end] == fence:
                blocks.append("\n".join(lines[index : end + 1]))
                break
        else:
            raise AssertionError(f"unclosed {directive} directive at line {index + 1}")
    return blocks


def test_final_toc_has_implicit_home_and_exact_eight_visible_groups() -> None:
    config = yaml.safe_load((DOCS / "myst.yml").read_text(encoding="utf-8"))
    toc = config["project"]["toc"]

    assert toc[0] == {"file": "index.md"}
    assert tuple(item["title"] for item in toc[1:]) == TOP_LEVEL
    assert all("hidden" not in item for item in toc[1:])
    assert config["site"]["options"]["style"] == "site.css"


def test_final_routes_are_semantic_and_internal_sources_are_excluded() -> None:
    config = yaml.safe_load((DOCS / "myst.yml").read_text(encoding="utf-8"))
    manifest = _manifest()

    assert {
        "/start-here",
        "/methods",
        "/representations",
        "/workflows",
        "/api",
        "/validation",
        "/project",
    } <= set(manifest.values())
    forbidden_routes = {
        *(f"/index-{index}" for index in range(1, 12)),
        "/assessment-rubric",
        "/instructor-resources",
        "/teaching-with-jaxstro",
    }
    assert not (set(manifest.values()) & forbidden_routes)
    assert len(manifest) == 163
    assert set(config["project"]["exclude"]) == {
        "audits/**",
        "plans/**",
        "superpowers/**",
        "_build/**",
    }


def test_cards_and_grids_are_restricted_to_explicit_landing_pages() -> None:
    for relative, text in _routed_markdown().items():
        has_choice_ui = bool(re.search(r"^:{3,}\{(?:grid|card)\}", text, re.MULTILINE))
        if has_choice_ui:
            assert relative in LANDING_PAGES, relative


def test_section_landings_share_choice_note_and_status_contracts() -> None:
    routed = _routed_markdown()
    for relative in SECTION_LANDINGS:
        text = routed[relative]
        assert re.search(r"^:{4,}\{grid\}", text, re.MULTILINE), relative
        assert re.search(r"^:{3}\{card\}\s+", text, re.MULTILINE), relative
        assert re.search(r"^:{3}\{note\}", text, re.MULTILINE), relative
        assert re.search(r"^\|[^\n]*Status[^\n]*\|$", text, re.MULTILINE), relative


def test_status_pages_use_their_exact_single_status_class() -> None:
    routed = _routed_markdown()
    for relative, status in STATUS_PAGES.items():
        marker = f":::{'{'}important{'}'} {status}"
        assert routed[relative].count(marker) == 1, relative


def test_tabs_are_forbidden_without_a_recorded_exception() -> None:
    for relative, text in _routed_markdown().items():
        assert not re.search(r"^:{3,}\{(?:tabs|tab-item)\}", text, re.MULTILINE), (
            relative
        )


def test_core_scientific_sections_remain_outside_dropdowns() -> None:
    for relative, text in _routed_markdown().items():
        dropdowns = _directive_blocks(text, "dropdown")
        for heading in CRITICAL_VISIBLE_HEADINGS:
            if re.search(rf"^##\s+{re.escape(heading)}", text, re.MULTILINE):
                assert all(
                    not re.search(rf"^##\s+{re.escape(heading)}", block, re.MULTILINE)
                    for block in dropdowns
                ), (relative, heading)


def test_equation_and_figure_labels_are_globally_unique() -> None:
    owners: dict[str, str] = {}
    for relative, text in _routed_markdown().items():
        labels = re.findall(
            r"^:(?:label|name):\s*((?:eq|fig)-[a-z0-9-]+)\s*$",
            text,
            re.MULTILINE,
        )
        for label in labels:
            assert label not in owners, (label, owners.get(label), relative)
            owners[label] = relative
    assert owners


def test_load_bearing_figures_have_names_alt_text_captions_and_references() -> None:
    for relative, text in _routed_markdown().items():
        for block in _directive_blocks(text, "figure"):
            name = re.search(r"^:name:\s*(fig-[a-z0-9-]+)\s*$", block, re.MULTILINE)
            alt = re.search(r"^:alt:\s*(.+)\s*$", block, re.MULTILINE)
            assert name is not None, relative
            assert alt is not None and len(alt.group(1).split()) >= 6, relative

            content = [
                line.strip()
                for line in block.splitlines()[1:-1]
                if line.strip() and not line.startswith(":")
            ]
            assert content, relative
            assert f"](#{name.group(1)})" in text, (relative, name.group(1))


def test_planned_derivations_reference_at_least_one_labeled_equation() -> None:
    routed = _routed_markdown()
    for relative in STATUS_PAGES:
        labels = re.findall(
            r"^:label:\s*(eq-[a-z0-9-]+)\s*$", routed[relative], re.MULTILINE
        )
        assert labels, relative
        assert any(f"](#{label})" in routed[relative] for label in labels), relative
