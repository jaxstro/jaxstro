"""Source-aware contracts for the final researcher-first MyST site."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
    "20-methods/linear-structure/iterative-linear-solvers.md": "Ecosystem guide",
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

CURRENT_METHOD_PAGES = {
    "20-methods/change-constraints-evolution/autodiff.md",
    "20-methods/change-constraints-evolution/rootfinding.md",
    "20-methods/change-constraints-evolution/optimization.md",
    "20-methods/change-constraints-evolution/ode.md",
    "20-methods/approximation-integration/interpolation.md",
    "20-methods/approximation-integration/regular-grid.md",
    "20-methods/approximation-integration/bsplines.md",
    "20-methods/approximation-integration/cumulative-trapz.md",
    "20-methods/approximation-integration/quadrature.md",
    "20-methods/approximation-integration/adaptive-quadrature.md",
    "20-methods/linear-structure/linear-algebra.md",
    "20-methods/linear-structure/operators.md",
    "20-methods/linear-structure/special-functions.md",
    "20-methods/probability-sampling/distributions.md",
    "20-methods/probability-sampling/random.md",
    "20-methods/probability-sampling/sampling.md",
    "20-methods/probability-sampling/quasi-monte-carlo.md",
    "20-methods/discrete-space/grids.md",
    "20-methods/discrete-space/meshes.md",
    "20-methods/discrete-space/spatial.md",
}

SCIENTIFIC_ML_PAGES = {
    "40-workflows/scientific-ml/preprocessing.md",
    "40-workflows/scientific-ml/data-plans.md",
    "40-workflows/scientific-ml/auditable-training.md",
    "40-workflows/scientific-ml/ecosystem-boundaries.md",
}

LITERAL_BODY_DIRECTIVES = {"code", "code-block", "literalinclude"}


@dataclass(frozen=True)
class HeadingNode:
    level: int
    text: str
    line: int
    heading_ancestry: tuple[str, ...]
    directive_ancestry: tuple[str, ...]


@dataclass(frozen=True)
class DirectiveNode:
    name: str
    argument: str
    options: dict[str, str]
    body: str
    line_start: int
    line_end: int
    heading_ancestry: tuple[str, ...]
    parent_directives: tuple[str, ...]


@dataclass(frozen=True)
class MystDocument:
    source: str
    lines: tuple[str, ...]
    directives: tuple[DirectiveNode, ...]
    headings: tuple[HeadingNode, ...]
    literal_ranges: tuple[tuple[int, int], ...]


@dataclass
class _OpenDirective:
    name: str
    argument: str
    fence_char: str
    fence_length: int
    indent: int
    start_index: int
    heading_ancestry: tuple[str, ...]
    parent_directives: tuple[str, ...]


def _append_literal_range(ranges: list[tuple[int, int]], start: int, end: int) -> None:
    if ranges and ranges[-1][1] + 1 == start:
        ranges[-1] = (ranges[-1][0], end)
    else:
        ranges.append((start, end))


def _inside_list_context(lines: tuple[str, ...], index: int, indent: int) -> bool:
    """Return whether an indented line belongs to a preceding list item."""
    if indent < 2:
        return False
    list_item = re.compile(r"^( {0,3})(?:[-+*]|\d{1,9}[.)])(?:[ \t]+)")
    for previous in range(index - 1, -1, -1):
        line = lines[previous]
        if not line.strip():
            continue
        match = list_item.match(line)
        if match:
            return indent >= len(match.group(1)) + 2
        previous_indent = len(line) - len(line.lstrip(" "))
        if previous_indent < indent:
            return False
    return False


def _directive_parts(
    lines: tuple[str, ...], start: int, end: int
) -> tuple[dict[str, str], str]:
    options: dict[str, str] = {}
    body_start = start
    in_options = True
    for index in range(start, end):
        stripped = lines[index].strip()
        if in_options and not stripped:
            body_start = index + 1
            continue
        option = re.match(r"^:([A-Za-z0-9_-]+):(?:\s*(.*))?$", stripped)
        if in_options and option:
            options[option.group(1)] = (option.group(2) or "").strip()
            body_start = index + 1
            continue
        in_options = False
        break
    return options, "\n".join(lines[body_start:end]).strip()


def _parse_myst_source(source: str) -> MystDocument:
    """Tokenize headings and MyST directives while excluding literal fences."""
    lines = tuple(source.splitlines())
    directives: list[DirectiveNode] = []
    headings: list[HeadingNode] = []
    literal_ranges: list[tuple[int, int]] = []
    directive_stack: list[_OpenDirective] = []
    visible_heading_stack: list[HeadingNode] = []
    literal: tuple[str, int, int, int] | None = None

    directive_open = re.compile(
        r"^(\s*)(:{3,}|`{3,})\{([A-Za-z0-9_-]+)\}(?:\s+(.*?))?\s*$"
    )
    generic_fence = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
    heading_pattern = re.compile(r"^(\s*)(#{1,6})\s+(.+?)\s*#*\s*$")

    for index, line in enumerate(lines):
        if literal is not None:
            char, length, indent, start = literal
            close = re.match(
                rf"^\s{{0,{indent}}}{re.escape(char)}{{{length},}}\s*$", line
            )
            if close:
                literal_ranges.append((start + 1, index + 1))
                literal = None
            continue

        if directive_stack:
            top = directive_stack[-1]
            close = re.match(
                rf"^\s{{0,{top.indent}}}{re.escape(top.fence_char)}"
                rf"{{{top.fence_length},}}\s*$",
                line,
            )
            if close:
                frame = directive_stack.pop()
                options, body = _directive_parts(lines, frame.start_index + 1, index)
                directives.append(
                    DirectiveNode(
                        name=frame.name,
                        argument=frame.argument,
                        options=options,
                        body=body,
                        line_start=frame.start_index + 1,
                        line_end=index + 1,
                        heading_ancestry=frame.heading_ancestry,
                        parent_directives=frame.parent_directives,
                    )
                )
                if frame.name in LITERAL_BODY_DIRECTIVES:
                    _append_literal_range(
                        literal_ranges, frame.start_index + 1, index + 1
                    )
                continue
            if top.name in LITERAL_BODY_DIRECTIVES:
                continue

        indent = len(line) - len(line.lstrip(" "))
        in_list = _inside_list_context(lines, index, indent)
        if indent >= 4 and not directive_stack and not in_list:
            _append_literal_range(literal_ranges, index + 1, index + 1)
            continue

        opener = directive_open.match(line)
        if opener:
            fence = opener.group(2)
            directive_stack.append(
                _OpenDirective(
                    name=opener.group(3),
                    argument=(opener.group(4) or "").strip(),
                    fence_char=fence[0],
                    fence_length=len(fence),
                    indent=len(opener.group(1)),
                    start_index=index,
                    heading_ancestry=tuple(node.text for node in visible_heading_stack),
                    parent_directives=tuple(node.name for node in directive_stack),
                )
            )
            continue

        fence = generic_fence.match(line)
        if fence:
            marker = fence.group(2)
            literal = (marker[0], len(marker), len(fence.group(1)), index)
            continue

        heading = heading_pattern.match(line)
        if heading:
            level = len(heading.group(2))
            text = heading.group(3).strip()
            ancestry = tuple(
                node.text for node in visible_heading_stack if node.level < level
            )
            node = HeadingNode(
                level=level,
                text=text,
                line=index + 1,
                heading_ancestry=ancestry,
                directive_ancestry=tuple(item.name for item in directive_stack),
            )
            headings.append(node)
            if not directive_stack:
                visible_heading_stack = [
                    previous
                    for previous in visible_heading_stack
                    if previous.level < level
                ]
                visible_heading_stack.append(node)

    if literal is not None:
        raise AssertionError(f"unclosed literal fence at line {literal[3] + 1}")
    if directive_stack:
        raise AssertionError(
            f"unclosed {directive_stack[-1].name} directive "
            f"at line {directive_stack[-1].start_index + 1}"
        )

    directives.sort(key=lambda node: node.line_start)
    return MystDocument(
        source=source,
        lines=lines,
        directives=tuple(directives),
        headings=tuple(headings),
        literal_ranges=tuple(literal_ranges),
    )


def _critical_headings_are_visible(
    document: MystDocument, expected: tuple[str, ...]
) -> bool:
    visible_h2 = {
        node.text
        for node in document.headings
        if node.level == 2 and not node.directive_ancestry
    }
    return all(heading in visible_h2 for heading in expected)


def _has_reference_outside_owner(
    document: MystDocument, label: str, *, owner: str
) -> bool:
    excluded = list(document.literal_ranges)
    excluded.extend(
        (node.line_start, node.line_end)
        for node in document.directives
        if node.name in {owner, "figure", "math"}
    )
    reference = re.compile(rf"\]\(#{re.escape(label)}\)")
    return any(
        reference.search(line)
        and not any(start <= line_number <= end for start, end in excluded)
        for line_number, line in enumerate(document.lines, start=1)
    )


def _manifest() -> dict[str, str]:
    return json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))


def _toc_page_descendants(node: object) -> int:
    if isinstance(node, list):
        return sum(_toc_page_descendants(item) for item in node)
    if not isinstance(node, dict):
        return 0
    return int("file" in node) + _toc_page_descendants(node.get("children", []))


def _toc_structure_errors(
    nodes: object, *, title_depth: int = 0, ancestry: tuple[str, ...] = ()
) -> list[str]:
    """Return semantic-depth and subsection-size violations recursively."""
    if isinstance(nodes, list):
        return [
            error
            for node in nodes
            for error in _toc_structure_errors(
                node, title_depth=title_depth, ancestry=ancestry
            )
        ]
    if not isinstance(nodes, dict):
        return []

    title = nodes.get("title")
    node_title_depth = title_depth + int(title is not None)
    node_ancestry = ancestry + ((str(title),) if title is not None else ())
    errors: list[str] = []
    if title is not None and node_title_depth > 2:
        errors.append(f"TOC title is too deep: {' > '.join(node_ancestry)}")
    if (
        title is not None
        and node_title_depth == 2
        and not nodes.get("hidden", False)
        and _toc_page_descendants(nodes) < 3
    ):
        errors.append(
            f"TOC subsection has fewer than 3 pages: {' > '.join(node_ancestry)}"
        )
    errors.extend(
        _toc_structure_errors(
            nodes.get("children", []),
            title_depth=node_title_depth,
            ancestry=node_ancestry,
        )
    )
    return errors


def _routed_markdown() -> dict[str, str]:
    return {
        relative: (DOCS / relative).read_text(encoding="utf-8")
        for relative in _manifest()
        if relative.endswith(".md")
    }


def _routed_documents() -> dict[str, MystDocument]:
    return {
        relative: _parse_myst_source(text)
        for relative, text in _routed_markdown().items()
    }


def test_final_toc_has_implicit_home_and_exact_eight_visible_groups() -> None:
    config = yaml.safe_load((DOCS / "myst.yml").read_text(encoding="utf-8"))
    toc = config["project"]["toc"]

    assert toc[0] == {"file": "index.md"}
    assert tuple(item["title"] for item in toc[1:]) == TOP_LEVEL
    assert all("hidden" not in item for item in toc[1:])
    assert config["site"]["options"]["style"] == "site.css"


def test_visible_toc_respects_semantic_depth_and_minimum_subsection_size() -> None:
    config = yaml.safe_load((DOCS / "myst.yml").read_text(encoding="utf-8"))

    assert _toc_structure_errors(config["project"]["toc"]) == []


def test_toc_structure_check_recurses_through_nested_mutations() -> None:
    undersized = [
        {
            "title": "Top",
            "children": [
                {
                    "title": "Small",
                    "children": [{"file": "one.md"}, {"file": "two.md"}],
                }
            ],
        }
    ]
    overdeep = [
        {
            "title": "Top",
            "children": [
                {
                    "title": "Large",
                    "children": [
                        {"file": "one.md"},
                        {"file": "two.md"},
                        {
                            "title": "Nested",
                            "children": [{"file": "three.md"}],
                        },
                    ],
                }
            ],
        }
    ]

    assert any(
        "fewer than 3 pages" in error for error in _toc_structure_errors(undersized)
    )
    assert any("too deep" in error for error in _toc_structure_errors(overdeep))


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
    assert len(manifest) == 181
    assert set(config["project"]["exclude"]) == {
        "audits/**",
        "plans/**",
        "superpowers/**",
        "_build/**",
    }


def test_cards_and_grids_are_restricted_to_explicit_landing_pages() -> None:
    for relative, document in _routed_documents().items():
        has_choice_ui = any(
            node.name in {"grid", "card"} for node in document.directives
        )
        if has_choice_ui:
            assert relative in LANDING_PAGES, relative


def test_section_landings_share_choice_note_and_status_contracts() -> None:
    routed = _routed_documents()
    for relative in SECTION_LANDINGS:
        document = routed[relative]
        names = [node.name for node in document.directives]
        assert "grid" in names, relative
        assert "card" in names, relative
        assert "note" in names, relative
        assert re.search(r"^\|[^\n]*Status[^\n]*\|$", document.source, re.MULTILINE), (
            relative
        )


def test_status_pages_use_their_exact_single_status_class() -> None:
    routed = _routed_documents()
    for relative, status in STATUS_PAGES.items():
        markers = [
            node
            for node in routed[relative].directives
            if node.name == "important" and node.argument == status
        ]
        assert len(markers) == 1, relative


def test_tabs_are_forbidden_without_a_recorded_exception() -> None:
    for relative, document in _routed_documents().items():
        assert not any(
            node.name in {"tabs", "tab-item"} for node in document.directives
        ), relative


def test_core_scientific_sections_remain_outside_dropdowns() -> None:
    routed = _routed_documents()
    for relative in CURRENT_METHOD_PAGES:
        assert _critical_headings_are_visible(
            routed[relative], ("What JAX differentiates", "Where the claim stops")
        ), relative
    for relative in STATUS_PAGES:
        expected = ("Core derivation",)
        if relative in SCIENTIFIC_ML_PAGES:
            expected += ("Assumptions and failure boundaries", "Where the claim stops")
        assert _critical_headings_are_visible(routed[relative], expected), relative


def test_equation_and_figure_labels_are_globally_unique() -> None:
    owners: dict[str, str] = {}
    for relative, document in _routed_documents().items():
        for node in document.directives:
            label = node.options.get("label") or node.options.get("name")
            if label is None or not re.fullmatch(r"(?:eq|fig)-[a-z0-9-]+", label):
                continue
            assert label not in owners, (label, owners.get(label), relative)
            owners[label] = relative
    assert owners


def test_load_bearing_figures_have_names_alt_text_captions_and_references() -> None:
    for relative, document in _routed_documents().items():
        for figure in (node for node in document.directives if node.name == "figure"):
            name = figure.options.get("name")
            alt = figure.options.get("alt")
            assert name is not None and re.fullmatch(r"fig-[a-z0-9-]+", name), relative
            assert alt is not None and len(alt.split()) >= 6, relative
            assert figure.body, relative
            assert _has_reference_outside_owner(document, name, owner="figure"), (
                relative,
                name,
            )


def test_planned_derivations_reference_at_least_one_labeled_equation() -> None:
    routed = _routed_documents()
    for relative in STATUS_PAGES:
        labels = [
            node.options["label"]
            for node in routed[relative].directives
            if node.name == "math"
            and re.fullmatch(r"eq-[a-z0-9-]+", node.options.get("label", ""))
        ]
        assert labels, relative
        assert any(
            _has_reference_outside_owner(routed[relative], label, owner="math")
            for label in labels
        ), relative


def test_tokenizer_supports_backtick_directives_options_ranges_and_ancestry() -> None:
    source = """# Page

## Evidence

```{figure} result.webp
:name: fig-result
:alt: Six words describing the scientific result clearly

Measured result caption.
```

```{card} Follow the evidence
:link: /validation
Card body.
```

```{tabs}
````{tab-item} First
Nested body.
````
```

```{dropdown} Details
Hidden details.
```
"""
    document = _parse_myst_source(source)

    assert [node.name for node in document.directives] == [
        "figure",
        "card",
        "tabs",
        "tab-item",
        "dropdown",
    ]
    figure = document.directives[0]
    assert figure.options == {
        "name": "fig-result",
        "alt": "Six words describing the scientific result clearly",
    }
    assert figure.body == "Measured result caption."
    assert figure.line_start == 5
    assert figure.line_end == 10
    assert figure.heading_ancestry == ("Page", "Evidence")


def test_tokenizer_supports_indented_nested_colon_directives() -> None:
    source = """## Choices
  ::::{grid}
  :columns: 2

    :::{card} One
    Body.
    :::
  ::::
"""
    document = _parse_myst_source(source)

    assert [node.name for node in document.directives] == ["grid", "card"]
    assert document.directives[0].options == {"columns": "2"}
    assert document.directives[1].parent_directives == ("grid",)


def test_tokenizer_ignores_directive_examples_inside_literal_fences() -> None:
    source = """````text
:::{card} Not live
:::
```{tabs}
```
````

    ~~~markdown
    :::{figure} not-live.webp
    :name: fig-not-live
    :::
    ~~~
"""
    document = _parse_myst_source(source)

    assert document.directives == ()


def test_tokenizer_treats_literal_body_directives_as_opaque() -> None:
    for name in ("code", "code-block", "literalinclude"):
        source = f"""## Visible heading

````{{{name}}} example.py
:::{"{"}card{"}"} Not live
:::
```{{tabs}}
```
## Core derivation
````
"""
        document = _parse_myst_source(source)

        assert [node.name for node in document.directives] == [name]
        assert [node.text for node in document.headings] == ["Visible heading"]
        assert document.literal_ranges == ((3, 9),)


def test_tokenizer_ignores_top_level_four_space_indented_code() -> None:
    source = """## Visible heading

    ## Core derivation
    :::{card} Not live
    :::
    ```{tabs}
    ```
"""
    document = _parse_myst_source(source)

    assert document.directives == ()
    assert [node.text for node in document.headings] == ["Visible heading"]
    assert document.literal_ranges == ((3, 7),)


def test_tokenizer_preserves_indented_live_directives_in_list_context() -> None:
    source = """## Choices

- Compare approaches.

    :::{card} Live nested choice
    Body.
    :::
"""
    document = _parse_myst_source(source)

    assert [node.name for node in document.directives] == ["card"]
    assert document.directives[0].argument == "Live nested choice"
    assert document.directives[0].body == "Body."


def test_critical_heading_contract_rejects_missing_renamed_nested_or_wrong_level() -> (
    None
):
    assert _critical_headings_are_visible(
        _parse_myst_source("## Core derivation\nBody.\n"), ("Core derivation",)
    )
    for mutation in (
        "## Derivation\nBody.\n",
        "### Core derivation\nBody.\n",
        ":::{dropdown} Hidden\n## Core derivation\n:::\n",
        "",
    ):
        assert not _critical_headings_are_visible(
            _parse_myst_source(mutation), ("Core derivation",)
        )


def test_figure_self_reference_does_not_satisfy_page_reference() -> None:
    self_reference = """```{figure} result.webp
:name: fig-result
:alt: Six words describing the scientific result clearly

Caption with a [self reference](#fig-result).
```
"""
    external_reference = f"{self_reference}\nSee [the measured result](#fig-result).\n"

    assert not _has_reference_outside_owner(
        _parse_myst_source(self_reference), "fig-result", owner="figure"
    )
    assert _has_reference_outside_owner(
        _parse_myst_source(external_reference), "fig-result", owner="figure"
    )
