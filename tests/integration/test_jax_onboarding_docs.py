"""Contracts for the beginner JAX onboarding route."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START_HERE = ROOT / "docs" / "00-start-here"

PAGES = {
    "why-jax.md": ("# Why JAX?", "## When JAX is the wrong tool"),
    "jax-from-first-principles.md": (
        "# JAX from first principles",
        "## A Python function, a mathematical map, and a traced program",
    ),
    "ways-to-use-these-docs.md": (
        "# Ways to use these docs",
        "## Research-question first",
    ),
}

ONBOARDING_ROUTE = (
    "00-start-here/start-here.md",
    "00-start-here/why-jax.md",
    "00-start-here/jax-from-first-principles.md",
    "00-start-here/choose-your-path.md",
    "00-start-here/first-research-calculation.md",
)

TASK_1_ROUTED_PAGES = (
    "index.md",
    *ONBOARDING_ROUTE,
    "00-start-here/ways-to-use-these-docs.md",
    "10-foundations/foundations.md",
    "40-workflows/differentiable-research/science-patterns.md",
    "40-workflows/workflows.md",
)


def _assert_links_in_order(text: str, links: tuple[str, ...]) -> None:
    positions = [text.index(f"]({link})") for link in links]
    assert positions == sorted(positions)


def _python_cells(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL))


def test_beginner_pages_exist_and_name_their_boundaries() -> None:
    for name, phrases in PAGES.items():
        text = (START_HERE / name).read_text(encoding="utf-8")
        for phrase in phrases:
            assert phrase in text


def test_why_jax_does_not_promise_automatic_correctness_or_speed() -> None:
    text = (START_HERE / "why-jax.md").read_text(encoding="utf-8")
    for phrase in (
        "JAX does not make an algorithm correct",
        "JAX does not make every program faster",
        "JAX does not make every derivative scientifically meaningful",
    ):
        assert phrase in text


def test_start_here_keeps_one_ordered_route_without_expanding_the_homepage() -> None:
    homepage = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    start_here = (START_HERE / "start-here.md").read_text(encoding="utf-8")
    usage_router = (START_HERE / "ways-to-use-these-docs.md").read_text(
        encoding="utf-8"
    )

    assert "](./00-start-here/start-here.md)" in homepage
    assert "](./00-start-here/why-jax.md)" in homepage
    assert "](./00-start-here/jax-from-first-principles.md)" not in homepage
    card_positions = [
        start_here.index(f":link: ./{Path(page).name}") for page in ONBOARDING_ROUTE[1:]
    ]
    assert card_positions == sorted(card_positions)
    _assert_links_in_order(
        usage_router,
        tuple(f"./{Path(page).name}" for page in ONBOARDING_ROUTE),
    )
    assert "](./ways-to-use-these-docs.md)" in start_here
    assert "Jaxstro owns" in start_here


def test_precision_is_enabled_before_the_first_jax_evaluation() -> None:
    text = (START_HERE / "jax-from-first-principles.md").read_text(encoding="utf-8")
    precision = text.index("enable_high_precision()")
    definition = text.index("def scaled_luminosity(")
    first_array = text.index("jnp.array(")
    first_evaluation = text.index("scaled_luminosity(2.0, 0.5)")

    assert precision < definition < first_array
    assert precision < first_evaluation


def test_first_principles_has_one_executable_scaled_luminosity_owner() -> None:
    text = (START_HERE / "jax-from-first-principles.md").read_text(encoding="utf-8")
    owners = 0
    for cell in _python_cells(text):
        tree = ast.parse(cell)
        owners += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "scaled_luminosity"
            for node in ast.walk(tree)
        )
        owners += sum(
            isinstance(node, ast.ImportFrom)
            and node.module == "examples.onboarding.first_jax_map"
            and any(alias.name == "scaled_luminosity" for alias in node.names)
            for node in ast.walk(tree)
        )

    assert owners == 1


def test_first_principles_python_cells_run_in_source_order() -> None:
    text = (START_HERE / "jax-from-first-principles.md").read_text(encoding="utf-8")
    subprocess.run(
        [sys.executable, "-c", "\n".join(_python_cells(text))],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_why_jax_has_a_rendered_three_way_decision_table() -> None:
    source = (START_HERE / "why-jax.md").read_text(encoding="utf-8")
    required = (
        "NumPy-style script",
        "Direct JAX",
        "Jaxstro",
        "Program transformations",
        "State and compilation constraints",
        "Units and conventions",
        "Derivative and evidence contracts",
        "Best fit",
    )
    assert "```{list-table} Choosing a research-programming layer" in source
    for phrase in required:
        assert phrase in source

    numpy_transform_cell = source.split("* - Program transformations", 1)[1].split(
        "* - State and compilation constraints", 1
    )[0]
    normalized_cell = numpy_transform_cell.lower()
    assert "eager vectorized and broadcast array batching is native" in normalized_cell
    assert "array-oriented" in numpy_transform_cell
    assert "general `vmap` lifting" in normalized_cell
    assert "automatic differentiation require other machinery" in normalized_cell
    assert "batching and derivatives are separate implementations" not in source

    subprocess.run(
        ["myst", "build", "--html", "--ci", "--strict"],
        cwd=ROOT / "docs",
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = (ROOT / "docs" / "_build" / "html" / "why-jax" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "<table" in rendered
    for choice in ("NumPy-style script", "Direct JAX", "Jaxstro"):
        assert choice in rendered


def test_substantial_beginner_pages_route_readers_onward() -> None:
    for name in PAGES:
        text = (START_HERE / name).read_text(encoding="utf-8")
        assert "Use this page when" in text, name

    first_principles = (START_HERE / "jax-from-first-principles.md").read_text(
        encoding="utf-8"
    )
    assert "](../10-foundations/foundations.md)" in first_principles
    assert (
        "](../20-methods/change-constraints-evolution/autodiff.md)" in first_principles
    )


def test_task_1_routed_sources_use_ascii_punctuation() -> None:
    for relative in TASK_1_ROUTED_PAGES:
        text = (ROOT / "docs" / relative).read_text(encoding="utf-8")
        assert text.isascii(), relative


def test_start_here_pages_have_canonical_routes_and_toc_entries() -> None:
    myst = (ROOT / "docs" / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads(
        (ROOT / "docs" / "route-manifest.json").read_text(encoding="utf-8")
    )
    routes = {
        "00-start-here/start-here.md": "/start-here",
        "00-start-here/first-research-calculation.md": ("/first-research-calculation"),
        "00-start-here/choose-your-path.md": "/choose-your-path",
        "00-start-here/why-jax.md": "/why-jax",
        "00-start-here/jax-from-first-principles.md": ("/jax-from-first-principles"),
        "00-start-here/ways-to-use-these-docs.md": "/ways-to-use-these-docs",
    }

    assert myst.count("title: Start here") == 1
    for page, route in routes.items():
        assert myst.count(f"file: {page}") == 1
        assert manifest[page] == route
    assert "00-getting-started/index.md" not in manifest
    assert "/how-to-learn" not in manifest.values()


def test_entry_pages_were_moved_out_of_the_old_directories() -> None:
    assert not (ROOT / "docs" / "00-getting-started" / "index.md").exists()
    assert not (ROOT / "docs" / "00-getting-started" / "how-to-learn.md").exists()
    assert not (ROOT / "docs" / "05-foundations" / "choose-your-path.md").exists()
