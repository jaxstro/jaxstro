"""End-to-end enforcement for the provenance-card registry."""

from __future__ import annotations

import importlib
import importlib.util
import tomllib
from pathlib import Path

from jaxstro.testing import (
    ASSERT_HELPERS,
    resolve_node_ids,
    validate_card,
)
from jaxstro.testing import (
    test_body_has_assert as body_has_assert,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_provenance_registry.py"
REGISTRY_DIR = REPO_ROOT / "docs" / "provenance" / "registry"
OUTPUT_DIR = (
    REPO_ROOT / "docs" / "50-api" / "research-infrastructure" / "source-provenance"
)
EXPECTED_FAMILIES = {"atmospheres", "constants", "transforms"}


def _load_builder():
    assert SCRIPT_PATH.exists(), f"missing registry builder: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location(
        "build_provenance_registry", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _families():
    return _load_builder().load_registry()


def _cards():
    return [card for cards in _families().values() for card in cards]


def test_registry_has_only_approved_families_and_unique_card_ids():
    families = _families()
    assert set(families) == EXPECTED_FAMILIES
    ids = [validate_card(card).id for cards in families.values() for card in cards]
    assert len(ids) == len(set(ids)), f"duplicate card ids: {ids}"


def test_incomplete_atmosphere_family_is_represented_honestly():
    families = _families()
    atmosphere_cards = families["atmospheres"]
    rendered = _load_builder().render_outputs(families)["atmospheres.md"]
    if not atmosphere_cards:
        assert "No source-verified cards are registered" in rendered


def test_code_references_resolve_to_real_jaxstro_symbols():
    problems = []
    for raw in _cards():
        card = validate_card(raw)
        for ref in card.code_refs:
            path_text, qualname = ref.split("::", 1)
            path = REPO_ROOT / path_text
            if not path.is_file():
                problems.append(f"{card.id}: missing code path {path_text}")
                continue
            relative = path.relative_to(REPO_ROOT / "src").with_suffix("")
            if relative.name == "__init__":
                relative = relative.parent
            module = importlib.import_module(".".join(relative.parts))
            obj = module
            try:
                for part in qualname.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                problems.append(f"{card.id}: unresolved code symbol {ref}")
    assert not problems, "\n".join(problems)


def test_validation_references_collect_and_assert_behavior():
    node_ids = sorted(
        {ref for card in _cards() for ref in validate_card(card).validation}
    )
    resolved = resolve_node_ids(node_ids, rootdir=str(REPO_ROOT))
    missing = sorted(set(node_ids) - resolved)
    assert not missing, f"validation node ids not collectable: {missing}"
    helpers = ASSERT_HELPERS + ("pytest.warns",)
    toothless = [
        ref for ref in node_ids if not body_has_assert(ref, assert_helpers=helpers)
    ]
    assert not toothless, f"validation tests without assertions: {toothless}"


def test_generated_pages_equal_fresh_rendering():
    rendered = _load_builder().render_outputs(_families())
    stale = [
        name
        for name, text in rendered.items()
        if not (OUTPUT_DIR / name).is_file()
        or (OUTPUT_DIR / name).read_text(encoding="utf-8") != text
    ]
    assert not stale, f"stale generated provenance pages: {stale}"


def test_yaml_parser_is_development_only():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = config["project"]["dependencies"]
    dev_extra = config["project"]["optional-dependencies"]["dev"]
    dev_group = config["dependency-groups"]["dev"]
    assert not any(item.lower().startswith("pyyaml") for item in runtime)
    assert any(item.lower().startswith("pyyaml") for item in dev_extra)
    assert any(item.lower().startswith("pyyaml") for item in dev_group)
