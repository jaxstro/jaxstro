"""Executable public-surface contracts for the API reference page."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

import jaxstro
from jaxstro.contracts import ADSemantics, SupportLevel, get_callable_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
API_PAGE = REPO_ROOT / "docs" / "40-api" / "index.md"


def _api_text() -> str:
    return API_PAGE.read_text(encoding="utf-8")


def _api_section(module: str) -> str:
    text = _api_text()
    match = re.search(
        rf"^### `{re.escape(module)}`\n(?P<body>.*?)(?=^### |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing API section for {module}"
    return match.group("body")


def test_spatial_is_an_eager_top_level_public_module_in_a_clean_process() -> None:
    code = """
import jaxstro
assert hasattr(jaxstro, "spatial")
assert "spatial" in jaxstro.__all__
assert jaxstro.spatial.__name__ == "jaxstro.spatial"
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_documented_public_import_surface_is_executable() -> None:
    public_modules = (
        "astrometry",
        "atmospheres",
        "constants",
        "coords",
        "geometry",
        "numerics",
        "params",
        "provenance",
        "quantity",
        "spatial",
        "testing",
        "units",
    )

    for module in public_modules:
        imported = importlib.import_module(f"jaxstro.{module}")
        assert getattr(jaxstro, module) is imported
        assert module in jaxstro.__all__

    from jaxstro.jaxconfig import enable_high_precision

    assert callable(enable_high_precision)


def test_api_table_has_structured_ownership_boundary_and_evidence_fields() -> None:
    text = _api_text()

    assert "```{list-table} Public modules" in text
    assert "  - Ownership" in text
    assert "  - Runtime / preprocessing boundary" in text
    assert "  - Evidence and status" in text
    assert "`jaxstro.units` is the current ecosystem contract" in text
    assert "`jaxstro.quantity` is implemented" in text
    assert "ecosystem adoption and any replacement cutover remain deferred" in text
    assert "Implemented with explicit policy gaps" in text
    assert "Sonora and BSTAR adapters are present" in text
    assert "../20-methods/discrete-space/spatial.md" in text


def test_api_reference_exposes_provenance_card_tooling_and_routes() -> None:
    text = _api_text()

    for symbol in ("ProvenanceCard", "validate_card", "render_card", "render_registry"):
        assert getattr(jaxstro.testing, symbol) is not None
        assert symbol in jaxstro.testing.__all__
        assert f"`{symbol}" in text

    assert "./provenance/index.md" in text
    assert "source-backed provenance cards" in text
    assert "runtime manifests" in text


def test_random_api_section_links_each_symbol_family_to_its_method_scope() -> None:
    section = _api_section("jaxstro.numerics.random")
    paragraphs = [paragraph for paragraph in section.split("\n\n") if paragraph]

    prng_paragraph = next(
        paragraph for paragraph in paragraphs if "`key_stream(...)`" in paragraph
    )
    assert "`fold_in_stream(...)`" in prng_paragraph
    assert "`seed_manifest(...)`" in prng_paragraph
    assert "../20-methods/probability-sampling/random.md" in prng_paragraph
    assert "../20-methods/probability-sampling/sampling.md" not in prng_paragraph

    resampling_paragraph = next(
        paragraph
        for paragraph in paragraphs
        if "`systematic_resample(...)`" in paragraph
    )
    assert "`stratified_resample(...)`" in resampling_paragraph
    assert "`residual_resample(...)`" in resampling_paragraph
    assert "../20-methods/probability-sampling/sampling.md" in resampling_paragraph
    assert "../20-methods/probability-sampling/random.md" not in resampling_paragraph


def test_interpolation_reference_does_not_duplicate_symbol_descriptions() -> None:
    text = _api_text()

    assert text.count("`pchip_slopes(...)`") == 1
    assert text.count("`monotone_cubic_interp(...)`") == 1


def test_safeguarded_rootfinding_public_surface_is_documented_and_executable() -> None:
    symbols = (
        "PROPOSAL_NONE",
        "PROPOSAL_SECANT",
        "PROPOSAL_MIDPOINT",
        "PROPOSAL_LO_ENDPOINT",
        "PROPOSAL_HI_ENDPOINT",
        "PROPOSAL_INVERSE_QUADRATIC",
        "ROOT_STATUS_RUNNING",
        "ROOT_STATUS_EXACT_LO",
        "ROOT_STATUS_EXACT_HI",
        "ROOT_STATUS_EXACT_INTERIOR",
        "ROOT_STATUS_WIDTH_CONVERGED",
        "ROOT_STATUS_MISSING_BRACKET",
        "ROOT_STATUS_NONFINITE_EVALUATION",
        "ROOT_STATUS_MAX_STEPS",
        "DERIVATIVE_STATUS_CERTIFIED",
        "DERIVATIVE_STATUS_PRIMAL_FAILED",
        "DERIVATIVE_STATUS_ASSUMPTIONS_REJECTED",
        "DERIVATIVE_STATUS_NONFINITE",
        "DERIVATIVE_STATUS_RESIDUAL_TOO_LARGE",
        "DERIVATIVE_STATUS_SLOPE_ILL_CONDITIONED",
        "DERIVATIVE_STATUS_BRACKET_TOO_WIDE",
        "BracketState",
        "BracketHistory",
        "BracketedRootState",
        "BracketProposal",
        "RootTrace",
        "BracketedRootResult",
        "ImplicitRootAssumptions",
        "ImplicitRootCertificate",
        "ImplicitRootResult",
        "initialize_bracket",
        "initialize_bracketed_root_state",
        "update_bracket",
        "propose_bracketed",
        "advance_bracketed_root",
        "safeguarded_bracketed_root",
        "map_safeguarded_bracketed_root",
        "implicit_bracketed_root",
    )
    text = _api_text()

    for symbol in symbols:
        assert getattr(jaxstro.numerics, symbol) is not None
        assert symbol in jaxstro.numerics.__all__
        assert f"`{symbol}" in text

    assert "value-first" in text
    assert "implicit-root derivative" in text


def test_universal_kepler_public_surface_is_documented_and_contracted() -> None:
    symbols = (
        "KEPLER_STATUS_CONVERGED",
        "KEPLER_STATUS_INVALID_INPUT",
        "KEPLER_STATUS_NONFINITE_ITERATION",
        "KEPLER_STATUS_SINGULAR_RADIUS",
        "KEPLER_STATUS_MAX_STEPS",
        "UniversalKeplerResult",
        "universal_kepler_step",
    )
    text = _api_text()

    for symbol in symbols:
        assert getattr(jaxstro.numerics, symbol) is not None
        assert symbol in jaxstro.numerics.__all__
        assert f"`{symbol}" in text

    contract = get_callable_contract("jaxstro.numerics.universal_kepler_step")
    transforms = {item.transform: item for item in contract.transforms}
    assert contract.ad_semantics is ADSemantics.SMOOTH_PATHWISE
    assert transforms["jit"].support is SupportLevel.SUPPORTED
    assert transforms["vmap"].support is SupportLevel.SUPPORTED
    assert transforms["jvp"].support is SupportLevel.CONDITIONAL
    assert transforms["vjp"].support is SupportLevel.CONDITIONAL
    assert "fixed" in transforms["jvp"].conditions.lower()
    assert "fixed" in transforms["vjp"].conditions.lower()
