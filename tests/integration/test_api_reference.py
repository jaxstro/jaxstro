"""Executable public-surface contracts for owner-qualified API pages."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import jaxstro
from jaxstro.contracts import ADSemantics, SupportLevel, get_callable_contract
from jaxstro.numerics import kepler, rootfinding

REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = REPO_ROOT / "docs" / "50-api"


def _page(relative: str) -> str:
    return (API_ROOT / relative).read_text(encoding="utf-8")


def test_spatial_is_an_eager_top_level_public_module_in_a_clean_process() -> None:
    code = """
import jaxstro
assert hasattr(jaxstro, "spatial")
assert "spatial" in jaxstro.__all__
assert jaxstro.spatial.__name__ == "jaxstro.spatial"
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_documented_public_module_surface_is_executable() -> None:
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


def test_api_landing_states_current_ownership_and_migration_boundary() -> None:
    text = _page("api.md")

    assert "## Canonical import policy" in text
    assert "## Method owners" in text
    assert "## Representation and data owners" in text
    assert "## Research infrastructure owners" in text
    assert "legacy inventory awaiting" in text
    assert "This reference does not change runtime exports" in text
    assert "Only current importable surfaces appear here" in text


def test_api_reference_exposes_provenance_card_tooling_and_routes() -> None:
    text = _page("research-infrastructure/testing.md")

    for symbol in ("ProvenanceCard", "validate_card", "render_card", "render_registry"):
        assert getattr(jaxstro.testing, symbol) is not None
        assert symbol in jaxstro.testing.__all__
        assert f"`{symbol}" in text

    assert "./source-provenance/source-provenance.md" in text
    assert "source-backed provenance cards" in text


def test_random_api_page_links_each_symbol_family_to_its_method_scope() -> None:
    text = _page("randomness/random.md")
    records = text.split("## Shape and dtype expectations", 1)[0]

    assert "`key_stream`" in records
    assert "`fold_in_stream`" in records
    assert "`seed_manifest`" in records
    assert "`systematic_resample`" in records
    assert "`stratified_resample`" in records
    assert "`residual_resample`" in records
    assert "../../20-methods/probability-sampling/random.md" in text
    assert "../../20-methods/probability-sampling/sampling.md" in text


def test_interpolation_reference_does_not_duplicate_symbol_descriptions() -> None:
    text = _page("approximation-integration/interpolation.md")

    assert text.count("`pchip_slopes(...)`") == 1
    assert text.count("`monotone_cubic_interp(...)`") == 1


def test_safeguarded_rootfinding_surface_is_documented_from_its_owner() -> None:
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
    owner_text = _page("change-constraints/rootfinding.md")
    contract_text = _page("research-infrastructure/contracts.md")

    for symbol in symbols:
        assert getattr(rootfinding, symbol) is not None
        assert (
            f"jaxstro.numerics.{symbol}" in contract_text or f"`{symbol}" in owner_text
        )

    assert "value-first" in owner_text
    assert "implicit derivative" in owner_text


def test_universal_kepler_surface_is_documented_and_contracted_by_owner() -> None:
    symbols = (
        "KEPLER_STATUS_CONVERGED",
        "KEPLER_STATUS_INVALID_INPUT",
        "KEPLER_STATUS_NONFINITE_ITERATION",
        "KEPLER_STATUS_SINGULAR_RADIUS",
        "KEPLER_STATUS_MAX_STEPS",
        "UniversalKeplerResult",
        "universal_kepler_step",
    )
    text = _page("change-constraints/kepler.md")

    for symbol in symbols:
        assert getattr(kepler, symbol) is not None
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
