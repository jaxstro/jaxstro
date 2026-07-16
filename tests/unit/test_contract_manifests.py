"""Coverage and import-isolation contracts for module manifests."""

import subprocess
import sys

import jaxstro
from jaxstro.contracts import collect_contracts
from jaxstro.contracts.registry import resolve_import_path

PUBLIC = {
    f"jaxstro.{name}"
    for name in (
        "astrometry",
        "atmospheres",
        "constants",
        "contracts",
        "coords",
        "evidence",
        "geometry",
        "jaxconfig",
        "numerics",
        "params",
        "provenance",
        "quantity",
        "quad",
        "spatial",
        "spectra",
        "testing",
        "units",
    )
}


def test_every_public_module_has_one_contract() -> None:
    inventory = collect_contracts(source_revision="test")
    assert {record.import_path for record in inventory.modules} == PUBLIC


def test_collection_does_not_import_optional_packages() -> None:
    code = """
import sys
import jaxstro.contracts
from jaxstro.contracts import collect_contracts
assert 'jaxstro.atmospheres' not in sys.modules
assert 'jaxstro.numerics' not in sys.modules
assert 'jaxstro.quantity' not in sys.modules
assert 'jaxstro.spectra' not in sys.modules
collect_contracts()
assert 'jaxstro.atmospheres' not in sys.modules
assert 'jaxstro.numerics' not in sys.modules
assert 'jaxstro.quantity' not in sys.modules
assert 'jaxstro.spectra' not in sys.modules
assert all(name not in sys.modules for name in ('polars', 'numpyro', 'optax'))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_quad_contract_collection_does_not_import_runtime_quad() -> None:
    code = """
import sys
from jaxstro.contracts import collect_contracts
assert 'jaxstro.quad' not in sys.modules
collect_contracts()
assert 'jaxstro.quad' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_contracts_is_public() -> None:
    assert jaxstro.contracts.__name__ == "jaxstro.contracts"
    assert "contracts" in jaxstro.__all__


def test_quad_contract_registers_fixed_evaluation() -> None:
    records = {item.import_path: item for item in collect_contracts().modules}
    fixed = {item.import_path: item for item in records["jaxstro.quad"].callables}[
        "jaxstro.quad.fixed"
    ]
    assert fixed.ad_semantics.value == "smooth_pathwise"
    assert {item.transform for item in fixed.transforms} == {"jax.jit", "jax.vmap"}


def test_quad_contract_registers_adaptive_integration() -> None:
    records = {item.import_path: item for item in collect_contracts().modules}
    adaptive = {item.import_path: item for item in records["jaxstro.quad"].callables}[
        "jaxstro.quad.integrate"
    ]
    assert adaptive.ad_semantics.value == "smooth_pathwise"
    assert {item.transform for item in adaptive.transforms} == {
        "jax.jit",
        "jax.vmap",
        "jvp",
        "vjp",
        "jacfwd/jacrev",
    }
    assert {item.transform: item.support.value for item in adaptive.transforms} == {
        "jax.jit": "supported",
        "jax.vmap": "supported",
        "jvp": "conditional",
        "vjp": "conditional",
        "jacfwd/jacrev": "conditional",
    }
    assert {item.kind.value for item in adaptive.evidence} == {
        "integration_test",
        "validation_test",
        "artifact",
    }


def test_contract_resolution_prefers_public_callable_over_same_named_module() -> None:
    assert resolve_import_path("jaxstro.quad.fixed") is jaxstro.quad.fixed
    assert resolve_import_path("jaxstro.quad.integrate") is jaxstro.quad.integrate


def test_core_dimensional_and_ownership_policies_are_specific() -> None:
    records = {item.import_path: item for item in collect_contracts().modules}
    assert "CGS" in records["jaxstro.constants"].dimensional_policy
    assert "pc" in records["jaxstro.coords"].dimensional_policy
    assert records["jaxstro.astrometry"].ownership == "Astrometric constants."
