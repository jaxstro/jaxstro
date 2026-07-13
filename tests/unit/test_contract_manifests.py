"""Coverage and import-isolation contracts for module manifests."""

import subprocess
import sys

import jaxstro
from jaxstro.contracts import collect_contracts

PUBLIC = {
    f"jaxstro.{name}"
    for name in (
        "astrometry",
        "atmospheres",
        "constants",
        "coords",
        "geometry",
        "jaxconfig",
        "numerics",
        "params",
        "provenance",
        "quantity",
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
from jaxstro.contracts import collect_contracts
collect_contracts()
assert all(name not in sys.modules for name in ('polars', 'numpyro', 'optax'))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_contracts_is_public() -> None:
    assert jaxstro.contracts.__name__ == "jaxstro.contracts"
    assert "contracts" in jaxstro.__all__
