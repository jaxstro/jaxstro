"""Fail-closed registry and deterministic rendering contracts."""

import pytest

from jaxstro.contracts import (
    ADSemantics,
    CallableContract,
    ContractInventory,
    ExecutionBoundary,
    MaturityLevel,
    ModuleContract,
)
from jaxstro.contracts.registry import resolve_import_path, validate_inventory
from jaxstro.contracts.render import inventory_to_json


def _callable(identifier: str, path: str) -> CallableContract:
    return CallableContract(
        id=identifier,
        import_path=path,
        purpose="Test fixture.",
        domain="Finite scalar inputs.",
        outputs="Scalar output.",
        ad_semantics=ADSemantics.UNVERIFIED,
        precision="Unverified.",
        maturity=MaturityLevel.IMPLEMENTED,
    )


def _inventory(*callables: CallableContract) -> ContractInventory:
    module = ModuleContract(
        id="numerics",
        import_path="jaxstro.numerics",
        ownership="Generic numerical primitives.",
        non_ownership="Domain acceptance and physical state.",
        intended_uses=("Differentiable scientific computing",),
        execution_boundary=ExecutionBoundary.RUNTIME,
        dimensional_policy="Caller-owned units.",
        maturity=MaturityLevel.VALIDATED,
        callables=callables,
    )
    return ContractInventory("1", "0.1.0", "test", (module,))


def test_import_resolution_fails_closed() -> None:
    assert callable(resolve_import_path("jaxstro.numerics.safeguarded_bracketed_root"))
    with pytest.raises(ValueError, match="cannot resolve"):
        resolve_import_path("jaxstro.numerics.not_a_symbol")


def test_registry_rejects_duplicate_callable_ids() -> None:
    inventory = _inventory(
        _callable("duplicate", "jaxstro.numerics.safeguarded_bracketed_root"),
        _callable("duplicate", "jaxstro.numerics.newton_ppf"),
    )
    with pytest.raises(ValueError, match="duplicate contract id"):
        validate_inventory(inventory)


def test_json_is_deterministic_and_portable() -> None:
    inventory = _inventory(
        _callable("root", "jaxstro.numerics.safeguarded_bracketed_root")
    )
    validate_inventory(inventory)
    first = inventory_to_json(inventory)
    assert first == inventory_to_json(inventory)
    assert "/Users/" not in first
