"""Fail-closed registry and deterministic rendering contracts."""

from pathlib import Path

import pytest

from jaxstro.contracts import (
    ADSemantics,
    CallableContract,
    ContractInventory,
    ExecutionBoundary,
    MaturityLevel,
    ModuleContract,
    SupportLevel,
    TransformContract,
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


def test_registry_rejects_unknown_runtime_vocabulary() -> None:
    inventory = _inventory(
        _callable("root", "jaxstro.numerics.safeguarded_bracketed_root")
    )
    object.__setattr__(inventory.modules[0], "maturity", "invented")
    with pytest.raises(ValueError, match="maturity"):
        validate_inventory(inventory)


def test_registry_rejects_noncallable_callable_target() -> None:
    inventory = _inventory(_callable("constant", "jaxstro.constants.G_CGS"))
    with pytest.raises(ValueError, match="not callable"):
        validate_inventory(inventory, resolve_paths=True)


def test_registry_rejects_private_callable_target() -> None:
    inventory = _inventory(
        _callable("private", "jaxstro.numerics.rootfinding._normalize_bracket_state")
    )
    with pytest.raises(ValueError, match="not public"):
        validate_inventory(inventory, resolve_paths=True)


def test_supported_transform_requires_evidence() -> None:
    record = _callable("root", "jaxstro.numerics.safeguarded_bracketed_root")
    object.__setattr__(
        record, "transforms", (TransformContract("jit", SupportLevel.SUPPORTED),)
    )
    with pytest.raises(ValueError, match="no evidence"):
        validate_inventory(_inventory(record))


def test_repository_audit_rejects_missing_evidence_target(tmp_path: Path) -> None:
    record = _callable("root", "jaxstro.numerics.safeguarded_bracketed_root")
    from jaxstro.contracts import EvidenceKind, EvidenceReference

    object.__setattr__(
        record,
        "evidence",
        (
            EvidenceReference(
                "missing",
                EvidenceKind.UNIT_TEST,
                "does/not/exist.py",
                "missing fixture",
            ),
        ),
    )
    with pytest.raises(ValueError, match="evidence target does not exist"):
        validate_inventory(_inventory(record), evidence_root=tmp_path)
