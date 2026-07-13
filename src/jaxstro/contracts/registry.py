"""Fail-closed collection validation for scientific contracts."""

from __future__ import annotations

import importlib
from typing import Any

from .schema import ContractInventory, SupportLevel


def resolve_import_path(path: str) -> object:
    """Resolve a dotted public path or raise a portable validation error."""

    parts = path.split(".")
    module: Any | None = None
    remainder: list[str] = []
    for stop in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:stop]))
        except ModuleNotFoundError:
            continue
        remainder = parts[stop:]
        break
    if module is None:
        raise ValueError(f"cannot resolve public import path: {path}")
    value: object = module
    try:
        for name in remainder:
            value = getattr(value, name)
    except AttributeError as exc:
        raise ValueError(f"cannot resolve public import path: {path}") from exc
    return value


def validate_inventory(inventory: ContractInventory) -> None:
    """Validate identities, public paths, evidence links, and support claims."""

    seen_contracts: set[str] = set()
    seen_evidence: set[str] = set()
    for module in inventory.modules:
        _claim_unique(module.id, seen_contracts, "contract")
        _require_text(module.ownership, f"{module.id} ownership")
        _require_text(module.non_ownership, f"{module.id} non-ownership")
        resolve_import_path(module.import_path)
        module_evidence = {item.id for item in module.evidence}
        for evidence in module.evidence:
            _validate_evidence(evidence.id, evidence.target, evidence.claim, seen_evidence)
        for callable_contract in module.callables:
            _claim_unique(callable_contract.id, seen_contracts, "contract")
            _require_text(callable_contract.purpose, f"{callable_contract.id} purpose")
            resolve_import_path(callable_contract.import_path)
            callable_evidence = {item.id for item in callable_contract.evidence}
            for evidence in callable_contract.evidence:
                _validate_evidence(
                    evidence.id, evidence.target, evidence.claim, seen_evidence
                )
            available = module_evidence | callable_evidence
            seen_transforms: set[str] = set()
            for transform in callable_contract.transforms:
                _claim_unique(transform.transform, seen_transforms, "transform")
                if transform.support in {
                    SupportLevel.SUPPORTED,
                    SupportLevel.CONDITIONAL,
                } and not transform.evidence_ids:
                    raise ValueError(
                        f"{callable_contract.id} {transform.transform} support has no evidence"
                    )
                if (
                    transform.support is SupportLevel.CONDITIONAL
                    and not transform.conditions.strip()
                ):
                    raise ValueError(
                        f"{callable_contract.id} conditional support has no conditions"
                    )
                _require_evidence(transform.evidence_ids, available)
            for boundary in callable_contract.boundaries:
                _require_text(boundary.summary, f"{callable_contract.id} boundary")
                _require_evidence(boundary.evidence_ids, available)


def _claim_unique(value: str, seen: set[str], kind: str) -> None:
    if value in seen:
        raise ValueError(f"duplicate {kind} id: {value}")
    _require_text(value, f"{kind} id")
    seen.add(value)


def _require_text(value: str, identity: str) -> None:
    if not value.strip():
        raise ValueError(f"empty {identity}")


def _validate_evidence(
    identifier: str, target: str, claim: str, seen: set[str]
) -> None:
    _claim_unique(identifier, seen, "evidence")
    _require_text(target, f"{identifier} target")
    _require_text(claim, f"{identifier} claim")
    if target.startswith("/"):
        raise ValueError(f"absolute evidence path is not portable: {target}")


def _require_evidence(references: tuple[str, ...], available: set[str]) -> None:
    missing = set(references) - available
    if missing:
        raise ValueError(f"unknown evidence ids: {sorted(missing)}")

