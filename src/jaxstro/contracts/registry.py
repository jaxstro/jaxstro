"""Fail-closed collection validation for scientific contracts."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from .schema import (
    ADSemantics,
    CallableContract,
    ContractInventory,
    EvidenceKind,
    ExecutionBoundary,
    FailureMode,
    MaturityLevel,
    ModuleContract,
    SupportLevel,
)


def collect_contracts(*, source_revision: str = "unknown") -> ContractInventory:
    """Collect the explicit lightweight module manifests."""
    from jaxstro import __version__

    from ._core import CORE_CONTRACTS

    sidecars = tuple(
        _load_sidecar(name)
        for name in (
            "atmospheres",
            "numerics",
            "params",
            "quantity",
            "quad",
            "spatial",
            "spectra",
            "testing",
        )
    )

    inventory = ContractInventory(
        "1",
        __version__,
        source_revision,
        tuple(
            sorted(
                (*CORE_CONTRACTS, *sidecars),
                key=lambda item: item.id,
            )
        ),
    )
    validate_inventory(inventory)
    return inventory


def _load_sidecar(name: str) -> ModuleContract:
    """Load a manifest without importing its parent runtime package."""
    path = Path(__file__).parents[1] / name / "_contracts.py"
    spec = importlib.util.spec_from_file_location(f"_jaxstro_contract_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load contract sidecar: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = getattr(module, "MODULE_CONTRACT", None)
    if not isinstance(contract, ModuleContract):
        raise TypeError(f"invalid module contract sidecar: {name}")
    return contract


def get_module_contract(import_path: str) -> ModuleContract:
    """Return one registered module or fail closed."""
    for module in collect_contracts().modules:
        if module.import_path == import_path:
            return module
    raise KeyError(import_path)


def get_callable_contract(import_path: str) -> CallableContract:
    """Return one registered callable or fail closed."""
    for module in collect_contracts().modules:
        for contract in module.callables:
            if contract.import_path == import_path:
                return contract
    raise KeyError(import_path)


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


def validate_inventory(
    inventory: ContractInventory,
    *,
    evidence_root: Path | None = None,
    resolve_paths: bool = False,
    evidence_index: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Validate identities, public paths, evidence links, and support claims."""

    seen_contracts: set[str] = set()
    seen_evidence: set[str] = set()
    for module in inventory.modules:
        _claim_unique(module.id, seen_contracts, "contract")
        _require_text(module.ownership, f"{module.id} ownership")
        _require_text(module.non_ownership, f"{module.id} non-ownership")
        _require_enum(module.maturity, MaturityLevel, "maturity")
        _require_enum(
            module.execution_boundary, ExecutionBoundary, "execution boundary"
        )
        if resolve_paths:
            target = resolve_import_path(module.import_path)
            if not isinstance(target, ModuleType):
                raise ValueError(
                    f"module contract target is not a module: {module.import_path}"
                )
            _require_public_module(module.import_path)
        module_evidence = {item.id for item in module.evidence}
        for evidence in module.evidence:
            _require_enum(evidence.kind, EvidenceKind, "evidence kind")
            _validate_evidence(
                evidence.id,
                evidence.target,
                evidence.claim,
                evidence.artifact_id,
                evidence.evidence_class,
                evidence.artifact_comparison_ids,
                seen_evidence,
                evidence_root,
                evidence_index,
            )
        for callable_contract in module.callables:
            _claim_unique(callable_contract.id, seen_contracts, "contract")
            _require_text(callable_contract.purpose, f"{callable_contract.id} purpose")
            _require_enum(callable_contract.maturity, MaturityLevel, "maturity")
            _require_enum(callable_contract.ad_semantics, ADSemantics, "AD semantics")
            if resolve_paths:
                target = resolve_import_path(callable_contract.import_path)
                if not callable(target):
                    raise ValueError(
                        f"callable contract target is not callable: {callable_contract.import_path}"
                    )
                _require_public_callable(callable_contract.import_path)
            callable_evidence = {item.id for item in callable_contract.evidence}
            for evidence in callable_contract.evidence:
                _require_enum(evidence.kind, EvidenceKind, "evidence kind")
                _validate_evidence(
                    evidence.id,
                    evidence.target,
                    evidence.claim,
                    evidence.artifact_id,
                    evidence.evidence_class,
                    evidence.artifact_comparison_ids,
                    seen_evidence,
                    evidence_root,
                    evidence_index,
                )
            available = module_evidence | callable_evidence
            seen_transforms: set[str] = set()
            for transform in callable_contract.transforms:
                _require_enum(transform.support, SupportLevel, "transform support")
                _claim_unique(transform.transform, seen_transforms, "transform")
                if (
                    transform.support
                    in {
                        SupportLevel.SUPPORTED,
                        SupportLevel.CONDITIONAL,
                    }
                    and not transform.evidence_ids
                ):
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
                _require_enum(boundary.failure_mode, FailureMode, "failure mode")
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


def _require_enum(value: object, enum_type: type, identity: str) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"unknown {identity}: {value!r}")


def _require_public_module(path: str) -> None:
    import jaxstro

    name = path.removeprefix("jaxstro.")
    if "." in name or (name not in jaxstro.__all__ and name != "jaxconfig"):
        raise ValueError(f"module contract target is not public: {path}")


def _require_public_callable(path: str) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = resolve_import_path(parent_path)
    exports = getattr(parent, "__all__", ())
    if name.startswith("_") or (exports and name not in exports):
        raise ValueError(f"callable contract target is not public: {path}")


def _validate_evidence(
    identifier: str,
    target: str,
    claim: str,
    artifact_id: str,
    evidence_class: str,
    artifact_comparison_ids: tuple[str, ...],
    seen: set[str],
    evidence_root: Path | None,
    evidence_index: Mapping[str, Mapping[str, object]] | None,
) -> None:
    _claim_unique(identifier, seen, "evidence")
    _require_text(target, f"{identifier} target")
    _require_text(claim, f"{identifier} claim")
    if target.startswith("/"):
        raise ValueError(f"absolute evidence path is not portable: {target}")
    if evidence_root is not None and not (evidence_root / target).is_file():
        raise ValueError(f"evidence target does not exist: {target}")
    if bool(artifact_id) != bool(evidence_class):
        raise ValueError(
            f"{identifier} must declare artifact_id and evidence_class together"
        )
    if artifact_comparison_ids and not artifact_id:
        raise ValueError(f"{identifier} comparison ids require an artifact_id")
    if not artifact_id or evidence_index is None:
        return
    indexed = evidence_index.get(artifact_id)
    if indexed is None:
        raise ValueError(f"indexed evidence artifact does not exist: {artifact_id}")
    indexed_class = indexed.get("evidence_class", "")
    if indexed_class != evidence_class:
        raise ValueError(
            "indexed evidence class mismatch: "
            f"{artifact_id} declares {evidence_class}, index has {indexed_class}"
        )
    if evidence_class == "computational" and not artifact_comparison_ids:
        raise ValueError(
            f"computational evidence link has no comparison ids: {artifact_id}"
        )
    statuses = indexed.get("comparison_statuses", {})
    if not isinstance(statuses, Mapping):
        raise ValueError(f"indexed comparison statuses are invalid: {artifact_id}")
    for comparison_id in artifact_comparison_ids:
        status = statuses.get(comparison_id)
        if status != "pass":
            raise ValueError(
                "indexed evidence gate did not pass: "
                f"{artifact_id}:{comparison_id} status={status!r}"
            )


def audit_runtime_inventory(
    inventory: ContractInventory,
    *,
    repository_root: Path,
    evidence_index: Mapping[str, Mapping[str, object]] | None = None,
) -> ContractInventory:
    """Run explicit heavyweight path/evidence checks and classify public callables."""
    validate_inventory(
        inventory,
        evidence_root=repository_root,
        resolve_paths=True,
        evidence_index=evidence_index,
    )
    registered = {
        contract.import_path
        for module in inventory.modules
        for contract in module.callables
    }
    unclassified: list[str] = []
    inherited: list[str] = []
    for module_contract in inventory.modules:
        module = resolve_import_path(module_contract.import_path)
        assert isinstance(module, ModuleType)
        exports = getattr(module, "__all__", None)
        names = (
            exports
            if exports is not None
            else [
                name
                for name, value in vars(module).items()
                if not name.startswith("_")
                and getattr(value, "__module__", None) == module.__name__
            ]
        )
        for name in names:
            path = f"{module_contract.import_path}.{name}"
            value = getattr(module, name)
            if path in registered or not callable(value):
                continue
            if isinstance(value, type):
                inherited.append(path)
            else:
                unclassified.append(path)
    return replace(
        inventory,
        unclassified_callables=tuple(sorted(set(unclassified))),
        inherited_symbols=tuple(sorted(set(inherited))),
    )


def _require_evidence(references: tuple[str, ...], available: set[str]) -> None:
    missing = set(references) - available
    if missing:
        raise ValueError(f"unknown evidence ids: {sorted(missing)}")
