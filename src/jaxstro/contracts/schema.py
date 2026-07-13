"""Dependency-light vocabulary for public scientific contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MaturityLevel(str, Enum):
    """Evidence-calibrated maturity of a public surface."""

    RATIFIED = "ratified"
    VALIDATED = "validated"
    IMPLEMENTED = "implemented"
    EXPERIMENTAL = "experimental"
    PLANNED = "planned"


class SupportLevel(str, Enum):
    """Status of one explicitly named execution or transform claim."""

    SUPPORTED = "supported"
    CONDITIONAL = "conditional"
    UNSUPPORTED = "unsupported"
    VALIDATION_ONLY = "validation_only"
    UNVERIFIED = "unverified"


class ADSemantics(str, Enum):
    """Meaning, if any, assigned to an automatic derivative."""

    SMOOTH_PATHWISE = "smooth_pathwise"
    KNOWN_ZERO = "known_zero"
    KNOWN_BLOCKED = "known_blocked"
    SURROGATE = "surrogate"
    VALIDATION_ONLY = "validation_only"
    VALUE_FIRST = "value_first"
    CERTIFIED_IMPLICIT = "certified_implicit"
    NOT_APPLICABLE = "not_applicable"
    UNVERIFIED = "unverified"


class EvidenceKind(str, Enum):
    """Independent evidence classes that must not be conflated."""

    ANALYTIC = "analytic"
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    VALIDATION_TEST = "validation_test"
    BENCHMARK = "benchmark"
    ARTIFACT = "artifact"
    PROVENANCE = "provenance"
    DOWNSTREAM = "downstream"


class ExecutionBoundary(str, Enum):
    """Where a module's work occurs relative to transformed runtime."""

    RUNTIME = "runtime"
    PREPROCESSING = "preprocessing"
    MIXED = "mixed"
    TOOLING = "tooling"
    STATIC = "static"


class FailureMode(str, Enum):
    """Observable behavior when a contract's boundary is crossed."""

    RAISES = "raises"
    STRUCTURED_RESULT = "structured_result"
    SATURATES = "saturates"
    RETURNS_NAN = "returns_nan"
    RETURNS_SENTINEL = "returns_sentinel"
    UNDEFINED = "undefined"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class EvidenceReference:
    """Reference to evidence supporting one named claim."""

    id: str
    kind: EvidenceKind
    target: str
    claim: str
    artifact_id: str = ""
    evidence_class: str = ""


@dataclass(frozen=True)
class TransformContract:
    """Support state and evidence for one JAX execution transform."""

    transform: str
    support: SupportLevel
    conditions: str = ""
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundaryContract:
    """Named domain boundary and its observable failure behavior."""

    summary: str
    failure_mode: FailureMode
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CallableContract:
    """Scientific and execution contract for one public callable."""

    id: str
    import_path: str
    purpose: str
    domain: str
    outputs: str
    ad_semantics: ADSemantics
    precision: str
    maturity: MaturityLevel
    transforms: tuple[TransformContract, ...] = ()
    boundaries: tuple[BoundaryContract, ...] = ()
    evidence: tuple[EvidenceReference, ...] = ()
    limitations: tuple[str, ...] = ()
    cost_notes: str = ""


@dataclass(frozen=True)
class ModuleContract:
    """Ownership and evidence contract for one public module."""

    id: str
    import_path: str
    ownership: str
    non_ownership: str
    intended_uses: tuple[str, ...]
    execution_boundary: ExecutionBoundary
    dimensional_policy: str
    maturity: MaturityLevel
    callables: tuple[CallableContract, ...] = ()
    evidence: tuple[EvidenceReference, ...] = ()


@dataclass(frozen=True)
class ContractInventory:
    """Portable, versioned collection of public scientific contracts."""

    schema_version: str
    package_version: str
    source_revision: str
    modules: tuple[ModuleContract, ...]
    unclassified_callables: tuple[str, ...] = ()
    inherited_symbols: tuple[str, ...] = ()
