"""Frozen records for portable scientific evidence artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class EvidenceStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INFO = "info"


class ComparisonRelation(str, Enum):
    LESS_EQUAL = "less_equal"
    GREATER_EQUAL = "greater_equal"
    CLOSE = "close"
    EQUAL = "equal"
    INFORMATIONAL = "informational"


@dataclass(frozen=True)
class MetricRecord:
    identity: str
    symbol: str
    value: int | float
    units: str
    status: EvidenceStatus = EvidenceStatus.INFO
    note: str = ""


@dataclass(frozen=True)
class ComparisonRecord:
    identity: str
    metric_id: str
    relation: ComparisonRelation
    reference: int | float
    units: str
    status: EvidenceStatus
    atol: float = 0.0
    rtol: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class EnvironmentRecord:
    policy: str
    values: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EvidenceArtifact:
    schema_version: str
    artifact_id: str
    artifact_version: str
    package_version: str
    source_revision: str
    generation_command: str
    precision: str
    deterministic_config: tuple[tuple[str, Any], ...]
    environment: EnvironmentRecord
    metrics: tuple[MetricRecord, ...] = ()
    comparisons: tuple[ComparisonRecord, ...] = ()
    limitations: tuple[str, ...] = ()
    method_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen_config = tuple(
            (str(key), _freeze_portable(value))
            for key, value in self.deterministic_config
        )
        frozen_payload = _freeze_portable(dict(self.method_payload))
        object.__setattr__(self, "deterministic_config", frozen_config)
        object.__setattr__(self, "method_payload", frozen_payload)

    @classmethod
    def fixture(
        cls,
        artifact_id: str,
        *,
        metrics: tuple[MetricRecord, ...] = (),
        method_payload: Mapping[str, Any] | None = None,
    ) -> EvidenceArtifact:
        """Construct a complete minimal artifact for tests and examples."""
        return cls(
            schema_version="1",
            artifact_id=artifact_id,
            artifact_version="1",
            package_version="0.1.0",
            source_revision="test",
            generation_command="test fixture",
            precision="float64",
            deterministic_config=(),
            environment=EnvironmentRecord("excluded-test-fixture"),
            metrics=metrics,
            method_payload={} if method_payload is None else method_payload,
        )


def _freeze_portable(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("portable evidence values must be finite")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_portable(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_portable(item) for item in value)
    raise TypeError(f"unsupported portable evidence value: {type(value).__name__}")
