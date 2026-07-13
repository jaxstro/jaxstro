"""Frozen records for portable scientific evidence artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
    method_payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fixture(
        cls, artifact_id: str, *, metrics: tuple[MetricRecord, ...] = ()
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
        )
