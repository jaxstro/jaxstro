"""Portable schemas and tools for scientific evidence artifacts."""

from .files import EvidenceFreshnessError, check_artifact, emit_artifact
from .index import (
    EvidenceClass,
    EvidenceIndex,
    EvidenceIndexEntry,
    build_evidence_index,
)
from .render import (
    artifact_from_dict,
    artifact_to_dict,
    artifact_to_json,
    artifact_to_markdown,
)
from .schema import (
    ComparisonRecord,
    ComparisonRelation,
    EnvironmentRecord,
    EvidenceArtifact,
    EvidenceStatus,
    MetricRecord,
)
from .validation import validate_artifact

__all__ = [
    "ComparisonRecord",
    "ComparisonRelation",
    "EnvironmentRecord",
    "EvidenceArtifact",
    "EvidenceFreshnessError",
    "EvidenceClass",
    "EvidenceIndex",
    "EvidenceIndexEntry",
    "EvidenceStatus",
    "MetricRecord",
    "artifact_to_dict",
    "artifact_from_dict",
    "artifact_to_json",
    "artifact_to_markdown",
    "check_artifact",
    "build_evidence_index",
    "emit_artifact",
    "validate_artifact",
]
