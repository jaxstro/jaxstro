"""Portable schemas and tools for scientific evidence artifacts."""

from .files import EvidenceFreshnessError, check_artifact, emit_artifact
from .render import artifact_to_dict, artifact_to_json, artifact_to_markdown
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
    "EvidenceStatus",
    "MetricRecord",
    "artifact_to_dict",
    "artifact_to_json",
    "artifact_to_markdown",
    "check_artifact",
    "emit_artifact",
    "validate_artifact",
]
