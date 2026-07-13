"""Deterministic JSON and Markdown rendering for evidence artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from .schema import (
    ComparisonRecord,
    ComparisonRelation,
    EnvironmentRecord,
    EvidenceArtifact,
    EvidenceStatus,
    MetricRecord,
)
from .validation import validate_artifact


def artifact_to_dict(artifact: EvidenceArtifact) -> dict[str, object]:
    """Return a validated, deterministically ordered JSON-ready mapping."""
    validate_artifact(artifact)
    result = _normalize(artifact)
    if not isinstance(result, dict):
        raise TypeError("evidence artifact did not normalize to a mapping")
    return result


def artifact_to_json(artifact: EvidenceArtifact) -> str:
    """Render deterministic portable JSON with one terminal newline."""
    return (
        json.dumps(
            artifact_to_dict(artifact), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )


def artifact_from_dict(payload: Mapping[str, Any]) -> EvidenceArtifact:
    """Reconstruct and validate an artifact from portable JSON-ready data."""
    metrics = tuple(
        MetricRecord(
            identity=item["identity"],
            symbol=item["symbol"],
            value=item["value"],
            units=item["units"],
            status=EvidenceStatus(item["status"]),
            note=item.get("note", ""),
        )
        for item in payload.get("metrics", ())
    )
    comparisons = tuple(
        ComparisonRecord(
            identity=item["identity"],
            metric_id=item["metric_id"],
            relation=ComparisonRelation(item["relation"]),
            reference=item["reference"],
            units=item["units"],
            status=EvidenceStatus(item["status"]),
            atol=item.get("atol", 0.0),
            rtol=item.get("rtol", 0.0),
            note=item.get("note", ""),
        )
        for item in payload.get("comparisons", ())
    )
    environment_payload = payload["environment"]
    artifact = EvidenceArtifact(
        schema_version=payload["schema_version"],
        artifact_id=payload["artifact_id"],
        artifact_version=payload["artifact_version"],
        package_version=payload["package_version"],
        source_revision=payload["source_revision"],
        generation_command=payload["generation_command"],
        precision=payload["precision"],
        deterministic_config=tuple(
            tuple(item) for item in payload.get("deterministic_config", ())
        ),
        environment=EnvironmentRecord(
            environment_payload["policy"],
            tuple(tuple(item) for item in environment_payload.get("values", ())),
        ),
        metrics=metrics,
        comparisons=comparisons,
        limitations=tuple(payload.get("limitations", ())),
        method_payload=dict(payload.get("method_payload", {})),
    )
    validate_artifact(artifact)
    return artifact


def artifact_to_markdown(artifact: EvidenceArtifact) -> str:
    """Render a human-auditable metric and comparison report."""
    validate_artifact(artifact)
    lines = [
        f"# {artifact.artifact_id}",
        "",
        f"Artifact version: `{artifact.artifact_version}`",
        "",
        "## Metrics",
        "",
        "| Metric identity | Symbol | Value | Units | Status |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for metric in sorted(artifact.metrics, key=lambda item: item.identity):
        lines.append(
            f"| {metric.identity} | `{metric.symbol}` | {metric.value} | "
            f"{metric.units} | {metric.status.value} |"
        )
    lines.extend(["", "## Comparisons", ""])
    if artifact.comparisons:
        lines.extend(
            [
                "| Comparison | Metric | Relation | Reference | Units | Absolute tolerance | Relative tolerance | Status | Note |",
                "| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |",
            ]
        )
        for item in sorted(artifact.comparisons, key=lambda value: value.identity):
            lines.append(
                f"| {item.identity} | `{item.metric_id}` | {item.relation.value} | "
                f"{item.reference} | {item.units} | {item.atol} | {item.rtol} | "
                f"{item.status.value} | {item.note} |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Environment policy", "", artifact.environment.policy])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in artifact.limitations)
    if not artifact.limitations:
        lines.append("- none registered")
    lines.extend(["", "## Method payload", "", "```json"])
    lines.append(
        json.dumps(_normalize(artifact.method_payload), indent=2, sort_keys=True)
    )
    lines.extend(["```", ""])
    return "\n".join(lines)


def _normalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        result = {
            item.name: _normalize(getattr(value, item.name)) for item in fields(value)
        }
        for key in ("metrics", "comparisons"):
            if key in result:
                result[key] = sorted(result[key], key=lambda item: item["identity"])
        return result
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value
