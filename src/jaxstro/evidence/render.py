"""Deterministic JSON and Markdown rendering for evidence artifacts."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from .schema import EvidenceArtifact
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
    return json.dumps(artifact_to_dict(artifact), indent=2, sort_keys=True) + "\n"


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
                "| Comparison | Metric | Relation | Reference | Status |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for item in sorted(artifact.comparisons, key=lambda value: value.identity):
            lines.append(
                f"| {item.identity} | `{item.metric_id}` | {item.relation.value} | "
                f"{item.reference} | {item.status.value} |"
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
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value
