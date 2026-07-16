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
    _require_exact_fields(
        payload,
        {
            "schema_version",
            "artifact_id",
            "artifact_version",
            "package_version",
            "source_revision",
            "generation_command",
            "precision",
            "deterministic_config",
            "environment",
            "metrics",
            "comparisons",
            "limitations",
            "method_payload",
        },
        "evidence artifact",
    )
    for item in payload["metrics"]:
        _require_exact_fields(
            item,
            {"identity", "symbol", "value", "units", "status", "note"},
            "metric",
        )
    for item in payload["comparisons"]:
        _require_exact_fields(
            item,
            {
                "identity",
                "metric_id",
                "relation",
                "reference",
                "units",
                "status",
                "atol",
                "rtol",
                "note",
            },
            "comparison",
        )
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
    _require_exact_fields(environment_payload, {"policy", "values"}, "environment")
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


def _require_exact_fields(
    payload: Mapping[str, Any], expected: set[str], identity: str
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{identity} fields mismatch; missing={missing}, extra={extra}"
        )


def artifact_to_markdown(artifact: EvidenceArtifact) -> str:
    """Render a human-auditable metric and comparison report."""
    validate_artifact(artifact)
    if artifact.method_payload.get("report_mode") == "progressive":
        return _progressive_artifact_to_markdown(artifact)

    lines = [
        f"# {artifact.artifact_id}",
        "",
        f"Artifact version: `{artifact.artifact_version}`",
    ]
    _append_gate_tables(lines, artifact)
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


def _progressive_artifact_to_markdown(artifact: EvidenceArtifact) -> str:
    payload = artifact.method_payload
    lines = [
        f"# {artifact.artifact_id}",
        "",
        f"Artifact version: `{artifact.artifact_version}`",
        "",
        "## What this evidence tests",
        "",
        f"This report audits **{payload.get('claim', artifact.artifact_id)}**. "
        "Each case compares the accepted replay derivative with an analytic "
        "reference and, where applicable, a finite difference of the frozen "
        "accepted formula. Adaptive-rerun finite differences remain diagnostic "
        "because the controller may choose different evidence at nearby inputs.",
        "",
        "The case map is the researcher-facing summary. Complete numerical gates "
        "and the machine-readable payload remain available below without "
        "overloading the main reading path.",
        "",
        "## Case map",
        "",
        "| Method | Case family | Variant | Status | Accepted evidence | Gates |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in payload.get("cases", ()):
        gates = case.get("gates", ())
        gate_status = (
            "pass" if all(item.get("passed", False) for item in gates) else "fail"
        )
        evidence = (
            f"regions={case.get('accepted_regions', 'n/a')}; "
            f"level={case.get('accepted_level', 'n/a')}"
        )
        lines.append(
            f"| {_table_cell(str(case.get('method', 'n/a')))} | "
            f"{_table_cell(str(case.get('family', 'n/a')))} | "
            f"{_table_cell(str(case.get('variant', 'not applicable')))} | "
            f"{_table_cell(str(case.get('status_name', 'not recorded')))} | "
            f"{_table_cell(evidence)} | {gate_status} |"
        )
    lines.extend(["", "## Main limitations", ""])
    lines.extend(f"- {item}" for item in artifact.limitations)
    if not artifact.limitations:
        lines.append("- none registered")
    lines.extend(
        [
            "",
            ":::{dropdown} Complete gate records",
            ":class: full-width",
            "",
        ]
    )
    _append_gate_tables(lines, artifact)
    lines.extend(
        [
            ":::",
            "",
            ":::{dropdown} Complete machine-readable method payload",
            "",
            "```json",
            json.dumps(_normalize(payload), indent=2, sort_keys=True),
            "```",
            ":::",
            "",
            "## Environment policy",
            "",
            artifact.environment.policy,
            "",
        ]
    )
    return "\n".join(lines)


def _append_gate_tables(lines: list[str], artifact: EvidenceArtifact) -> None:
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric identity | Symbol | Value | Units | Status |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for metric in sorted(artifact.metrics, key=lambda item: item.identity):
        lines.append(
            f"| {_table_cell(metric.identity)} | `{_table_cell(metric.symbol)}` | "
            f"{metric.value} | {_table_cell(metric.units)} | "
            f"{_table_cell(metric.status.value)} |"
        )
    lines.extend(["", "## Comparisons", ""])
    if not artifact.comparisons:
        lines.append("- none")
        return
    lines.extend(
        [
            "| Comparison | Metric | Relation | Reference | Units | Absolute tolerance | Relative tolerance | Status | Note |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for item in sorted(artifact.comparisons, key=lambda value: value.identity):
        lines.append(
            f"| {_table_cell(item.identity)} | `{_table_cell(item.metric_id)}` | "
            f"{_table_cell(item.relation.value)} | {item.reference} | "
            f"{_table_cell(item.units)} | {item.atol} | {item.rtol} | "
            f"{_table_cell(item.status.value)} | {_table_cell(item.note)} |"
        )


def _table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


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
