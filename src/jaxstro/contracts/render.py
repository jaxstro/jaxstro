"""Deterministic JSON and MyST rendering for scientific contracts."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from .schema import ContractInventory


def inventory_to_dict(inventory: ContractInventory) -> dict[str, object]:
    """Return a deterministically ordered, JSON-ready inventory."""

    normalized = _normalize(inventory)
    if not isinstance(normalized, dict):
        raise TypeError("contract inventory did not normalize to a mapping")
    return normalized


def inventory_to_json(inventory: ContractInventory) -> str:
    """Render portable deterministic JSON with a terminal newline."""

    return json.dumps(inventory_to_dict(inventory), indent=2, sort_keys=True) + "\n"


def render_contract_reference(inventory: ContractInventory) -> str:
    """Render generated ownership, transform, evidence, and limitation tables."""

    lines = [
        "---",
        "title: Scientific contract registry",
        "---",
        "",
        "# Scientific contract registry",
        "",
        "Unverified does not mean unsupported; it means no claim is registered.",
        "This generated page does not infer support from importability or an unrelated passing test.",
        "",
        "## Module ownership",
        "",
        "| Module | Maturity | Boundary | Dimensional policy | Ownership | Non-ownership |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for module in sorted(inventory.modules, key=lambda item: item.id):
        lines.append(
            f"| `{module.import_path}` | {module.maturity.value} | {module.execution_boundary.value} | "
            f"{_cell(module.dimensional_policy)} | {_cell(module.ownership)} | {_cell(module.non_ownership)} |"
        )
    lines.extend(
        [
            "",
            "## Transform and AD contracts",
            "",
            "| Callable | Maturity | AD semantics | Transform claims | Boundaries | Evidence | Limitations and cost |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for module in sorted(inventory.modules, key=lambda item: item.id):
        for contract in sorted(module.callables, key=lambda item: item.id):
            transforms = (
                "; ".join(
                    f"`{item.transform}`: {item.support.value}"
                    + (f" ({item.conditions})" if item.conditions else "")
                    for item in contract.transforms
                )
                or "none claimed"
            )
            boundaries = (
                "; ".join(
                    f"{item.summary} [{item.failure_mode.value}]"
                    for item in contract.boundaries
                )
                or "none registered"
            )
            evidence = (
                "; ".join(
                    f"[`{item.id}`](https://github.com/drannarosen/jaxstro/blob/main/{item.target}) ({item.kind.value})"
                    + (
                        f" -> [`{item.artifact_id}`](../evidence-index)"
                        f" gates `{', '.join(item.artifact_comparison_ids)}`"
                        if item.artifact_id
                        else ""
                    )
                    for item in contract.evidence
                )
                or "none registered"
            )
            notes = (
                "; ".join((*contract.limitations, contract.cost_notes)).strip("; ")
                or "none registered"
            )
            lines.append(
                f"| `{contract.import_path}` | {contract.maturity.value} | {contract.ad_semantics.value} | "
                f"{_cell(transforms)} | {_cell(boundaries)} | {_cell(evidence)} | {_cell(notes)} |"
            )
    lines.extend(
        [
            "",
            "## Unclassified callable surfaces",
            "",
            f"The runtime export audit found **{len(inventory.unclassified_callables)}** public callables without callable-level records:",
            "",
            *[f"- `{path}`" for path in inventory.unclassified_callables],
            "",
            "The absence from the table is not a support or maturity claim.",
            "",
            "## Module-inherited public types",
            "",
            "These immutable record or type constructors inherit their module-level contract:",
            "",
            *[f"- `{path}`" for path in inventory.inherited_symbols],
        ]
    )
    return "\n".join(lines) + "\n"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _normalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        result = {
            item.name: _normalize(getattr(value, item.name)) for item in fields(value)
        }
        if "modules" in result:
            result["modules"] = sorted(result["modules"], key=lambda item: item["id"])
        if "callables" in result:
            result["callables"] = sorted(
                result["callables"], key=lambda item: item["id"]
            )
        if "evidence" in result:
            result["evidence"] = sorted(result["evidence"], key=lambda item: item["id"])
        if result.get("artifact_id") == "":
            result.pop("artifact_id")
            result.pop("evidence_class")
            result.pop("artifact_comparison_ids")
        if "transforms" in result:
            result["transforms"] = sorted(
                result["transforms"], key=lambda item: item["transform"]
            )
        return result
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value
