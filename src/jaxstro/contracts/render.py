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
    """Render a compact generated MyST reference from validated records."""

    lines = [
        "---",
        "title: Scientific contract registry",
        "---",
        "",
        "# Scientific contract registry",
        "",
        "Unverified does not mean unsupported; it means no claim is registered.",
        "",
        "## Module ownership",
        "",
        "| Module | Maturity | Ownership | Non-ownership |",
        "| --- | --- | --- | --- |",
    ]
    for module in sorted(inventory.modules, key=lambda item: item.id):
        lines.append(
            f"| `{module.import_path}` | {module.maturity.value} | "
            f"{module.ownership} | {module.non_ownership} |"
        )
    lines.extend(["", "## Transform and AD contracts", ""])
    for module in sorted(inventory.modules, key=lambda item: item.id):
        for contract in sorted(module.callables, key=lambda item: item.id):
            lines.append(
                f"- `{contract.import_path}` — `{contract.ad_semantics.value}`"
            )
    lines.extend(["", "## Unclassified callable surfaces", "", "Reported by coverage ratchets."])
    return "\n".join(lines) + "\n"


def _normalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        result = {item.name: _normalize(getattr(value, item.name)) for item in fields(value)}
        if "modules" in result:
            result["modules"] = sorted(result["modules"], key=lambda item: item["id"])
        if "callables" in result:
            result["callables"] = sorted(result["callables"], key=lambda item: item["id"])
        if "evidence" in result:
            result["evidence"] = sorted(result["evidence"], key=lambda item: item["id"])
        if "transforms" in result:
            result["transforms"] = sorted(
                result["transforms"], key=lambda item: item["transform"]
            )
        return result
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value

