"""Validate and emit the executable research-workflow registry."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "docs/40-workflows/investigations/registry.json"
WORKFLOW_FIELDS = {
    "id",
    "title",
    "page",
    "example",
    "prerequisites",
    "public_apis",
    "contract_ids",
    "evidence_ids",
    "validation_targets",
    "limitations",
}


def validate_unique_references(
    identity: str,
    workflow: dict[str, Any],
    fields: Sequence[str],
) -> None:
    """Reject duplicate references before computing workflow coverage counts."""
    for field in fields:
        values = workflow[field]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate research workflow {field}: {identity}")


def load_and_validate_workflows(root: Path) -> list[dict[str, Any]]:
    """Load research workflows and fail closed on references or schema drift."""
    payload = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if (
        set(payload) != {"schema_version", "workflows"}
        or payload["schema_version"] != "2"
    ):
        raise ValueError("unsupported research workflow manifest schema")
    if not isinstance(payload["workflows"], list):
        raise ValueError("research workflows must be a list")

    contracts = json.loads(
        (root / "docs/validation/contracts.json").read_text(encoding="utf-8")
    )
    contract_records = {
        item["id"]: item
        for module in contracts["modules"]
        for item in module["callables"]
    }
    evidence = json.loads(
        (root / "docs/validation/evidence-index.json").read_text(encoding="utf-8")
    )
    evidence_ids = {item["id"] for item in evidence["entries"]}

    seen: set[str] = set()
    workflows = sorted(payload["workflows"], key=lambda item: item["id"])
    for workflow in workflows:
        if not isinstance(workflow, dict) or set(workflow) != WORKFLOW_FIELDS:
            identity = workflow.get("id", "?") if isinstance(workflow, dict) else "?"
            raise ValueError(f"research workflow fields mismatch: {identity}")
        identity = workflow["id"]
        if identity in seen:
            raise ValueError(f"duplicate research workflow id: {identity}")
        seen.add(identity)

        for field in ("id", "title", "page", "example"):
            if not isinstance(workflow[field], str) or not workflow[field].strip():
                raise ValueError(
                    f"research workflow {field} must be nonempty: {identity}"
                )
        for field in (
            "prerequisites",
            "public_apis",
            "contract_ids",
            "evidence_ids",
            "validation_targets",
            "limitations",
        ):
            if not isinstance(workflow[field], list) or not all(
                isinstance(value, str) and value.strip() for value in workflow[field]
            ):
                raise ValueError(
                    f"research workflow {field} must be a text list: {identity}"
                )

        validate_unique_references(
            identity,
            workflow,
            (
                "contract_ids",
                "public_apis",
                "evidence_ids",
                "validation_targets",
                "prerequisites",
            ),
        )
        if not workflow["contract_ids"] or not workflow["validation_targets"]:
            raise ValueError(
                f"research workflow lacks contract or validation: {identity}"
            )

        missing_contracts = set(workflow["contract_ids"]) - set(contract_records)
        if missing_contracts:
            raise ValueError(
                f"unknown research workflow contract ids: {sorted(missing_contracts)}"
            )
        contract_paths = {
            contract_records[contract_id]["import_path"]
            for contract_id in workflow["contract_ids"]
        }
        if set(workflow["public_apis"]) != contract_paths:
            raise ValueError(
                f"research workflow API and contract paths disagree: {identity}"
            )

        missing_evidence = set(workflow["evidence_ids"]) - evidence_ids
        if missing_evidence:
            raise ValueError(
                f"unknown research workflow evidence ids: {sorted(missing_evidence)}"
            )
        for field in ("page", "example"):
            if not (root / workflow[field]).is_file():
                raise ValueError(
                    f"research workflow {field} does not exist: {workflow[field]}"
                )
        for field in ("prerequisites", "validation_targets"):
            for target in workflow[field]:
                if not (root / target).is_file():
                    raise ValueError(
                        f"research workflow target does not exist: {target}"
                    )
    return workflows


def _render_investigations(workflows: list[dict[str, Any]]) -> str:
    lines = [
        "---",
        "title: Executable research investigations",
        "description: Predict, compute, audit, and state the warranted claim using public Jaxstro APIs.",
        "---",
        "",
        "# Executable research investigations",
        "",
        "Each investigation imports public APIs from one repository-owned Python module. The page explains the reasoning; the example is the executable source of truth.",
        "",
        "| Investigation | Public contracts | Indexed evidence | Known limitations |",
        "| --- | --- | --- | --- |",
    ]
    for workflow in workflows:
        page = Path(workflow["page"]).name
        contracts = ", ".join(f"`{item}`" for item in workflow["contract_ids"])
        evidence = (
            ", ".join(f"`{item}`" for item in workflow["evidence_ids"])
            or "No standalone indexed artifact; callable validation only"
        )
        limitations = "; ".join(workflow["limitations"])
        lines.append(
            f"| [{workflow['title']}](./{page}) | {contracts} | {evidence} | {limitations} |"
        )
    lines.extend(
        [
            "",
            "Every command prints a complete prediction, metric, audit, and claim report. Measured results use the required form:",
            "",
            "| Metric identity | Symbol | Value | Units |",
            "| --- | --- | ---: | --- |",
            "| example metric | `m` | measured | explicit units |",
            "",
            "The research workflow registry is checked against the scientific contract registry and evidence index. Missing indexed evidence remains visible rather than being inferred from a passing example.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_coverage(workflows: list[dict[str, Any]]) -> str:
    payload = {
        "schema_version": "2",
        "workflow_count": len(workflows),
        "workflows": [
            {
                "contract_count": len(workflow["contract_ids"]),
                "evidence_count": len(workflow["evidence_ids"]),
                "id": workflow["id"],
                "limitation_count": len(workflow["limitations"]),
                "validation_target_count": len(workflow["validation_targets"]),
            }
            for workflow in workflows
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_outputs(root: Path = ROOT) -> dict[Path, str]:
    """Render every generated research-workflow product from one manifest."""
    workflows = load_and_validate_workflows(root)
    return {
        root / "docs/40-workflows/investigations/investigations.md": (
            _render_investigations(workflows)
        ),
        root / "docs/validation/research-workflow-coverage.json": (
            _render_coverage(workflows)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = render_outputs()
    if args.emit:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        print("research workflow registry emitted")
        return 0
    stale = [
        path
        for path, content in outputs.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    if stale:
        for path in stale:
            print(f"stale: {path.relative_to(ROOT)}")
        return 1
    print("research workflow registry fresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
