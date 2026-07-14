"""Emit or check Jaxstro's generated scientific-contract inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from jaxstro.contracts import audit_runtime_inventory, collect_contracts
from jaxstro.contracts.render import inventory_to_json, render_contract_reference

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {
    ROOT / "docs/validation/contracts.json": inventory_to_json,
    ROOT
    / "docs/50-api/research-infrastructure/contracts.md": render_contract_reference,
}


def render_outputs() -> dict[Path, str]:
    """Render every committed output from the same validated inventory."""
    inventory = audit_runtime_inventory(
        collect_contracts(source_revision="pending-content-digest"),
        repository_root=ROOT,
        evidence_index=_load_evidence_index(),
    )
    digest_payload = inventory_to_json(replace(inventory, source_revision=""))
    digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    inventory = replace(inventory, source_revision=f"sha256:{digest}")
    return {path: renderer(inventory) for path, renderer in OUTPUTS.items()}


def _load_evidence_index() -> dict[str, dict[str, object]]:
    path = ROOT / "docs/validation/evidence-index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {entry["id"]: entry for entry in payload["entries"]}
    for entry in result.values():
        if entry["evidence_class"] != "computational":
            continue
        artifact = json.loads((ROOT / str(entry["target"])).read_text(encoding="utf-8"))
        entry["comparison_statuses"] = {
            item["identity"]: item["status"] for item in artifact["comparisons"]
        }
    return result


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
        print("scientific contract artifacts emitted")
        return 0
    stale = [
        path
        for path, content in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]
    if stale:
        for path in stale:
            print(f"stale: {path.relative_to(ROOT)}")
        return 1
    print("scientific contract artifacts fresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
