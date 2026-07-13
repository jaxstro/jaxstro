"""Emit or check Jaxstro's generated scientific-contract inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

from jaxstro.contracts import collect_contracts
from jaxstro.contracts.render import inventory_to_json, render_contract_reference

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {
    ROOT / "docs/validation/contracts.json": inventory_to_json,
    ROOT / "docs/40-api/contracts.md": render_contract_reference,
}


def render_outputs() -> dict[Path, str]:
    """Render every committed output from the same validated inventory."""
    inventory = collect_contracts(source_revision="repository-versioned")
    return {path: renderer(inventory) for path, renderer in OUTPUTS.items()}


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
