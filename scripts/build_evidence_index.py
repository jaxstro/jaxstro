"""Emit or check the generated cross-class scientific evidence index."""

from __future__ import annotations

import argparse
from pathlib import Path

from jaxstro.evidence.index import (
    build_evidence_index,
    evidence_index_to_json,
    evidence_index_to_markdown,
)

ROOT = Path(__file__).resolve().parents[1]


def render_outputs() -> dict[Path, str]:
    index = build_evidence_index(ROOT)
    return {
        ROOT / "docs/validation/evidence-index.json": evidence_index_to_json(index),
        ROOT / "docs/60-validation/evidence-index.md": evidence_index_to_markdown(
            index
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
        print("scientific evidence index emitted")
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
    print("scientific evidence index fresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
