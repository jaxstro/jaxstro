"""Build and check jaxstro's generated provenance-card reference pages.

YAML is repository input policy, so parsing lives in this development script. The
installed :mod:`jaxstro.testing.provenance_cards` module accepts mappings and remains
serialization- and filesystem-independent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from jaxstro.testing.provenance_cards import render_registry, validate_card

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = REPO_ROOT / "docs" / "provenance" / "registry"
OUTPUT_DIR = (
    REPO_ROOT / "docs" / "50-api" / "research-infrastructure" / "source-provenance"
)

FAMILY_TITLES = {
    "atmospheres": "Atmosphere boundaries",
    "constants": "Constants and unit conventions",
    "transforms": "Coordinate and astrometric transforms",
    "lane_emden": "Lane-Emden self-gravitating spheres",
}


def load_registry() -> dict[str, list[dict[str, object]]]:
    """Load, validate, and deterministically order all registry families."""

    families: dict[str, list[dict[str, object]]] = {}
    for path in sorted(REGISTRY_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{path}: expected a top-level list of cards")
        cards: list[dict[str, object]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: card[{index}] must be a mapping")
            validate_card(item, context=f"{path}: card[{index}]")
            cards.append(item)
        families[path.stem] = cards
    return families


def render_outputs(
    families: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, str]:
    """Return every generated filename and its deterministic content."""

    outputs = render_registry(
        families if families is not None else load_registry(),
        family_titles=FAMILY_TITLES,
    )
    outputs["source-provenance.md"] = outputs.pop("index.md")
    return outputs


def emit(outputs: dict[str, str]) -> None:
    """Write generated pages to the committed documentation directory."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        path = OUTPUT_DIR / name
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)}")


def check(outputs: dict[str, str]) -> bool:
    """Return whether committed pages exactly match fresh rendering."""

    stale = [
        name
        for name, content in outputs.items()
        if not (OUTPUT_DIR / name).is_file()
        or (OUTPUT_DIR / name).read_text(encoding="utf-8") != content
    ]
    if stale:
        print(
            f"stale provenance pages (run --emit): {stale}",
            file=sys.stderr,
        )
        return False
    print("provenance pages fresh")
    return True


def main(argv: list[str] | None = None) -> int:
    """Run the emit/check command-line adapter."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="write generated pages")
    parser.add_argument(
        "--check", action="store_true", help="fail when committed pages are stale"
    )
    args = parser.parse_args(argv)
    if not (args.emit or args.check):
        parser.error("choose --emit, --check, or both")

    outputs = render_outputs()
    if args.emit:
        emit(outputs)
    if args.check and not check(outputs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
