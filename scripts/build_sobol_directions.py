#!/usr/bin/env python3
"""Build the checked compact Joe-Kuo Sobol direction-number table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from pprint import pformat

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src/jaxstro/quad/data"
SOURCE = DATA_DIR / "new-joe-kuo-6.21201"
LICENSE = DATA_DIR / "JOE_KUO_LICENSE"
METADATA = DATA_DIR / "joe-kuo-metadata.json"
OUTPUT = ROOT / "src/jaxstro/quad/_sobol_data.py"

SOURCE_SHA256 = "68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441"
LICENSE_SHA256 = "9d10226b50eeb34be0ab06bfa3392c7bd1f04bf602f9af4343295d1fd003d0e3"
MAX_SOBOL_DIMENSION = 21201


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one immutable source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_line(line: str) -> tuple[int, int, int, tuple[int, ...]]:
    """Parse one Joe-Kuo direction-number record."""
    fields = [int(field) for field in line.split()]
    dimension, degree, coefficient, *initial = fields
    if len(initial) != degree:
        raise ValueError(
            f"dimension {dimension} declares degree {degree} "
            f"but has {len(initial)} initial values"
        )
    return dimension, degree, coefficient, tuple(initial)


def load_records() -> tuple[tuple[int, int, int, tuple[int, ...]], ...]:
    """Load and structurally validate all dimensions after the first."""
    if file_sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("Joe-Kuo source checksum does not match reviewed provenance")
    if file_sha256(LICENSE) != LICENSE_SHA256:
        raise ValueError("Joe-Kuo license checksum does not match reviewed provenance")
    lines = SOURCE.read_text(encoding="ascii").splitlines()
    records = tuple(parse_line(line) for line in lines[1:] if line.strip())
    dimensions = tuple(record[0] for record in records)
    expected = tuple(range(2, MAX_SOBOL_DIMENSION + 1))
    if dimensions != expected:
        raise ValueError("Joe-Kuo dimensions must be consecutive from 2 through 21201")
    return records


def render_table() -> str:
    """Render the deterministic importable runtime table."""
    records = load_records()
    polynomials = tuple((degree, coefficient) for _, degree, coefficient, _ in records)
    initial = tuple(values for _, _, _, values in records)
    return (
        '"""Generated Joe-Kuo Sobol polynomial and initial-direction data.\n\n'
        "Do not edit by hand. Regenerate with scripts/build_sobol_directions.py.\n"
        '"""\n\n'
        "# fmt: off\n"
        f'SOURCE_SHA256 = "{SOURCE_SHA256}"\n'
        f"MAX_SOBOL_DIMENSION = {MAX_SOBOL_DIMENSION}\n\n"
        f"SOBOL_POLYNOMIALS = {pformat(polynomials, width=88)}\n\n"
        f"SOBOL_INITIAL_DIRECTIONS = {pformat(initial, width=88)}\n\n"
        "__all__ = [\n"
        '    "MAX_SOBOL_DIMENSION",\n'
        '    "SOBOL_INITIAL_DIRECTIONS",\n'
        '    "SOBOL_POLYNOMIALS",\n'
        '    "SOURCE_SHA256",\n'
        "]\n"
        "# fmt: on\n"
    )


def metadata_payload() -> dict:
    """Return the reviewed machine-readable source provenance."""
    return {
        "citation": {
            "authors": ["Stephen Joe", "Frances Y. Kuo"],
            "doi": "10.1137/070709359",
            "title": (
                "Constructing Sobol sequences with better two-dimensional projections"
            ),
            "year": 2008,
        },
        "criterion": 6,
        "dimension": MAX_SOBOL_DIMENSION,
        "license": {
            "sha256": LICENSE_SHA256,
            "url": "https://web.maths.unsw.edu.au/~fkuo/sobol/licence",
        },
        "source": {
            "sha256": SOURCE_SHA256,
            "updated": "2010-09-16",
            "url": ("https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201"),
        },
    }


def render_metadata() -> str:
    """Render deterministic JSON provenance."""
    return json.dumps(metadata_payload(), indent=2, sort_keys=True) + "\n"


def check_or_emit(*, emit: bool) -> None:
    """Write generated outputs or fail if tracked bytes are stale."""
    expected = {
        OUTPUT: render_table(),
        METADATA: render_metadata(),
    }
    stale: list[Path] = []
    for path, content in expected.items():
        if emit:
            path.write_text(content, encoding="utf-8")
        elif not path.is_file() or path.read_text(encoding="utf-8") != content:
            stale.append(path)
    if stale:
        paths = ", ".join(str(path.relative_to(ROOT)) for path in stale)
        raise SystemExit(f"stale Sobol generated artifacts: {paths}")
    action = "emitted" if emit else "fresh"
    print(f"Sobol direction artifacts {action}")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    check_or_emit(emit=arguments.emit)


if __name__ == "__main__":
    main()
