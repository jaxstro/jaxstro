import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.build_sobol_directions import parse_line

from jaxstro.quad import _sobol_data

ROOT = Path(__file__).parents[3]
DATA = ROOT / "src/jaxstro/quad/data/new-joe-kuo-6.21201"
LICENSE = ROOT / "src/jaxstro/quad/data/JOE_KUO_LICENSE"
METADATA = ROOT / "src/jaxstro/quad/data/joe-kuo-metadata.json"
GENERATOR = ROOT / "scripts/build_sobol_directions.py"

SOURCE_SHA256 = "68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441"
LICENSE_SHA256 = "9d10226b50eeb34be0ab06bfa3392c7bd1f04bf602f9af4343295d1fd003d0e3"


def test_vendored_sobol_sources_have_reviewed_checksums():
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == SOURCE_SHA256
    assert hashlib.sha256(LICENSE.read_bytes()).hexdigest() == LICENSE_SHA256


def test_generated_table_covers_declared_dimension():
    assert _sobol_data.MAX_SOBOL_DIMENSION == 21201
    assert _sobol_data.SOURCE_SHA256 == SOURCE_SHA256
    assert len(_sobol_data.SOBOL_POLYNOMIALS) == 21200
    assert len(_sobol_data.SOBOL_INITIAL_DIRECTIONS) == 21200


def test_parser_requires_one_initial_value_per_declared_degree():
    assert parse_line("3 2 1 1 3") == (3, 2, 1, (1, 3))
    with pytest.raises(ValueError, match="declares degree 2 but has 1"):
        parse_line("3 2 1 1")


@pytest.mark.parametrize("dimension", (2, 10000, 21201))
def test_raw_records_map_exactly_to_generated_runtime_fields(dimension):
    raw_line = DATA.read_text(encoding="ascii").splitlines()[dimension - 1]
    parsed_dimension, degree, coefficient, initial = parse_line(raw_line)
    generated_row = dimension - 2
    assert parsed_dimension == dimension
    assert _sobol_data.SOBOL_POLYNOMIALS[generated_row] == (degree, coefficient)
    assert _sobol_data.SOBOL_INITIAL_DIRECTIONS[generated_row] == initial


def test_metadata_records_complete_source_and_license_provenance():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata == {
        "citation": {
            "authors": ["Stephen Joe", "Frances Y. Kuo"],
            "doi": "10.1137/070709359",
            "title": (
                "Constructing Sobol sequences with better two-dimensional projections"
            ),
            "year": 2008,
        },
        "criterion": 6,
        "dimension": 21201,
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


def test_generated_direction_table_is_byte_exact_fresh():
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
