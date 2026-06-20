"""Tests for the host-side atmosphere grid index helpers."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from jaxstro.atmospheres import (
    build_newera_lowres_index,
    discover_newera_lowres_files,
    parse_newera_lowres_filename,
    read_newera_lowres_header,
    resolve_data_dir,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "newera_lowres"


def test_parse_newera_filename_with_alpha():
    meta = parse_newera_lowres_filename(
        "PHOENIX-NewEraV3-LowRes-SPECTRA.Z-0.5.alpha=0.4.txt"
    )

    assert meta.version == "V3"
    assert meta.m_h == -0.5
    assert meta.alpha_m == 0.4


def test_parse_newera_filename_without_alpha_defaults_to_zero():
    meta = parse_newera_lowres_filename("PHOENIX-NewEraV3-LowRes-SPECTRA.Z+0.5.txt")

    assert meta.version == "V3"
    assert meta.m_h == 0.5
    assert meta.alpha_m == 0.0


def test_parse_newera_filename_rejects_non_lowres_name():
    with pytest.raises(ValueError, match="NewEra low-resolution"):
        parse_newera_lowres_filename("PHOENIX-NewEraV3-GAIA-DR4-SPECTRA.Z-0.5.txt")


def test_discover_newera_lowres_files_ignores_tarballs_and_other_products():
    files = discover_newera_lowres_files(FIXTURE_DIR)

    assert [path.name for path in files] == [
        "PHOENIX-NewEraV3-LowRes-SPECTRA.Z-0.5.alpha=0.4.txt",
        "PHOENIX-NewEraV3-LowRes-SPECTRA.Z+0.5.txt",
    ]


def test_read_newera_header_reads_only_first_line_metadata():
    header = read_newera_lowres_header(
        FIXTURE_DIR / "PHOENIX-NewEraV3-LowRes-SPECTRA.Z-0.5.alpha=0.4.txt"
    )

    assert header.column_names == ("teff", "logg", "lambda_angstrom", "flux_cgs")
    assert header.raw.startswith("# teff logg")


def test_build_newera_lowres_index_from_synthetic_fixture():
    index = build_newera_lowres_index(FIXTURE_DIR)

    assert index.root == FIXTURE_DIR
    assert index.product == "LowRes-SPECTRA"
    assert index.versions == ("V3",)
    assert index.m_h_values == (-0.5, 0.5)
    assert index.alpha_m_values == (0.0, 0.4)
    assert len(index.files) == 2
    assert all(point.header is not None for point in index.files)


def test_resolve_data_dir_uses_explicit_path_before_env(tmp_path, monkeypatch):
    env_path = tmp_path / "env-cache"
    explicit_path = tmp_path / "explicit-cache"
    monkeypatch.setenv("JAXSTRO_DATA_DIR", str(env_path))

    assert resolve_data_dir(explicit_path) == explicit_path.expanduser()


def test_resolve_data_dir_uses_env(monkeypatch, tmp_path):
    env_path = tmp_path / "env-cache"
    monkeypatch.setenv("JAXSTRO_DATA_DIR", str(env_path))

    assert resolve_data_dir() == env_path


def test_atmospheres_import_has_no_pytest_dependency():
    module = importlib.import_module("jaxstro.atmospheres")

    assert "pytest" not in module.__dict__


def test_fixture_files_stay_tiny():
    fixture_files = list(FIXTURE_DIR.glob("*.txt"))

    assert fixture_files
    assert sum(path.stat().st_size for path in fixture_files) < 10_000
