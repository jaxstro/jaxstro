"""Tests for the local BOSZ data-conversion script."""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("polars")
pytest.importorskip("zarr")


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "convert_bosz_resampled.py"
)


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_bosz_resampled", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bosz_name(teff: int, logg: float) -> str:
    return (
        f"bosz2024_ap_t{teff}_g+{logg:.1f}_m+0.00_a+0.00_c+0.00_v2_r10000_resam.txt.gz"
    )


def _write_gzip_table(path: Path, offset: float) -> None:
    rows = [
        f"{1.0 + offset:.6e} {10.0 + offset:.6e}",
        f"{2.0 + offset:.6e} {20.0 + offset:.6e}",
        f"{3.0 + offset:.6e} {30.0 + offset:.6e}",
    ]
    with gzip.open(path, "wt", encoding="ascii") as handle:
        handle.write("\n".join(rows) + "\n")


def test_converter_writes_zarr_parquet_and_deletes_validated_raw(tmp_path):
    converter = _load_converter()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_files = []
    for offset, (teff, logg) in enumerate(
        [(12000, 3.0), (12000, 4.0), (13000, 3.0), (13000, 4.0)]
    ):
        path = raw_dir / _bosz_name(teff, logg)
        _write_gzip_table(path, float(offset))
        source_files.append(path)

    wavelength = tmp_path / "bosz2024_wave_r10000.txt"
    wavelength.write_text("5.0000000e+02\n5.0100000e+02\n5.0200000e+02\n")

    processed_dir = tmp_path / "processed"
    result = converter.convert_source_files(
        source_files,
        wavelength_path=wavelength,
        processed_dir=processed_dir,
        group_name="synthetic_bridge",
        chunk_models=2,
        delete_raw_after_validate=True,
    )

    assert result["raw_deleted"] is True
    assert all(not path.exists() for path in source_files)
    assert result["source_count"] == 4
    assert result["n_wave"] == 3
    assert result["readback_ok"] is True
    assert (processed_dir / "catalog.parquet").exists()

    import zarr

    root = zarr.open_group(processed_dir / converter.DEFAULT_BOSZ_ZARR, mode="r")
    np.testing.assert_allclose(root["wavelength"][:], [500.0, 501.0, 502.0])
    flux = root["synthetic_bridge"]["flux"]
    continuum = root["synthetic_bridge"]["continuum"]
    assert flux.dtype == np.dtype("float32")
    assert continuum.dtype == np.dtype("float32")
    assert flux.shape == (4, 3)
    np.testing.assert_allclose(flux[1, :], np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(continuum[1, :], np.array([11.0, 21.0, 31.0]))
