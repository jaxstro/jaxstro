"""Tests for the host-side BOSZ index and processed backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxstro.atmospheres import (
    AtmosphereParams,
    BoszBackend,
    build_bosz_index,
    discover_bosz_files,
    parse_bosz_filename,
)


def _bosz_name(teff: int, logg: float) -> str:
    return (
        f"bosz2024_ap_t{teff}_g+{logg:.1f}_m+0.00_a+0.00_c+0.00_v2_r10000_resam.txt.gz"
    )


def _write_processed_artifact(processed_dir: Path) -> None:
    pl = pytest.importorskip("polars")
    zarr = pytest.importorskip("zarr")

    zarr_path = processed_dir / "bosz_2025_recomputed.zarr"
    root = zarr.open_group(zarr_path, mode="w", zarr_format=2)
    root.create_array(
        "wavelength",
        data=np.array([500.0, 501.0, 502.0], dtype=np.float64),
        chunks=(3,),
        overwrite=True,
    )
    group = root.create_group("synthetic")
    group.create_array(
        "flux",
        data=np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        chunks=(1, 3),
        overwrite=True,
    )
    group.create_array(
        "continuum",
        data=np.ones((4, 3), dtype=np.float32),
        chunks=(1, 3),
        overwrite=True,
    )

    rows = []
    for row, (teff, logg) in enumerate(
        [(12000.0, 3.0), (12000.0, 4.0), (13000.0, 3.0), (13000.0, 4.0)]
    ):
        rows.append(
            {
                "filename": _bosz_name(int(teff), logg),
                "source_path": _bosz_name(int(teff), logg),
                "source_size_bytes": 1,
                "source_sha256": "synthetic",
                "atmosphere": "ap",
                "teff": teff,
                "logg": logg,
                "m_h": 0.0,
                "alpha_m": 0.0,
                "c_m": 0.0,
                "vturb_km_s": 2.0,
                "resolution": "r10000",
                "product": "resam",
                "n_wave": 3,
                "wavelength_min": 500.0,
                "wavelength_max": 502.0,
                "wavelength_unit": "angstrom",
                "flux_unit": "bosz_resampled_column0",
                "continuum_unit": "bosz_resampled_column1",
                "zarr_group": "synthetic",
                "zarr_row": row,
            }
        )
    pl.DataFrame(rows).write_parquet(processed_dir / "catalog.parquet")


def test_parse_bosz_filename_extracts_recomputed_grid_metadata():
    meta = parse_bosz_filename(_bosz_name(12500, 4.5))

    assert meta.atmosphere == "ap"
    assert meta.teff == 12500.0
    assert meta.logg == 4.5
    assert meta.m_h == 0.0
    assert meta.alpha_m == 0.0
    assert meta.c_m == 0.0
    assert meta.vturb_km_s == 2.0
    assert meta.resolution == "r10000"
    assert meta.product == "resam"


def test_parse_bosz_filename_rejects_non_bosz_name():
    with pytest.raises(ValueError, match="BOSZ 2024"):
        parse_bosz_filename("PHOENIX-NewEraV3-LowRes-SPECTRA.Z+0.5.txt")


def test_discover_and_build_bosz_index(tmp_path):
    raw = tmp_path / "raw" / "r10000" / "m+0.00"
    raw.mkdir(parents=True)
    for name in [_bosz_name(12000, 3.0), _bosz_name(12000, 4.0)]:
        (raw / name).write_bytes(b"")
    (raw / "not-a-bosz.txt.gz").write_bytes(b"")

    files = discover_bosz_files(tmp_path)
    index = build_bosz_index(tmp_path)

    assert [path.name for path in files] == [
        _bosz_name(12000, 3.0),
        _bosz_name(12000, 4.0),
    ]
    assert index.resolutions == ("r10000",)
    assert index.m_h_values == (0.0,)
    assert index.c_m_values == (0.0,)
    assert index.vturb_values_km_s == (2.0,)
    assert len(index.files) == 2


def test_bosz_backend_opens_processed_artifact_and_interpolates(tmp_path):
    _write_processed_artifact(tmp_path)
    backend = BoszBackend.open(tmp_path)

    result = backend.spectrum(AtmosphereParams(teff=12500.0, logg=3.5))

    np.testing.assert_allclose(result.spectrum.wavelength, [500.0, 501.0, 502.0])
    np.testing.assert_allclose(result.spectrum.flux_lambda, [2.5, 3.5, 4.5])
    assert result.spectrum.wavelength_unit == "angstrom"
    assert result.spectrum.flux_unit == "bosz_resampled_column0"
    assert bool(result.status.in_grid)


def test_bosz_backend_prepare_rejects_missing_carbon_plane(tmp_path):
    _write_processed_artifact(tmp_path)
    backend = BoszBackend.open(tmp_path)

    with pytest.raises(ValueError, match="No BOSZ plane"):
        backend.prepare(AtmosphereParams(teff=12500.0, logg=3.5, c_m=0.25))
