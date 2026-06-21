"""Tests for the host-side processed NewEra backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxstro.atmospheres import AtmosphereParams, NewEraBackend


def _write_processed_artifact(processed_dir: Path) -> None:
    pl = pytest.importorskip("polars")
    zarr = pytest.importorskip("zarr")

    zarr_path = processed_dir / "newera_lowres_v3.zarr"
    root = zarr.open_group(zarr_path, mode="w", zarr_format=2)
    files = root.require_group("files")
    group = files.create_group("synthetic")
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

    rows = []
    for row, (teff, logg) in enumerate(
        [(5000.0, 4.0), (5000.0, 5.0), (6000.0, 4.0), (6000.0, 5.0)]
    ):
        rows.append(
            {
                "source_file": "synthetic.txt",
                "source_row": row,
                "version": "V3",
                "product": "LowRes-SPECTRA",
                "m_h": 0.0,
                "alpha_m": 0.0,
                "n_wave": 3,
                "lambda_min": 100.0,
                "lambda_max": 102.0,
                "lambda_step": 1.0,
                "teff": teff,
                "logg": logg,
                "mass": 1.0,
                "row_abundance_anchor": 7.5,
                "row_alpha_m": 0.0,
                "raw_metadata": "synthetic",
                "zarr_group": "files/synthetic",
                "zarr_row": row,
            }
        )
    pl.DataFrame(rows).write_parquet(processed_dir / "catalog.parquet")


def test_newera_backend_opens_processed_artifact_and_interpolates(tmp_path):
    _write_processed_artifact(tmp_path)
    backend = NewEraBackend.open(tmp_path)

    result = backend.spectrum(AtmosphereParams(teff=5500.0, logg=4.5))

    np.testing.assert_allclose(result.spectrum.wavelength, [100.0, 101.0, 102.0])
    np.testing.assert_allclose(result.spectrum.flux_lambda, [2.5, 3.5, 4.5])
    assert bool(result.status.in_grid)


def test_newera_backend_prepare_rejects_missing_abundance_plane(tmp_path):
    _write_processed_artifact(tmp_path)
    backend = NewEraBackend.open(tmp_path)

    with pytest.raises(ValueError, match="No NewEra abundance plane"):
        backend.prepare(AtmosphereParams(teff=5500.0, logg=4.5, m_h=0.5))
