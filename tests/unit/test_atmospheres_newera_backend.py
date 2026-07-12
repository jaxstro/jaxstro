"""Tests for the host-side processed NewEra backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxstro.atmospheres import AtmosphereParams, AtmosphereQuery, NewEraBackend
from jaxstro.spectra import (
    SpectralAxis,
    SpectralCoordinate,
    SpectralPlan,
    SpectrumStatusCode,
)


def _write_processed_artifact(
    processed_dir: Path,
    *,
    points: tuple[tuple[float, float], ...] = (
        (5000.0, 4.0),
        (5000.0, 5.0),
        (6000.0, 4.0),
        (6000.0, 5.0),
    ),
) -> None:
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
    for row, (teff, logg) in enumerate(points):
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


def _query(teff: float = 5500.0, logg: float = 4.5) -> AtmosphereQuery:
    return AtmosphereQuery(
        params=AtmosphereParams(teff=teff, logg=logg),
        product_id="newera-v3-lowres",
        family="newera",
        spectral_plan=SpectralPlan(
            SpectralAxis.points(
                np.array([100.0, 101.0, 102.0]),
                coordinate=SpectralCoordinate.WAVELENGTH,
                unit="nm",
            )
        ),
        requested_parameter_names=("teff", "logg"),
    )


def test_newera_backend_opens_processed_artifact_and_interpolates(tmp_path):
    _write_processed_artifact(tmp_path)
    backend = NewEraBackend.open(tmp_path)

    prepared = backend.prepare(_query())
    assert prepared.prepared is not None
    result = prepared.prepared.evaluate(_query().params)

    np.testing.assert_allclose(result.spectrum.axis.values, [100.0, 101.0, 102.0])
    np.testing.assert_allclose(result.spectrum.values, [2500.0, 3500.0, 4500.0])
    assert result.spectrum.value_unit == "erg s^-1 cm^-2 nm^-1"
    assert int(result.status.code) == SpectrumStatusCode.OK
    assert result.spectrum.provenance.product_id == "newera-v3-lowres"


def test_newera_backend_prepare_rejects_missing_abundance_plane(tmp_path):
    _write_processed_artifact(tmp_path)
    backend = NewEraBackend.open(tmp_path)

    query = _query()
    query = AtmosphereQuery(
        params=AtmosphereParams(teff=5500.0, logg=4.5, m_h=0.5),
        product_id=query.product_id,
        family=query.family,
        spectral_plan=query.spectral_plan,
        requested_parameter_names=query.requested_parameter_names,
    )

    result = backend.prepare(query)

    assert result.status is SpectrumStatusCode.NO_DATASET
    assert result.prepared is None


def test_newera_sparse_cell_never_reports_false_success(tmp_path):
    _write_processed_artifact(
        tmp_path,
        points=((5000.0, 4.0), (5000.0, 5.0), (6000.0, 4.0)),
    )
    backend = NewEraBackend.open(tmp_path)

    result = backend.prepare(_query(teff=5250.0, logg=4.25))

    assert result.status is SpectrumStatusCode.NO_COMPLETE_CELL
    assert result.prepared is None


def test_newera_sparse_cell_uses_only_explicitly_approved_simplex(tmp_path):
    _write_processed_artifact(
        tmp_path,
        points=((5000.0, 4.0), (5000.0, 5.0), (6000.0, 4.0)),
    )
    backend = NewEraBackend.open(tmp_path, approved_simplices=((0, 1, 2),))

    prepared = backend.prepare(_query(teff=5250.0, logg=4.25))

    assert prepared.status is SpectrumStatusCode.OK
    assert prepared.prepared is not None
    result = prepared.prepared.evaluate(_query(teff=5250.0, logg=4.25).params)
    np.testing.assert_allclose(result.spectrum.values, [1750.0, 2750.0, 3750.0])
