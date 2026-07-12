"""Tests for Sonora 2024 raw metadata parsing."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from jaxstro.atmospheres import AtmosphereParams, AtmosphereQuery
from jaxstro.atmospheres.sonora import SonoraBackend, parse_sonora_2024_filename
from jaxstro.spectra import (
    SpectralAxis,
    SpectralCoordinate,
    SpectralPlan,
    SpectrumStatusCode,
)


def _write_processed_artifact(processed_dir: Path) -> None:
    pl = pytest.importorskip("polars")
    zarr = pytest.importorskip("zarr")
    root = zarr.open_group(processed_dir / "sonora_2024.zarr", mode="w", zarr_format=2)
    root.create_array(
        "wavelength",
        data=np.array([0.1, 0.2, 0.4]),
        chunks=(3,),
        overwrite=True,
    )
    group = root.create_group("spectra")
    group.create_array(
        "flux",
        data=np.array(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
            dtype=np.float32,
        ),
        chunks=(1, 3),
        overwrite=True,
    )
    rows = []
    for row, (teff, logg) in enumerate(
        ((900.0, 4.0), (900.0, 5.0), (1100.0, 4.0), (1100.0, 5.0))
    ):
        rows.append(
            {
                "filename": f"synthetic-{row}",
                "teff": teff,
                "g_m_s2": 100.0,
                "logg": logg,
                "m_h": 0.0,
                "c_o": 1.0,
                "cloud_label": "f1",
                "n_wave": 3,
                "wavelength_min": 0.1,
                "wavelength_max": 0.4,
                "wavelength_unit": "micron",
                "flux_unit": "W/m2/m",
                "source_zip_sha256": "synthetic",
                "zarr_group": "spectra",
                "zarr_row": row,
            }
        )
    pl.DataFrame(rows).write_parquet(processed_dir / "catalog.parquet")


def _query(
    product_id: str = "sonora-diamondback-2024:f1:m+0:co1",
) -> AtmosphereQuery:
    return AtmosphereQuery(
        params=AtmosphereParams(teff=1000.0, logg=4.5, m_h=0.0, c_o=1.0),
        product_id=product_id,
        family="sonora",
        cloud_label="f1",
        spectral_plan=SpectralPlan(
            SpectralAxis.points(
                np.array([100.0, 200.0, 400.0]),
                coordinate=SpectralCoordinate.WAVELENGTH,
                unit="nm",
            )
        ),
        requested_parameter_names=("teff", "logg"),
    )


def test_parse_sonora_filename_preserves_source_gravity_and_derives_cgs_logg():
    metadata = parse_sonora_2024_filename("spectra/t1300g3160f1_m+0.5_co1.0.spec")

    assert metadata.teff == 1300.0
    assert metadata.g_m_s2 == 3160.0
    assert metadata.cloud_label == "f1"
    assert metadata.m_h == 0.5
    assert metadata.c_o == 1.0
    assert math.isclose(metadata.logg, math.log10(3160.0 * 100.0))


def test_parse_sonora_filename_accepts_cloud_free_label():
    metadata = parse_sonora_2024_filename("t900g31nc_m-0.5_co1.0.spec")

    assert metadata.cloud_label == "nc"
    assert metadata.g_m_s2 == 31.0
    assert math.isclose(metadata.logg, math.log10(31.0 * 100.0))


def test_parse_sonora_filename_rejects_non_sonora_name():
    with pytest.raises(ValueError, match="Sonora 2024"):
        parse_sonora_2024_filename("not-a-spectrum.txt")


def test_sonora_backend_prepares_explicit_product_and_canonical_flux(tmp_path):
    _write_processed_artifact(tmp_path)
    backend = SonoraBackend.open(tmp_path, cloud_label="f1", m_h=0.0, c_o=1.0)

    prepared = backend.prepare(_query())
    assert prepared.prepared is not None
    result = prepared.prepared.evaluate(_query().params)

    np.testing.assert_allclose(result.spectrum.axis.values, [100.0, 200.0, 400.0])
    np.testing.assert_allclose(result.spectrum.values, [2.5e-6, 3.5e-6, 4.5e-6])
    assert result.spectrum.provenance.canonical_conversion == "multiply by 1e-6"
    assert int(result.status.code) == SpectrumStatusCode.OK


def test_sonora_product_plane_must_match_cloud_and_c_o(tmp_path):
    _write_processed_artifact(tmp_path)
    backend = SonoraBackend.open(tmp_path, cloud_label="f1", m_h=0.0, c_o=1.0)

    cloud_result = backend.prepare(_query("sonora-diamondback-2024:nc:m+0:co1"))
    c_o_query = _query("sonora-diamondback-2024:f1:m+0:co0.55")
    c_o_result = backend.prepare(c_o_query)

    assert cloud_result.status is SpectrumStatusCode.NO_DATASET
    assert c_o_result.status is SpectrumStatusCode.NO_DATASET
