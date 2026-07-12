"""Host-side PHOENIX/NewEra processed-artifact backend."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from jaxstro.spectra import (
    FluxInterpolation,
    PreparedRectilinearStencil,
    PreparedSimplexStencil,
    SpectralAxis,
    SpectralCoordinate,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
    SpectrumStatusCode,
    resample_spectrum,
)

from .adapters import PreparationResult, PreparedAtmosphere
from .params import AtmosphereQuery
from .products import ArtifactReport, ProductDescriptor
from .topology import GridTopology, TopologyKind, select_topology

DEFAULT_NEWERA_ZARR = "newera_lowres_v3.zarr"
DEFAULT_NEWERA_CATALOG = "catalog.parquet"


@dataclass(frozen=True)
class NewEraBackend:
    """Lazy host-side backend for processed NewEra low-resolution spectra."""

    processed_dir: Path
    catalog_rows: tuple[dict[str, Any], ...]
    zarr_path: Path
    _store: Any = field(repr=False, compare=False)
    _artifact_report: ArtifactReport = field(repr=False)
    approved_simplices: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def open(
        cls,
        processed_dir: str | os.PathLike[str] | None = None,
        *,
        catalog_name: str = DEFAULT_NEWERA_CATALOG,
        zarr_name: str = DEFAULT_NEWERA_ZARR,
        approved_simplices: tuple[tuple[int, ...], ...] = (),
    ) -> "NewEraBackend":
        """Open a processed NewEra artifact directory.

        Optional data dependencies are imported here, not at package import time.
        """
        pl, zarr = _load_optional_backend_deps()

        if processed_dir is None:
            from . import resolve_data_dir

            root = resolve_data_dir() / "atmospheres" / "newera" / "processed"
        else:
            root = Path(processed_dir).expanduser()

        catalog_path = root / catalog_name
        zarr_path = root / zarr_name
        if not catalog_path.exists():
            raise FileNotFoundError(f"NewEra catalog not found: {catalog_path}")
        if not zarr_path.exists():
            raise FileNotFoundError(f"NewEra Zarr store not found: {zarr_path}")

        catalog_rows = tuple(pl.read_parquet(catalog_path).to_dicts())
        store = zarr.open_group(zarr_path, mode="r")
        with catalog_path.open("rb") as handle:
            catalog_digest = hashlib.file_digest(handle, "sha256").hexdigest()
        artifact_report = ArtifactReport(
            valid=bool(catalog_rows),
            digest=f"sha256:{catalog_digest}" if catalog_rows else None,
            schema="newera-lowres-v3" if catalog_rows else None,
            message="" if catalog_rows else "NewEra artifact is empty",
        )
        return cls(
            processed_dir=root,
            catalog_rows=catalog_rows,
            zarr_path=zarr_path,
            _store=store,
            _artifact_report=artifact_report,
            approved_simplices=approved_simplices,
        )

    @staticmethod
    def product_descriptor() -> ProductDescriptor:
        """Return the exact NewEra low-resolution product contract."""
        return ProductDescriptor(
            product_id="newera-v3-lowres",
            family="newera",
            parameter_names=("teff", "logg"),
            topology_policy="complete-cell-or-approved-simplex",
            flux_interpolation_policy="positive_log",
            provenance_id="newera-v3-lowres",
        )

    def describe_product(self) -> ProductDescriptor:
        return self.product_descriptor()

    def validate_artifact(self) -> ArtifactReport:
        """Return deterministic schema and catalog evidence for this artifact."""
        return self._artifact_report

    def prepare(self, query: AtmosphereQuery) -> PreparationResult:
        """Prepare a complete cell or approved simplex on the requested axis."""
        descriptor = self.product_descriptor()
        if query.product_id != descriptor.product_id:
            return PreparationResult.failure(
                SpectrumStatusCode.NO_DATASET,
                "NewEra adapter requires product newera-v3-lowres",
            )
        params = query.params
        teff = _as_host_float(params.teff, "teff")
        logg = _as_host_float(params.logg, "logg")
        m_h = _as_host_float(params.m_h, "m_h")
        alpha_m = _as_host_float(params.alpha_m, "alpha_m")

        rows = [
            row
            for row in self.catalog_rows
            if math.isclose(float(row["m_h"]), m_h)
            and math.isclose(float(row["alpha_m"]), alpha_m)
        ]
        if not rows:
            return PreparationResult.failure(
                SpectrumStatusCode.NO_DATASET,
                f"No NewEra abundance plane for m_h={m_h}, alpha_m={alpha_m}",
            )
        topology = GridTopology(
            parameter_names=descriptor.parameter_names,
            points=tuple((float(row["teff"]), float(row["logg"])) for row in rows),
            approved_simplices=self.approved_simplices,
        )
        selection = select_topology(topology, (teff, logg))
        if selection.status is not SpectrumStatusCode.OK:
            return PreparationResult.failure(
                selection.status,
                "NewEra query has no valid prepared topology",
            )

        vertex_rows = [rows[index] for index in selection.vertex_indices]
        vertex_values = []
        template = None
        for record in vertex_rows:
            wavelength = _wavelength_grid(record)
            provenance = SpectrumProvenance(
                source_id="phoenix-newera-v3",
                product_id=descriptor.product_id,
                native_coordinate="wavelength_nm",
                native_density="F_lambda",
                native_unit="W m^-2 nm^-1",
                canonical_conversion="multiply by 1e3",
                citations=("https://doi.org/10.1051/0004-6361/202554171",),
                artifact_digest=self.validate_artifact().digest,
            )
            native = Spectrum(
                axis=SpectralAxis.points(
                    jnp.asarray(wavelength, dtype=jnp.float64),
                    coordinate=SpectralCoordinate.WAVELENGTH,
                    unit="nm",
                ),
                values=jnp.asarray(self._read_flux(record) * 1.0e3, dtype=jnp.float64),
                semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
                provenance=provenance,
            )
            remapped = resample_spectrum(native, query.spectral_plan)
            if int(remapped.status.code) != SpectrumStatusCode.OK:
                return PreparationResult.failure(
                    SpectrumStatusCode.UNSUPPORTED_SPECTRAL_WINDOW,
                    "NewEra vertex does not cover the requested spectral plan",
                )
            template = remapped.spectrum
            vertex_values.append(remapped.spectrum.values)
        if template is None:  # protected by successful topology selection
            raise ValueError("NewEra topology selected no vertices")
        stacked = jnp.stack(vertex_values)
        stencil: PreparedRectilinearStencil | PreparedSimplexStencil
        if selection.kind is TopologyKind.RECTILINEAR:
            teff_axis = jnp.asarray(
                sorted(
                    {
                        point[0]
                        for point in topology.points
                        if point[0]
                        in {topology.points[i][0] for i in selection.vertex_indices}
                    }
                )
            )
            logg_axis = jnp.asarray(
                sorted(
                    {
                        point[1]
                        for point in topology.points
                        if point[1]
                        in {topology.points[i][1] for i in selection.vertex_indices}
                    }
                )
            )
            stencil = PreparedRectilinearStencil(
                parameter_axes=(teff_axis, logg_axis),
                vertex_values=stacked.reshape(
                    (teff_axis.shape[0], logg_axis.shape[0], stacked.shape[-1])
                ),
                template=template,
                interpolation=FluxInterpolation.POSITIVE_LOG,
            )
        else:
            stencil = PreparedSimplexStencil(
                vertices=jnp.asarray(
                    [topology.points[index] for index in selection.vertex_indices]
                ),
                vertex_values=stacked,
                template=template,
                interpolation=FluxInterpolation.POSITIVE_LOG,
            )
        prepared = PreparedAtmosphere(
            stencil=stencil,
            parameter_names=descriptor.parameter_names,
            spectral_plan=query.spectral_plan,
            provenance=template.provenance,
        )
        return PreparationResult.success(prepared)

    def _read_flux(self, record: dict[str, Any]) -> np.ndarray:
        group = self._store
        for part in str(record["zarr_group"]).split("/"):
            group = group[part]
        flux = group["flux"][int(record["zarr_row"]), :]
        return np.asarray(flux, dtype=np.float64)


def _load_optional_backend_deps():
    try:
        pl = import_module("polars")
        zarr = import_module("zarr")
    except ImportError as exc:  # pragma: no cover - depends on local extra
        raise ImportError(
            "NewEraBackend requires the optional data dependencies. "
            "Install or run with: uv run --extra data ..."
        ) from exc
    return pl, zarr


def _as_host_float(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"NewEraBackend.prepare expects scalar {name}")
    return float(array)


def _bounding_pair(values: list[float], target: float) -> tuple[float, float]:
    unique = sorted(set(values))
    if len(unique) < 2:
        raise ValueError("NewEra interpolation requires at least two axis values")
    if target <= unique[0]:
        return unique[0], unique[1]
    if target >= unique[-1]:
        return unique[-2], unique[-1]
    for lower, upper in zip(unique[:-1], unique[1:]):
        if lower <= target <= upper:
            return lower, upper
    raise ValueError(f"Could not bracket target value {target}")


def _wavelength_grid(record: dict[str, Any]) -> np.ndarray:
    n_wave = int(record["n_wave"])
    lambda_min = float(record["lambda_min"])
    lambda_step = float(record["lambda_step"])
    return lambda_min + np.arange(n_wave, dtype=np.float64) * lambda_step


__all__ = [
    "DEFAULT_NEWERA_CATALOG",
    "DEFAULT_NEWERA_ZARR",
    "NewEraBackend",
]
