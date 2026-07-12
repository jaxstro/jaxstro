"""Host-side Sonora 2024 raw-file metadata helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
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

DEFAULT_SONORA_ZARR = "sonora_2024.zarr"
DEFAULT_SONORA_CATALOG = "catalog.parquet"

_SONORA_2024_PATTERN = re.compile(
    r"^t(?P<teff>\d+)"
    r"g(?P<g_m_s2>\d+)"
    r"(?P<cloud_label>nc|f\d+)"
    r"_m(?P<m_h>[+-]?\d+(?:\.\d+)?)"
    r"_co(?P<c_o>\d+(?:\.\d+)?)"
    r"\.spec$"
)


@dataclass(frozen=True)
class Sonora2024Metadata:
    """Metadata encoded in a Sonora 2024 Diamondback spectrum filename."""

    filename: str
    teff: float
    g_m_s2: float
    logg: float
    m_h: float
    c_o: float
    cloud_label: str


@dataclass(frozen=True)
class SonoraBackend:
    """Exact-product adapter for processed Sonora Diamondback spectra."""

    processed_dir: Path
    catalog_rows: tuple[dict[str, Any], ...]
    zarr_path: Path
    _store: Any = field(repr=False, compare=False)
    cloud_label: str = "f1"
    m_h: float = 0.0
    c_o: float = 1.0
    approved_simplices: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def open(
        cls,
        processed_dir: str | os.PathLike[str] | None = None,
        *,
        catalog_name: str = DEFAULT_SONORA_CATALOG,
        zarr_name: str = DEFAULT_SONORA_ZARR,
        cloud_label: str,
        m_h: float,
        c_o: float,
        approved_simplices: tuple[tuple[int, ...], ...] = (),
    ) -> SonoraBackend:
        pl, zarr = _load_optional_backend_deps()
        if processed_dir is None:
            from . import resolve_data_dir

            root = resolve_data_dir() / "atmospheres" / "sonora" / "2024" / "processed"
        else:
            root = Path(processed_dir).expanduser()
        catalog_path = root / catalog_name
        zarr_path = root / zarr_name
        if not catalog_path.exists():
            raise FileNotFoundError(f"Sonora catalog not found: {catalog_path}")
        if not zarr_path.exists():
            raise FileNotFoundError(f"Sonora Zarr store not found: {zarr_path}")
        return cls(
            processed_dir=root,
            catalog_rows=tuple(pl.read_parquet(catalog_path).to_dicts()),
            zarr_path=zarr_path,
            _store=zarr.open_group(zarr_path, mode="r"),
            cloud_label=cloud_label,
            m_h=float(m_h),
            c_o=float(c_o),
            approved_simplices=approved_simplices,
        )

    @staticmethod
    def product_descriptor(
        cloud_label: str,
        m_h: float,
        c_o: float,
    ) -> ProductDescriptor:
        return ProductDescriptor(
            product_id=(f"sonora-diamondback-2024:{cloud_label}:m{m_h:+g}:co{c_o:g}"),
            family="sonora",
            parameter_names=("teff", "logg"),
            topology_policy="complete-cell-or-approved-simplex",
            flux_interpolation_policy="linear",
            provenance_id="sonora-diamondback-2024",
        )

    def describe_product(self) -> ProductDescriptor:
        return self.product_descriptor(self.cloud_label, self.m_h, self.c_o)

    def validate_artifact(self) -> ArtifactReport:
        valid = self.zarr_path.exists() and bool(self.catalog_rows)
        payload = json.dumps(self.catalog_rows, sort_keys=True, default=str).encode()
        digest = f"sha256:{hashlib.sha256(payload).hexdigest()}" if valid else None
        return ArtifactReport(
            valid=valid,
            digest=digest,
            schema="sonora-diamondback-2024" if valid else None,
            message="" if valid else "Sonora artifact is unavailable or empty",
        )

    def prepare(self, query: AtmosphereQuery) -> PreparationResult:
        descriptor = self.describe_product()
        if (
            query.product_id != descriptor.product_id
            or query.cloud_label != self.cloud_label
        ):
            return PreparationResult.failure(
                SpectrumStatusCode.NO_DATASET,
                "Sonora query does not match cloud, metallicity, and C/O product",
            )
        params = query.params
        if not math.isclose(
            float(np.asarray(params.m_h)), self.m_h
        ) or not math.isclose(float(np.asarray(params.c_o)), self.c_o):
            return PreparationResult.failure(
                SpectrumStatusCode.NO_DATASET,
                "Sonora parameters do not match the exact product plane",
            )
        rows = [
            row
            for row in self.catalog_rows
            if str(row["cloud_label"]) == self.cloud_label
            and math.isclose(float(row["m_h"]), self.m_h)
            and math.isclose(float(row["c_o"]), self.c_o)
        ]
        if not rows:
            return PreparationResult.failure(
                SpectrumStatusCode.NO_DATASET,
                "No Sonora spectra match the exact product plane",
            )
        topology = GridTopology(
            parameter_names=descriptor.parameter_names,
            points=tuple((float(row["teff"]), float(row["logg"])) for row in rows),
            approved_simplices=self.approved_simplices,
        )
        point = (float(np.asarray(params.teff)), float(np.asarray(params.logg)))
        selection = select_topology(topology, point)
        if selection.status is not SpectrumStatusCode.OK:
            return PreparationResult.failure(
                selection.status, "No valid Sonora topology"
            )
        wavelength_nm = np.asarray(self._store["wavelength"][:], dtype=float) * 1.0e3
        descending = bool(np.all(np.diff(wavelength_nm) < 0.0))
        if descending:
            wavelength_nm = wavelength_nm[::-1]
        elif not np.all(np.diff(wavelength_nm) > 0.0):
            raise ValueError("Sonora wavelength grid must be monotonic")
        artifact = self.validate_artifact()
        vertex_values = []
        template = None
        for index in selection.vertex_indices:
            record = rows[index]
            group = self._store
            for part in str(record["zarr_group"]).split("/"):
                group = group[part]
            flux = np.asarray(group["flux"][int(record["zarr_row"]), :], dtype=float)
            if descending:
                flux = flux[::-1]
            provenance = SpectrumProvenance(
                source_id="sonora-diamondback-2024",
                product_id=descriptor.product_id,
                native_coordinate="wavelength_micron",
                native_density="wavelength-density flux",
                native_unit="W m^-2 m^-1",
                canonical_conversion="multiply by 1e-6",
                citations=("https://doi.org/10.5281/zenodo.12735103",),
                artifact_digest=artifact.digest,
            )
            native = Spectrum(
                axis=SpectralAxis.points(
                    jnp.asarray(wavelength_nm),
                    coordinate=SpectralCoordinate.WAVELENGTH,
                    unit="nm",
                ),
                values=jnp.asarray(flux * 1.0e-6),
                semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
                provenance=provenance,
            )
            remapped = resample_spectrum(native, query.spectral_plan)
            if int(remapped.status.code) != SpectrumStatusCode.OK:
                return PreparationResult.failure(
                    SpectrumStatusCode.UNSUPPORTED_SPECTRAL_WINDOW,
                    "Sonora sampling does not cover the explicit spectral plan",
                )
            template = remapped.spectrum
            vertex_values.append(template.values)
        if template is None:
            raise ValueError("Sonora topology selected no vertices")
        stacked = jnp.stack(vertex_values)
        stencil: PreparedRectilinearStencil | PreparedSimplexStencil
        if selection.kind is TopologyKind.RECTILINEAR:
            selected = [topology.points[i] for i in selection.vertex_indices]
            teff_axis = jnp.asarray(sorted({value[0] for value in selected}))
            logg_axis = jnp.asarray(sorted({value[1] for value in selected}))
            stencil = PreparedRectilinearStencil(
                parameter_axes=(teff_axis, logg_axis),
                vertex_values=stacked.reshape(
                    (teff_axis.shape[0], logg_axis.shape[0], stacked.shape[-1])
                ),
                template=template,
                interpolation=FluxInterpolation.LINEAR,
            )
        else:
            stencil = PreparedSimplexStencil(
                vertices=jnp.asarray(
                    [topology.points[index] for index in selection.vertex_indices]
                ),
                vertex_values=stacked,
                template=template,
            )
        return PreparationResult.success(
            PreparedAtmosphere(
                stencil=stencil,
                parameter_names=descriptor.parameter_names,
                spectral_plan=query.spectral_plan,
                provenance=template.provenance,
            )
        )


def _load_optional_backend_deps():
    try:
        pl = import_module("polars")
        zarr = import_module("zarr")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SonoraBackend requires the optional data dependencies"
        ) from exc
    return pl, zarr


def parse_sonora_2024_filename(path: str | Path) -> Sonora2024Metadata:
    """Parse Sonora 2024 spectrum filename coordinates.

    The released filenames encode gravity in ``m/s2``. jaxstro also records the
    cgs ``logg`` coordinate used by ``AtmosphereParams`` as ``log10(g_m_s2*100)``.
    """
    filename = Path(path).name
    match = _SONORA_2024_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Not a Sonora 2024 spectrum filename: {filename}")

    g_m_s2 = float(match.group("g_m_s2"))
    return Sonora2024Metadata(
        filename=filename,
        teff=float(match.group("teff")),
        g_m_s2=g_m_s2,
        logg=math.log10(g_m_s2 * 100.0),
        m_h=float(match.group("m_h")),
        c_o=float(match.group("c_o")),
        cloud_label=match.group("cloud_label"),
    )


__all__ = [
    "DEFAULT_SONORA_CATALOG",
    "DEFAULT_SONORA_ZARR",
    "SonoraBackend",
    "Sonora2024Metadata",
    "parse_sonora_2024_filename",
]
