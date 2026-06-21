"""Host-side atmosphere grid discovery and indexing helpers.

This package keeps host-side data ingestion separate from JAX-side spectral
interpolation. It can discover raw PHOENIX/NewEra files, open processed local
artifacts, and return raw spectra. Downstream packages own filter projection,
photometry, bolometric corrections, and survey-specific observables.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from .bosz import (
    BOSZ_2025_RECOMPUTED_NOTE,
    DEFAULT_BOSZ_CATALOG,
    DEFAULT_BOSZ_ZARR,
    BoszBackend,
    BoszFile,
    BoszIndex,
    BoszMetadata,
    build_bosz_index,
    discover_bosz_files,
    parse_bosz_filename,
)
from .newera import DEFAULT_NEWERA_CATALOG, DEFAULT_NEWERA_ZARR, NewEraBackend
from .spectra import (
    STATUS_MISSING_ABUNDANCE,
    STATUS_OK,
    STATUS_OUT_OF_GRID,
    AtmosphereBackend,
    AtmosphereParams,
    PreparedSpectralGrid,
    Spectrum,
    SpectrumResult,
    SpectrumStatus,
)

PathLike: TypeAlias = str | os.PathLike[str]

NEWERA_DATA_ENV = "JAXSTRO_DATA_DIR"
NEWERA_LOWRES_PRODUCT = "LowRes-SPECTRA"
_NEWERA_LOWRES_PATTERN = re.compile(
    r"^PHOENIX-NewEra"
    r"(?P<version>V\d+(?:\.\d+)?)"
    r"-LowRes-SPECTRA"
    r"\.Z(?P<m_h>[+-]\d+(?:\.\d+)?)"
    r"(?:\.alpha=(?P<alpha_m>[+-]?\d+(?:\.\d+)?))?"
    r"\.txt$"
)


@dataclass(frozen=True)
class NewEraLowResMetadata:
    """Metadata encoded in a PHOENIX/NewEra low-resolution spectra filename."""

    filename: str
    version: str
    product: str
    m_h: float
    alpha_m: float


@dataclass(frozen=True)
class NewEraLowResHeader:
    """First-line header metadata from a NewEra low-resolution text product."""

    raw: str
    column_names: tuple[str, ...]


@dataclass(frozen=True)
class NewEraLowResFile:
    """One discovered NewEra low-resolution spectra file."""

    path: Path
    metadata: NewEraLowResMetadata
    header: NewEraLowResHeader | None = None


@dataclass(frozen=True)
class NewEraLowResIndex:
    """Immutable summary of a local NewEra low-resolution spectra directory."""

    root: Path
    product: str
    files: tuple[NewEraLowResFile, ...]
    versions: tuple[str, ...]
    m_h_values: tuple[float, ...]
    alpha_m_values: tuple[float, ...]


def resolve_data_dir(path: PathLike | None = None) -> Path:
    """Resolve the jaxstro data cache root without creating directories."""
    if path is not None:
        return Path(path).expanduser()

    env_path = os.environ.get(NEWERA_DATA_ENV)
    if env_path:
        return Path(env_path).expanduser()

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_root = (
        Path(xdg_cache_home).expanduser()
        if xdg_cache_home
        else Path("~/.cache").expanduser()
    )
    return cache_root / "jaxstro"


def parse_newera_lowres_filename(path: PathLike) -> NewEraLowResMetadata:
    """Parse metadata from a PHOENIX/NewEra low-resolution spectra filename."""
    filename = Path(path).name
    match = _NEWERA_LOWRES_PATTERN.match(filename)
    if match is None:
        raise ValueError(
            f"Not a PHOENIX/NewEra low-resolution spectra filename: {filename}"
        )

    alpha = match.group("alpha_m")
    return NewEraLowResMetadata(
        filename=filename,
        version=match.group("version"),
        product=NEWERA_LOWRES_PRODUCT,
        m_h=float(match.group("m_h")),
        alpha_m=0.0 if alpha is None else float(alpha),
    )


def discover_newera_lowres_files(root: PathLike) -> tuple[Path, ...]:
    """Discover local PHOENIX/NewEra low-resolution spectra text files."""
    root_path = Path(root).expanduser()
    candidates = root_path.glob("PHOENIX-NewEraV*-LowRes-SPECTRA.Z*.txt")
    files = []
    for path in candidates:
        try:
            metadata = parse_newera_lowres_filename(path)
        except ValueError:
            continue
        files.append((metadata.m_h, metadata.alpha_m, metadata.filename, path))

    return tuple(path for _, _, _, path in sorted(files))


def read_newera_lowres_header(path: PathLike) -> NewEraLowResHeader:
    """Read the first-line column header from a NewEra low-resolution text file."""
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        raw = handle.readline().strip()

    column_line = raw[1:].strip() if raw.startswith("#") else raw
    return NewEraLowResHeader(raw=raw, column_names=tuple(column_line.split()))


def build_newera_lowres_index(
    root: PathLike,
    *,
    read_headers: bool = True,
) -> NewEraLowResIndex:
    """Build an immutable local index for NewEra low-resolution spectra files."""
    root_path = Path(root).expanduser()
    files = []
    for path in discover_newera_lowres_files(root_path):
        metadata = parse_newera_lowres_filename(path)
        header = read_newera_lowres_header(path) if read_headers else None
        files.append(NewEraLowResFile(path=path, metadata=metadata, header=header))

    file_tuple = tuple(files)
    versions = tuple(sorted({entry.metadata.version for entry in file_tuple}))
    m_h_values = tuple(sorted({entry.metadata.m_h for entry in file_tuple}))
    alpha_m_values = tuple(sorted({entry.metadata.alpha_m for entry in file_tuple}))

    return NewEraLowResIndex(
        root=root_path,
        product=NEWERA_LOWRES_PRODUCT,
        files=file_tuple,
        versions=versions,
        m_h_values=m_h_values,
        alpha_m_values=alpha_m_values,
    )


__all__ = [
    "NEWERA_DATA_ENV",
    "NEWERA_LOWRES_PRODUCT",
    "DEFAULT_NEWERA_CATALOG",
    "DEFAULT_NEWERA_ZARR",
    "BOSZ_2025_RECOMPUTED_NOTE",
    "DEFAULT_BOSZ_CATALOG",
    "DEFAULT_BOSZ_ZARR",
    "STATUS_MISSING_ABUNDANCE",
    "STATUS_OK",
    "STATUS_OUT_OF_GRID",
    "AtmosphereBackend",
    "AtmosphereParams",
    "BoszBackend",
    "BoszFile",
    "BoszIndex",
    "BoszMetadata",
    "NewEraBackend",
    "NewEraLowResFile",
    "NewEraLowResHeader",
    "NewEraLowResIndex",
    "NewEraLowResMetadata",
    "PreparedSpectralGrid",
    "Spectrum",
    "SpectrumResult",
    "SpectrumStatus",
    "build_bosz_index",
    "build_newera_lowres_index",
    "discover_bosz_files",
    "discover_newera_lowres_files",
    "parse_bosz_filename",
    "parse_newera_lowres_filename",
    "read_newera_lowres_header",
    "resolve_data_dir",
]
