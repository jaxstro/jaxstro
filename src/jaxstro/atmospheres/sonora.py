"""Host-side Sonora 2024 raw-file metadata helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

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
    "Sonora2024Metadata",
    "parse_sonora_2024_filename",
]
