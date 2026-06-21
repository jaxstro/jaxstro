"""Host-side TLUSTY raw flux metadata helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TLUSTY_FLUX_PATTERN = re.compile(
    r"^(?P<prefix>[A-Z]+)"
    r"(?P<teff>\d+)"
    r"g(?P<logg_code>\d+)"
    r"v(?P<vturb_km_s>\d+)"
    r"(?P<cn>CN)?"
    r"\.flux\.gz$"
)
_BARE_EXPONENT_PATTERN = re.compile(
    r"^(?P<mantissa>[+-]?(?:\d+(?:\.\d*)?|\.\d+))(?P<exponent>[+-]\d+)$"
)


@dataclass(frozen=True)
class TlustyFluxMetadata:
    """Metadata encoded in a TLUSTY flux filename."""

    filename: str
    prefix: str
    teff: float
    logg: float
    vturb_km_s: float
    cn_altered: bool


def parse_tlusty_flux_filename(path: str | Path) -> TlustyFluxMetadata:
    """Parse TLUSTY ``*.flux.gz`` filename coordinates."""
    filename = Path(path).name
    match = _TLUSTY_FLUX_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Not a TLUSTY flux filename: {filename}")

    return TlustyFluxMetadata(
        filename=filename,
        prefix=match.group("prefix"),
        teff=float(match.group("teff")),
        logg=float(match.group("logg_code")) / 100.0,
        vturb_km_s=float(match.group("vturb_km_s")),
        cn_altered=match.group("cn") is not None,
    )


def parse_tlusty_float(token: str) -> float:
    """Parse TLUSTY Fortran-style floats, including bare signed exponents."""
    normalized = token.replace("D", "E").replace("d", "E")
    match = _BARE_EXPONENT_PATTERN.match(normalized)
    if match is not None and "E" not in normalized and "e" not in normalized:
        normalized = f"{match.group('mantissa')}E{match.group('exponent')}"
    return float(normalized)


__all__ = [
    "TlustyFluxMetadata",
    "parse_tlusty_float",
    "parse_tlusty_flux_filename",
]
