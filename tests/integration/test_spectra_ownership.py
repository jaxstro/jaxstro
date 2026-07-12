"""Architecture guard for the spectra hard cutover."""

from __future__ import annotations

from pathlib import Path

import jaxstro
from jaxstro.spectra import Spectrum

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_IMPORT_ALLOWLIST = {
    "src/jaxstro/atmospheres/__init__.py",
    "src/jaxstro/atmospheres/bosz.py",
    "src/jaxstro/atmospheres/library.py",
    "src/jaxstro/atmospheres/newera.py",
    "src/jaxstro/atmospheres/overlap.py",
}


def test_generic_spectrum_owner_is_top_level_spectra_package() -> None:
    assert jaxstro.spectra.Spectrum is Spectrum
    assert Spectrum.__module__ == "jaxstro.spectra.types"


def test_no_new_production_module_imports_legacy_spectral_owner() -> None:
    observed = {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "src").rglob("*.py")
        if "from .spectra import" in path.read_text(encoding="utf-8")
    }

    assert observed <= LEGACY_IMPORT_ALLOWLIST
