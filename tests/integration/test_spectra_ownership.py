"""Architecture guard for the spectra hard cutover."""

from __future__ import annotations

from pathlib import Path

import jaxstro
import jaxstro.atmospheres as atmospheres
from jaxstro.spectra import Spectrum

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_generic_spectrum_owner_is_top_level_spectra_package() -> None:
    assert jaxstro.spectra.Spectrum is Spectrum
    assert Spectrum.__module__ == "jaxstro.spectra.types"


def test_legacy_spectral_owner_is_deleted_without_public_aliases() -> None:
    assert not (REPO_ROOT / "src/jaxstro/atmospheres/spectra.py").exists()
    assert not hasattr(atmospheres, "PreparedSpectralGrid")
    assert not hasattr(atmospheres, "STATUS_MISSING_ABUNDANCE")


def test_no_production_module_imports_legacy_spectral_owner() -> None:
    observed = {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "src").rglob("*.py")
        if "from .spectra import" in path.read_text(encoding="utf-8")
    }

    assert observed == set()
