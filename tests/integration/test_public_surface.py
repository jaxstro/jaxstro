from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path

import jaxstro
from jaxstro._public import PUBLIC_MODULES

ROOT = Path(__file__).resolve().parents[2]


def test_root_exports_the_canonical_public_modules() -> None:
    expected = {
        "astrometry",
        "atmospheres",
        "constants",
        "contracts",
        "coords",
        "evidence",
        "geometry",
        "jaxconfig",
        "numerics",
        "params",
        "provenance",
        "quad",
        "quantity",
        "spatial",
        "spectra",
        "testing",
        "units",
    }
    assert set(PUBLIC_MODULES) == expected
    assert set(jaxstro.__all__) == {"DEFAULT_UNITS", *expected}
    for name in PUBLIC_MODULES:
        assert getattr(jaxstro, name) is importlib.import_module(f"jaxstro.{name}")


def test_release_support_page_names_only_qualified_support() -> None:
    text = (ROOT / "docs/70-project/release/support.md").read_text(encoding="utf-8")
    for phrase in (
        "CPython 3.13",
        "Ubuntu x86_64 CPU",
        "JAX_ENABLE_X64=1",
        "not a qualified support claim",
        "GPU",
        "experimental",
    ):
        assert phrase in text

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "Operating System :: OS Independent" not in project["project"]["classifiers"]
    assert (
        "Programming Language :: Python :: 3" not in project["project"]["classifiers"]
    )
    assert "Programming Language :: Python :: 3.13" in project["project"]["classifiers"]

    routes = json.loads((ROOT / "docs/route-manifest.json").read_text(encoding="utf-8"))
    assert routes["70-project/release/support.md"] == "/support"
