from __future__ import annotations

import importlib.util
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_distribution", ROOT / "scripts/check_distribution.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_check_wheel_requires_typing_license_and_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "jaxstro-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("jaxstro/__init__.py", "")
        archive.writestr("jaxstro/py.typed", "")
        archive.writestr(
            "jaxstro-0.1.0.dist-info/METADATA",
            "Name: jaxstro\nVersion: 0.1.0\nRequires-Python: >=3.13\n"
            "License-Expression: Apache-2.0\n",
        )
        archive.writestr("jaxstro-0.1.0.dist-info/licenses/LICENSE", "Apache-2.0")
    checker.check_wheel(wheel, expected_version="0.1.0")


def test_check_sdist_requires_metadata_and_rejects_generated_tree(
    tmp_path: Path,
) -> None:
    sdist = tmp_path / "jaxstro-0.1.0.tar.gz"
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "PKG-INFO").write_text(
        "Name: jaxstro\nVersion: 0.1.0\nRequires-Python: >=3.13\n"
        "License-Expression: Apache-2.0\n",
        encoding="utf-8",
    )
    (payload / "LICENSE").write_text("Apache-2.0", encoding="utf-8")
    (payload / "pyproject.toml").write_text("[project]", encoding="utf-8")
    package = payload / "src/jaxstro"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "py.typed").write_text("", encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        for source in payload.rglob("*"):
            if source.is_file():
                archive.add(
                    source, arcname=f"jaxstro-0.1.0/{source.relative_to(payload)}"
                )
    checker.check_sdist(sdist, expected_version="0.1.0")


def test_check_sdist_rejects_internal_workspaces(tmp_path: Path) -> None:
    sdist = tmp_path / "jaxstro-0.1.0.tar.gz"
    payload = tmp_path / "AGENTS.md"
    payload.write_text("internal", encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(payload, arcname="jaxstro-0.1.0/AGENTS.md")
    with pytest.raises(ValueError, match="forbidden release member"):
        checker.check_sdist(sdist, expected_version="0.1.0")


def test_check_sdist_rejects_generated_documentation(tmp_path: Path) -> None:
    sdist = tmp_path / "jaxstro-0.1.0.tar.gz"
    payload = tmp_path / "index.html"
    payload.write_text("generated", encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(payload, arcname="jaxstro-0.1.0/docs/_build/html/index.html")
    with pytest.raises(ValueError, match="forbidden release member"):
        checker.check_sdist(sdist, expected_version="0.1.0")


def test_check_sdist_rejects_a_wrong_archive_root(tmp_path: Path) -> None:
    sdist = tmp_path / "jaxstro-0.1.0.tar.gz"
    payload = tmp_path / "PKG-INFO"
    payload.write_text("Name: jaxstro\n", encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(payload, arcname="wrong-root/PKG-INFO")
    with pytest.raises(ValueError, match="invalid sdist root"):
        checker.check_sdist(sdist, expected_version="0.1.0")


def test_check_provenance_requires_pinned_backend(tmp_path: Path) -> None:
    provenance = tmp_path / "build-provenance.txt"
    provenance.write_text(
        "uv 0.8.22\nPython 3.13.0\nhatchling==1.31.0\n", encoding="utf-8"
    )
    checker.check_provenance(provenance)
