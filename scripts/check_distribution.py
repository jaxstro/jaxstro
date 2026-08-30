#!/usr/bin/env python3
"""Validate release artifacts and their clean installed public surface."""

from __future__ import annotations

import argparse
import subprocess
import tarfile
import tomllib
import zipfile
from collections.abc import Callable
from email import message_from_bytes
from pathlib import Path, PurePosixPath

FORBIDDEN_PREFIXES = (
    ".github/",
    "docs/audits/",
    "docs/plans/",
    "docs/superpowers/",
    "docs/_build/",
    "laboratory/",
    "tests/",
    ".mypy_cache/",
    ".pytest_cache/",
)
FORBIDDEN_NAMES = {"AGENTS.md", "CLAUDE.md", "STATUS.md"}


def _require(names: set[str], name: str) -> None:
    if name not in names:
        raise ValueError(f"required release member is missing: {name}")


def _reject_forbidden(names: set[str]) -> None:
    for name in names:
        if (
            name in FORBIDDEN_NAMES
            or name.startswith(FORBIDDEN_PREFIXES)
            or "/__pycache__/" in name
            or name.startswith("__pycache__/")
        ):
            raise ValueError(f"forbidden release member: {name}")


def _require_metadata(
    names: set[str],
    read: Callable[[str], bytes],
    expected_version: str,
    *,
    metadata_name: str | None = None,
) -> None:
    if metadata_name is None:
        candidates = sorted(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        if len(candidates) != 1:
            raise ValueError("wheel must contain exactly one dist-info METADATA file")
        metadata_name = candidates[0]
    _require(names, metadata_name)
    metadata = message_from_bytes(read(metadata_name))
    expected = {
        "Name": "jaxstro",
        "Version": expected_version,
        "Requires-Python": ">=3.13",
        "License-Expression": "Apache-2.0",
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"invalid {metadata_name} {field}: {metadata.get(field)!r}; expected {value!r}"
            )


def _require_license(names: set[str]) -> None:
    if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
        raise ValueError("wheel is missing a bundled LICENSE")


def _read_tar_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"cannot read sdist member: {member.name}")
    return extracted.read()


def check_wheel(path: Path, *, expected_version: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        _require(names, "jaxstro/__init__.py")
        _require(names, "jaxstro/py.typed")
        _require_metadata(names, archive.read, expected_version)
        _require_license(names)
        _reject_forbidden(names)


def check_sdist(path: Path, *, expected_version: str) -> None:
    with tarfile.open(path, "r:gz") as archive:
        expected_root = f"jaxstro-{expected_version}"
        members: dict[str, tarfile.TarInfo] = {}
        for member in archive.getmembers():
            if member.isdir():
                continue
            member_path = PurePosixPath(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or not member_path.parts
                or member_path.parts[0] != expected_root
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise ValueError(f"invalid sdist root or member: {member.name}")
            relative_name = member_path.relative_to(expected_root).as_posix()
            if relative_name in members:
                raise ValueError(f"duplicate sdist member: {relative_name}")
            members[relative_name] = member

        names = set(members)
        _reject_forbidden(names)
        for name in (
            "src/jaxstro/__init__.py",
            "src/jaxstro/py.typed",
            "LICENSE",
            "pyproject.toml",
            "PKG-INFO",
        ):
            _require(names, name)
        pkg_info = members["PKG-INFO"]
        _require_metadata(
            names,
            lambda _: _read_tar_member(archive, pkg_info),
            expected_version,
            metadata_name="PKG-INFO",
        )


def check_provenance(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not any(line.startswith("uv ") for line in lines):
        raise ValueError("build provenance is missing uv version")
    if not any(line.startswith("Python 3.13") for line in lines):
        raise ValueError("build provenance is missing Python 3.13 version")
    if "hatchling==1.31.0" not in lines:
        raise ValueError("build provenance is missing pinned hatchling backend")


def check_clean_imports(python: Path) -> None:
    code = """
import importlib
from jaxstro._public import PUBLIC_MODULES

for module in PUBLIC_MODULES:
    importlib.import_module(f\"jaxstro.{module}\")
"""
    subprocess.run([str(python), "-c", code], check=True)


def _project_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()

    expected_version = _project_version()
    check_wheel(args.wheel, expected_version=expected_version)
    check_sdist(args.sdist, expected_version=expected_version)
    check_provenance(args.provenance)
    check_clean_imports(args.python)


if __name__ == "__main__":
    main()
