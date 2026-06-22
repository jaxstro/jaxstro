"""Runtime provenance helpers for deterministic scientific artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactHash:
    """Hash record for a generated or consumed artifact."""

    path: str
    digest: str
    algorithm: str = "sha256"
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready dictionary."""
        return {
            "algorithm": self.algorithm,
            "digest": self.digest,
            "path": self.path,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Small environment snapshot for provenance manifests."""

    python: dict[str, str]
    platform: dict[str, str]
    packages: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready dictionary."""
        return {
            "python": dict(sorted(self.python.items())),
            "platform": dict(sorted(self.platform.items())),
            "packages": dict(sorted(self.packages.items())),
        }


@dataclass(frozen=True)
class MethodManifest:
    """Deterministic manifest for a scientific method run."""

    name: str
    version: str
    parameters: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[ArtifactHash, ...] = ()
    environment: EnvironmentSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready dictionary."""
        payload: dict[str, Any] = {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "inputs": dict(sorted(self.inputs.items())),
            "name": self.name,
            "parameters": dict(sorted(self.parameters.items())),
            "version": self.version,
        }
        if self.environment is not None:
            payload["environment"] = self.environment.to_dict()
        return payload


def hash_artifact(path: str | Path, *, algorithm: str = "sha256") -> ArtifactHash:
    """Hash a file artifact and return its digest metadata."""
    artifact_path = Path(path)
    hasher = hashlib.new(algorithm)
    with artifact_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return ArtifactHash(
        path=str(artifact_path),
        digest=hasher.hexdigest(),
        algorithm=algorithm,
        size_bytes=artifact_path.stat().st_size,
    )


def environment_snapshot(
    packages: tuple[str, ...] = ("jax", "jaxlib"),
) -> EnvironmentSnapshot:
    """Capture a small deterministic Python/platform/package snapshot."""
    package_versions: dict[str, str] = {}
    for package in packages:
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"
    return EnvironmentSnapshot(
        python={
            "implementation": platform.python_implementation(),
            "version": sys.version.split()[0],
        },
        platform={
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
        },
        packages=package_versions,
    )


def manifest_to_json(manifest: MethodManifest) -> str:
    """Render a method manifest as deterministic pretty JSON."""
    return json.dumps(manifest.to_dict(), indent=2, sort_keys=True)


def manifest_to_markdown(manifest: MethodManifest) -> str:
    """Render a method manifest as deterministic Markdown."""
    lines = [f"# {manifest.name}", "", f"version: `{manifest.version}`", ""]
    lines.extend(["## Inputs", ""])
    if manifest.inputs:
        for key, value in sorted(manifest.inputs.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Parameters", ""])
    if manifest.parameters:
        for key, value in sorted(manifest.parameters.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Artifacts", ""])
    if manifest.artifacts:
        lines.append("| Path | Algorithm | Digest | Size (bytes) |")
        lines.append("| --- | --- | --- | --- |")
        for artifact in manifest.artifacts:
            lines.append(
                f"| `{artifact.path}` | `{artifact.algorithm}` | "
                f"`{artifact.digest}` | {artifact.size_bytes} |"
            )
    else:
        lines.append("- none")
    if manifest.environment is not None:
        lines.extend(["", "## Environment", ""])
        for section, values in manifest.environment.to_dict().items():
            lines.append(f"### {section}")
            for key, value in values.items():
                lines.append(f"- `{key}`: `{value}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ArtifactHash",
    "EnvironmentSnapshot",
    "MethodManifest",
    "hash_artifact",
    "environment_snapshot",
    "manifest_to_json",
    "manifest_to_markdown",
]
