"""Strict write and byte-freshness helpers for evidence artifacts."""

from pathlib import Path

from .render import artifact_to_json, artifact_to_markdown
from .schema import EvidenceArtifact


class EvidenceFreshnessError(ValueError):
    """Raised when a committed evidence artifact differs from fresh rendering."""


def render_for_path(path: str | Path, artifact: EvidenceArtifact) -> str:
    """Select the deterministic renderer from the output suffix."""
    suffix = Path(path).suffix
    if suffix == ".json":
        return artifact_to_json(artifact)
    if suffix in {".md", ".markdown"}:
        return artifact_to_markdown(artifact)
    raise ValueError(f"unsupported evidence artifact suffix: {suffix}")


def emit_artifact(path: str | Path, artifact: EvidenceArtifact) -> None:
    """Write a freshly rendered artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_for_path(output, artifact), encoding="utf-8")


def check_artifact(path: str | Path, artifact: EvidenceArtifact) -> None:
    """Fail when an artifact is missing or differs byte-for-byte."""
    output = Path(path)
    expected = render_for_path(output, artifact)
    if not output.is_file() or output.read_text(encoding="utf-8") != expected:
        raise EvidenceFreshnessError(f"stale evidence artifact: {output}")
