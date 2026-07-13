"""Cross-class index for computational, source, and policy evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from .render import artifact_from_dict


class EvidenceClass(str, Enum):
    COMPUTATIONAL = "computational"
    SOURCE_PROVENANCE = "source_provenance"
    SCIENTIFIC_POLICY = "scientific_policy"


@dataclass(frozen=True)
class EvidenceIndexEntry:
    id: str
    evidence_class: EvidenceClass
    target: str
    schema_version: str
    source_revision: str
    content_digest: str
    optional_data_policy: str
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceIndex:
    schema_version: str
    entries: tuple[EvidenceIndexEntry, ...]


_TARGETS = (
    (
        "rootfinding.performance",
        EvidenceClass.COMPUTATIONAL,
        "docs/validation/rootfinding-performance.json",
        "No external data required.",
    ),
    (
        "rootfinding.implicit-gradients",
        EvidenceClass.COMPUTATIONAL,
        "docs/validation/implicit-root-gradients.json",
        "No external data required.",
    ),
    (
        "spectra.performance",
        EvidenceClass.COMPUTATIONAL,
        "docs/validation/spectra-performance.json",
        "Requires the local NewEra artifact and the declared data extra.",
    ),
    (
        "provenance.cards",
        EvidenceClass.SOURCE_PROVENANCE,
        "docs/40-api/provenance/index.md",
        "Uses repository-owned source cards; no runtime dataset required.",
    ),
    (
        "atmosphere.interpolation-policy",
        EvidenceClass.SCIENTIFIC_POLICY,
        "docs/validation/atmosphere-interpolation.json",
        "Policy regeneration requires approved local atmosphere holdouts.",
    ),
)


def build_evidence_index(root: str | Path) -> EvidenceIndex:
    """Build and validate an index without conflating evidence classes."""
    root_path = Path(root)
    entries = []
    for identity, evidence_class, target, optional_policy in _TARGETS:
        path = root_path / target
        if not path.is_file():
            raise ValueError(f"evidence target does not exist: {target}")
        content = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        if identity == "provenance.cards":
            digest = _digest_files(
                tuple(sorted((root_path / "docs/40-api/provenance").glob("*.md")))
            )
        if evidence_class is EvidenceClass.COMPUTATIONAL:
            artifact = artifact_from_dict(json.loads(content))
            if artifact.artifact_id != identity:
                raise ValueError(f"evidence artifact identity mismatch: {identity}")
            schema_version = artifact.schema_version
            source_revision = artifact.source_revision
            limitations = artifact.limitations
        else:
            payload = json.loads(content) if path.suffix == ".json" else {}
            schema_version = str(payload.get("schema_version", "source-card-v1"))
            source_revision = digest
            limitations = (
                "This evidence class is indexed but not rewritten as a computational envelope.",
            )
        entries.append(
            EvidenceIndexEntry(
                identity,
                evidence_class,
                target,
                schema_version,
                source_revision,
                digest,
                optional_policy,
                limitations,
            )
        )
    return EvidenceIndex("1", tuple(sorted(entries, key=lambda item: item.id)))


def _digest_files(paths: tuple[Path, ...]) -> str:
    """Hash an ordered file set, including identities and complete contents."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def evidence_index_to_json(index: EvidenceIndex) -> str:
    """Render a deterministic machine-readable evidence index."""
    payload = asdict(index)
    for entry in payload["entries"]:
        entry["evidence_class"] = entry["evidence_class"].value
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def evidence_index_to_markdown(index: EvidenceIndex) -> str:
    """Render a compact human-facing evidence map."""
    lines = [
        "---",
        "title: Scientific evidence index",
        "---",
        "",
        "# Scientific evidence index",
        "",
        "Evidence classes remain distinct: a source citation is not a numerical validation, and a benchmark is not a physical acceptance result.",
        "",
        "| Evidence ID | Class | Artifact | Source revision | Optional-data policy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in index.entries:
        source_url = "https://github.com/drannarosen/jaxstro/blob/main/" + entry.target
        lines.append(
            f"| `{entry.id}` | {entry.evidence_class.value} | "
            f"[artifact source]({source_url}) | `{entry.source_revision}` | "
            f"{entry.optional_data_policy} |"
        )
    return "\n".join(lines) + "\n"
