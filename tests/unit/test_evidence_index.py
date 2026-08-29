"""Resolution and freshness contracts for the unified evidence index."""

import subprocess
import sys
from pathlib import Path

from jaxstro.evidence.index import EvidenceClass, _digest_files, build_evidence_index

ROOT = Path(__file__).resolve().parents[2]


def test_evidence_index_preserves_distinct_evidence_classes() -> None:
    index = build_evidence_index(ROOT)
    entries = {item.id: item for item in index.entries}
    assert entries["quad.performance"].evidence_class is EvidenceClass.COMPUTATIONAL
    for identity in (
        "quad.multidim.truth",
        "quad.multidim.replay-and-astro",
        "quad.rqmc-calibration",
        "quad.multidim.comparisons",
        "quad.multidim.observed-memory",
        "quad.multidim.performance-baseline",
    ):
        assert entries[identity].evidence_class is EvidenceClass.COMPUTATIONAL
    assert entries["quad.multidim.truth"].source_revision == "not-recorded"
    assert entries["quad.multidim.comparisons"].source_revision == (
        "e091640eefc7322714126e06e380ca3f9486af07"
    )
    assert entries["quad.multidim.observed-memory"].source_revision == (
        "1048b5b5fab2efb8e2c418c5249afbe5c715d603"
    )
    assert (
        entries["rootfinding.performance"].evidence_class is EvidenceClass.COMPUTATIONAL
    )
    assert entries["provenance.cards"].evidence_class is EvidenceClass.SOURCE_PROVENANCE
    assert (
        entries["atmosphere.interpolation-policy"].evidence_class
        is EvidenceClass.SCIENTIFIC_POLICY
    )
    assert all(item.content_digest.startswith("sha256:") for item in index.entries)
    assert all((ROOT / item.target).is_file() for item in index.entries)


def test_evidence_index_artifacts_are_fresh() -> None:
    subprocess.run(
        [sys.executable, "scripts/build_evidence_index.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_provenance_digest_changes_when_same_count_card_content_changes(
    tmp_path: Path,
) -> None:
    index = tmp_path / "index.md"
    family = tmp_path / "constants.md"
    index.write_text("one family, one card\n", encoding="utf-8")
    family.write_text("source locator A\n", encoding="utf-8")
    before = _digest_files((index, family))
    family.write_text("source locator B\n", encoding="utf-8")
    assert _digest_files((index, family)) != before
