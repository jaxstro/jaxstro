"""Tests for runtime provenance helpers."""

import json

import pytest

from jaxstro import provenance


class TestArtifactHashing:
    """Tests for deterministic artifact hashing."""

    def test_hash_artifact_reports_sha256_size_and_path(self, tmp_path):
        artifact = tmp_path / "artifact.txt"
        artifact.write_text("jaxstro\n", encoding="utf-8")
        record = provenance.hash_artifact(artifact)
        assert record.path.endswith("artifact.txt")
        assert record.algorithm == "sha256"
        assert record.size_bytes == 8
        assert len(record.digest) == 64

    def test_hash_artifact_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            provenance.hash_artifact(tmp_path / "missing.txt")


class TestEnvironmentSnapshot:
    """Tests for stable environment snapshots."""

    def test_environment_snapshot_has_stable_schema(self):
        snapshot = provenance.environment_snapshot(packages=("jax",))
        payload = snapshot.to_dict()
        assert set(payload) == {"python", "platform", "packages"}
        assert "version" in payload["python"]
        assert "system" in payload["platform"]
        assert "jax" in payload["packages"]


class TestMethodManifestRendering:
    """Tests for deterministic manifest JSON and Markdown."""

    def test_manifest_json_is_deterministic_and_sorted(self, tmp_path):
        artifact = tmp_path / "result.dat"
        artifact.write_text("42", encoding="utf-8")
        manifest = provenance.MethodManifest(
            name="demo",
            version="1",
            parameters={"b": 2, "a": 1},
            inputs={"z": "last", "a": "first"},
            artifacts=(provenance.hash_artifact(artifact),),
        )
        first = provenance.manifest_to_json(manifest)
        second = provenance.manifest_to_json(manifest)
        assert first == second
        assert json.loads(first)["parameters"] == {"a": 1, "b": 2}

    def test_manifest_markdown_contains_core_sections(self, tmp_path):
        artifact = tmp_path / "result.dat"
        artifact.write_text("42", encoding="utf-8")
        manifest = provenance.MethodManifest(
            name="demo",
            version="1",
            parameters={"alpha": 0.5},
            inputs={"source": "synthetic"},
            artifacts=(provenance.hash_artifact(artifact),),
        )
        markdown = provenance.manifest_to_markdown(manifest)
        assert markdown.startswith("# demo")
        assert "## Inputs" in markdown
        assert "## Parameters" in markdown
        assert "## Artifacts" in markdown
        assert "sha256" in markdown
