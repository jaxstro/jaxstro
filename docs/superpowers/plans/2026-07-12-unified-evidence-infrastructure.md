# Unified Evidence Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one dependency-light evidence envelope for numerical metrics, comparisons, environment policy, deterministic emit/check, and contract-linked artifacts, then migrate representative root, spectra, provenance, and atmosphere evidence without changing scientific thresholds.

**Architecture:** A public `jaxstro.evidence` subpackage owns frozen schemas, validation, deterministic JSON/Markdown rendering, and file freshness helpers. Method scripts retain their scientific payloads and acceptance rules while emitting a shared envelope. Contract evidence IDs resolve through a generated artifact index; source cards remain distinct from computational evidence.

**Tech Stack:** Python 3.11+, frozen dataclasses, string-backed enums, JSON, hashlib, pathlib, pytest, Ruff, MyPy, JAX benchmark scripts, MyST.

## Global Constraints

- Work inline in the normal checkout; preserve all unrelated and untracked work.
- Use `env -u VIRTUAL_ENV uv run --no-sync` for every Python, pytest, Ruff, and MyPy command.
- TDD every task and commit coherent slices.
- Add no dependency, downstream import, network access, or domain-specific acceptance rule.
- Preserve all existing scientific thresholds and algorithmic comparison rules during migration.
- Keep analytic, implementation, AD/FD, convergence, performance, source-provenance, and downstream evidence distinct.
- Hardware-dependent wall time is informational unless an artifact explicitly defines a hardware policy.
- Report measured numerical results only in tables with metric identity, symbol, value, and units.
- Use one targeted reviewer after C1 and two focused completion reviewers after C3.

---

### Task 1: Define evidence schemas and fail-closed validation

**Files:**
- Create: `src/jaxstro/evidence/__init__.py`
- Create: `src/jaxstro/evidence/schema.py`
- Create: `src/jaxstro/evidence/validation.py`
- Create: `tests/unit/test_evidence_schema.py`
- Modify: `src/jaxstro/__init__.py`
- Modify: `src/jaxstro/contracts/_core.py`

**Interfaces:**
- Produces: `EvidenceStatus`, `ComparisonRelation`, `MetricRecord`, `ComparisonRecord`, `EnvironmentRecord`, `EvidenceArtifact`, and `validate_artifact(artifact) -> None`.

- [ ] **Step 1: Write failing schema tests**

```python
import dataclasses
import pytest
from jaxstro.evidence import EvidenceArtifact, EvidenceStatus, MetricRecord, validate_artifact


def test_metric_requires_identity_symbol_value_and_units() -> None:
    metric = MetricRecord("root.residual", "abs(f(x_star))", 1.0e-14, "function units", EvidenceStatus.PASS)
    artifact = EvidenceArtifact.fixture("rootfinding.performance", metrics=(metric,))
    validate_artifact(artifact)
    with pytest.raises(dataclasses.FrozenInstanceError):
        metric.units = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("units", ["", "unitless", None])
def test_metric_rejects_missing_or_ambiguous_units(units) -> None:
    artifact = EvidenceArtifact.fixture("bad", metrics=(MetricRecord("m", "m", 1.0, units, EvidenceStatus.INFO),))
    with pytest.raises(ValueError, match="units"):
        validate_artifact(artifact)
```

- [ ] **Step 2: Verify imports fail**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_evidence_schema.py`

Expected: FAIL because `jaxstro.evidence` does not exist.

- [ ] **Step 3: Implement frozen records and validation**

`EvidenceArtifact` fields are schema version, artifact ID/version, package version, source revision, generation command, precision, deterministic configuration, environment record, metrics, comparisons, limitations, and method payload. Metric units must be nonempty and use `dimensionless` rather than `unitless`. Numeric values must be finite. Comparison metric IDs must resolve; PASS/FAIL must agree with the declared relation and tolerances. Payload keys are sorted during rendering but remain method-owned.

- [ ] **Step 4: Run tests and static gates**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_evidence_schema.py
env -u VIRTUAL_ENV uv run --no-sync ruff check src/jaxstro/evidence tests/unit/test_evidence_schema.py
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro/evidence
```

- [ ] **Step 5: Commit**

```bash
git add src/jaxstro/evidence src/jaxstro/__init__.py src/jaxstro/contracts/_core.py tests/unit/test_evidence_schema.py
git commit -m "feat: define unified scientific evidence schema"
```

### Task 2: Add deterministic rendering and freshness helpers

**Files:**
- Create: `src/jaxstro/evidence/render.py`
- Create: `src/jaxstro/evidence/files.py`
- Create: `tests/unit/test_evidence_rendering.py`
- Modify: `src/jaxstro/evidence/__init__.py`

**Interfaces:**
- Produces: `artifact_to_dict`, `artifact_to_json`, `artifact_to_markdown`, `emit_artifact`, and `check_artifact`.

- [ ] **Step 1: Write failing deterministic-output tests**

```python
def test_json_and_markdown_are_deterministic(valid_artifact) -> None:
    assert artifact_to_json(valid_artifact) == artifact_to_json(valid_artifact)
    markdown = artifact_to_markdown(valid_artifact)
    assert "| Metric identity | Symbol | Value | Units | Status |" in markdown
    assert "/Users/" not in artifact_to_json(valid_artifact)


def test_check_artifact_rejects_stale_bytes(tmp_path, valid_artifact) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvidenceFreshnessError, match="stale"):
        check_artifact(path, valid_artifact)
```

- [ ] **Step 2: Verify renderer imports fail**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_evidence_rendering.py`

- [ ] **Step 3: Implement portable deterministic rendering**

Sort mapping keys and record collections by stable ID. Markdown emits the required metric table and separate comparison, environment, limitation, and method-payload sections. `emit_artifact` writes exactly one terminal newline; `check_artifact` compares bytes and never mutates.

- [ ] **Step 4: Run focused gates and commit**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_evidence_schema.py tests/unit/test_evidence_rendering.py
env -u VIRTUAL_ENV uv run --no-sync ruff check src/jaxstro/evidence tests/unit/test_evidence_*.py
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro/evidence
git add src/jaxstro/evidence tests/unit/test_evidence_rendering.py
git commit -m "feat: render and check scientific evidence"
```

### Checkpoint C1: schema review

Use one targeted reviewer for schema flexibility, units, nonfinite policy, comparison truth, determinism, and separation from scientific acceptance. Resolve all Critical and Important findings.

### Task 3: Migrate value-first rootfinding evidence

**Files:**
- Modify: `scripts/benchmark_rootfinding.py`
- Modify: `tests/unit/test_benchmark_rootfinding_script.py`
- Modify: `docs/validation/rootfinding-performance.json`
- Create: `docs/validation/rootfinding-performance.md`
- Modify: `docs/10-theory/rootfinding.md`

**Interfaces:**
- Produces: the same case metrics and comparison verdicts inside artifact ID `rootfinding.performance`.

- [ ] **Step 1: Ratchet existing scientific payload before migration**

Add tests asserting the fresh artifact retains every existing case name, control, evaluation count, iteration count, residual, relative residual, warm timing, and the rule that hybrid evaluations do not exceed bisection evaluations.

- [ ] **Step 2: Verify the legacy artifact lacks the envelope**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_benchmark_rootfinding_script.py`

Expected: FAIL on absent `artifact_id`, `metrics`, and `method_payload`.

- [ ] **Step 3: Adapt the script to shared records**

Keep measurement functions unchanged. Convert measured case values to `MetricRecord`s, algorithmic gates to `ComparisonRecord`s, controls/cases to method payload, wall timing to informational metrics, and existing environment fields to `EnvironmentRecord`. Use shared emit/check helpers.

- [ ] **Step 4: Emit, test, and commit**

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/benchmark_rootfinding.py --emit
env -u VIRTUAL_ENV uv run --no-sync python scripts/benchmark_rootfinding.py --check
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_benchmark_rootfinding_script.py tests/validation/test_bracketed_root_algorithms.py
git add scripts/benchmark_rootfinding.py tests/unit/test_benchmark_rootfinding_script.py docs/validation/rootfinding-performance.json docs/validation/rootfinding-performance.md docs/10-theory/rootfinding.md
git commit -m "refactor: migrate rootfinding evidence envelope"
```

### Task 4: Migrate implicit-root derivative evidence

**Files:**
- Modify: `scripts/benchmark_implicit_root.py`
- Modify: `tests/unit/test_implicit_root_evidence.py`
- Modify: `docs/validation/implicit-root-gradients.json`
- Create: `docs/validation/implicit-root-gradients.md`
- Modify: `docs/10-theory/rootfinding.md`

**Interfaces:**
- Produces: artifact ID `rootfinding.implicit-gradients` with analytic, AD, central-FD, residual, width, slope, and certification metrics.

- [ ] **Step 1: Add failing envelope and parity assertions**

Require every existing case and value, all derivative controls, explicit `dimensionless` or coordinate/parameter units, comparison records for AD/analytic and AD/FD gates, and rejected flat-slope evidence.

- [ ] **Step 2: Adapt without changing solver or thresholds**

Preserve the current solve, FD step, residual limit, width limit, slope floor, and fail-closed cases. Only construct and render the shared envelope around them.

- [ ] **Step 3: Emit, verify, and commit**

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/benchmark_implicit_root.py --emit
env -u VIRTUAL_ENV uv run --no-sync python scripts/benchmark_implicit_root.py --check
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_implicit_root_evidence.py tests/validation/test_implicit_root_gradients.py tests/integration/test_rootfinding_docs.py
git add scripts/benchmark_implicit_root.py tests/unit/test_implicit_root_evidence.py docs/validation/implicit-root-gradients.json docs/validation/implicit-root-gradients.md docs/10-theory/rootfinding.md
git commit -m "refactor: migrate implicit-root evidence envelope"
```

### Task 5: Migrate spectra performance and conservation evidence

**Files:**
- Modify: `scripts/benchmark_spectra.py`
- Modify: `tests/unit/test_benchmark_spectra_script.py`
- Modify: `docs/validation/spectra-performance.json`
- Create: `docs/validation/spectra-performance.md`
- Modify: `docs/60-validation/index.md`

**Interfaces:**
- Produces: artifact ID `spectra.performance` while retaining shapes, dtype, backend, memory scope, timings, and conservation semantics.

- [ ] **Step 1: Write failing envelope/parity tests**

Pin every current field and assert compile/first-call, warm runtime, batch shape, and host-memory metrics remain separately identified. Wall time and memory remain informational rather than fabricated pass thresholds.

- [ ] **Step 2: Adapt, emit, and verify**

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/benchmark_spectra.py --emit
env -u VIRTUAL_ENV uv run --no-sync python scripts/benchmark_spectra.py --check
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_benchmark_spectra_script.py tests/validation/test_spectra_remap_conservation.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/benchmark_spectra.py tests/unit/test_benchmark_spectra_script.py docs/validation/spectra-performance.json docs/validation/spectra-performance.md docs/60-validation/index.md
git commit -m "refactor: migrate spectral evidence envelope"
```

### Task 6: Index source provenance and atmosphere policy artifacts

**Files:**
- Create: `src/jaxstro/evidence/index.py`
- Create: `scripts/build_evidence_index.py`
- Create: `tests/unit/test_evidence_index.py`
- Create: `docs/validation/evidence-index.json`
- Create: `docs/60-validation/evidence-index.md`
- Modify: `docs/myst.yml`
- Modify: `scripts/check.sh`, `scripts/check_docs.sh`

**Interfaces:**
- Produces: `EvidenceIndexEntry`, `build_evidence_index`, and strict `--emit|--check` artifacts mapping stable evidence IDs to computational envelopes, source-card registries, and atmosphere-policy artifacts.

- [ ] **Step 1: Write failing index-resolution tests**

Require unique IDs, existing targets, artifact/schema identity, evidence class, source revision/content digest, and explicit optional-data policy. A source provenance card is indexed as `source_provenance`, not rewritten as a computational envelope. Atmosphere interpolation remains `scientific_policy` with its existing selection rule and thresholds.

- [ ] **Step 2: Implement and generate the index**

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_evidence_index.py --emit
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_evidence_index.py --check
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_evidence_index.py tests/validation/provenance_cards/test_registry.py tests/validation/test_atmosphere_holdouts.py
```

- [ ] **Step 3: Wire freshness and navigation, then commit**

```bash
git add src/jaxstro/evidence/index.py scripts/build_evidence_index.py tests/unit/test_evidence_index.py docs/validation/evidence-index.json docs/60-validation/evidence-index.md docs/myst.yml scripts/check.sh scripts/check_docs.sh
git commit -m "feat: publish unified scientific evidence index"
```

### Task 7: Link contracts to indexed evidence

**Files:**
- Modify: `src/jaxstro/contracts/registry.py`
- Modify: `scripts/build_contract_registry.py`
- Modify: `tests/unit/test_contract_registry.py`
- Modify: `tests/integration/test_contract_docs.py`
- Regenerate: `docs/validation/contracts.json`, `docs/40-api/contracts.md`

**Interfaces:**
- Consumes: `docs/validation/evidence-index.json` in explicit repository audit mode.
- Produces: fail-closed validation that every artifact evidence ID resolves to the correct class and target.

- [ ] **Step 1: Write failing missing/wrong-class tests**

Create fixtures where an evidence ID is absent and where a benchmark claim points to source provenance. Both must fail with stable messages.

- [ ] **Step 2: Implement explicit index-aware audit**

Lightweight installed queries remain import- and filesystem-light. The builder's repository audit loads the evidence index, resolves targets, and annotates generated links. Do not make ordinary `collect_contracts()` depend on committed docs.

- [ ] **Step 3: Regenerate and verify**

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_contract_registry.py --emit
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_contract_registry.py --check
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_contract_registry.py tests/integration/test_contract_docs.py
git add src/jaxstro/contracts/registry.py scripts/build_contract_registry.py tests/unit/test_contract_registry.py tests/integration/test_contract_docs.py docs/validation/contracts.json docs/40-api/contracts.md
git commit -m "feat: resolve contracts through evidence index"
```

### Checkpoint C3: completion reviews

Use two focused reviewers: one for scientific-threshold preservation and evidence-class honesty; one for public API, determinism, downstream portability, and pedagogy. Resolve all Critical and Important findings.

### Task 8: Close Phase C and prepare Phase B

**Files:**
- Modify: `docs/90-development-log/package-assessment-scorecard.md`
- Modify: `docs/90-development-log/sota-assessment.md`
- Modify: `STATUS.md`
- Modify: `tests/integration/test_assessment_scorecard.py`

- [ ] **Step 1: Add a failing closeout contract**

Require `Unified evidence infrastructure: implemented`, artifact counts derived from the index, an explicit statement that method thresholds remain method-owned, and Phase B as the single next action.

- [ ] **Step 2: Update conservatively and verify**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_evidence_*.py tests/unit/test_contract_*.py tests/integration/test_contract_*.py tests/integration/test_assessment_scorecard.py
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_evidence_index.py --check
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_contract_registry.py --check
env -u VIRTUAL_ENV uv run --no-sync ruff check --no-fix src tests scripts
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
bash scripts/check_docs.sh
```

- [ ] **Step 3: Commit**

```bash
git add STATUS.md docs/90-development-log/package-assessment-scorecard.md docs/90-development-log/sota-assessment.md tests/integration/test_assessment_scorecard.py
git commit -m "docs: close unified evidence infrastructure phase"
```

## Completion criteria

- Shared schemas validate units, finiteness, comparisons, environment policy, and limitations.
- Emit/check and JSON/Markdown rendering are deterministic and portable.
- Root, implicit-root, and spectra artifacts preserve all prior metrics and thresholds.
- Source provenance and atmosphere policy remain distinct evidence classes.
- A generated evidence index resolves contract claims fail closed.
- Ordinary metadata queries remain lightweight; runtime/filesystem audits are explicit tooling operations.
- All Critical and Important review findings are resolved; Phase B remains separate.
