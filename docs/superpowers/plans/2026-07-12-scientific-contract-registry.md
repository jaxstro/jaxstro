# Scientific Contract Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tiered public registry that makes Jaxstro's ownership, JAX transforms, AD semantics, boundaries, maturity, limitations, and supporting evidence machine-readable and generates the corresponding website matrices.

**Architecture:** Frozen dependency-light records and fail-closed validation live in `jaxstro.contracts`. Public subpackages keep small `_contracts.py` manifests beside their owners; single-file core modules use one explicit core manifest to avoid runtime circular imports. A deterministic collector validates imports and evidence links, emits JSON and MyST, and reports missing callable records as unclassified rather than supported.

**Tech Stack:** Python 3.11+, frozen dataclasses, string-backed enums, importlib, JSON, pytest, Ruff, MyPy, MyST.

## Global Constraints

- Work inline in `/Users/anna/projects/jaxstro-dev/jaxstro`; do not create a worktree.
- Preserve existing and untracked changes; never reset, clean, or overwrite unrelated work.
- Use `env -u VIRTUAL_ENV uv run --no-sync` for every Python, pytest, Ruff, and MyPy command.
- Write failing tests before implementation and commit each coherent task.
- Use targeted independent reviewers only after checkpoints A0, A1, and A3.
- Add no runtime dependency, downstream import, network access, dataset load, decorator, or JAX tracing behavior.
- Every public module gets a module contract; missing callable records remain explicitly unclassified.
- Keep unsupported, conditional, validation-only, and unverified distinct.
- Report measured numerical results only in tables with metric identity, symbol, value, and units.
- Phase C evidence migration and Phase B curriculum implementation require later plans.

---

## File map

New installed files:

- `src/jaxstro/contracts/{__init__,schema,registry,render,_core}.py`
- `src/jaxstro/{atmospheres,numerics,params,quantity,spatial,spectra,testing}/_contracts.py`

New tooling, docs, and tests:

- `scripts/build_contract_registry.py`
- `docs/validation/contracts.json`
- `docs/40-api/contracts.md`
- `docs/90-development-log/package-assessment-scorecard.md`
- `tests/unit/test_contract_{schema,registry,manifests}.py`
- `tests/unit/test_build_contract_registry_script.py`
- `tests/integration/test_contract_{rootfinding,exemplars,docs}.py`
- `tests/integration/test_{assessment_scorecard,agent_guidance}.py`

Existing files modified:

- `src/jaxstro/__init__.py`
- `CLAUDE.md`, `STATUS.md`
- `docs/myst.yml`, `docs/40-api/index.md`, `docs/60-validation/index.md`
- `docs/90-development-log/sota-assessment.md`
- `scripts/check.sh`, `scripts/check_docs.sh`

---

### Task 1: Add the living assessment scorecard

**Files:**
- Create: `tests/integration/test_assessment_scorecard.py`
- Create: `docs/90-development-log/package-assessment-scorecard.md`
- Modify: `docs/myst.yml`
- Modify: `docs/90-development-log/sota-assessment.md`

**Interfaces:**
- Consumes: approved grades and promotion policy from the design specification.
- Produces: `/package-assessment-scorecard` and a stable editorial rubric.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORECARD = ROOT / "docs/90-development-log/package-assessment-scorecard.md"


def test_scorecard_has_grades_evidence_and_promotion_rules() -> None:
    text = SCORECARD.read_text(encoding="utf-8")
    for phrase in (
        "# Jaxstro package assessment scorecard",
        "## Grading rubric",
        "## Current grades",
        "## Coverage by scientific area",
        "## Grade-change policy",
        "Deficiency preventing the next grade",
        "Promotion evidence required",
        "Scientific contract registry",
    ):
        assert phrase in text


def test_scorecard_is_navigable_and_linked_from_sota() -> None:
    myst = (ROOT / "docs/myst.yml").read_text(encoding="utf-8")
    sota = (ROOT / "docs/90-development-log/sota-assessment.md").read_text(encoding="utf-8")
    assert myst.count("90-development-log/package-assessment-scorecard.md") == 1
    assert "package-assessment-scorecard.md" in sota
```

- [ ] **Step 2: Verify the missing-file failure**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_assessment_scorecard.py`

Expected: FAIL because the scorecard does not exist.

- [ ] **Step 3: Write the scorecard and navigation**

Copy the approved assessment into a durable scorecard. For every dimension include current grade, dated rationale, evidence, deficiency preventing the next grade, and promotion evidence. Include this exact policy:

```md
## Grade-change policy

A grade does not improve merely because a feature or documentation claim lands.
Promotion requires the relevant scientific contract, independent validation,
limitation statement, reproducible artifact where metrics matter, and downstream
adoption evidence where reuse is the justification.
```

Add the page once under Development log and link it from the SOTA roadmap.

- [ ] **Step 4: Run focused tests**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_assessment_scorecard.py tests/integration/test_sota_assessment.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/90-development-log/package-assessment-scorecard.md docs/90-development-log/sota-assessment.md docs/myst.yml tests/integration/test_assessment_scorecard.py
git commit -m "docs: add living package assessment scorecard"
```

### Task 2: Make `CLAUDE.md` current and bounded

**Files:**
- Create: `tests/integration/test_agent_guidance.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: live exports, `docs/10-theory/rootfinding.md`, and current project commands.
- Produces: active guidance without historical task logs or obsolete AD advice.

- [ ] **Step 1: Write failing currency tests**

```python
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[2] / "CLAUDE.md"


def test_guide_names_current_architecture_and_derivative_targets() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    for module in ("atmospheres", "numerics", "params", "provenance", "quantity", "spatial", "spectra", "testing"):
        assert f"`jaxstro.{module}`" in text
    assert "finite executed iteration" in text
    assert "certified implicit derivative" in text
    assert "Use `newton` / `newton_with_grad` / `newton_ppf`" not in text


def test_guide_is_not_a_historical_status_log() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    for stale in ("Phase B working decisions", "T7b", "feature/consolidate-harden-release"):
        assert stale not in text
```

- [ ] **Step 2: Verify stale-guide failures**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_agent_guidance.py`

Expected: FAIL on omissions, legacy derivative advice, or historical text.

- [ ] **Step 3: Rewrite the guide with these exact sections**

```md
# CLAUDE.md
## Purpose and ownership
## Read first
## Current package map
## Commands
## Units policy
## JAX and AD contracts
## Load-bearing numerical invariants
## Evidence and documentation
## Change discipline
```

State that Newton exposes sensitivity of its smooth finite executed iteration; value-first roots make no implicit claim; and `implicit_bracketed_root` exposes a certified implicit derivative only when every gate passes. Retain the dx-outside trapezoid, probabilists' Hermite, singular condition-number sentinel, and explicit-units invariants. Link historical decisions rather than copying task logs.

- [ ] **Step 4: Run currency tests**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_agent_guidance.py tests/integration/test_architecture_docs.py tests/integration/test_rootfinding_docs.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md tests/integration/test_agent_guidance.py
git commit -m "docs: make agent guidance current"
```

### Checkpoint A0: current-truth review

Request one targeted documentation/API reviewer. Address every Critical or Important finding, rerun Tasks 1–2 tests, and commit fixes before Task 3.

### Task 3: Define vocabulary and frozen schemas

**Files:**
- Create: `src/jaxstro/contracts/schema.py`
- Create: `src/jaxstro/contracts/__init__.py`
- Create: `tests/unit/test_contract_schema.py`

**Interfaces:**
- Produces: `MaturityLevel`, `SupportLevel`, `ADSemantics`, `EvidenceKind`, `ExecutionBoundary`, `EvidenceReference`, `TransformContract`, `BoundaryContract`, `CallableContract`, `ModuleContract`, and `ContractInventory`.

- [ ] **Step 1: Write failing schema tests**

```python
import dataclasses
import pytest
from jaxstro.contracts import ADSemantics, CallableContract, EvidenceKind, EvidenceReference, MaturityLevel, SupportLevel, TransformContract


def test_callable_contract_is_frozen_and_uses_validated_vocabulary() -> None:
    evidence = EvidenceReference("root.value.quadratic", EvidenceKind.VALIDATION_TEST, "tests/validation/test_bracketed_root_algorithms.py", "analytic quadratic root")
    contract = CallableContract(
        id="numerics.safeguarded_bracketed_root",
        import_path="jaxstro.numerics.safeguarded_bracketed_root",
        purpose="Auditable value-first scalar root solve.",
        domain="Finite scalar endpoints and residuals.",
        outputs="BracketedRootResult",
        transforms=(TransformContract("jit", SupportLevel.SUPPORTED, evidence_ids=(evidence.id,)),),
        ad_semantics=ADSemantics.VALUE_FIRST,
        precision="float32 and float64",
        maturity=MaturityLevel.VALIDATED,
        evidence=(evidence,),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.purpose = "changed"  # type: ignore[misc]
    assert SupportLevel.UNVERIFIED is not SupportLevel.UNSUPPORTED
```

- [ ] **Step 2: Verify imports fail**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_contract_schema.py`

Expected: FAIL because `jaxstro.contracts` does not exist.

- [ ] **Step 3: Implement minimal frozen records**

Use `class Name(str, Enum)` for approved vocabulary and `@dataclass(frozen=True)` for records. Tuple fields default to `()`. Callable records include limitations, cost notes, and boundaries. Module records include ownership, non-ownership, intended uses, execution boundary, dimensional policy, maturity, callables, and evidence. Inventory records schema version, package version, source revision, and modules.

- [ ] **Step 4: Run tests and static gates**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_contract_schema.py
env -u VIRTUAL_ENV uv run --no-sync ruff check src/jaxstro/contracts tests/unit/test_contract_schema.py
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro/contracts
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaxstro/contracts tests/unit/test_contract_schema.py
git commit -m "feat: define scientific contract schemas"
```

### Task 4: Add fail-closed collection and deterministic rendering

**Files:**
- Create: `src/jaxstro/contracts/registry.py`
- Create: `src/jaxstro/contracts/render.py`
- Create: `tests/unit/test_contract_registry.py`

**Interfaces:**
- Produces: `validate_inventory(inventory) -> None`, `resolve_import_path(path) -> object`, `inventory_to_dict(inventory) -> dict[str, object]`, `inventory_to_json(inventory) -> str`, and `render_contract_reference(inventory) -> str`.

- [ ] **Step 1: Write failing validation tests**

```python
import pytest
from jaxstro.contracts import (
    ADSemantics,
    CallableContract,
    ContractInventory,
    ExecutionBoundary,
    MaturityLevel,
    ModuleContract,
)
from jaxstro.contracts.registry import resolve_import_path, validate_inventory
from jaxstro.contracts.render import inventory_to_json


def _inventory(*callables: CallableContract) -> ContractInventory:
    module = ModuleContract(
        id="numerics",
        import_path="jaxstro.numerics",
        ownership="Generic numerical primitives.",
        non_ownership="Domain acceptance and physical state.",
        intended_uses=("Differentiable scientific computing",),
        execution_boundary=ExecutionBoundary.RUNTIME,
        dimensional_policy="Caller-owned units.",
        maturity=MaturityLevel.VALIDATED,
        callables=callables,
    )
    return ContractInventory("1", "0.1.0", "test", (module,))


def _callable(identifier: str, path: str) -> CallableContract:
    return CallableContract(
        id=identifier,
        import_path=path,
        purpose="Test fixture.",
        domain="Finite scalar inputs.",
        outputs="Scalar output.",
        ad_semantics=ADSemantics.UNVERIFIED,
        precision="Unverified.",
        maturity=MaturityLevel.IMPLEMENTED,
    )


def test_import_resolution_fails_closed() -> None:
    assert callable(resolve_import_path("jaxstro.numerics.bisect"))
    with pytest.raises(ValueError, match="cannot resolve"):
        resolve_import_path("jaxstro.numerics.not_a_symbol")


def test_registry_rejects_duplicate_callable_ids() -> None:
    inventory = _inventory(
        _callable("duplicate", "jaxstro.numerics.bisect"),
        _callable("duplicate", "jaxstro.numerics.newton"),
    )
    with pytest.raises(ValueError, match="duplicate contract id"):
        validate_inventory(inventory)


def test_json_is_deterministic_and_portable() -> None:
    inventory = _inventory(_callable("bisect", "jaxstro.numerics.bisect"))
    validate_inventory(inventory)
    first = inventory_to_json(inventory)
    assert first == inventory_to_json(inventory)
    assert "/Users/" not in first
```

Extend the same explicit constructors with tests showing that duplicate module and
evidence IDs fail, transform evidence IDs must resolve, and supported or
conditional transforms require linked evidence plus conditions where applicable.

- [ ] **Step 2: Verify missing implementations**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_contract_registry.py`

Expected: FAIL because registry/render functions do not exist.

- [ ] **Step 3: Implement validation and rendering**

Resolve dotted paths by importing the longest valid module prefix and traversing attributes. Reject duplicate IDs, unresolved paths, empty purpose/ownership, and dangling evidence. Sort modules, callables, transforms, and evidence before rendering; render enum values as strings and append one newline.

- [ ] **Step 4: Run focused gates**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_contract_schema.py tests/unit/test_contract_registry.py
env -u VIRTUAL_ENV uv run --no-sync ruff check src/jaxstro/contracts tests/unit/test_contract_*.py
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro/contracts
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaxstro/contracts tests/unit/test_contract_registry.py
git commit -m "feat: validate and render scientific contracts"
```

### Task 5: Register every public module without optional imports

**Files:**
- Create: `src/jaxstro/contracts/_core.py`
- Create: `src/jaxstro/{atmospheres,numerics,params,quantity,spatial,spectra,testing}/_contracts.py`
- Modify: `src/jaxstro/contracts/{registry,__init__}.py`
- Modify: `src/jaxstro/__init__.py`
- Create: `tests/unit/test_contract_manifests.py`

**Interfaces:**
- Produces: `collect_contracts(source_revision: str = "unknown") -> ContractInventory` and `get_module_contract(import_path: str) -> ModuleContract`.

- [ ] **Step 1: Write failing coverage and isolation tests**

```python
import subprocess
import sys
import jaxstro
from jaxstro.contracts import collect_contracts

PUBLIC = {f"jaxstro.{name}" for name in ("astrometry", "atmospheres", "constants", "coords", "geometry", "jaxconfig", "numerics", "params", "provenance", "quantity", "spatial", "spectra", "testing", "units")}


def test_every_public_module_has_one_contract() -> None:
    inventory = collect_contracts(source_revision="test")
    assert {record.import_path for record in inventory.modules} == PUBLIC


def test_collection_does_not_import_optional_packages() -> None:
    code = "from jaxstro.contracts import collect_contracts; import sys; collect_contracts(); assert all(x not in sys.modules for x in ('polars','numpyro','optax'))"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_contracts_is_public() -> None:
    assert jaxstro.contracts.__name__ == "jaxstro.contracts"
    assert "contracts" in jaxstro.__all__
```

- [ ] **Step 2: Verify coverage fails**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_contract_manifests.py`

Expected: FAIL because manifests and collection are absent.

- [ ] **Step 3: Add explicit module records**

Record current boundaries: `units` is canonical; `quantity` adoption is deferred; numerics owns generic mechanics but no domain acceptance; spatial separates discrete preprocessing from evaluation; spectra excludes filters/photometry; atmospheres separates host artifacts from evidence-gated evaluation; params is not an inference framework; testing does not own scientific acceptance. The collector imports only manifest modules.

- [ ] **Step 4: Run module and import gates**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_contract_manifests.py tests/integration/test_api_reference.py tests/integration/test_architecture_docs.py
env -u VIRTUAL_ENV uv run --no-sync ruff check src/jaxstro/contracts src/jaxstro/*/_contracts.py tests/unit/test_contract_manifests.py
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaxstro tests/unit/test_contract_manifests.py
git commit -m "feat: register public module contracts"
```

### Checkpoint A1: schema and architecture review

Request one targeted API/architecture reviewer. Address Critical and Important findings about schema minimality, imports, naming, failure behavior, and ownership. Commit fixes before callable records.

### Task 6: Register rootfinding callable contracts

**Files:**
- Modify: `src/jaxstro/numerics/_contracts.py`
- Modify: `tests/unit/test_contract_manifests.py`
- Create: `tests/integration/test_contract_rootfinding.py`

**Interfaces:**
- Produces: `get_callable_contract(import_path: str) -> CallableContract` and records for safeguarded/map/implicit root solvers plus initialize/update/propose primitives.

- [ ] **Step 1: Write failing semantic tests**

```python
from jaxstro.contracts import ADSemantics, SupportLevel, get_callable_contract


def test_value_root_separates_transform_and_cost_claims() -> None:
    record = get_callable_contract("jaxstro.numerics.safeguarded_bracketed_root")
    transforms = {item.transform: item for item in record.transforms}
    assert record.ad_semantics is ADSemantics.VALUE_FIRST
    assert transforms["jit"].support is SupportLevel.SUPPORTED
    assert transforms["vmap"].support is SupportLevel.CONDITIONAL
    assert "physical per-lane skipping" in transforms["vmap"].conditions
    assert "lax.map" in record.cost_notes


def test_implicit_root_requires_certification() -> None:
    record = get_callable_contract("jaxstro.numerics.implicit_bracketed_root")
    assert record.ad_semantics is ADSemantics.CERTIFIED_IMPLICIT
    assert any("unique" in item.lower() for item in record.limitations)
```

- [ ] **Step 2: Verify lookup fails**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_contract_rootfinding.py`

Expected: FAIL because callable records are absent.

- [ ] **Step 3: Add records using existing root evidence**

Link stable tests in `test_bracketed_root.py`, `test_bracketed_root_algorithms.py`, `test_implicit_root.py`, and `test_implicit_root_gradients.py`. Encode `valid=False`, exact roots, missing brackets, exhaustion, trace shape, scalar conditional skipping, VMAP cost caveat, and fail-closed implicit certification.

- [ ] **Step 4: Run root gates**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_contract_rootfinding.py tests/unit/test_bracketed_root.py tests/unit/test_implicit_root.py tests/validation/test_bracketed_root_algorithms.py tests/validation/test_implicit_root_gradients.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/jaxstro/numerics/_contracts.py tests/unit/test_contract_manifests.py tests/integration/test_contract_rootfinding.py
git commit -m "feat: register rootfinding contracts"
```

### Task 7: Register distribution, interpolation, and evidence exemplars

**Files:**
- Modify: `src/jaxstro/numerics/_contracts.py`
- Modify: `src/jaxstro/testing/_contracts.py`
- Create: `tests/integration/test_contract_exemplars.py`

**Interfaces:**
- Produces: records for finite power-law normalization/logpdf/CDF/PPF, interpolation APIs, gradient audits, and provenance-card entry points.

- [ ] **Step 1: Write failing exemplar tests**

```python
from jaxstro.contracts import ADSemantics, get_callable_contract


def test_powerlaw_names_removable_limit_and_support() -> None:
    record = get_callable_contract("jaxstro.numerics.powerlaw_cdf")
    assert record.ad_semantics is ADSemantics.SMOOTH_PATHWISE
    assert "alpha=-1" in record.domain.replace(" ", "")
    assert any("support" in item.summary.lower() for item in record.boundaries)


def test_regular_grid_keeps_boundary_policies_distinct() -> None:
    record = get_callable_contract("jaxstro.numerics.regular_grid_interp")
    assert any("clamp" in item.summary for item in record.boundaries)
    assert any("reject" in item.summary for item in record.boundaries)


def test_gradient_audit_is_validation_only() -> None:
    record = get_callable_contract("jaxstro.testing.compare_gradients")
    assert record.ad_semantics is ADSemantics.VALIDATION_ONLY
```

- [ ] **Step 2: Verify missing-record failures**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_contract_exemplars.py`

Expected: FAIL.

- [ ] **Step 3: Add records with analytic and FD evidence**

Use distribution, interpolation, regular-grid, gradient-audit, and provenance registry tests. Preserve distinct clamp/fill/reject behavior. Do not infer any transform without linked evidence.

- [ ] **Step 4: Run exemplar gates**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_contract_exemplars.py tests/unit/test_distributions.py tests/unit/test_interpolation_shape_preserving.py tests/unit/test_regular_grid.py tests/integration/test_grad_audit.py tests/validation/provenance_cards/test_registry.py`

Expected: PASS or existing optional-data skips only.

- [ ] **Step 5: Commit**

```bash
git add src/jaxstro/numerics/_contracts.py src/jaxstro/testing/_contracts.py tests/integration/test_contract_exemplars.py
git commit -m "feat: register numerical contract exemplars"
```

### Task 8: Generate and gate the public inventory

**Files:**
- Create: `scripts/build_contract_registry.py`
- Create: `tests/unit/test_build_contract_registry_script.py`
- Create: `tests/integration/test_contract_docs.py`
- Create: `docs/validation/contracts.json`
- Create: `docs/40-api/contracts.md`
- Modify: `docs/myst.yml`, `docs/40-api/index.md`, `docs/60-validation/index.md`
- Modify: `scripts/check.sh`, `scripts/check_docs.sh`

**Interfaces:**
- Produces: `python scripts/build_contract_registry.py --emit|--check`, normalized JSON, and `/contracts`.

- [ ] **Step 1: Write failing freshness and docs tests**

```python
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_contract_artifacts_are_fresh() -> None:
    subprocess.run([sys.executable, "scripts/build_contract_registry.py", "--check"], cwd=ROOT, check=True)
```

```python
def test_generated_page_explains_unverified() -> None:
    page = (ROOT / "docs/40-api/contracts.md").read_text(encoding="utf-8")
    assert "# Scientific contract registry" in page
    assert "Unverified does not mean unsupported" in page
    assert "## Transform and AD contracts" in page
    assert "## Unclassified callable surfaces" in page
```

- [ ] **Step 2: Verify missing builder/artifacts**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_build_contract_registry_script.py tests/integration/test_contract_docs.py`

Expected: FAIL.

- [ ] **Step 3: Implement strict emit/check**

Require exactly one mode, validate the inventory, and compare JSON and MyST
byte-for-byte in check mode. The committed inventory uses
`source_revision="repository-versioned"`: its exact source revision is the git
commit containing it, avoiding an impossible self-referential commit hash that
would make every post-commit freshness check stale. Render ownership, transforms,
AD, evidence, limitations, and unclassified coverage.

- [ ] **Step 4: Emit and wire gates**

Run: `env -u VIRTUAL_ENV uv run --no-sync python scripts/build_contract_registry.py --emit`

Add the corresponding `--check` command to `scripts/check.sh` and `scripts/check_docs.sh`. Add the page once under API Reference and route duplicated transform claims to it without removing signatures or ownership context.

- [ ] **Step 5: Run Phase A focused verification**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_contract_*.py tests/unit/test_build_contract_registry_script.py tests/integration/test_contract_*.py tests/integration/test_assessment_scorecard.py tests/integration/test_agent_guidance.py tests/integration/test_api_reference.py tests/integration/test_architecture_docs.py tests/integration/test_validation_docs.py
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_contract_registry.py --check
env -u VIRTUAL_ENV uv run --no-sync ruff check src/jaxstro/contracts src/jaxstro/*/_contracts.py scripts/build_contract_registry.py tests/unit/test_contract_*.py tests/integration/test_contract_*.py
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
bash scripts/check_docs.sh
```

Expected: focused tests, freshness, Ruff, MyPy, and docs gates pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_contract_registry.py scripts/check.sh scripts/check_docs.sh docs/validation/contracts.json docs/40-api/contracts.md docs/40-api/index.md docs/60-validation/index.md docs/myst.yml tests/unit/test_build_contract_registry_script.py tests/integration/test_contract_docs.py
git commit -m "docs: publish scientific contract registry"
```

### Checkpoint A3: completion reviews

Use targeted reviewers conservatively for schema/API cleanliness, JAX/AD honesty, evidence/docs pedagogy, and downstream suitability. Address every Critical and Important finding, then repeat Task 8 Step 5.

### Task 9: Close Phase A without overclaiming

**Files:**
- Modify: `docs/90-development-log/package-assessment-scorecard.md`
- Modify: `docs/90-development-log/sota-assessment.md`
- Modify: `STATUS.md`
- Modify: `tests/integration/test_assessment_scorecard.py`

**Interfaces:**
- Produces: honest Phase A status and Phase C as the single next action.

- [ ] **Step 1: Add a failing closeout assertion**

```python
def test_scorecard_separates_registry_delivery_from_uniform_evidence() -> None:
    text = SCORECARD.read_text(encoding="utf-8")
    assert "Scientific contract registry: implemented" in text
    assert "Evidence depth remains uneven" in text
    assert "Phase C" in text
```

- [ ] **Step 2: Verify it fails before the status update**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_assessment_scorecard.py`

Expected: FAIL.

- [ ] **Step 3: Update scorecard, roadmap, and status**

Record registry delivery without promoting grades that require unified evidence or adoption. Replace the active `STATUS.md` `next:` with writing and executing the Phase C plan; do not accumulate another active instruction.

- [ ] **Step 4: Run final bounded gate**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/test_contract_*.py tests/integration/test_contract_*.py tests/integration/test_assessment_scorecard.py tests/integration/test_agent_guidance.py
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_contract_registry.py --check
env -u VIRTUAL_ENV uv run --no-sync ruff check src tests scripts/build_contract_registry.py
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
git diff --check
git status --short
```

Expected: all gates pass and only closeout files remain modified.

- [ ] **Step 5: Commit**

```bash
git add STATUS.md docs/90-development-log/package-assessment-scorecard.md docs/90-development-log/sota-assessment.md tests/integration/test_assessment_scorecard.py
git commit -m "docs: close scientific contract registry phase"
```

## Phase A completion criteria

- Every current public module has one validated module contract.
- Rootfinding, finite power-law distributions, interpolation, and evidence tooling have callable records.
- Supported claims link to evidence; absent claims remain unverified or unclassified.
- Collection imports no optional data, ML, or downstream package.
- JSON and MyST are deterministic and freshness-gated.
- The scorecard and current `CLAUDE.md` are published and tested.
- All checkpoint Critical and Important findings are resolved.
- Phase C remains pending; no curriculum or evidence-envelope implementation is smuggled into Phase A.
