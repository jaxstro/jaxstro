# Jaxstro.quad Phase A4 Performance Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a fair, reproducible Jaxstro-versus-Quadax one-dimensional quadrature comparison, harden or optimize only measured deficiencies, publish the evidence on the MyST website, and close the Phase A release gate.

**Architecture:** A benchmark-only case catalog and two thin library adapters feed one deterministic accuracy/work evaluator and one explicitly synchronized timing evaluator. The CLI emits a shared `EvidenceArtifact`; CI checks deterministic numerical evidence but never gates noisy wall time. Runtime optimization is a measured decision checkpoint: if a trigger fires, write and execute a focused evidence-derived addendum before touching the numerical engine.

**Tech Stack:** Python 3.11+, JAX 0.10.1, Jaxstro `quad`, Quadax 0.2.13 in a PEP 735 benchmark-only dependency group, pytest, Ruff, MyPy, MyST.

## Global Constraints

- Preserve all Phase A3 value, status, work, replay-AD, quantity, PyTree, and compatibility contracts.
- Pin `quadax==0.2.13` only in `[dependency-groups].benchmark`; never add it to runtime dependencies or public extras.
- Use the approved family-matched and practical best-method lanes.
- Use float64 as the primary scientific lane and a bounded dtype-aware float32 lane.
- Emit the authoritative CPU baseline from a clean committed tree and record complete environment provenance.
- Follow the official JAX timing protocol: outer JIT, device-resident inputs, separate compile timing, explicit synchronization, and at least 21 interleaved warm repetitions.
- Treat correctness and false convergence as release blockers regardless of speed.
- Optimize only when an approved trigger fires; do not pre-emptively rewrite a controller.
- Keep timing informational; freshness checks cover deterministic configuration, accuracy, status, work, and schema.
- Use research-software language, LaTeX mathematics, supported MyST elements, and no course framing.
- Do not add multidimensional, sparse-grid, QMC, oscillatory-specialist, or other new runtime methods in Phase A4.
- Do not modify sibling repositories, publish, push, or change the live website in this plan.

---

## File structure

- `scripts/quad_benchmark_cases.py`: immutable case definitions, analytic truths, method suitability, and fairness labels.
- `scripts/quad_benchmark_adapters.py`: thin Jaxstro and Quadax execution adapters with normalized result records.
- `scripts/quad_benchmark_timing.py`: compile, warm, VMAP, JVP, and supported reverse-mode timing with synchronization.
- `scripts/benchmark_quad.py`: CLI, deterministic evaluation, artifact validation, rendering, `--emit`, and `--check`.
- `tests/unit/test_benchmark_quad_cases.py`: catalog truth and predeclared method-choice tests.
- `tests/unit/test_benchmark_quad_adapters.py`: cross-library normalization and semantic-difference tests.
- `tests/unit/test_benchmark_quad_timing.py`: timing protocol tests using small JAX callables.
- `tests/unit/test_benchmark_quad_script.py`: artifact schema, deterministic freshness, CLI, and dependency-boundary tests.
- `docs/validation/quad-performance.json`: emitted machine-readable evidence.
- `docs/60-validation/numerical/quadrature-performance.md`: emitted researcher-facing report.

### Task 1: Close and commit the Phase A3 review fixes

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `STATUS.md`
- Modify: `docs/20-methods/approximation-integration/adaptive-quadrature.md`
- Modify: `docs/20-methods/approximation-integration/differentiating-an-integral.md`
- Modify: `docs/50-api/approximation-integration/quad.md`
- Modify: `docs/50-api/research-infrastructure/contracts.md`
- Modify: `docs/superpowers/specs/2026-07-15-jaxstro-quad-phase-a3-replay-quantity-design.md`
- Modify: `docs/validation/contracts.json`
- Modify: `src/jaxstro/quad/_adaptive.py`
- Modify: `src/jaxstro/quad/_contracts.py`
- Modify: `src/jaxstro/quad/_quantity.py`
- Modify: `src/jaxstro/quad/adaptive.py`
- Modify: `src/jaxstro/quad/domains.py`
- Modify: `src/jaxstro/quad/transforms.py`
- Modify: `tests/integration/test_quad_quantity_transforms.py`
- Modify: `tests/unit/quad/test_adaptive_substrate.py`
- Modify: `tests/unit/quad/test_adaptive_tanh_sinh.py`
- Modify: `tests/unit/quad/test_domains.py`
- Modify: `tests/unit/quad/test_integrate_gk.py`
- Modify: `tests/unit/test_contract_manifests.py`
- Modify: `tests/unit/test_quad_quantity.py`

**Interfaces:**
- Consumes: the already implemented explicit improper-domain `scale` contract and review regressions.
- Produces: a committed, verified Phase A3 baseline on which authoritative benchmark provenance can depend.

- [ ] **Step 0: Verify the approved design and executable plan are committed**

Run:

```bash
git log -1 --oneline -- docs/superpowers/specs/2026-07-16-jaxstro-quad-phase-a4-performance-design.md
git log -1 --oneline -- docs/superpowers/plans/2026-07-16-jaxstro-quad-phase-a4-performance.md
git status --short
```

Expected: both documents resolve to commits; status contains only the approved
Phase A3 review-fix files listed in this task.

- [ ] **Step 1: Re-run the focused corrected gate**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync pytest \
  tests/unit/quad \
  tests/unit/test_quad_quantity.py \
  tests/unit/test_quad_replay_substrate.py \
  tests/integration/test_quad_adaptive_transforms.py \
  tests/integration/test_quad_compatibility.py \
  tests/integration/test_quad_fixed_transforms.py \
  tests/integration/test_quad_quantity_transforms.py \
  tests/integration/test_quad_replay_artifact.py \
  tests/integration/test_quad_replay_docs.py \
  tests/integration/test_quad_replay_transforms.py \
  tests/validation/test_quad_adaptive_reference.py \
  tests/validation/test_quad_fixed_reference.py \
  tests/validation/test_quad_gk_tables.py \
  tests/validation/test_quad_replay_derivatives.py \
  tests/unit/test_contract_manifests.py -q
```

Expected: all tests pass; the previously measured count is 443.

- [ ] **Step 2: Re-run static and artifact checks**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync ruff format --check src/jaxstro/quad tests/unit/quad tests/unit/test_quad_quantity.py tests/integration/test_quad_quantity_transforms.py
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync ruff check src/jaxstro/quad tests/unit/quad tests/unit/test_quad_quantity.py tests/integration/test_quad_quantity_transforms.py
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync mypy src/jaxstro/quad
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync python scripts/build_contract_registry.py --check
git diff --check
```

Expected: all commands exit zero and the contract script prints `scientific contract artifacts fresh`.

- [ ] **Step 3: Commit the Phase A3 review fixes without including the Phase A4 plan**

Run:

```bash
git add CHANGELOG.md STATUS.md docs/20-methods/approximation-integration docs/50-api/approximation-integration/quad.md docs/50-api/research-infrastructure/contracts.md docs/superpowers/specs/2026-07-15-jaxstro-quad-phase-a3-replay-quantity-design.md docs/validation/contracts.json src/jaxstro/quad tests/integration/test_quad_quantity_transforms.py tests/unit/quad tests/unit/test_contract_manifests.py tests/unit/test_quad_quantity.py
git commit -m "fix(quad): make improper quantity maps unit invariant"
```

Expected: one coherent review-fix commit and a clean status because the Phase A4 design and plan are committed separately before execution.

### Task 2: Add the benchmark-only dependency boundary and catalog

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `scripts/quad_benchmark_cases.py`
- Create: `tests/unit/test_benchmark_quad_cases.py`

**Interfaces:**
- Consumes: `jaxstro.quad` domains and methods.
- Produces: `ComparisonLabel`, `LibraryMethod`, `TruthProvenance`, `BenchmarkCase`, `MethodPair`, `BestMethodChoice`, `CASES`, `METHOD_PAIRS`, and `BEST_METHODS` for all later tasks.

- [ ] **Step 1: Write failing dependency and catalog tests**

Create tests that require the exact dependency boundary and immutable catalog:

```python
from __future__ import annotations

import math
import tomllib
from pathlib import Path

import jax.numpy as jnp

from scripts.quad_benchmark_cases import (
    BEST_METHODS,
    CASES,
    METHOD_PAIRS,
    ComparisonLabel,
    LibraryMethod,
)

ROOT = Path(__file__).resolve().parents[2]


def test_quadax_is_benchmark_only_and_exactly_pinned() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert payload["dependency-groups"]["benchmark"] == ["quadax==0.2.13"]
    assert all("quadax" not in item for item in payload["project"]["dependencies"])
    assert all(
        "quadax" not in item
        for extra in payload["project"]["optional-dependencies"].values()
        for item in extra
    )


def test_case_truths_match_analytic_definitions() -> None:
    by_name = {case.name: case for case in CASES}
    assert math.isclose(by_name["smooth_exponential"].truth, math.e - 1.0)
    assert math.isclose(by_name["endpoint_sqrt"].truth, 2.0 / 3.0)
    assert math.isclose(by_name["semi_infinite_exponential"].truth, 1.0)
    assert math.isclose(by_name["full_line_gaussian"].truth, math.sqrt(math.pi))
    value = by_name["vector_polynomial_exponential"].fun(jnp.asarray(0.5), ())
    assert value.shape == (2,)


def test_method_pairs_and_best_choices_are_predeclared() -> None:
    assert {pair.label for pair in METHOD_PAIRS} == {
        ComparisonLabel.EXACT,
        ComparisonLabel.STRONG_MATCH,
        ComparisonLabel.NODE_MATCHED,
        ComparisonLabel.FAMILY_MATCHED,
        ComparisonLabel.CAPABILITY,
    }
    assert set(BEST_METHODS) == {case.name for case in CASES}
    for choice in BEST_METHODS.values():
        assert isinstance(choice.jaxstro_method, LibraryMethod)
        assert isinstance(choice.quadax_method, LibraryMethod)
        assert choice.rationale
        assert choice.source


def test_every_case_has_portable_truth_provenance() -> None:
    for case in CASES:
        assert case.truth_provenance.kind in {"analytic", "reference"}
        assert case.truth_provenance.expression
        assert case.truth_provenance.source
        assert case.truth_provenance.atol > 0.0
        assert case.truth_provenance.rtol >= 0.0
    by_name = {case.name: case for case in CASES}
    for name in ("expensive_identity", "narrow_gaussian"):
        assert by_name[name].truth_provenance.kind == "reference"
        assert "NumPy Gauss-Legendre" in by_name[name].truth_provenance.source
        assert by_name[name].truth_provenance.reference_version
        assert by_name[name].truth_provenance.reference_orders == (256, 512, 1024)
```

- [ ] **Step 2: Run the catalog tests to verify they fail**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync pytest tests/unit/test_benchmark_quad_cases.py -q
```

Expected: import or missing benchmark-group failure.

- [ ] **Step 3: Add the exact benchmark dependency**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv add --group benchmark "quadax==0.2.13"
```

Then confirm:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark python -c "import quadax; print(quadax.__version__)"
```

Expected: `0.2.13`.

- [ ] **Step 4: Implement the immutable case and pair declarations**

Create these public internal records and exact enum values:

```python
class ComparisonLabel(str, Enum):
    EXACT = "exact"
    STRONG_MATCH = "strong_match"
    NODE_MATCHED = "node_matched"
    FAMILY_MATCHED = "family_matched"
    CAPABILITY = "capability"


class LibraryMethod(str, Enum):
    GAUSS_KRONROD = "gauss_kronrod"
    CLENSHAW_CURTIS = "clenshaw_curtis"
    TANH_SINH = "tanh_sinh"
    ROMBERG = "romberg"
    ROMBERG_TANH_SINH = "romberg_tanh_sinh"


@dataclass(frozen=True)
class TruthProvenance:
    kind: str
    expression: str
    source: str
    reference_version: str
    atol: float
    rtol: float
    reference_orders: tuple[int, ...] = ()
    reference_values: tuple[float, ...] = ()
    convergence_delta: float | None = None
    analytic_crosscheck: float | tuple[float, ...] | None = None


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    family: str
    fun: Callable[[jax.Array, tuple[Any, ...]], jax.Array]
    domain: Interval | RightInfinite | LeftInfinite | Infinite
    truth: float | tuple[float, ...] | None
    derivative_truth: float | tuple[float, ...] | None
    truth_provenance: TruthProvenance
    theta: float
    expected: str
    supported_methods: tuple[LibraryMethod, ...]


@dataclass(frozen=True)
class MethodPair:
    family: LibraryMethod
    variant: str
    label: ComparisonLabel
    jaxstro_config: tuple[tuple[str, str | int | float | bool], ...]
    quadax_config: tuple[tuple[str, str | int | float | bool], ...]
    note: str


@dataclass(frozen=True)
class BestMethodChoice:
    case: str
    jaxstro_method: LibraryMethod
    jaxstro_config: tuple[tuple[str, str | int | float | bool], ...]
    quadax_method: LibraryMethod
    quadax_config: tuple[tuple[str, str | int | float | bool], ...]
    rationale: str
    source: str
```

Define the 11 approved cases with exact analytic formulas or independent
references. Use `math.erf` for an analytic cross-check of the localized and
narrow Gaussian truths, `math.sin(50.0) / 50.0` for the oscillatory truth, and
`truth=None`, `expected="fail_closed"` for the nonfinite case. The expensive case evaluates eight shifted
`sin(x + k) ** 2 + cos(x + k) ** 2` identities multiplied by
`exp(-theta * x)` and therefore retains the analytic truth
`(1 - exp(-theta)) / theta`. Store the exact LaTeX expression and either
`analytic definition in this catalog` or the named independent reference
procedure in every `TruthProvenance`.

For `expensive_identity` and `narrow_gaussian`, compute independent reference
values on the host with `numpy.polynomial.legendre.leggauss` at orders 256, 512,
and 1024, using NumPy evaluation rather than either JAX library adapter. Require
the 512-versus-1024 difference to be at most `1.0e-13`, store the order-1024
value as benchmark truth, and require agreement with the available analytic
cross-check to `1.0e-12`. Record all three orders, values, convergence delta,
NumPy version, and cross-check expression in `TruthProvenance` and the artifact.

Define `BEST_METHODS` as `dict[str, BestMethodChoice]` with independent Jaxstro
and Quadax choices, configurations, rationale, and public/source basis for every
case. Return immutable records from catalog accessors so timing results cannot
mutate method selection.

Use this exact catalog; formulas are evaluated at the listed default `theta`:

| Name | Integrand and domain | Truth and derivative truth | Jaxstro / Quadax practical choice |
| --- | --- | --- | --- |
| `smooth_exponential` | $\exp(\theta x)$ on $[0,1]$, $\theta=1$ | $(e^\theta-1)/\theta$; $((\theta-1)e^\theta+1)/\theta^2$ | GK / GK |
| `vector_polynomial_exponential` | $[x^2,\exp(\theta x)]$ on $[0,1]$, $\theta=1$ | $[1/3,(e^\theta-1)/\theta]$; $[0,((\theta-1)e^\theta+1)/\theta^2]$ | GK / GK |
| `localized_gaussian` | $\exp[-400(x-\theta)^2]$ on $[0,1]$, $\theta=0.37$ | error-function expression recorded in provenance; derivative omitted | CC / CC |
| `breakpoint_kink` | $|x-\theta|$ on $[0,1]$ with breakpoint $\theta=0.3$ | $[\theta^2+(1-\theta)^2]/2$; $2\theta-1$ | CC / CC |
| `endpoint_sqrt` | $\sqrt{x}$ on $[0,1]$ | $2/3$; derivative omitted | TS / TS |
| `semi_infinite_exponential` | $\exp(-\theta x)$ on $[0,\infty)$, $\theta=1$ | $1/\theta$; $-1/\theta^2$ | TS / TS |
| `full_line_gaussian` | $\exp[-(x/\theta)^2]$ on $(-\infty,\infty)$, $\theta=1$ | $\theta\sqrt{\pi}$; $\sqrt{\pi}$ | RTS / RTS |
| `oscillatory_cosine` | $\cos(\theta x)$ on $[0,1]$, $\theta=50$ | $\sin(\theta)/\theta$; $[\theta\cos(\theta)-\sin(\theta)]/\theta^2$ | GK / GK |
| `expensive_identity` | $\exp(-\theta x)$ times the mean of eight shifted $\sin^2+\cos^2$ identities on $[0,1]$, $\theta=1$ | converged independent NumPy Gauss-Legendre value, cross-checked by $(1-e^{-\theta})/\theta$; derivative cross-check $[(\theta+1)e^{-\theta}-1]/\theta^2$ | GK / GK |
| `narrow_gaussian` | $\exp[-10000(x-0.501)^2]$ on $[0,1]$ | converged independent NumPy Gauss-Legendre value, cross-checked by the error-function expression; derivative omitted | CC / CC |
| `nonfinite_band` | NaN for $|x-0.5|<0.05$, otherwise $\exp(-x)$ on $[0,1]$ | expected fail-closed classification; no finite truth | GK / GK |

Here GK is Gauss-Kronrod, CC is adaptive Clenshaw-Curtis, TS is adaptive
tanh-sinh, and RTS is the library's Romberg-tanh-sinh capability. Store the
full enum values rather than these abbreviations in executable records.

Define exactly these pair variants:

```python
(
    ("gauss_kronrod", "pair21", "exact", {"pair": 21}, {"order": 21}),
    (
        "clenshaw_curtis",
        "nodes17",
        "node_matched",
        {"initial_order": 17},
        {"order": 16},
    ),
    (
        "tanh_sinh",
        "closest_work",
        "family_matched",
        {"initial_level": 2},
        {"order": 61},
    ),
    (
        "tanh_sinh",
        "native_default",
        "family_matched",
        {"initial_level": 3},
        {"order": 61},
    ),
    (
        "romberg",
        "divmax10",
        "strong_match",
        {"initial_level": 1, "max_evaluations": 1025},
        {"divmax": 10},
    ),
    (
        "romberg_tanh_sinh",
        "divmax10",
        "capability",
        {"initial_level": 1},
        {"divmax": 10},
    ),
)
```

- [ ] **Step 5: Run the catalog tests**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark pytest tests/unit/test_benchmark_quad_cases.py -q
```

Expected: all catalog tests pass.

- [ ] **Step 6: Commit the dependency boundary and catalog**

```bash
git add pyproject.toml uv.lock scripts/quad_benchmark_cases.py tests/unit/test_benchmark_quad_cases.py
git commit -m "test(quad): freeze Phase A4 benchmark catalog"
```

### Task 3: Implement normalized Jaxstro and Quadax adapters

**Files:**
- Create: `scripts/quad_benchmark_adapters.py`
- Create: `tests/unit/test_benchmark_quad_adapters.py`

**Interfaces:**
- Consumes: `BenchmarkCase`, `LibraryMethod`, and `MethodPair` from Task 2.
- Produces: traceable `RawBenchmarkResult`, host-only `NormalizedResult`, `raw_jaxstro(case, family, controls)`, `raw_quadax(case, family, controls)`, `normalize_result(...)`, `portable_numeric(...)`, `matched_capacities(...)`, and `normalize_quadax_evaluations(...)`.

- [ ] **Step 1: Write failing adapter tests**

```python
import math

import jax.numpy as jnp

from scripts.quad_benchmark_adapters import (
    RunControls,
    normalize_quadax_evaluations,
    normalize_result,
    matched_capacities,
    portable_numeric,
    raw_jaxstro,
    raw_quadax,
)
from scripts.quad_benchmark_cases import CASES, LibraryMethod


def test_exact_gk_pair_converges_to_the_same_truth() -> None:
    case = next(case for case in CASES if case.name == "smooth_exponential")
    controls = RunControls(epsabs=1.0e-10, epsrel=1.0e-10, max_regions=64)
    ours = normalize_result(
        raw_jaxstro(case, LibraryMethod.GAUSS_KRONROD, controls)(jnp.asarray(case.theta)),
        library="jaxstro",
        family=LibraryMethod.GAUSS_KRONROD,
    )
    theirs = normalize_result(
        raw_quadax(case, LibraryMethod.GAUSS_KRONROD, controls)(jnp.asarray(case.theta)),
        library="quadax",
        family=LibraryMethod.GAUSS_KRONROD,
    )
    assert ours.converged and theirs.converged
    assert math.isclose(float(ours.value), case.truth, rel_tol=1.0e-10)
    assert math.isclose(float(theirs.value), case.truth, rel_tol=1.0e-10)


def test_clenshaw_curtis_normalizes_actual_node_count() -> None:
    assert normalize_quadax_evaluations(
        LibraryMethod.CLENSHAW_CURTIS, reported=32, order=16
    ) == 34


def test_nonfinite_semantics_are_not_collapsed() -> None:
    case = next(case for case in CASES if case.name == "nonfinite_band")
    controls = RunControls(epsabs=1.0e-8, epsrel=1.0e-8, max_regions=32)
    ours = normalize_result(
        raw_jaxstro(case, LibraryMethod.GAUSS_KRONROD, controls)(jnp.asarray(case.theta)),
        library="jaxstro",
        family=LibraryMethod.GAUSS_KRONROD,
    )
    theirs = normalize_result(
        raw_quadax(case, LibraryMethod.GAUSS_KRONROD, controls)(jnp.asarray(case.theta)),
        library="quadax",
        family=LibraryMethod.GAUSS_KRONROD,
    )
    assert ours.semantic_status == "nonfinite_integrand"
    assert theirs.semantic_status != ours.semantic_status


def test_nonfinite_values_have_portable_explicit_classifications() -> None:
    assert portable_numeric(jnp.asarray(jnp.nan)) == {
        "finite": False,
        "classification": "nan",
    }
    assert portable_numeric(jnp.asarray(jnp.inf)) == {
        "finite": False,
        "classification": "posinf",
    }
    assert portable_numeric(jnp.asarray(-jnp.inf)) == {
        "finite": False,
        "classification": "neginf",
    }


def test_breakpoint_region_and_evaluation_capacities_are_matched() -> None:
    case = next(case for case in CASES if case.name == "breakpoint_kink")
    controls = RunControls(epsabs=1.0e-8, epsrel=1.0e-8, max_regions=64)
    matched = matched_capacities(
        case,
        LibraryMethod.GAUSS_KRONROD,
        node_cost=21,
        controls=controls,
    )
    assert matched.jaxstro_max_regions == matched.quadax_max_ninter == 64
    assert matched.initial_segments == 2
    assert matched.jaxstro_max_evaluations >= 21 * (2 * 64 - 2)
```

- [ ] **Step 2: Run adapter tests to verify they fail**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark pytest tests/unit/test_benchmark_quad_adapters.py -q
```

Expected: missing adapter module.

- [ ] **Step 3: Implement controls and normalized records**

```python
@dataclass(frozen=True)
class RunControls:
    epsabs: float
    epsrel: float
    max_regions: int
    max_evaluations: int = 16384


@dataclass(frozen=True)
class MatchedCapacities:
    initial_segments: int
    jaxstro_max_regions: int
    jaxstro_max_evaluations: int
    quadax_max_ninter: int


class RawBenchmarkResult(NamedTuple):
    value: Any
    error: Any
    status: jax.Array
    reported_evaluations: jax.Array
    normalized_evaluations: jax.Array
    refinements: jax.Array
    active_regions: jax.Array
    levels: jax.Array


@dataclass(frozen=True)
class NormalizedResult:
    value: Any
    error: Any
    converged: bool
    raw_status: int
    semantic_status: str
    reported_evaluations: int
    normalized_evaluations: int
    refinements: int
    active_regions: int
    levels: int
```

`RawBenchmarkResult` contains only arrays and remains JIT/VMAP/JVP compatible.
Use `-1` for a work field unavailable from a library; interpret it only after
synchronization. `NormalizedResult` is host-only. Map Jaxstro `QuadStatus`
values to their lowercase names there. Preserve Quadax's raw integer status and
translate only documented success and capacity states; use
`quadax_status_<integer>` for unknown states rather than inventing equivalence.

`portable_numeric` converts finite scalar/vector leaves to JSON-safe values and
encodes each nonfinite leaf as `{"finite": false, "classification": "nan"}`,
`posinf`, or `neginf`. The artifact never contains a raw NaN or infinity and
never substitutes zero for failure evidence.

- [ ] **Step 4: Implement the Jaxstro adapter**

Map each `LibraryMethod` to the approved immutable method object.
`raw_jaxstro(...)` returns a callable `theta -> RawBenchmarkResult`. It calls
`quad.integrate` with explicit `args=theta`, `MaxNorm`, tolerances, capacities,
and `gradient="replay"`. Return all `QuadResult` diagnostics as arrays without
host conversion or estimated work fields.

- [ ] **Step 5: Implement the Quadax adapter**

`raw_quadax(...)` also returns a callable `theta -> RawBenchmarkResult`.
Convert domains to Quadax interval arrays with ordered breakpoints and explicit
infinities. Call `quadgk`, `quadcc`, `quadts`, `romberg`, or `rombergts` with
explicit tolerances, infinity norm, matched order/capacity, and
`full_output=False`. Retain `value`, `info.err`, `info.neval`, and `info.status`.
Normalize Clenshaw-Curtis actual nodes as:

```python
def normalize_quadax_evaluations(
    family: LibraryMethod, reported: int, order: int | None
) -> int:
    if family is LibraryMethod.CLENSHAW_CURTIS:
        if order is None or reported % order:
            raise ValueError("Quadax Clenshaw-Curtis work is not rule-aligned")
        return (order + 1) * (reported // order)
    return reported
```

- [ ] **Step 6: Run the adapter and relevant quadrature tests**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark pytest tests/unit/test_benchmark_quad_adapters.py tests/unit/quad/test_integrate_gk.py tests/unit/quad/test_adaptive_tanh_sinh.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit adapters**

```bash
git add scripts/quad_benchmark_adapters.py tests/unit/test_benchmark_quad_adapters.py
git commit -m "feat(quad): add matched Quadax benchmark adapters"
```

### Task 4: Implement synchronized compile, warm, VMAP, and AD measurements

**Files:**
- Create: `scripts/quad_benchmark_timing.py`
- Create: `tests/unit/test_benchmark_quad_timing.py`

**Interfaces:**
- Consumes: scalar callables with one device-resident parameter and PyTree results.
- Produces: `TimingRecord`, `measure_callable`, `measure_pair_interleaved`, and `ready_tree`.

- [ ] **Step 1: Write failing timing-protocol tests**

```python
import jax
import jax.numpy as jnp

from scripts.quad_benchmark_timing import measure_callable, measure_pair_interleaved


def test_measure_callable_separates_compile_and_warm_samples() -> None:
    record = measure_callable(
        lambda x: (jnp.sin(x), jnp.asarray(0, dtype=jnp.int32)),
        jax.device_put(jnp.asarray(0.5)),
        repeats=5,
    )
    assert record.lower_seconds >= 0.0
    assert record.compile_seconds >= 0.0
    assert len(record.warm_seconds) == 5
    assert record.median_warm_seconds > 0.0
    assert record.mad_warm_seconds >= 0.0


def test_interleaved_measurement_preserves_both_sample_counts() -> None:
    records = measure_pair_interleaved(
        {"jaxstro": lambda x: jnp.exp(x), "quadax": lambda x: jnp.exp(x)},
        jax.device_put(jnp.asarray(0.5)),
        repeats=5,
    )
    assert set(records) == {"jaxstro", "quadax"}
    assert all(len(record.warm_seconds) == 5 for record in records.values())
```

- [ ] **Step 2: Run timing tests to verify they fail**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark pytest tests/unit/test_benchmark_quad_timing.py -q
```

Expected: missing timing module.

- [ ] **Step 3: Implement synchronized timing records**

```python
@dataclass(frozen=True)
class TimingRecord:
    lower_seconds: float
    compile_seconds: float
    warm_seconds: tuple[float, ...]
    median_warm_seconds: float
    mad_warm_seconds: float
    minimum_warm_seconds: float
    maximum_warm_seconds: float


def ready_tree(value: Any) -> Any:
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )
```

In `measure_callable`, time `jax.jit(fun).lower(argument)` as `lower_seconds`,
time `.compile()` separately as `compile_seconds`, execute the compiled callable
once, synchronize every leaf, then record synchronized warm samples with
`time.perf_counter`. Compute median absolute deviation as
`median(abs(sample - median_warm_seconds))`. In
`measure_pair_interleaved`, compile both callables first and alternate library
order on every repetition.

- [ ] **Step 4: Add VMAP and derivative factories**

Implement helpers over a traceable `theta -> RawBenchmarkResult` callable. The
primal timing returns the complete raw result. JVP timing uses:

```python
def jvp_kernel(theta):
    primal, tangent = jax.jvp(
        raw_callable,
        (theta,),
        (jnp.ones_like(theta),),
    )
    return primal, tangent.value
```

This retains value, error, status, and work in the timed output while storing
only the mathematically meaningful value tangent. Reverse mode uses
`jax.value_and_grad` with `has_aux=True`, returning the raw result as auxiliary
diagnostics and differentiating `jnp.real(result.value)` only for scalar-output
cases. Use `jax.vmap(raw_callable)` for batch timing. Use JVP for all five
families. Mark reverse mode unsupported for Quadax `romberg` and `rombergts`;
do not catch that limitation and convert it into a slow timing.

- [ ] **Step 5: Run timing tests and Ruff**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark pytest tests/unit/test_benchmark_quad_timing.py -q
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync ruff check scripts/quad_benchmark_timing.py tests/unit/test_benchmark_quad_timing.py
```

Expected: all tests and lint pass.

- [ ] **Step 6: Commit timing infrastructure**

```bash
git add scripts/quad_benchmark_timing.py tests/unit/test_benchmark_quad_timing.py
git commit -m "feat(quad): add synchronized benchmark timing"
```

### Task 5: Build the deterministic evaluator, artifact, and CLI

**Files:**
- Create: `scripts/benchmark_quad.py`
- Create: `tests/unit/test_benchmark_quad_script.py`

**Interfaces:**
- Consumes: `CASES`, `METHOD_PAIRS`, adapters, and timing helpers.
- Produces: `run_deterministic_suite`, `run_timing_suite`, `build_artifact`, `render_report`, `algorithmic_metrics_match`, `--emit`, `--check`, and `--timing-only PATH`.

- [ ] **Step 1: Write failing artifact and CLI tests**

```python
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from jaxstro.evidence import artifact_from_dict, artifact_to_json

from scripts import benchmark_quad

ROOT = Path(__file__).resolve().parents[2]


def test_deterministic_suite_contains_both_lanes_and_all_cases() -> None:
    payload = benchmark_quad.run_deterministic_suite()
    assert payload["schema_version"] == 1
    assert {record["precision"] for record in payload["records"]} == {
        "float32",
        "float64",
    }
    assert {record["lane"] for record in payload["records"]} == {
        "family_matched",
        "best_method",
    }
    assert {record["case"] for record in payload["records"]} == {
        case.name for case in benchmark_quad.CASES
    }
    assert {
        (
            record["lane"],
            record["case"],
            record["family"],
            record["pair_variant"],
            record["precision"],
        )
        for record in payload["records"]
    } == benchmark_quad.expected_record_keys()


def test_derivative_gate_rejects_a_wrong_finite_derivative() -> None:
    assert benchmark_quad.derivative_gate(
        measured=1.0,
        truth=1.0,
        atol=1.0e-10,
        rtol=1.0e-10,
    )["passed"]
    assert not benchmark_quad.derivative_gate(
        measured=1.1,
        truth=1.0,
        atol=1.0e-10,
        rtol=1.0e-10,
    )["passed"]


def test_standalone_entrypoint_enables_real_float64() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts import benchmark_quad; "
                "print(benchmark_quad.lane_dtype('float64'))"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "float64" in completed.stdout


def test_nonfinite_failure_evidence_round_trips_without_json_nan() -> None:
    payload = benchmark_quad.run_deterministic_suite()
    artifact = benchmark_quad.build_artifact(payload)
    rendered = artifact_to_json(artifact)
    assert "NaN" not in rendered
    restored = artifact_from_dict(json.loads(rendered))
    failure = next(
        record
        for record in restored.method_payload["baseline"]["records"]
        if record["case"] == "nonfinite_band"
    )
    assert failure["jaxstro"]["value"] == {
        "finite": False,
        "classification": "nan",
    }


def test_optimized_merge_preserves_baseline_byte_for_byte() -> None:
    baseline = benchmark_quad.run_deterministic_suite()
    baseline_bytes = json.dumps(baseline, sort_keys=True, allow_nan=False)
    optimized = copy.deepcopy(baseline)
    optimized["source_revision"] = "optimized-revision"
    payload = benchmark_quad.merge_optimized(
        baseline=baseline,
        optimized=optimized,
        ratios={"warm": 0.8},
        contract_parity=True,
    )
    assert json.dumps(payload["baseline"], sort_keys=True, allow_nan=False) == baseline_bytes
    assert payload["optimized"]["source_revision"] == "optimized-revision"
    assert payload["ratios"] == {"warm": 0.8}
    assert payload["contract_parity"] is True


def test_freshness_ignores_timings_but_rejects_accuracy_drift() -> None:
    current = benchmark_quad.run_deterministic_suite()
    stored = copy.deepcopy(current)
    stored["timings"] = [{"median_warm_seconds": 99.0}]
    assert benchmark_quad.algorithmic_metrics_match(stored, current)
    stored["records"][0]["jaxstro"]["absolute_error"] += 1.0e-4
    assert not benchmark_quad.algorithmic_metrics_match(stored, current)


def test_check_mode_requires_the_emitted_artifact() -> None:
    completed = subprocess.run(
        [sys.executable, ROOT / "scripts" / "benchmark_quad.py", "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode in {0, 1}
    assert "quadrature performance evidence" in completed.stdout
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark pytest tests/unit/test_benchmark_quad_script.py -q
```

Expected: missing benchmark CLI.

- [ ] **Step 3: Implement deterministic records and gates**

The entry point must enable high precision before importing JAX or any
JAX-dependent benchmark module:

```python
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from scripts.quad_benchmark_adapters import raw_jaxstro, raw_quadax  # noqa: E402
from scripts.quad_benchmark_cases import BEST_METHODS, CASES, METHOD_PAIRS  # noqa: E402
from scripts.quad_benchmark_timing import measure_pair_interleaved  # noqa: E402
```

Construct every parameter and domain leaf explicitly with the selected lane
dtype. `lane_dtype("float64")` must return an actual `jnp.dtype("float64")` in
a standalone subprocess, not only under pytest configuration.

Freeze the bounded float32 catalog before evaluation:

```python
FLOAT32_CASES = (
    "smooth_exponential",
    "breakpoint_kink",
    "endpoint_sqrt",
    "semi_infinite_exponential",
    "oscillatory_cosine",
)
```

`expected_record_keys()` constructs the exact set from every supported float64
case/pair/lane plus only these float32 cases and their supported pairs. A missing
or duplicate lane, case, pair variant, or dtype is a schema failure.

For every declared lane, case, and supported pair, store:

```python
{
    "lane": lane,
    "case": case.name,
    "family": family.value,
    "pair_variant": pair.variant,
    "comparison_label": label.value,
    "truth": metric(truth, "integral units"),
    "jaxstro": normalized_payload(ours, truth),
    "quadax": normalized_payload(theirs, truth),
    "derivatives": derivative_payload,
    "warranted": correctness_gate(ours, theirs, truth, case.expected),
}
```

Emit one record for each approved float64 comparison and each explicitly
supported bounded float32 comparison. `correctness_gate` must judge observed
error against dtype-aware tolerance and must reject a nonfinite or false-success
result even when the library reports success. Never use agreement between
libraries as truth.

For every case with `derivative_truth`, compute deterministic JVP values for
both libraries, compare each independently with scalar or vector analytic truth
using the case provenance tolerances, and record policy labels
`accepted_formula_replay` and `adaptive_loop_ad`. Record supported reverse-mode
values and errors separately. Ratchet Quadax Romberg reverse mode as
`{"supported": false, "reason": "forward_mode_only"}`. A derivative failure
blocks any derivative performance interpretation.

Observed vector error and derivative error use the same declared infinity norm
as the runtime comparison. Store reported-error calibration as the finite ratio
`reported_error / max(observed_error, dtype_tiny)` plus a classification; do not
claim that an estimator is a rigorous bound unless its method contract says so.

The top-level method payload has a stable before/after shape from its first
emission:

```python
{
    "schema_version": 1,
    "controls": controls,
    "baseline": {
        "source_revision": revision,
        "records": records,
        "timings": timings,
    },
    "optimized": None,
    "ratios": None,
    "contract_parity": None,
    "optimization_decision": decision,
}
```

After an approved optimization, `baseline` remains byte-for-byte unchanged and
the clean rerun is written to `optimized`; `ratios` contains the predeclared
warm, compile, VMAP, AD, and comparable-work ratios, and `contract_parity` is a
boolean supported by the unchanged deterministic gates. Freshness compares the
live tree to `optimized` when present and otherwise to `baseline`.

- [ ] **Step 4: Implement the timing suite**

Use repeats 21 and VMAP batch sizes `(16, 128)`. Record scalar value, both VMAP
sizes, JVP, and supported reverse mode. Record memory as:

```python
{
    "status": "unavailable",
    "reason": "No backend-portable peak device-memory metric is available in this CPU artifact.",
}
```

unless a backend-supported measurement is proven reproducible during
implementation.

- [ ] **Step 5: Wrap the payload in the shared evidence envelope**

Use artifact identity `quad.performance`, schema version `1`, package version,
source revision, deterministic controls, environment fields, informational
timing metrics, PASS/FAIL accuracy comparisons, limitations, and generation
command:

```text
uv run --group benchmark python scripts/benchmark_quad.py --emit
```

For nonfinite cases, emit only finite indicator metrics such as `value_finite=0`
and store `nan`/`posinf`/`neginf` classifications in the portable method
payload. Never pass a nonfinite float to `MetricRecord`, `ComparisonRecord`, or
`EvidenceArtifact`.

`algorithmic_metrics_match` compares schema, controls, case/pair identities,
truth, observed errors, reported errors, statuses, and work within declared
numeric tolerances. It explicitly ignores environment timestamps and all timing
values.

Implement `render_report(artifact) -> str` as a deterministic authored MyST
report with the ten approved sections: purpose, label definitions, cases and
truth, accuracy and calibration, work, compile/warm/VMAP/AD timing, failure
semantics, environment, optimization decision, and warranted limitations. Do
not rely on the generic evidence renderer to supply the researcher-facing
explanation.

- [ ] **Step 6: Implement `--emit`, `--check`, and `--timing-only PATH`**

`--emit` must refuse to emit authoritative timings when `git status --porcelain`
is nonempty before execution. `--check` runs only the deterministic suite,
validates the stored artifact and Markdown rendering, and prints exactly one of:

```text
quadrature performance evidence healthy: deterministic metrics match fresh run
quadrature performance evidence is missing or stale
```

`--timing-only PATH` requires a clean tree, runs only the timing suite in the
fresh process, and writes JSON to the caller-supplied path outside the
repository. It does not rewrite authoritative evidence. Reject a path under
`REPO_ROOT` so confirmation runs cannot dirty the source tree.

- [ ] **Step 7: Run focused CLI tests**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark pytest tests/unit/test_benchmark_quad_cases.py tests/unit/test_benchmark_quad_adapters.py tests/unit/test_benchmark_quad_timing.py tests/unit/test_benchmark_quad_script.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit the executable benchmark before emission**

```bash
git add scripts/benchmark_quad.py tests/unit/test_benchmark_quad_script.py
git commit -m "feat(quad): add Phase A4 comparison harness"
```

Expected: `git status --short` is empty before authoritative emission.

### Task 6: Emit and independently review the clean baseline

**Files:**
- Create: `docs/validation/quad-performance.json`
- Create: `docs/60-validation/numerical/quadrature-performance.md`
- Create: `docs/superpowers/reviews/2026-07-16-quad-phase-a4-baseline-review.md`

**Interfaces:**
- Consumes: the committed benchmark CLI and clean worktree.
- Produces: the immutable baseline evidence and a classified optimization decision.

- [ ] **Step 1: Confirm the benchmark source tree is clean**

Run:

```bash
git status --short
git rev-parse HEAD
```

Expected: no status output and one revision recorded by the artifact.

- [ ] **Step 2: Run the authoritative CPU benchmark**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark python scripts/benchmark_quad.py --emit
```

Expected: both artifact paths are written, all correctness gates pass or the
command exits nonzero with the exact blocking records named.

- [ ] **Step 3: Run the deterministic freshness check**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark python scripts/benchmark_quad.py --check
```

Expected: `quadrature performance evidence healthy: deterministic metrics match fresh run`.

- [ ] **Step 4: Dispatch an independent fairness review checkpoint**

The reviewer must inspect method labels, node/work normalization, truth sources,
status semantics, timing synchronization, AD comparability, and every proposed
comparative claim. Record Critical, Important, and Minor findings in
`docs/superpowers/reviews/2026-07-16-quad-phase-a4-baseline-review.md`.

- [ ] **Step 5: Correct every Critical or Important review finding test-first**

For each finding, add one named failing regression to the relevant benchmark
test file, run that test to observe failure, implement the smallest correction,
and rerun the complete benchmark-focused suite. Do not weaken case truth or
tolerances.

- [ ] **Step 6: Commit corrected harness and tests before re-emission**

Commit the first emitted artifacts, review, and any corrected source/tests so a
new authoritative run can begin from a clean tree:

```bash
git add scripts/benchmark_quad.py scripts/quad_benchmark_*.py tests/unit/test_benchmark_quad_*.py docs/validation/quad-performance.json docs/60-validation/numerical/quadrature-performance.md docs/superpowers/reviews/2026-07-16-quad-phase-a4-baseline-review.md
git commit -m "fix(quad): record reviewed Phase A4 baseline corrections"
git status --short
```

Expected: the worktree is clean before the authoritative rerun. This checkpoint
commit may contain the first artifact revision; Step 7 replaces it with evidence
generated from this clean reviewed source revision.

- [ ] **Step 7: Re-emit the reviewed baseline from the clean revision**

Run the exact `--emit` and `--check` commands from Steps 2 and 3. Confirm the
artifact's `source_revision` is the current clean revision and the baseline
review refers to that revision.

- [ ] **Step 8: Commit the reviewed baseline evidence**

```bash
git add docs/validation/quad-performance.json docs/60-validation/numerical/quadrature-performance.md docs/superpowers/reviews/2026-07-16-quad-phase-a4-baseline-review.md scripts tests/unit/test_benchmark_quad_*.py
git commit -m "test(quad): publish reviewed Phase A4 baseline"
```

### Task 7: Apply the conditional optimization and hardening decision

**Files:**
- Create only if a trigger fires: `docs/superpowers/plans/2026-07-16-jaxstro-quad-phase-a4-optimization-addendum.md`
- Create only if a trigger fires: `docs/superpowers/reviews/2026-07-16-quad-phase-a4-optimization-review.md`
- Modify only if a trigger fires: the exact profiled owner under `src/jaxstro/quad/`
- Modify only if a trigger fires: a targeted regression under `tests/unit/quad/`, `tests/integration/`, or `tests/validation/`
- Modify: `docs/validation/quad-performance.json`
- Modify: `docs/60-validation/numerical/quadrature-performance.md`

**Interfaces:**
- Consumes: reviewed baseline records and approved thresholds.
- Produces: either a documented `no_optimization_required` decision or a separately reviewed, contract-preserving optimized result.

- [ ] **Step 1: Evaluate the approved triggers mechanically**

Freeze these automatic-trigger controls in the artifact before inspecting the
baseline:

```python
REPRESENTATIVE_CASES = (
    "smooth_exponential",
    "localized_gaussian",
    "breakpoint_kink",
    "endpoint_sqrt",
    "semi_infinite_exponential",
    "oscillatory_cosine",
    "expensive_identity",
)
RATIO_ELIGIBLE_LABELS = ("exact", "strong_match", "node_matched")
WARM_RATIO_TRIGGER = 1.25
COMPILE_RATIO_TRIGGER = 2.0
WORK_RATIO_TRIGGER = 1.50
VMAP_RATIO_TRIGGER = 1.50
AD_RATIO_TRIGGER = 1.50
MIN_WARM_CASES = 3
MIN_COMPILE_CASES = 2
MIN_OTHER_CASES = 3
```

A warm, VMAP, or AD timing ratio is eligible only when both results converge,
both pass truth gates, the case is representative, the comparison label is
eligible, and
`jaxstro_median - quadax_median > 2 * max(jaxstro_mad, quadax_mad)`. Work ratios
require semantically comparable normalized evaluations. Memory cannot trigger
on the CPU artifact while its status is `unavailable`.

Compilation has one independent measurement per method-case process and is
exempt from the warm-sample MAD rule. A ratio above `COMPILE_RATIO_TRIGGER` in
at least `MIN_COMPILE_CASES` creates `review_required`, not an automatic runtime
change. Confirm it by running the benchmark's `--timing-only` mode in a second
fresh process from the same clean revision; both processes must identify the
same two or more cases above the factor-of-two threshold before the optimization
addendum may classify compilation as a fired trigger.

Run the confirmation as:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark python scripts/benchmark_quad.py --timing-only /tmp/jaxstro-quad-a4-compile-confirmation.json
```

The decision code reads the confirmation file, verifies the same source
revision, controls, device, and dtype, and intersects its above-threshold compile
cases with the authoritative baseline before evaluating `MIN_COMPILE_CASES`.

Compute and store:

```python
{
    "warm_regression_cases_over_25_percent": tuple(warm_cases),
    "compile_or_memory_cases_over_2x": tuple(compile_or_memory_cases),
    "work_inefficiency_cases": tuple(work_cases),
    "vmap_regression_cases": tuple(vmap_cases),
    "ad_regression_cases": tuple(ad_cases),
    "decision": decision,
}
```

The warm trigger requires at least three representative matched converged cases.
The compile or memory trigger requires at least two. One noisy case cannot fire a
trigger. Work, VMAP, and AD ratios require at least three eligible cases.
Profiling observations outside these automatic rules are recorded as
`review_required`; they can authorize code only through the separately reviewed
optimization addendum.

- [ ] **Step 2A: If no trigger fires, ratchet the no-change decision**

Add a test asserting the stored decision is `no_optimization_required`, rerun
the deterministic benchmark, update the report if its decision prose changed,
then commit the decision without touching runtime code:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark python scripts/benchmark_quad.py --check
git add tests/unit/test_benchmark_quad_script.py docs/validation/quad-performance.json docs/60-validation/numerical/quadrature-performance.md
git commit -m "docs(quad): record Phase A4 parity decision"
git status --short
```

Expected: deterministic evidence is fresh and the worktree is clean before
Task 8. If the artifact/report are already byte-identical, the commit contains
the new decision ratchet test only.

- [ ] **Step 2B: If a trigger fires, profile before proposing code**

Use JAX lowering text, cost analysis, work records, and focused timing to name
the exact cost owner. Write an optimization addendum containing the measured
trigger, root cause, exact files, failing regression, proposed minimal change,
contract invariants, and unchanged rebenchmark command. Run writing-plans
self-review and independent numerical/JAX review on the addendum before code.

- [ ] **Step 3B: Implement each approved optimization test-first**

Add the failing performance/work/trace regression, prove it fails on the
baseline, implement the minimum owner-local change, then run all Phase A
correctness and replay/quantity tests. Never encode raw wall time as a normal CI
assertion; use deterministic work, trace, or structure when possible.

- [ ] **Step 4B: Commit the reviewed runtime change before re-emission**

Commit the approved addendum, runtime owner, and regression while retaining the
already committed baseline artifact:

```bash
git add docs/superpowers/plans/2026-07-16-jaxstro-quad-phase-a4-optimization-addendum.md src/jaxstro/quad tests
git commit -m "perf(quad): optimize measured adaptive bottleneck"
git status --short
```

Expected: the worktree is clean and the baseline artifact is unchanged.

- [ ] **Step 5B: Re-emit the unchanged benchmark and retain both records**

Run the exact Task 6 command from a clean committed optimization revision. The
artifact must retain baseline metrics, optimized metrics, ratios, and
`contract_parity=true`.

- [ ] **Step 6B: Dispatch independent post-optimization review**

The reviewer checks numerical parity, work semantics, replay AD, quantity
behavior, JIT/VMAP, benchmark fairness, and the absence of claim inflation.

- [ ] **Step 7B: Correct post-optimization review findings before final emission**

If review requires source changes, add a failing regression, implement the
minimum correction, then checkpoint source, tests, the current optimized
artifact/report, and the review together:

```bash
git add src/jaxstro/quad tests docs/validation/quad-performance.json docs/60-validation/numerical/quadrature-performance.md docs/superpowers/reviews/2026-07-16-quad-phase-a4-optimization-review.md
git commit -m "fix(quad): correct reviewed Phase A4 optimization"
git status --short
```

Confirm a clean tree, then repeat Steps 5B and 6B. Never emit authoritative
optimized evidence from a dirty tree.

- [ ] **Step 8: Commit the decision or reviewed optimized evidence**

Use one of:

```bash
git commit -m "docs(quad): record Phase A4 parity decision"
git commit -m "test(quad): publish reviewed optimized evidence"
```

### Task 8: Integrate the evidence throughout the website and registries

**Files:**
- Modify: `src/jaxstro/evidence/index.py`
- Modify: `src/jaxstro/quad/_contracts.py`
- Modify: `docs/validation/evidence-index.json`
- Modify: `docs/60-validation/evidence-index.md`
- Modify: `docs/validation/contracts.json`
- Modify: `docs/50-api/research-infrastructure/contracts.md`
- Modify: `docs/60-validation/validation.md`
- Modify: `docs/20-methods/approximation-integration/adaptive-quadrature.md`
- Modify: `docs/50-api/approximation-integration/quad.md`
- Modify: `docs/70-project/development/future-capabilities-roadmap.md`
- Modify: `docs/70-project/development/sota-assessment.md`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`
- Modify: `tests/unit/test_evidence_index.py`
- Modify: `tests/unit/test_contract_manifests.py`
- Modify: `tests/integration/test_docs_gate_wiring.py`
- Modify: `CHANGELOG.md`
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: reviewed `quad.performance` evidence and its warranted claims.
- Produces: one discoverable public comparison page and synchronized project truth.

- [ ] **Step 1: Write failing registry and route tests**

Require:

```python
assert "quad.performance" in {entry.id for entry in index.entries}
assert (
    route_manifest["60-validation/numerical/quadrature-performance.md"]
    == "/quadrature-performance"
)
assert "comparison_label" in performance_payload["method_payload"]["baseline"]["records"][0]
assert "exact" in api_text
assert "node-matched" in api_text
assert "family-matched" in api_text
assert "capability comparison" in api_text
assert (
    "[quadrature performance and comparison]"
    "(../../60-validation/numerical/quadrature-performance.md)"
    in adaptive_text
)
assert "shipped and validated" in api_text
assert "benchmarking" in api_text
assert "alpha" in api_text
assert "approved but planned" in api_text
assert "intentionally unsupported" in api_text
assert "Migrating to `jaxstro.quad`" in api_text
assert "jaxstro.numerics.integration" in api_text
assert "quad.performance" in validation_text
for unsupported_claim in (
    "Jaxstro is universally fastest",
    "Jaxstro is universally best",
    "Jaxstro is universally SOTA",
):
    assert unsupported_claim not in combined_public_text
```

Extend the quad callable contract to cite `docs/validation/quad-performance.json`
without asserting universal superiority. Define `api_text`, `adaptive_text`,
`validation_text`, `combined_public_text`, and the parsed route manifest from
their exact repository paths in the focused tests rather than using loose grep
tokens.

- [ ] **Step 2: Run the focused registry tests to verify they fail**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark pytest tests/unit/test_evidence_index.py tests/unit/test_contract_manifests.py tests/integration/test_docs_gate_wiring.py -q
```

Expected: missing evidence identity, contract reference, and route failures.

- [ ] **Step 3: Register and regenerate evidence**

Add `quad.performance` to `_TARGETS`, cite it from the adaptive contract, then run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync python scripts/build_evidence_index.py --emit
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync python scripts/build_contract_registry.py --emit
```

- [ ] **Step 4: Add the website navigation and benchmark-status callout**

Add the performance report under numerical validation in `docs/myst.yml`. Add a
MyST `important` admonition to the adaptive overview containing the measured
claim, hardware boundary, comparison-label explanation, and a link to the full
report. Add the capability map states `shipped and validated`, `benchmarking`,
`alpha`, `approved but planned`, and `intentionally unsupported` to the API page.

- [ ] **Step 5: Update roadmap, SOTA assessment, changelog, and status honestly**

Check the Phase A comparison item only after all numerical gates pass. State the
measured result, do not use `fastest`, `best`, or `SOTA` beyond the exact evidence
envelope, and leave Phase B/C methods visibly planned.

Add a `Migrating to jaxstro.quad` section to the API page that lists each legacy
`jaxstro.numerics.integration` and `jaxstro.numerics.quadrature` re-export, its
canonical `jaxstro.quad` owner, the no-behavior-change guarantee during sibling
migration, and the rule that removal remains deferred until downstream audits
are complete.

- [ ] **Step 6: Run focused evidence and documentation tests**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark pytest tests/unit/test_evidence_index.py tests/unit/test_contract_manifests.py tests/integration/test_docs_gate_wiring.py tests/unit/test_benchmark_quad_script.py -q
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync python scripts/build_evidence_index.py --check
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync python scripts/build_contract_registry.py --check
bash scripts/check_docs.sh
```

Expected: tests pass, artifacts are fresh, and all rendered routes pass strict
link, identifier, alternative-text, and accessibility checks.

- [ ] **Step 7: Commit website and evidence integration**

```bash
git add src/jaxstro/evidence/index.py src/jaxstro/quad/_contracts.py docs tests/unit/test_evidence_index.py tests/unit/test_contract_manifests.py tests/integration/test_docs_gate_wiring.py CHANGELOG.md STATUS.md
git commit -m "docs(quad): publish Phase A4 comparison evidence"
```

### Task 9: Final A+ verification and release-boundary review

**Files:**
- Modify only for verified corrections: files already owned by Tasks 2 through 8.
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: the complete Phase A4 implementation and public evidence.
- Produces: a verified Phase A release candidate and the next brainstorming boundary.

- [ ] **Step 1: Run the complete benchmark-focused suite**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark pytest \
  tests/unit/test_benchmark_quad_cases.py \
  tests/unit/test_benchmark_quad_adapters.py \
  tests/unit/test_benchmark_quad_timing.py \
  tests/unit/test_benchmark_quad_script.py \
  tests/unit/quad \
  tests/integration/test_quad_adaptive_transforms.py \
  tests/integration/test_quad_compatibility.py \
  tests/integration/test_quad_quantity_transforms.py \
  tests/integration/test_quad_replay_transforms.py \
  tests/validation/test_quad_adaptive_reference.py \
  tests/validation/test_quad_replay_derivatives.py -q
```

Expected: all tests pass without weakened tolerances.

- [ ] **Step 2: Run repository-wide static, artifact, and test gates**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync ruff format --check src tests scripts laboratory
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync ruff check src tests scripts laboratory
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync mypy src/jaxstro
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --group benchmark python scripts/benchmark_quad.py --check
bash scripts/check.sh
bash scripts/check_docs.sh
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 3: Run the full repository regression with local data attached when available**

Run:

```bash
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/jaxstro-uv-cache uv run --no-sync pytest -q
```

Expected: all code-backed tests pass; if the isolated worktree lacks the local
atmosphere corpus, attach the existing main-checkout corpus read-only and rerun
only the named data-backed failures.

- [ ] **Step 4: Dispatch final independent code and scientific review**

Review the complete diff for Critical, Important, and Minor findings. Require
explicit verdicts on numerical correctness, benchmark fairness, JAX transforms,
AD semantics, quantity behavior, public claims, generated evidence, website
clarity, dependency boundaries, and maintainability.

- [ ] **Step 5: Correct every Critical and Important finding test-first**

Add a failing regression, implement the minimum correction, rerun its focused
gate, and then repeat the complete Task 9 gate. Do not merge with unresolved
Critical or Important findings.

- [ ] **Step 6: Record completion and next boundary**

Update `STATUS.md` with exact test counts, benchmark revision, hardware, measured
results, optimization decision, limitations, and:

```text
next: Brainstorm and approve the next jaxstro.quad capability family; do not implement Phase B or Phase C before its design gate.
```

- [ ] **Step 7: Commit the verified closeout**

```bash
git add STATUS.md
git commit -m "docs(quad): close Phase A4 verification"
```

The branch remains local. Merging, pushing, publication, sibling migration, and
the next capability implementation require their separately approved actions.
