---
title: Jaxstro quad Phase A3 replay and quantity implementation plan
description: Test-driven execution plan for replay derivatives, quantity-aware adaptive integration, validation evidence, and replay-default promotion.
---

# Jaxstro quad Phase A3 Replay and Quantity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver replay-differentiable adaptive one-dimensional quadrature for all five Phase A2 methods, add an alpha opt-in quantity boundary over the same raw engine, publish independent derivative evidence, and promote replay to the default only after the complete gate passes.

**Architecture:** Keep the Phase A2 adaptive controllers as the sole owners of primal convergence and expose private family-specific replay evidence. A private all-positional `jax.custom_jvp` core returns the exact primal `QuadResult` while differentiating a fixed formula reconstructed from stopped segment-local or accepted-level evidence. A thin eager quantity adapter validates units, unwraps values before the raw core, and restores units without creating a second quadrature engine.

**Tech Stack:** Python 3.11-3.13, JAX and JAXlib 0.10.1 or newer, `jax.numpy`, `jax.lax`, `jax.custom_jvp`, `jax.jvp`, `jax.vjp`, `jax.jacfwd`, `jax.jacrev`, PyTest, Ruff, MyPy, MyST Markdown, KaTeX/LaTeX, JSON validation artifacts.

## Global Constraints

- The governing design is `docs/superpowers/specs/2026-07-15-jaxstro-quad-phase-a3-replay-quantity-design.md` at commit `b7c217a`.
- Raw arrays remain the only hot-kernel representation; quantity support is a validation and normalization boundary.
- `quad.integrate` remains the single public adaptive entry point. Do not add quantity-only or derivative-only public integration functions.
- Keep `gradient="stop"` as the default until Task 10 proves every replay-default promotion gate; retain it permanently as an explicit mode afterward.
- Replay differentiates only `QuadResult.value`. Floating and complex diagnostic tangents are exact zeros; integer and Boolean diagnostic tangents are JAX `float0`.
- Never differentiate sorting, refinement, stopping, region ordering, capacity logic, breakpoint motion, status selection, error estimation, or the primal `jax.lax.while_loop`.
- Moving finite bounds use the signed affine replay map directly. Do not use `minimum`, `maximum`, `absolute`, or `sign` in that differentiable map.
- Physical breakpoints remain stopped. Regional evidence uses original-segment identity plus segment-local accepted coordinates.
- Invalid and nonfinite statuses return nonfinite primal values; their derivatives are undefined and must not receive a transform-invariant NaN promise.
- Direct quotient-unit algebra from `jax.grad` over a `Quantity` PyTree is outside Phase A3. Differentiate selected numerical values and record explicit derivative units.
- Quantity normalization belongs only to `quad.integrate`. `quad.fixed`, `map_domain`, and `map_interval` fail closed on quantity-valued domains and unit-bearing `Infinite`.
- No new runtime dependency, sibling migration, deprecation, multidimensional method, publication, push, live-site mutation, or comparative superiority claim enters this plan.
- Use LaTeX for mathematics and ASCII in prose and code. Use MyST admonitions and contract tables; use no course or instructor framing.
- Never weaken or delete a scientific regression to make a gate pass.
- Commit after every task and request read-only subagent code review at Checkpoints C1, C2, C3, and C4. Resolve every Critical or Important finding before continuing.

## File and Responsibility Map

- `src/jaxstro/quad/_adaptive.py`: segment-local initial partitions, propagated segment identity, transformed-integrand replay switch, and regional primal evidence.
- `src/jaxstro/quad/_replay.py`: private replay evidence records, static integration configuration, diagnostic tangent construction, regional/global replay dispatch, and the all-positional custom-JVP core.
- `src/jaxstro/quad/_romberg.py`: fixed accepted-level reconstruction for classical Romberg and global tanh-sinh.
- `src/jaxstro/quad/_quantity.py`: quantity-mode resolution, eager unit validation, integrand/density adapters, raw-call normalization, and result-unit restoration.
- `src/jaxstro/quad/adaptive.py`: public validation/dispatch, raw primal solver, stop/replay selection, failure-value masking, and quantity adapter entry.
- `src/jaxstro/quad/domains.py`: optional static `Infinite.unit` metadata.
- `src/jaxstro/quad/transforms.py`: signed replay map plus fail-closed raw transform boundaries.
- `src/jaxstro/quad/result.py`: quantity-capable annotations without changing the public field layout.
- `src/jaxstro/quad/_contracts.py`: current derivative and quantity capability declarations after evidence passes.
- `tests/unit/test_quad_replay_substrate.py`: evidence, signed-map, breakpoint, fixed-level, and diagnostic-tangent unit tests.
- `tests/unit/test_quad_quantity.py`: mode resolution, conversions, dimensional failures, weighted densities, and shared-domain rejection.
- `tests/integration/test_quad_replay_transforms.py`: JVP/VJP/Jacobian/JIT/VMAP/complex/closure composition tests.
- `tests/integration/test_quad_quantity_transforms.py`: fixed-unit quantity transform and derivative-rescaling tests.
- `tests/validation/test_quad_replay_derivatives.py`: analytic, frozen-formula finite-difference, adaptive-rerun, and stability-ladder gates.
- `scripts/generate_quad_replay_evidence.py`: deterministic machine-readable derivative artifact generator.
- `docs/validation/quad-replay-derivatives.json`: generated source artifact.
- `docs/60-validation/numerical/quadrature-replay-derivatives.md`: generated researcher-facing evidence report.
- `docs/20-methods/approximation-integration/differentiating-an-integral.md`: beginner-facing derivations and worked example.
- `docs/20-methods/approximation-integration/adaptive-quadrature.md`: replay/stop, status, and method-choice updates.
- `docs/50-api/approximation-integration/quad.md`: exact public signature and differentiable/static input contract.
- `docs/40-workflows/differentiable-research/auditing-derivatives.md`: analytic, frozen-formula FD, and adaptive-rerun workflow.
- `docs/60-validation/validation.md`, `docs/60-validation/evidence-index.md`, `docs/myst.yml`: validation ledger, evidence routing, and MyST navigation.
- `docs/70-project/development/future-capabilities-roadmap.md`, `docs/70-project/development/numerical-methods-roadmap.md`, `docs/70-project/development/sota-assessment.md`, `STATUS.md`: honest completion and next-slice tracking.

---

### Task 1: Segment-local primal evidence and fail-closed invalid values

**Files:**
- Modify: `src/jaxstro/quad/_adaptive.py`
- Modify: `src/jaxstro/quad/adaptive.py`
- Test: `tests/unit/test_quad_replay_substrate.py`
- Modify: `tests/unit/quad/test_adaptive_controller.py`
- Modify: `tests/unit/quad/test_integrate_gk.py`
- Test: `tests/validation/test_quad_adaptive_reference.py`

**Interfaces:**
- Consumes: `Interval`, `RightInfinite`, `LeftInfinite`, `Infinite`, `sorted_breakpoints`, `AdaptiveControllerResult`, and the existing Phase A2 local-estimator contract.
- Produces: `ReferencePartition(lower, upper, segment_id, valid)`, `interval_segment_bounds(domain)`, `select_segment(domain, segment_id)`, and controller outputs carrying `region_segment_id`.

- [ ] **Step 1: Write failing segment-local partition and propagation tests**

Create `tests/unit/test_quad_replay_substrate.py` with exact ascending, reversed, and split-propagation assertions:

```python
import jax
import jax.numpy as jnp

from jaxstro.quad import Interval
from jaxstro.quad._adaptive import (
    LocalEstimate,
    adaptive_controller,
    reference_partition,
    select_segment,
)
from jaxstro.quad.tolerance import MaxNorm


def test_reference_partition_is_local_to_each_original_segment():
    domain = Interval(0.0, 4.0, breakpoints=(1.0, 3.0))
    partition = reference_partition(domain)

    assert jnp.array_equal(partition.lower, -jnp.ones(3))
    assert jnp.array_equal(partition.upper, jnp.ones(3))
    assert jnp.array_equal(partition.segment_id, jnp.arange(3, dtype=jnp.int32))
    assert [(float(select_segment(domain, i).lower), float(select_segment(domain, i).upper)) for i in range(3)] == [(0.0, 1.0), (1.0, 3.0), (3.0, 4.0)]


def test_reversed_partition_preserves_integration_orientation():
    domain = Interval(4.0, 0.0, breakpoints=(1.0, 3.0))

    assert [(float(select_segment(domain, i).lower), float(select_segment(domain, i).upper)) for i in range(3)] == [(4.0, 3.0), (3.0, 1.0), (1.0, 0.0)]


def test_controller_propagates_parent_segment_identity():
    partition = reference_partition(Interval(0.0, 2.0, breakpoints=(1.0,)))

    def estimate(lower, upper, segment_id):
        width = upper - lower
        return LocalEstimate(
            value=width,
            error=jnp.asarray(1.0),
            nonfinite=jnp.asarray(False),
        )

    result = adaptive_controller(
        partition,
        estimate,
        node_cost=1,
        max_evaluations=4,
        max_regions=4,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    active_ids = result.region_segment_id[result.region_active]
    assert jnp.all((active_ids == 0) | (active_ids == 1))
    assert jnp.sum(active_ids == 0) + jnp.sum(active_ids == 1) == result.active_regions
```

- [ ] **Step 2: Run the new tests and verify structural failure**

Run:

```bash
uv run --no-sync pytest tests/unit/test_quad_replay_substrate.py -q
```

Expected: FAIL because `ReferencePartition.segment_id`, `select_segment`, the three-argument local estimator, and `region_segment_id` do not exist.

- [ ] **Step 3: Implement segment-local partition and controller provenance**

In `src/jaxstro/quad/_adaptive.py`, extend the records and add the segment helpers:

```python
class ReferencePartition(NamedTuple):
    lower: Array
    upper: Array
    segment_id: Array
    valid: Array


def interval_segment_bounds(domain: Interval) -> tuple[Array, Array]:
    points = sorted_breakpoints(domain)
    dtype = jnp.result_type(domain.lower, domain.upper, *domain.breakpoints, 0.0)
    start = jnp.concatenate((jnp.reshape(jnp.asarray(domain.lower, dtype=dtype), (1,)), points))
    stop = jnp.concatenate((points, jnp.reshape(jnp.asarray(domain.upper, dtype=dtype), (1,))))
    return start, stop


def select_segment(domain: Domain, segment_id: Array) -> Domain:
    if isinstance(domain, Interval):
        start, stop = interval_segment_bounds(domain)
        return Interval(start[segment_id], stop[segment_id])
    return domain
```

Import `sorted_breakpoints`, make `reference_partition` return one `[-1, 1]` region per original finite segment and `segment_id=jnp.arange(count, dtype=jnp.int32)`, and return segment zero for improper domains. Extend `AdaptiveControllerResult` and `_ControllerState` with `region_segment_id` and `segment_id`. Change every local-estimator call to `(lower, upper, segment_id)`. When splitting a region, assign the parent `segment_id` to both children.

In `src/jaxstro/quad/adaptive.py`, make segment selection part of the primal estimator itself:

```python
def local_estimator(lower, upper, segment_id):
    segment_domain = select_segment(domain, segment_id)
    transformed = transformed_integrand(
        fun,
        segment_domain,
        rule_nodes,
        region_lower=lower,
        region_upper=upper,
        args=args,
        measure=selected_measure,
        open_region=isinstance(method, AdaptiveTanhSinh),
    )
    estimate = reduce_values(transformed.values)
    return LocalEstimate(
        value=estimate.value,
        error=estimate.error,
        nonfinite=transformed.nonfinite | estimate.nonfinite,
        roundoff=transformed.roundoff,
    )
```

Update the existing controller-test estimators to accept and ignore `_segment_id`. Add a public breakpoint regression in `tests/unit/quad/test_integrate_gk.py` proving that a three-segment constant integral is evaluated once over each physical segment and still returns the original whole-domain value, including reversed orientation.

- [ ] **Step 4: Add fail-closed invalid/nonfinite value tests**

Append:

```python
import pytest

from jaxstro import quad


@pytest.mark.parametrize(
    "domain, fun, expected_status",
    [
        (quad.Interval(jnp.nan, 1.0), lambda x: x, quad.QuadStatus.INVALID_INPUT),
        (quad.Interval(0.0, 1.0), lambda x: jnp.where(x > 0.5, jnp.nan, x), quad.QuadStatus.NONFINITE_INTEGRAND),
    ],
)
def test_invalid_and_nonfinite_solves_return_nonfinite_values(domain, fun, expected_status):
    result = quad.integrate(
        fun,
        domain,
        method=quad.GaussKronrod(15),
        epsabs=1e-6,
        epsrel=1e-6,
        max_evaluations=45,
        max_regions=2,
        gradient="stop",
    )
    assert result.status == expected_status
    assert not jnp.all(jnp.isfinite(result.value))
```

- [ ] **Step 5: Implement one result-level failure mask**

In `src/jaxstro/quad/adaptive.py`, add and apply this helper after both regional and global assembly, before either stop or replay dispatch:

```python
def _fail_closed_value(result: QuadResult) -> QuadResult:
    failed = (result.status == QuadStatus.INVALID_INPUT) | (
        result.status == QuadStatus.NONFINITE_INTEGRAND
    )
    value = jax.tree.map(
        lambda leaf: jnp.where(failed, jnp.full_like(leaf, jnp.nan), leaf),
        result.value,
    )
    return result._replace(value=value)
```

- [ ] **Step 6: Run focused and Phase A2 regression tests**

Run:

```bash
uv run --no-sync pytest tests/unit/test_quad_replay_substrate.py tests/unit/quad/test_adaptive_controller.py tests/unit/quad/test_integrate_gk.py tests/validation/test_quad_adaptive_reference.py -q
```

Expected: PASS with no changed status, work-count, or accepted-error contract outside the new nonfinite-value requirement.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/jaxstro/quad/_adaptive.py src/jaxstro/quad/adaptive.py tests/unit/test_quad_replay_substrate.py tests/unit/quad/test_adaptive_controller.py tests/unit/quad/test_integrate_gk.py tests/validation/test_quad_adaptive_reference.py
git commit -m "refactor(quad): preserve segment-local replay evidence"
```

---

### Task 2: Signed replay substrate and Gauss-Kronrod custom JVP

**Files:**
- Create: `src/jaxstro/quad/_replay.py`
- Modify: `src/jaxstro/quad/transforms.py`
- Modify: `src/jaxstro/quad/_adaptive.py`
- Modify: `src/jaxstro/quad/adaptive.py`
- Test: `tests/unit/test_quad_replay_substrate.py`
- Test: `tests/integration/test_quad_replay_transforms.py`

**Interfaces:**
- Consumes: Task 1 segment-local controller evidence and the existing `gauss_kronrod_data` and `gauss_kronrod_estimate_values` owners.
- Produces: `map_interval_replay`, `map_domain_replay`, `RegionalReplayEvidence`, `GlobalReplayEvidence`, `PrimalSolve`, `IntegrateConfig`, `_integrate_replay_core`, and `_result_tangent`.

- [ ] **Step 1: Write failing signed-map and coincident-bound tests**

Append to `tests/unit/test_quad_replay_substrate.py`:

```python
from jaxstro.quad.transforms import map_interval_replay


def test_signed_replay_map_keeps_reversed_orientation_without_sign():
    mapped = map_interval_replay(Interval(2.0, -1.0), jnp.array([-1.0, 0.0, 1.0]))
    assert jnp.allclose(mapped.x, jnp.array([2.0, 0.5, -1.0]))
    assert mapped.jacobian == -1.5
    assert mapped.orientation == 1.0


def test_signed_replay_map_has_leibniz_tangent_at_coincident_bounds():
    nodes = jnp.array([-0.5, 0.5])
    weights = jnp.ones(2)

    def fixed_formula(lower, upper):
        mapped = map_interval_replay(Interval(lower, upper), nodes)
        return jnp.sum(weights * mapped.x**0 * mapped.jacobian)

    _, lower_tangent = jax.jvp(fixed_formula, (1.0, 1.0), (1.0, 0.0))
    _, upper_tangent = jax.jvp(fixed_formula, (1.0, 1.0), (0.0, 1.0))
    assert lower_tangent == -2.0
    assert upper_tangent == 2.0
```

- [ ] **Step 2: Run the signed-map tests and verify the missing symbol**

Run:

```bash
uv run --no-sync pytest tests/unit/test_quad_replay_substrate.py -q
```

Expected: FAIL because `map_interval_replay` is not defined.

- [ ] **Step 3: Implement signed replay maps and transformed-integrand selection**

Add to `src/jaxstro/quad/transforms.py`:

```python
def map_interval_replay(domain: Interval, reference: Array) -> AffineMapResult:
    lower = jnp.asarray(domain.lower)
    upper = jnp.asarray(domain.upper)
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    reference = jnp.asarray(reference)
    return AffineMapResult(
        x=midpoint + half_width * reference,
        jacobian=half_width,
        orientation=jnp.asarray(1.0, dtype=half_width.dtype),
        valid=interval_is_valid(domain),
    )


def map_domain_replay(domain: Domain, reference: Array) -> DomainMapResult:
    if isinstance(domain, Interval):
        return DomainMapResult(*map_interval_replay(domain, reference))
    return map_domain(domain, reference)
```

Export both helpers. Add `replay: bool = False` to `_adaptive.transformed_integrand` and select `map_domain_replay` only when `replay` is true.

- [ ] **Step 4: Write failing Gauss-Kronrod replay transform tests**

Create `tests/integration/test_quad_replay_transforms.py`:

```python
import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad


def _integrate(theta, *, lower=0.0, upper=1.0, gradient="replay"):
    return quad.integrate(
        lambda x, args: jnp.exp(args * x),
        quad.Interval(lower, upper),
        args=theta,
        method=quad.GaussKronrod(21),
        epsabs=1e-10,
        epsrel=1e-10,
        max_evaluations=147,
        max_regions=4,
        gradient=gradient,
    )


def test_gauss_kronrod_replay_matches_analytic_parameter_derivative():
    theta = 0.7
    expected = ((theta - 1.0) * jnp.exp(theta) + 1.0) / theta**2
    actual = jax.grad(lambda value: _integrate(value).value)(theta)
    assert jnp.allclose(actual, expected, rtol=2e-8, atol=2e-10)


def test_gauss_kronrod_replay_matches_moving_bound_identity():
    lower_grad, upper_grad = jax.jacrev(
        lambda lower, upper: _integrate(0.7, lower=lower, upper=upper).value,
        argnums=(0, 1),
    )(0.2, 1.3)
    assert jnp.allclose(lower_grad, -jnp.exp(0.7 * 0.2), rtol=2e-8)
    assert jnp.allclose(upper_grad, jnp.exp(0.7 * 1.3), rtol=2e-8)


def test_gauss_kronrod_coincident_bound_tangents_are_not_zeroed():
    lower_grad, upper_grad = jax.jacrev(
        lambda lower, upper: _integrate(0.7, lower=lower, upper=upper).value,
        argnums=(0, 1),
    )(0.4, 0.4)
    value = jnp.exp(0.7 * 0.4)
    assert jnp.allclose(lower_grad, -value, rtol=2e-8)
    assert jnp.allclose(upper_grad, value, rtol=2e-8)


def test_stop_mode_remains_exactly_zero():
    assert jax.grad(lambda theta: _integrate(theta, gradient="stop").value)(0.7) == 0.0


def test_unknown_gradient_mode_fails_eagerly():
    with pytest.raises(ValueError, match='gradient must be "replay" or "stop"'):
        _integrate(0.7, gradient="through")
```

- [ ] **Step 5: Run the replay tests and verify the public rejection**

Run:

```bash
uv run --no-sync pytest tests/integration/test_quad_replay_transforms.py -q
```

Expected: FAIL because `gradient="replay"` is rejected by the Phase A2 entry point.

- [ ] **Step 6: Implement the private replay records and exact diagnostic tangents**

Create `src/jaxstro/quad/_replay.py` with these interfaces:

```python
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .result import QuadError, QuadResult, QuadWork


class RegionalReplayEvidence(NamedTuple):
    segment_local_lower: Any
    segment_local_upper: Any
    segment_id: Any
    active_mask: Any


class GlobalReplayEvidence(NamedTuple):
    accepted_level: Any


class PrimalSolve(NamedTuple):
    result: QuadResult
    evidence: RegionalReplayEvidence | GlobalReplayEvidence


@dataclass(frozen=True)
class IntegrateConfig:
    fun: Callable
    method: Any
    measure: Any
    max_evaluations: int
    max_regions: int
    error_norm: Any


def _zero_tangent(leaf):
    leaf = jnp.asarray(leaf)
    if jnp.issubdtype(leaf.dtype, jnp.inexact):
        return jnp.zeros_like(leaf)
    return np.zeros(leaf.shape, dtype=jax.dtypes.float0)


def result_tangent(result: QuadResult, value_tangent) -> QuadResult:
    return QuadResult(
        value=value_tangent,
        error=QuadError(
            estimate=jax.tree.map(_zero_tangent, result.error.estimate),
            norm=_zero_tangent(result.error.norm),
            kind=_zero_tangent(result.error.kind),
            confidence_level=_zero_tangent(result.error.confidence_level),
        ),
        tolerance=_zero_tangent(result.tolerance),
        status=_zero_tangent(result.status),
        work=QuadWork(*(_zero_tangent(leaf) for leaf in result.work)),
    )
```

Keep replay evaluators in this focused module. Do not export any of these names from `jaxstro.quad`.

- [ ] **Step 7: Refactor one raw primal solver and add the all-positional custom JVP**

In `src/jaxstro/quad/adaptive.py`, create `_solve_raw(config: IntegrateConfig, domain: Domain, args, epsabs, epsrel) -> PrimalSolve` by relocating the complete current numerical body without changing its validation, dispatch, tolerance, estimator, or result-assembly statements. Replace references to the former public parameters with `config` fields; the public `integrate` wrapper must contain no copied numerical branch.

Return `RegionalReplayEvidence` from controller arrays and synthesize one active `[-1, 1]` region for the zero-width branch. Return `GlobalReplayEvidence(refined.levels - 1)` for global methods and the configured initial level for their zero-width branch. Preserve exact public primal values and all Phase A2 diagnostic leaves.

In `_replay.py`, define the positional core:

```python
@partial(jax.custom_jvp, nondiff_argnums=(0,))
def integrate_replay_core(config, domain, args, epsabs, epsrel):
    from .adaptive import _solve_raw

    return _solve_raw(config, domain, args, epsabs, epsrel).result


@integrate_replay_core.defjvp
def _integrate_replay_core_jvp(config, primals, tangents):
    from .adaptive import _solve_raw

    domain, args, epsabs, epsrel = primals
    domain_tangent, args_tangent, _epsabs_tangent, _epsrel_tangent = tangents
    solve = _solve_raw(config, domain, args, epsabs, epsrel)
    evidence = jax.tree.map(jax.lax.stop_gradient, solve.evidence)

    def replay(live_domain, live_args):
        return replay_value(config, live_domain, live_args, evidence, solve.result.value)

    _, value_tangent = jax.jvp(
        replay,
        (domain, args),
        (domain_tangent, args_tangent),
    )
    return solve.result, result_tangent(solve.result, value_tangent)
```

Implement `replay_value` for `GaussKronrod` first: stop breakpoint children, select each active original segment, evaluate the accepted regional formula through `transformed_integrand` with `replay=True`, reduce with `gauss_kronrod_estimate_values`, mask inactive capacity slots with `lax.cond`, and sum only active values.

- [ ] **Step 8: Dispatch public stop and replay modes**

Replace the Phase A2 rejection in `integrate` with:

```python
if gradient not in {"replay", "stop"}:
    raise ValueError('gradient must be "replay" or "stop"')

config = IntegrateConfig(
    fun=fun,
    method=method,
    measure=selected_measure,
    max_evaluations=max_evaluations,
    max_regions=max_regions,
    error_norm=error_norm,
)
if gradient == "stop":
    return jax.tree.map(jax.lax.stop_gradient, _solve_raw(config, domain, args, epsabs, epsrel).result)
return integrate_replay_core(config, domain, args, epsabs, epsrel)
```

Keep the default as `"stop"` in this task.

- [ ] **Step 9: Run the focused replay and primal regression gate**

Run:

```bash
uv run --no-sync pytest tests/unit/test_quad_replay_substrate.py tests/integration/test_quad_replay_transforms.py tests/unit/quad/test_adaptive_controller.py tests/unit/quad/test_integrate_gk.py tests/validation/test_quad_adaptive_reference.py -q
```

Expected: PASS. Confirm the coincident-bound derivatives are nonzero and the Phase A2 primal reference cases remain within their predeclared thresholds.

- [ ] **Step 10: Commit Task 2**

```bash
git add src/jaxstro/quad/_replay.py src/jaxstro/quad/transforms.py src/jaxstro/quad/_adaptive.py src/jaxstro/quad/adaptive.py tests/unit/test_quad_replay_substrate.py tests/integration/test_quad_replay_transforms.py
git commit -m "feat(quad): add Gauss-Kronrod replay derivatives"
```

---

### Task 3: Complete regional-family replay

**Files:**
- Modify: `src/jaxstro/quad/_replay.py`
- Modify: `src/jaxstro/quad/adaptive.py`
- Test: `tests/integration/test_quad_replay_transforms.py`

**Interfaces:**
- Consumes: Task 2 regional replay evaluator and the existing Clenshaw-Curtis and tanh-sinh pair reducers.
- Produces: replay derivatives for `AdaptiveClenshawCurtis` and `AdaptiveTanhSinh` on finite and supported improper domains.

- [ ] **Step 1: Add parameterized regional-method derivative tests**

Append:

```python
@pytest.mark.parametrize(
    "method, rtol",
    [
        (quad.GaussKronrod(21), 3e-8),
        (quad.AdaptiveClenshawCurtis(17), 2e-7),
        (quad.AdaptiveTanhSinh(3), 2e-7),
    ],
)
def test_regional_replay_parameter_derivative(method, rtol):
    def integral(theta):
        return quad.integrate(
            lambda x, args: jnp.exp(-args * x),
            quad.Interval(0.0, 2.0),
            args=theta,
            method=method,
            epsabs=1e-10,
            epsrel=1e-10,
            max_evaluations=600,
            max_regions=12,
            gradient="replay",
        ).value

    theta = 0.8
    expected = (2.0 * theta * jnp.exp(-2.0 * theta) + jnp.exp(-2.0 * theta) - 1.0) / theta**2
    assert jnp.allclose(jax.grad(integral)(theta), expected, rtol=rtol, atol=2e-9)


def test_adaptive_tanh_sinh_replay_on_right_infinite_domain():
    def integral(theta):
        return quad.integrate(
            lambda x, args: jnp.exp(-args * x),
            quad.RightInfinite(0.0),
            args=theta,
            method=quad.AdaptiveTanhSinh(3),
            epsabs=1e-9,
            epsrel=1e-9,
            max_evaluations=1200,
            max_regions=16,
            gradient="replay",
        ).value

    assert jnp.allclose(jax.grad(integral)(1.3), -1.0 / 1.3**2, rtol=3e-6)
```

- [ ] **Step 2: Run the regional test matrix and verify missing dispatch**

Run:

```bash
uv run --no-sync pytest tests/integration/test_quad_replay_transforms.py -q
```

Expected: FAIL for the two methods that the replay dispatcher does not yet implement.

- [ ] **Step 3: Add method-specific high-rule reduction without duplicating traversal**

In `_replay.py`, factor the shared traversal as:

```python
def _regional_rule(config, dtype):
    if isinstance(config.method, GaussKronrod):
        data = gauss_kronrod_data(config.method, dtype=dtype)
        return data.nodes, lambda values: gauss_kronrod_estimate_values(values, data).value, False
    if isinstance(config.method, AdaptiveClenshawCurtis):
        pair = clenshaw_curtis_pair_data(config.method, dtype=dtype)
        return pair.nodes, lambda values: nested_rule_estimate_values(values, pair).value, False
    if isinstance(config.method, AdaptiveTanhSinh):
        pair = tanh_sinh_pair_data(config.method, dtype=dtype)
        return pair.nodes, lambda values: tanh_sinh_estimate_values(values, pair).value, True
    raise TypeError(f"{type(config.method).__name__} is not a regional replay method")
```

Use the returned nodes, reducer, and `open_region` flag in the one shared region traversal. Do not copy controller or mapping logic by method.

- [ ] **Step 4: Add breakpoint-tangent and accepted-formula parity tests**

Append tests that (a) JVP tangent supplied only to an `Interval` breakpoint contributes exactly zero, (b) changing a breakpoint in a new primal call changes the stopped segment provenance but preserves the mathematical integral within the declared quadrature tolerance, and (c) the replay fixed value agrees with `result.value` within one ulp-scaled summation tolerance for all regional methods.

Use:

```python
def test_breakpoint_tangent_is_stopped():
    method = quad.GaussKronrod(21)

    def value(breakpoint):
        return quad.integrate(
            lambda x: jnp.exp(x),
            quad.Interval(0.0, 1.0, breakpoints=(breakpoint,)),
            method=method,
            epsabs=1e-10,
            epsrel=1e-10,
            max_evaluations=126,
            max_regions=4,
            gradient="replay",
        ).value

    _, tangent = jax.jvp(value, (0.4,), (1.0,))
    assert tangent == 0.0
```

- [ ] **Step 5: Run regional replay, adaptive-reference, and transform tests**

Run:

```bash
uv run --no-sync pytest tests/integration/test_quad_replay_transforms.py tests/integration/test_quad_adaptive_transforms.py tests/validation/test_quad_adaptive_reference.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/jaxstro/quad/_replay.py src/jaxstro/quad/adaptive.py tests/integration/test_quad_replay_transforms.py
git commit -m "feat(quad): complete regional replay derivatives"
```

## Checkpoint C1: Regional replay review

Dispatch a read-only code-review subagent over the Task 1-3 commit range. Require review of signed coincident-bound mathematics, breakpoint provenance, primal A2 regression risk, custom-JVP linearity, exact `float0` diagnostics, and hidden control-flow differentiation. Fix every Critical or Important finding, rerun the Task 1-3 gates, and commit corrections before Task 4.

---

### Task 4: Exact accepted-level replay for both global families

**Files:**
- Modify: `src/jaxstro/quad/_romberg.py`
- Modify: `src/jaxstro/quad/_replay.py`
- Test: `tests/unit/test_quad_replay_substrate.py`
- Test: `tests/integration/test_quad_replay_transforms.py`

**Interfaces:**
- Consumes: `GlobalReplayEvidence.accepted_level`, `_richardson_row`, `_masked_evaluate`, `_tanh_sinh_tables`, and the Task 2 custom-JVP dispatcher.
- Produces: `romberg_replay_value` and `romberg_tanh_sinh_replay_value`, both fixed-shape and free of adaptive `while_loop` control.

- [ ] **Step 1: Write failing accepted-level reconstruction tests**

Append to `tests/unit/test_quad_replay_substrate.py`:

```python
from jaxstro.quad._romberg import (
    romberg_refine,
    romberg_replay_value,
    romberg_tanh_sinh_refine,
    romberg_tanh_sinh_replay_value,
)


@pytest.mark.parametrize("initial_level", [1, 2, 3])
def test_romberg_replay_reconstructs_accepted_diagonal(initial_level):
    def evaluate(reference):
        return jnp.exp(reference), jnp.asarray(False), jnp.asarray(False)

    primal = romberg_refine(
        evaluate,
        jnp.asarray(0.0),
        initial_level=initial_level,
        max_evaluations=257,
        max_regions=1,
        epsabs=1e-12,
        epsrel=1e-12,
        error_norm=MaxNorm(),
        dtype=jnp.float64,
    )
    replay = romberg_replay_value(
        evaluate,
        jnp.asarray(0.0),
        initial_level=initial_level,
        accepted_level=primal.levels - 1,
        max_evaluations=257,
        dtype=jnp.float64,
    )
    assert jnp.allclose(replay, primal.value, rtol=2e-15, atol=2e-15)


def test_global_tanh_sinh_replay_reconstructs_accepted_weight_row():
    def evaluate(reference):
        return jnp.exp(-reference**2), jnp.asarray(False), jnp.asarray(False)

    primal = romberg_tanh_sinh_refine(
        evaluate,
        jnp.asarray(0.0),
        initial_level=2,
        max_evaluations=801,
        max_regions=1,
        epsabs=1e-11,
        epsrel=1e-11,
        error_norm=MaxNorm(),
        dtype=jnp.float64,
    )
    replay = romberg_tanh_sinh_replay_value(
        evaluate,
        jnp.asarray(0.0),
        initial_level=2,
        accepted_level=primal.levels - 1,
        max_evaluations=801,
        dtype=jnp.float64,
    )
    assert jnp.allclose(replay, primal.value, rtol=2e-15, atol=2e-15)
```

- [ ] **Step 2: Run the fixed-level unit tests and verify missing helpers**

Run:

```bash
uv run --no-sync pytest tests/unit/test_quad_replay_substrate.py -q
```

Expected: FAIL on imports for both replay helpers.

- [ ] **Step 3: Implement fixed accepted-level classical Romberg replay**

Add `romberg_replay_value` to `_romberg.py`. It must reproduce the initial-grid summation and Richardson-row order used by `romberg_refine`, then advance a fixed-capacity `lax.fori_loop` only through the stopped accepted level:

```python
def romberg_replay_value(
    evaluate_one,
    zero,
    *,
    initial_level: int,
    accepted_level,
    max_evaluations: int,
    dtype,
):
    max_level = (max_evaluations - 1).bit_length() - 1
    accepted_level = jax.lax.stop_gradient(jnp.asarray(accepted_level, dtype=jnp.int32))
    lane_count = 2 ** max(max_level - 1, 0)
    payload_shape = zero.shape
    value_dtype = zero.dtype
    real_dtype = jnp.real(zero).dtype
    table = jnp.zeros(
        (max_level + 1, max_level + 1) + payload_shape,
        dtype=value_dtype,
    )
    floors = jnp.zeros(
        (max_level + 1, max_level + 1) + payload_shape,
        dtype=real_dtype,
    )

    fine_count = 2**initial_level + 1
    lane = jnp.arange(max(lane_count, fine_count), dtype=jnp.int32)
    initial_active = lane < fine_count
    initial_nodes = -1.0 + 2.0 * lane.astype(dtype) / (fine_count - 1)
    initial_values, _, _ = _masked_evaluate(
        evaluate_one,
        initial_nodes,
        initial_active,
        zero,
    )

    def initialize(level, state):
        current_table, current_floors = state
        stride = 2 ** (initial_level - level)
        selected = initial_active & (lane % stride == 0)
        endpoint = (lane == 0) | (lane == fine_count - 1)
        coefficients = jnp.where(
            selected,
            jnp.where(endpoint, 0.5, 1.0),
            0.0,
        )
        shape = coefficients.shape + (1,) * len(payload_shape)
        step = jnp.asarray(2.0 / 2**level, dtype=dtype)
        weighted = initial_values * jnp.reshape(coefficients, shape)
        base = step * jnp.sum(weighted, axis=0)
        resabs = step * jnp.sum(jnp.abs(weighted), axis=0)
        base_floor = _gamma(2**level + 1, real_dtype) * resabs
        return _richardson_row(
            current_table,
            current_floors,
            level,
            base,
            base_floor,
            max_level,
        )

    table, floors = jax.lax.fori_loop(
        0,
        initial_level + 1,
        initialize,
        (table, floors),
    )

    def maybe_advance(level, state):
        def advance(current):
            current_table, current_floors = current
            new_count = 2 ** (level - 1)
            lane = jnp.arange(lane_count, dtype=jnp.int32)
            new_active = lane < new_count
            odd = 2 * lane + 1
            nodes = -1.0 + 2.0 * odd.astype(dtype) / (2**level)
            values, _, _ = _masked_evaluate(
                evaluate_one,
                nodes,
                new_active,
                zero,
            )
            shape = new_active.shape + (1,) * len(payload_shape)
            selected_values = jnp.where(
                jnp.reshape(new_active, shape),
                values,
                0.0,
            )
            step = jnp.asarray(2.0, dtype=dtype) / (2**level)
            base = 0.5 * current_table[level - 1, 0] + step * jnp.sum(
                selected_values,
                axis=0,
            )
            previous_resabs = current_floors[level - 1, 0] / _gamma(
                2 ** (level - 1) + 1,
                real_dtype,
            )
            resabs = 0.5 * previous_resabs + step * jnp.sum(
                jnp.abs(selected_values),
                axis=0,
            )
            base_floor = _gamma(2**level + 1, real_dtype) * resabs
            return _richardson_row(
                current_table,
                current_floors,
                level,
                base,
                base_floor,
                max_level,
            )

        return jax.lax.cond(level <= accepted_level, advance, lambda x: x, state)

    table, _ = jax.lax.fori_loop(
        initial_level + 1,
        max_level + 1,
        maybe_advance,
        (table, floors),
    )
    return table[accepted_level, accepted_level]
```

Use the exact initialization and update expressions shown above; do not call `romberg_refine`, do not use `while_loop`, and do not evaluate a level greater than `accepted_level`. During implementation, factor the duplicated initial-row and one-level arithmetic shared with `romberg_refine` into private helpers so the primal and replay formulas cannot drift.

- [ ] **Step 4: Implement fixed accepted-level global tanh-sinh replay**

Use the existing cached host tables and evaluate only the stopped active row:

```python
def romberg_tanh_sinh_replay_value(
    evaluate_one,
    zero,
    *,
    initial_level: int,
    accepted_level,
    max_evaluations: int,
    dtype,
):
    dtype_name = np.dtype(dtype).name
    _, nodes_host, weights_host, _, active_host, _, _, _ = _tanh_sinh_tables(
        initial_level, max_evaluations, dtype_name
    )
    nodes = jnp.asarray(nodes_host, dtype=dtype)
    weights = jnp.asarray(weights_host, dtype=dtype)
    active = jnp.asarray(active_host)
    level = jax.lax.stop_gradient(jnp.asarray(accepted_level, dtype=jnp.int32))
    values, _, _ = _masked_evaluate(evaluate_one, nodes, active[level], zero)
    shape = (nodes.shape[0],) + (1,) * zero.ndim
    return jnp.sum(values * jnp.reshape(weights[level], shape), axis=0)
```

Export both helpers privately through `_romberg.__all__` for tests and replay dispatch.

- [ ] **Step 5: Add public global-method derivative tests**

Append:

```python
@pytest.mark.parametrize(
    "method, max_evaluations, rtol",
    [
        (quad.Romberg(2), 257, 3e-7),
        (quad.RombergTanhSinh(2), 801, 3e-6),
    ],
)
def test_global_replay_parameter_derivative(method, max_evaluations, rtol):
    def integral(theta):
        return quad.integrate(
            lambda x, args: jnp.exp(args * x),
            quad.Interval(-0.5, 1.0),
            args=theta,
            method=method,
            epsabs=1e-10,
            epsrel=1e-10,
            max_evaluations=max_evaluations,
            max_regions=1,
            gradient="replay",
        ).value

    theta = 0.4
    expected = jax.grad(
        lambda t: (jnp.exp(t) - jnp.exp(-0.5 * t)) / t
    )(theta)
    assert jnp.allclose(jax.grad(integral)(theta), expected, rtol=rtol, atol=2e-8)
```

- [ ] **Step 6: Dispatch global replay and run the five-method matrix**

In `_replay.replay_value`, build the same signed or improper `evaluate_one` used by the primal global engine with `transformed_integrand` and `replay=True`, then dispatch to the method-selected fixed-level helper using `GlobalReplayEvidence.accepted_level`.

Run:

```bash
uv run --no-sync pytest tests/unit/test_quad_replay_substrate.py tests/integration/test_quad_replay_transforms.py tests/integration/test_quad_adaptive_transforms.py tests/validation/test_quad_adaptive_reference.py -q
```

Expected: PASS across all five adaptive methods.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/jaxstro/quad/_romberg.py src/jaxstro/quad/_replay.py tests/unit/test_quad_replay_substrate.py tests/integration/test_quad_replay_transforms.py
git commit -m "feat(quad): add accepted-level global replay"
```

## Checkpoint C2: Complete replay review

Dispatch a new read-only code-review subagent over the Task 4 range and the combined Task 1-4 architecture. Require exact formula parity, proof that replay contains no adaptive `while_loop`, masked evaluation of inactive global nodes, correct transpose behavior, and preservation of global error/work/status semantics. Resolve all Critical and Important findings and rerun the complete five-method matrix before Task 5.

---

### Task 5: Quantity-mode domains and eager normalization boundary

**Files:**
- Create: `src/jaxstro/quad/_quantity.py`
- Modify: `src/jaxstro/quad/domains.py`
- Modify: `src/jaxstro/quad/transforms.py`
- Modify: `src/jaxstro/quad/fixed.py`
- Modify: `src/jaxstro/quad/result.py`
- Modify: `src/jaxstro/quad/adaptive.py`
- Test: `tests/unit/test_quad_quantity.py`

**Interfaces:**
- Consumes: `jaxstro.quantity.Quantity`, `Unit`, `units.dimensionless`, `DimensionError`, the Task 2 public stop/replay dispatch, and existing `WeightedMeasure.density_unit`.
- Produces: `Infinite(unit: Unit | None = None)`, `quantity_mode`, `normalize_call -> NormalizedCall`, and `restore_result`.

- [ ] **Step 1: Write failing mode-resolution and shared-domain tests**

Create `tests/unit/test_quad_quantity.py`:

```python
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro import quantity as q
from jaxstro.quantity.errors import DimensionError


def test_quantity_mode_finite_interval_restores_all_scientific_units():
    result = quad.integrate(
        lambda x: 2.0 * x * q.s**-1,
        quad.Interval(0.0 * q.cm, 3.0 * q.cm),
        method=quad.GaussKronrod(21),
        epsabs=1e-10 * q.cm**2 / q.s,
        epsrel=1e-10,
        max_evaluations=63,
        max_regions=2,
        gradient="stop",
    )
    expected_unit = q.cm**2 / q.s
    assert result.value.unit == expected_unit
    assert result.error.estimate.unit == expected_unit
    assert result.error.norm.unit == expected_unit
    assert result.tolerance.unit == expected_unit


def test_quantity_mode_accepts_dimensionless_domain_with_dimensionful_integrand():
    result = quad.integrate(
        lambda x: jnp.ones_like(x.value) * q.erg,
        quad.Interval(0.0, 1.0),
        method=quad.GaussKronrod(15),
        epsabs=1e-9 * q.erg,
        epsrel=1e-9,
        max_evaluations=45,
        max_regions=2,
        gradient="stop",
    )
    assert result.value.unit == q.erg
    assert jnp.allclose(result.value.value, 1.0)


def test_fully_infinite_quantity_domain_requires_static_unit():
    domain = quad.Infinite(unit=q.cm)
    assert domain.unit == q.cm


def test_raw_fixed_and_transform_paths_reject_unit_bearing_infinite_domain():
    domain = quad.Infinite(unit=q.cm)
    with pytest.raises(TypeError, match="quantity-valued domains are supported only by quad.integrate"):
        quad.fixed(lambda x: x, domain, rule=quad.TanhSinhRule(3))
    with pytest.raises(TypeError, match="quantity-valued domains are supported only by quad.integrate"):
        quad.map_domain(domain, jnp.array([0.0]))


def test_quantity_epsabs_must_match_integral_unit():
    with pytest.raises(DimensionError):
        quad.integrate(
            lambda x: x,
            quad.Interval(0.0 * q.cm, 1.0 * q.cm),
            method=quad.GaussKronrod(15),
            epsabs=1e-6 * q.s,
            epsrel=1e-6,
            max_evaluations=45,
            max_regions=2,
        )
```

- [ ] **Step 2: Run quantity tests and verify raw conversion failures**

Run:

```bash
uv run --no-sync pytest tests/unit/test_quad_quantity.py -q
```

Expected: FAIL because `Infinite.unit` and quantity normalization are absent.

- [ ] **Step 3: Add static fully-infinite unit metadata and fail-closed shared APIs**

Change `Infinite` to:

```python
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Infinite:
    unit: Unit | None = None

    def tree_flatten(self):
        return (), self.unit

    @classmethod
    def tree_unflatten(cls, unit, _children):
        return cls(unit=unit)
```

Import `Unit` from `jaxstro.quantity`. Add a small eager `domain_has_quantity` helper in `_quantity.py`. Call a fail-closed validator at the start of `quad.fixed`, `map_domain`, and `map_interval`; reject quantity bounds and `Infinite(unit is not None)` with the exact message in the test. Preserve byte-for-byte raw `Infinite()` behavior.

- [ ] **Step 4: Implement explicit mode resolution and normalized-call record**

Create `_quantity.py` with:

```python
from dataclasses import dataclass
from typing import Any, Callable

import jax

from jaxstro.quantity import Quantity, Unit
from jaxstro.quantity import units as q_units


@dataclass(frozen=True)
class NormalizedCall:
    fun: Callable
    domain: Any
    args: Any
    measure: Any
    epsabs: Any
    epsrel: Any
    result_unit: Unit


def quantity_mode(domain, epsabs) -> bool:
    coordinates = domain_coordinates(domain)
    return (
        any(isinstance(value, Quantity) for value in coordinates)
        or (isinstance(domain, Infinite) and domain.unit is not None)
        or isinstance(epsabs, Quantity)
    )
```

Implement `coordinate_unit(domain, epsabs)` exactly from the design table: infer a compatible unit from quantity coordinates, use `Infinite.unit`, or use `q_units.dimensionless` for a raw domain activated by quantity `epsabs`. Reject mixed dimensional raw/quantity coordinates. Convert all domain coordinates into the selected representation before returning `NormalizedCall`.

- [ ] **Step 5: Infer the stable integrand unit and wrap one raw engine**

Use `jax.eval_shape` with a `Quantity` node PyTree to require a quantity output and capture its static unit. Define a wrapper that always receives raw nodes from the engine, constructs `Quantity(nodes, coordinate_unit)`, calls the user function with the original explicit `args` convention, validates the returned unit, and returns `output.to_value(output_unit)`.

For quantity `WeightedMeasure`, similarly wrap `density` so it receives `Quantity` nodes, requires a quantity output compatible with `density_unit`, and returns `density.to_value(density_unit)`. Raw mode leaves the original callable unchanged.

Compute:

```python
result_unit = integrand_unit * coordinate_unit
if isinstance(measure, WeightedMeasure):
    result_unit = result_unit * measure.density_unit
```

Convert `epsabs` to `result_unit`; require dimensionless `epsrel` and unwrap it.

- [ ] **Step 6: Restore quantity result leaves without changing field layout**

Update `QuadError.norm` and `QuadResult.tolerance` annotations from `Array` to `Any`. Implement:

```python
def restore_result(result: QuadResult, result_unit: Unit) -> QuadResult:
    return QuadResult(
        value=Quantity(result.value, result_unit),
        error=QuadError(
            estimate=Quantity(result.error.estimate, result_unit),
            norm=Quantity(result.error.norm, result_unit),
            kind=result.error.kind,
            confidence_level=result.error.confidence_level,
        ),
        tolerance=Quantity(result.tolerance, result_unit),
        status=result.status,
        work=result.work,
    )
```

At the start of public `integrate`, select raw or quantity mode. Normalize before any existing `jnp.asarray` or `jnp.result_type`, call the same raw stop/replay engine once, and restore only at the outer boundary.

- [ ] **Step 7: Run quantity and complete raw regression tests**

Run:

```bash
uv run --no-sync pytest tests/unit/test_quad_quantity.py tests/unit/quad/test_fixed.py tests/unit/quad/test_adaptive_controller.py tests/unit/quad/test_integrate_gk.py tests/integration/test_quad_replay_transforms.py tests/validation/test_quad_adaptive_reference.py -q
```

Expected: PASS. Raw result PyTrees and raw `Infinite()` behavior remain unchanged.

- [ ] **Step 8: Commit Task 5**

```bash
git add src/jaxstro/quad/_quantity.py src/jaxstro/quad/domains.py src/jaxstro/quad/transforms.py src/jaxstro/quad/fixed.py src/jaxstro/quad/result.py src/jaxstro/quad/adaptive.py tests/unit/test_quad_quantity.py
git commit -m "feat(quad): add opt-in quantity normalization"
```

---

### Task 6: Quantity-aware replay and dimensional derivative evidence

**Files:**
- Modify: `src/jaxstro/quad/_quantity.py`
- Modify: `src/jaxstro/quad/_replay.py`
- Create: `tests/integration/test_quad_quantity_transforms.py`
- Modify: `tests/unit/test_quad_quantity.py`

**Interfaces:**
- Consumes: Task 5 quantity adapter and Task 2-4 replay core.
- Produces: fixed-unit JVP/VJP workflows, explicit raw-value Jacobian scaling, and weighted-density representation invariance.

- [ ] **Step 1: Write failing metre-centimetre value and derivative invariance tests**

Create `tests/integration/test_quad_quantity_transforms.py`:

```python
import jax
import jax.numpy as jnp

from jaxstro import quad
from jaxstro import quantity as q


def _length_integral(bound_value, bound_unit):
    bound = q.Quantity(bound_value, bound_unit)
    return quad.integrate(
        lambda x: x,
        quad.Interval(q.Quantity(0.0, bound_unit), bound),
        method=quad.GaussKronrod(21),
        epsabs=q.Quantity(1e-10, bound_unit**2),
        epsrel=1e-10,
        max_evaluations=63,
        max_regions=2,
        gradient="replay",
    ).value


def test_quantity_replay_preserves_physical_value_across_length_units():
    metres = _length_integral(2.0, q.m).to_value(q.cm**2)
    centimetres = _length_integral(200.0, q.cm).to_value(q.cm**2)
    assert jnp.allclose(metres, centimetres, rtol=2e-12)


def test_raw_value_jacobians_rescale_with_declared_derivative_units():
    d_metres = jax.grad(lambda value: _length_integral(value, q.m).to_value(q.m**2))(2.0)
    d_centimetres = jax.grad(lambda value: _length_integral(value, q.cm).to_value(q.cm**2))(200.0)
    assert jnp.allclose(d_metres, 2.0, rtol=2e-10)
    assert jnp.allclose(d_centimetres, 200.0, rtol=2e-10)
    assert jnp.allclose(d_metres * 100.0, d_centimetres, rtol=2e-10)


def test_quantity_result_jvp_keeps_static_integral_unit():
    primal, tangent = jax.jvp(
        lambda value: _length_integral(value, q.cm),
        (200.0,),
        (1.0,),
    )
    assert primal.unit == q.cm**2
    assert tangent.unit == q.cm**2
```

- [ ] **Step 2: Add weighted-density coordinate representation tests**

Append:

```python
def _weighted_expectation(scale_value, length_unit):
    scale = q.Quantity(scale_value, length_unit)
    measure = quad.WeightedMeasure(
        lambda x, args: q.math.exp(-(x / args)),
        density_unit=q.dimensionless,
    )
    return quad.integrate(
        lambda x: x,
        quad.Interval(q.Quantity(0.0, length_unit), 2.0 * scale),
        args=scale,
        measure=measure,
        method=quad.GaussKronrod(21),
        epsabs=q.Quantity(1e-9, length_unit**2),
        epsrel=1e-9,
        max_evaluations=147,
        max_regions=4,
        gradient="replay",
    ).value


def test_quantity_weighted_density_sees_physical_coordinates():
    left = _weighted_expectation(1.0, q.m).to_value(q.cm**2)
    right = _weighted_expectation(100.0, q.cm).to_value(q.cm**2)
    assert jnp.allclose(left, right, rtol=2e-8)
```

- [ ] **Step 3: Run quantity transform tests and inspect failures**

Run:

```bash
uv run --no-sync pytest tests/integration/test_quad_quantity_transforms.py -q
```

Expected: initial failures identify any adapter closure, output-unit, or custom-JVP PyTree mismatch. Do not relax unit assertions.

- [ ] **Step 4: Make normalization functions JAX-transform safe**

Ensure `jax.eval_shape` receives `args` as an explicit operand rather than closing over traced array values. Keep only callable and unit metadata in static adapters. Ensure the public quantity wrapper restores units after the raw custom-JVP core so the core's tangent remains a raw `QuadResult` with valid `float0` diagnostic leaves.

Add a regression proving direct quantity-PyTree gradients are not documented as quotient-unit outputs; the supported example differentiates `bound_value`, not `Quantity` itself.

- [ ] **Step 5: Run raw and quantity transform matrices**

Run:

```bash
uv run --no-sync pytest tests/integration/test_quad_replay_transforms.py tests/integration/test_quad_quantity_transforms.py tests/validation/test_quantity_jax_transforms.py tests/validation/test_quantity_math_gradients.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

```bash
git add src/jaxstro/quad/_quantity.py src/jaxstro/quad/_replay.py tests/unit/test_quad_quantity.py tests/integration/test_quad_quantity_transforms.py
git commit -m "test(quad): validate dimensional replay derivatives"
```

## Checkpoint C3: Quantity and derivative-unit review

Dispatch a fresh read-only code-review subagent over Tasks 5-6. Require review of mode activation, dimensionless expectations, quantity-output inference, density coordinate semantics, `epsabs` dimensions, result PyTrees, direct-Quantity Jacobian claim boundaries, centimetre/metre scaling, and shared `Infinite` blast radius. Resolve all Critical and Important findings before Task 7.

---

### Task 7: Complete JAX, complex, closure, and failure envelopes

**Files:**
- Modify: `tests/integration/test_quad_replay_transforms.py`
- Modify: `tests/integration/test_quad_quantity_transforms.py`
- Modify: `src/jaxstro/quad/_replay.py`
- Modify: `src/jaxstro/quad/adaptive.py`

**Interfaces:**
- Consumes: the complete five-method replay and quantity boundaries from Tasks 1-6.
- Produces: executable evidence for value-only Jacobians, full-result JVP diagnostics, selected VJP projections, JIT, VMAP, complex envelopes, explicit-parameter ownership, and undefined invalid-status derivatives.

- [ ] **Step 1: Add complete-result JVP diagnostic tangent tests**

Append:

```python
def _assert_zero_or_float0(primal, tangent):
    for primal_leaf, tangent_leaf in zip(
        jax.tree.leaves(primal), jax.tree.leaves(tangent), strict=True
    ):
        if jnp.issubdtype(jnp.asarray(primal_leaf).dtype, jnp.inexact):
            assert jnp.all(jnp.asarray(tangent_leaf) == 0)
        else:
            assert jnp.asarray(tangent_leaf).dtype == jax.dtypes.float0


@pytest.mark.parametrize(
    "method,max_evaluations,max_regions",
    [
        (quad.GaussKronrod(21), 147, 4),
        (quad.AdaptiveClenshawCurtis(17), 153, 4),
        (quad.AdaptiveTanhSinh(3), 600, 8),
        (quad.Romberg(2), 257, 1),
        (quad.RombergTanhSinh(2), 801, 1),
    ],
)
def test_full_result_jvp_stops_every_diagnostic(method, max_evaluations, max_regions):
    def solve(theta):
        return quad.integrate(
            lambda x, args: jnp.exp(args * x),
            quad.Interval(0.0, 1.0),
            args=theta,
            method=method,
            epsabs=1e-9,
            epsrel=1e-9,
            max_evaluations=max_evaluations,
            max_regions=max_regions,
            gradient="replay",
        )

    primal, tangent = jax.jvp(solve, (0.4,), (1.0,))
    assert jnp.isfinite(tangent.value)
    _assert_zero_or_float0(primal._replace(value=0.0), tangent._replace(value=0.0))
```

- [ ] **Step 2: Add selected VJP, value-only Jacobian, JIT, and VMAP tests**

Use scalar objectives that project only `value`; prove a diagnostic-only floating projection returns zero cotangent. Apply `jacfwd` and `jacrev` only to `.value`:

```python
def test_value_only_jacobians_agree_under_jit_and_vmap():
    def value(theta):
        return _integrate(theta).value

    forward = jax.jit(jax.jacfwd(value))
    reverse = jax.jit(jax.jacrev(value))
    theta = jnp.array([0.2, 0.5, 0.8])
    assert jnp.allclose(jax.vmap(forward)(theta), jax.vmap(reverse)(theta), rtol=2e-8)


def test_diagnostic_vjp_projection_is_zero():
    def tolerance(theta):
        return _integrate(theta).tolerance

    _, pullback = jax.vjp(tolerance, 0.4)
    assert pullback(jnp.asarray(1.0))[0] == 0.0
```

Do not call `jacrev` on the integer-bearing full `QuadResult`.

- [ ] **Step 3: Add hidden-closure rejection and explicit-args success tests**

```python
def test_differentiated_parameter_hidden_in_integrand_closure_is_rejected():
    def hidden(theta):
        return quad.integrate(
            lambda x: theta * x,
            quad.Interval(0.0, 1.0),
            method=quad.GaussKronrod(15),
            epsabs=1e-9,
            epsrel=1e-9,
            max_evaluations=45,
            max_regions=2,
            gradient="replay",
        ).value

    with pytest.raises((jax.errors.UnexpectedTracerError, ValueError), match="closed-over|Tracer|nondiff"):
        jax.grad(hidden)(2.0)


def test_same_parameter_is_supported_through_explicit_args():
    def explicit(theta):
        return quad.integrate(
            lambda x, args: args * x,
            quad.Interval(0.0, 1.0),
            args=theta,
            method=quad.GaussKronrod(15),
            epsabs=1e-9,
            epsrel=1e-9,
            max_evaluations=45,
            max_regions=2,
            gradient="replay",
        ).value

    assert jax.grad(explicit)(2.0) == 0.5
```

If JAX's natural failure message is unstable, add one narrow eager tracer guard at the static-config boundary and assert Jaxstro's own stable message. Do not inspect arbitrary closure contents recursively beyond the supported tracer guard.

- [ ] **Step 4: Add the three complex differentiation envelopes**

```python
def test_real_parameter_to_complex_output_uses_realified_jacobian():
    def value(theta):
        result = quad.integrate(
            lambda x, args: jnp.exp(1j * args * x),
            quad.Interval(0.0, 1.0),
            args=theta,
            method=quad.GaussKronrod(21),
            epsabs=1e-10,
            epsrel=1e-10,
            max_evaluations=147,
            max_regions=4,
            gradient="replay",
        ).value
        return jnp.stack((jnp.real(result), jnp.imag(result)))

    theta = 0.7
    z = 1j * theta
    derivative = 1j * (((z - 1.0) * jnp.exp(z) + 1.0) / z**2)
    expected = jnp.stack((jnp.real(derivative), jnp.imag(derivative)))
    assert jnp.allclose(jax.jacrev(value)(theta), expected, rtol=2e-8, atol=2e-10)


def test_complex_parameter_to_real_output_uses_jax_cotangent_convention():
    def loss(theta):
        return jnp.real(_complex_integral(theta))

    theta = 0.7 + 0.2j
    expected = ((theta - 1.0) * jnp.exp(theta) + 1.0) / theta**2
    assert jnp.allclose(jax.grad(loss)(theta), expected, rtol=2e-8, atol=2e-10)


def test_complex_to_complex_is_realified_not_forced_holomorphic():
    def realified(parts):
        theta = parts[0] + 1j * parts[1]
        value = _complex_integral(theta)
        return jnp.stack((jnp.real(value), jnp.imag(value)))

    parts = jnp.array([0.7, 0.2])
    theta = parts[0] + 1j * parts[1]
    derivative = ((theta - 1.0) * jnp.exp(theta) + 1.0) / theta**2
    expected = jnp.asarray(
        [
            [jnp.real(derivative), -jnp.imag(derivative)],
            [jnp.imag(derivative), jnp.real(derivative)],
        ]
    )
    assert jnp.allclose(jax.jacrev(realified)(parts), expected, rtol=2e-8, atol=2e-10)
```

Define `_complex_integral(theta)` immediately above these tests as the same `quad.integrate` call used by `_integrate`, but with integrand `jnp.exp(args * x)` and a complex `args` value. This keeps the runtime under test while the expected derivative comes from the independent closed form.

- [ ] **Step 5: Add invalid/nonfinite status tests without derivative-layout promises**

Assert nonfinite primal values and exact statuses under eager, JIT, and VMAP. Do not assert a specific tangent/Jacobian NaN pattern. Add a documentation-facing test string explaining that derivatives are undefined for these statuses.

- [ ] **Step 6: Run the complete JAX envelope**

Run:

```bash
uv run --no-sync pytest tests/integration/test_quad_replay_transforms.py tests/integration/test_quad_quantity_transforms.py tests/integration/test_quad_adaptive_transforms.py -q
```

Expected: PASS. Successful reverse-mode execution through all five methods is the regression that proves JAX is not attempting to transpose the primal dynamic `while_loop`.

- [ ] **Step 7: Commit Task 7**

```bash
git add src/jaxstro/quad/_replay.py src/jaxstro/quad/adaptive.py tests/integration/test_quad_replay_transforms.py tests/integration/test_quad_quantity_transforms.py
git commit -m "test(quad): close replay transform envelopes"
```

---

### Task 8: Independent derivative validation and machine-readable evidence

**Files:**
- Create: `scripts/generate_quad_replay_evidence.py`
- Create: `docs/validation/quad-replay-derivatives.json`
- Create: `docs/60-validation/numerical/quadrature-replay-derivatives.md`
- Create: `tests/validation/test_quad_replay_derivatives.py`
- Create: `tests/integration/test_quad_replay_artifact.py`
- Modify: `src/jaxstro/evidence/index.py`
- Modify by generator: `docs/validation/evidence-index.json`
- Modify by generator: `docs/60-validation/evidence-index.md`

**Interfaces:**
- Consumes: all five replay methods, raw-value quantity parameterizations, and the existing `EvidenceArtifact`/`emit_artifact`/`check_artifact` architecture.
- Produces: one `quad.replay-derivatives` computational artifact rendered to deterministic JSON and Markdown, indexed through `build_evidence_index`, with case records separating analytic derivatives, frozen-formula finite differences, adaptive-rerun finite differences, partition/level evidence, unit metadata, and predeclared gates.

- [ ] **Step 1: Define the immutable evidence schema in a failing test**

Create `tests/validation/test_quad_replay_derivatives.py`:

```python
import json
from pathlib import Path

from jaxstro.evidence import EvidenceStatus, artifact_from_dict


ARTIFACT = Path("docs/validation/quad-replay-derivatives.json")


def test_replay_evidence_schema_and_required_case_families():
    artifact = artifact_from_dict(json.loads(ARTIFACT.read_text()))
    payload = artifact.method_payload
    assert artifact.artifact_id == "quad.replay-derivatives"
    assert artifact.schema_version == "1"
    assert payload["claim"] == "replay-differentiable adaptive one-dimensional quadrature"
    required = {
        "smooth_parameter",
        "vector_payload",
        "complex_payload",
        "moving_bounds",
        "reversed_bounds",
        "coincident_bounds",
        "improper_tail",
        "endpoint_singularity",
        "weighted_density",
        "exhausted_finite",
        "quantity_rescaling",
        "invalid_input",
        "nonfinite_integrand",
    }
    assert required <= {case["family"] for case in payload["cases"]}
    assert all(
        comparison.status is not EvidenceStatus.FAIL
        for comparison in artifact.comparisons
    )
    for case in payload["cases"]:
        assert {
            "method",
            "family",
            "dtype",
            "primal_value",
            "analytic_value",
            "observed_primal_error",
            "reported_primal_error",
            "replay_ad_derivative",
            "analytic_derivative",
            "frozen_formula_fd",
            "adaptive_rerun_fd",
            "accepted_regions",
            "accepted_level",
            "parameter_unit",
            "integral_unit",
            "derivative_unit",
            "gates",
        } <= case.keys()
```

- [ ] **Step 2: Run the schema test and verify the missing artifact**

Run:

```bash
uv run --no-sync pytest tests/validation/test_quad_replay_derivatives.py -q
```

Expected: FAIL because the artifact does not exist.

- [ ] **Step 3: Implement deterministic analytic and finite-difference helpers**

Create `scripts/generate_quad_replay_evidence.py` with exact central differences:

```python
def central_difference(fun, x, step):
    return (fun(x + step) - fun(x - step)) / (2.0 * step)


def relative_error(actual, expected):
    scale = max(abs(expected), 1.0e-300)
    return abs(actual - expected) / scale


def gate(name, observed, threshold):
    return {
        "name": name,
        "observed": float(observed),
        "threshold": float(threshold),
        "passed": bool(observed <= threshold),
    }
```

Use the repository high-precision helper, deterministic method declarations, no random inputs, and the existing evidence records:

```python
from jaxstro import __version__
from jaxstro.evidence import (
    ComparisonRecord,
    ComparisonRelation,
    EnvironmentRecord,
    EvidenceArtifact,
    EvidenceStatus,
    MetricRecord,
    artifact_from_dict,
    check_artifact,
    emit_artifact,
)
```

Implement `build_artifact` with `artifact_id="quad.replay-derivatives"`, the exact generation command, deterministic controls, portable finite metrics, explicit PASS comparisons, calibrated limitations, and the case records in `method_payload`. Follow `scripts/benchmark_implicit_root.py`: `--emit` must pass the same `EvidenceArtifact` to `emit_artifact` for both JSON and Markdown; `--check` must reconstruct fresh algorithmic metrics, use `check_artifact` for the Markdown generated from the stored JSON artifact, and reject stale deterministic measurements while ignoring only declared environment snapshot fields.

- [ ] **Step 4: Generate separate frozen-formula and adaptive-rerun comparisons**

For each method, obtain the stopped private evidence at the center parameter. Compute:

```python
replay_ad = jax.grad(public_value)(theta)
frozen_fd = central_difference(
    lambda value: replay_value(config, domain, value, stopped_evidence, primal.value),
    theta,
    fd_step,
)
adaptive_rerun_fd = central_difference(public_value, theta, fd_step)
```

The first two are correctness gates. Record the third as `diagnostic` unless the case is predeclared away from partition changes. Never label adaptive-rerun disagreement as a custom-JVP implementation failure without checking partition/level stability.

- [ ] **Step 5: Add tolerance and capacity ladders**

For every method, record at least three tolerances and two nonbinding capacities. Require the final two stable rungs to satisfy predeclared primal and analytic-derivative thresholds. Record accepted regions for regional methods and accepted levels for global methods. Include at least one case near, but not across, an observed partition change.

Use dtype-aware maximum relative thresholds of `5e-5` for float32 and method/case-specific float64 thresholds no looser than `5e-7`; declare tighter thresholds in the case table when the analytic fixture supports them.

Exercise all five methods on scalar smooth-parameter and vector-payload cases. Exercise every method over its supported domain envelope for complex payloads and failure statuses. For complex cases, compare the realified Jacobian against the independently evaluated closed-form derivative, not merely finiteness. Invalid-input and nonfinite-integrand records must gate exact status and nonfinite primal value while recording the derivative as undefined rather than inventing a tangent layout.

- [ ] **Step 6: Add quantity-rescaling records with explicit units**

Record metre and centimetre raw-value derivatives for the same physical integral. Set `parameter_unit`, `integral_unit`, and `derivative_unit` to serialized unit strings and gate the physical derivative after conversion to one common derivative unit.

- [ ] **Step 7: Write the artifact and enforce freshness**

Support:

```bash
uv run --no-sync python scripts/generate_quad_replay_evidence.py --emit
uv run --no-sync python scripts/generate_quad_replay_evidence.py --check
```

Add `("quad.replay-derivatives", EvidenceClass.COMPUTATIONAL, "docs/validation/quad-replay-derivatives.json", "No external data required.")` to `src/jaxstro/evidence/index.py`. Regenerate both evidence-index outputs after emitting the artifact.

In `--check` mode, enforce the stored JSON algorithmic comparison and byte-check the Markdown renderer. Create `tests/integration/test_quad_replay_artifact.py` with this complete freshness contract:

```python
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_quad_replay_derivative_artifact_is_fresh():
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_quad_replay_evidence.py",
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )


def test_evidence_index_is_fresh_with_quad_replay_artifact():
    subprocess.run(
        [sys.executable, "scripts/build_evidence_index.py", "--check"],
        cwd=ROOT,
        check=True,
    )
```

- [ ] **Step 8: Run validation and freshness tests**

Run:

```bash
uv run --no-sync python scripts/generate_quad_replay_evidence.py --emit
uv run --no-sync python scripts/build_evidence_index.py --emit
uv run --no-sync pytest tests/validation/test_quad_replay_derivatives.py tests/integration/test_quad_replay_artifact.py -q
```

Expected: PASS with every required gate marked `passed: true`.

- [ ] **Step 9: Commit Task 8**

```bash
git add scripts/generate_quad_replay_evidence.py src/jaxstro/evidence/index.py docs/validation/quad-replay-derivatives.json docs/60-validation/numerical/quadrature-replay-derivatives.md docs/validation/evidence-index.json docs/60-validation/evidence-index.md tests/validation/test_quad_replay_derivatives.py tests/integration/test_quad_replay_artifact.py
git commit -m "test(quad): publish replay derivative evidence"
```

---

### Task 9: Researcher-first derivations, API contracts, and evidence navigation

**Files:**
- Create: `docs/20-methods/approximation-integration/differentiating-an-integral.md`
- Modify: `docs/20-methods/approximation-integration/adaptive-quadrature.md`
- Modify: `docs/50-api/approximation-integration/quad.md`
- Modify: `docs/40-workflows/differentiable-research/auditing-derivatives.md`
- Modify: `docs/60-validation/validation.md`
- Modify: `docs/myst.yml`
- Test: `tests/integration/test_quad_replay_docs.py`

**Interfaces:**
- Consumes: the public behavior and artifact completed in Tasks 1-8.
- Produces: the exact MyST route `Quadrature -> Adaptive Quadrature -> Differentiating an Integral -> Auditing Derivatives -> Quadrature Replay Derivative Evidence`.

- [ ] **Step 1: Write failing content and navigation contract tests**

Create `tests/integration/test_quad_replay_docs.py`:

```python
from pathlib import Path


ROOT = Path("docs")


def test_differentiating_integral_page_contains_required_derivations():
    text = (ROOT / "20-methods/approximation-integration/differentiating-an-integral.md").read_text()
    for required in (
        "## The exact integral derivative",
        "## The accepted fixed-formula derivative",
        "## Why the two derivatives can differ",
        "## Moving bounds",
        "## Units of a derivative",
        "## A complete analytic, AD, and finite-difference audit",
        "```{math}",
        ":::{warning}",
        ":::{admonition}",
    ):
        assert required in text
    assert "course" not in text.lower()
    assert "instructor" not in text.lower()


def test_myst_toc_places_derivative_page_after_adaptive_quadrature():
    text = (ROOT / "myst.yml").read_text()
    adaptive = text.index("20-methods/approximation-integration/adaptive-quadrature.md")
    derivative = text.index("20-methods/approximation-integration/differentiating-an-integral.md")
    assert adaptive < derivative
    assert "60-validation/numerical/quadrature-replay-derivatives.md" in text


def test_docs_link_foundations_workflow_api_and_evidence():
    text = (ROOT / "20-methods/approximation-integration/differentiating-an-integral.md").read_text()
    for route in (
        "../../00-start-here/why-jax.md",
        "../../10-foundations/mathematical-objects/what-is-a-derivative.md",
        "../../40-workflows/differentiable-research/what-jax-differentiates.md",
        "../../30-representations/units-quantities/quantities.md",
        "../../50-api/approximation-integration/quad.md",
        "../../60-validation/numerical/quadrature-replay-derivatives.md",
    ):
        assert route in text
```

Adjust relative paths to the actual page depth while keeping all six destinations.

- [ ] **Step 2: Run the page tests and verify missing routes**

Run:

```bash
uv run --no-sync pytest tests/integration/test_quad_replay_docs.py -q
```

Expected: FAIL because both new pages and TOC entries are absent.

- [ ] **Step 3: Write the complete differentiating-an-integral derivation**

The page must follow this exact conceptual order:

1. big picture: exact integral, executed adaptive solve, accepted formula;
2. prerequisites linked to Why JAX?, What is a Derivative?, and quantities;
3. assumptions permitting differentiation under the integral sign;
4. exact derivative with the Leibniz rule in LaTeX;
5. fixed accepted-formula replay derivative in LaTeX;
6. stopped refinement and breakpoint decisions;
7. signed affine derivation including coincident bounds;
8. primal error versus derivative evidence;
9. JVP directional units and raw-value Jacobian quotient units;
10. a complete exponential fixture with analytic derivative, replay AD, frozen-formula FD, and adaptive-rerun FD;
11. method and failure-boundary contract table; and
12. links to the API, auditing workflow, and generated evidence.

Use MyST `note`, `warning`, `tip`, and titled `admonition` blocks only where they carry distinct scientific meaning. Use no Unicode mathematics.

- [ ] **Step 4: Update method, API, workflow, and validation pages**

Document the exact `gradient` modes; differentiable and static inputs; invalid-status derivative boundary; complex envelopes; quantity activation table; direct-Quantity Jacobian limitation; and replay-default promotion evidence. Treat the Task 8 Markdown report and evidence index as generated, read-only outputs; link them without hand-copying or editing measurements.

- [ ] **Step 5: Add TOC entries and cross-route tests**

Place the methods page immediately after adaptive quadrature and the evidence page after implicit-root gradients in the Validation section. Update any expected route counts and information-architecture inventories through their existing test owners; do not rename or remove a stable route.

- [ ] **Step 6: Run focused docs tests and strict MyST verification**

Run:

```bash
uv run --no-sync pytest tests/integration/test_quad_replay_docs.py tests/integration/test_methods_information_architecture.py tests/integration/test_research_workflows_information_architecture.py tests/integration/test_validation_docs.py tests/integration/test_api_reference.py -q
bash scripts/check_docs.sh
```

Expected: all tests pass; strict MyST build, route crawl, and accessibility checks pass with the two new routes and no broken links.

- [ ] **Step 7: Commit Task 9**

```bash
git add docs/20-methods/approximation-integration/differentiating-an-integral.md docs/20-methods/approximation-integration/adaptive-quadrature.md docs/50-api/approximation-integration/quad.md docs/40-workflows/differentiable-research/auditing-derivatives.md docs/60-validation/validation.md docs/myst.yml tests/integration/test_quad_replay_docs.py
git commit -m "docs(quad): teach replay-differentiable integration"
```

---

### Task 10: Replay-default promotion and current capability records

**Files:**
- Modify: `src/jaxstro/quad/adaptive.py`
- Modify: `src/jaxstro/quad/_contracts.py`
- Modify: `tests/integration/test_contract_docs.py`
- Modify: `tests/integration/test_quad_replay_transforms.py`
- Modify: `docs/50-api/approximation-integration/quad.md`
- Modify: `docs/70-project/development/future-capabilities-roadmap.md`
- Modify: `docs/70-project/development/numerical-methods-roadmap.md`
- Modify: `docs/70-project/development/sota-assessment.md`

**Interfaces:**
- Consumes: every Task 1-9 gate, especially the fresh machine-readable artifact and strict docs build.
- Produces: public `gradient="replay"` default, permanent explicit `gradient="stop"`, and current evidence-linked contracts without comparative superiority language.

- [ ] **Step 1: Run the promotion gate before changing the default**

Run:

```bash
uv run --no-sync pytest tests/unit/test_quad_replay_substrate.py tests/unit/test_quad_quantity.py tests/integration/test_quad_replay_transforms.py tests/integration/test_quad_quantity_transforms.py tests/validation/test_quad_replay_derivatives.py tests/validation/test_quad_adaptive_reference.py -q
uv run --no-sync python scripts/generate_quad_replay_evidence.py --check
```

Expected: PASS. If any gate fails, stop; retain `gradient="stop"` and record the blocker rather than continuing.

- [ ] **Step 2: Write a failing default-policy regression**

Add:

```python
def test_integrate_defaults_to_replay_after_promotion_gate():
    def value(theta):
        return quad.integrate(
            lambda x, args: args * x,
            quad.Interval(0.0, 1.0),
            args=theta,
            method=quad.GaussKronrod(15),
            epsabs=1e-9,
            epsrel=1e-9,
            max_evaluations=45,
            max_regions=2,
        ).value

    assert jax.grad(value)(2.0) == 0.5
```

Run the single test and expect FAIL because the default is still stop.

- [ ] **Step 3: Promote the default and preserve the explicit stop escape hatch**

Change only the public signature default:

```python
gradient: str = "replay",
```

Keep the exact mode validation and explicit stop branch. Rerun the new default test plus `test_stop_mode_remains_exactly_zero`.

- [ ] **Step 4: Mark only evidence-backed contracts current**

Update `_contracts.py` and project pages to state:

- all five adaptive methods support first-order replay derivatives in the documented envelope;
- replay is the default and stop remains explicit;
- quantity-aware adaptive integration is alpha and opt-in;
- direct Quantity-PyTree quotient-unit Jacobians, higher derivatives, multidimensional integration, matched Quadax comparison, sibling migrations, and superiority claims remain outside the current contract.

Every current statement links to the exact tests, artifact, method page, and validation page.

- [ ] **Step 5: Run contract, roadmap, and default-policy tests**

Run:

```bash
uv run --no-sync pytest tests/integration/test_contract_docs.py tests/integration/test_quad_replay_transforms.py tests/integration/test_quad_replay_docs.py tests/integration/test_future_representation_guides.py -q
```

Expected: PASS with no stale Phase A2 limitation strings.

- [ ] **Step 6: Commit Task 10**

```bash
git add src/jaxstro/quad/adaptive.py src/jaxstro/quad/_contracts.py tests/integration/test_contract_docs.py tests/integration/test_quad_replay_transforms.py docs/50-api/approximation-integration/quad.md docs/70-project/development/future-capabilities-roadmap.md docs/70-project/development/numerical-methods-roadmap.md docs/70-project/development/sota-assessment.md
git commit -m "feat(quad): promote verified replay derivatives"
```

## Checkpoint C4: Complete A3 code and claim review

Dispatch two independent read-only subagents over the complete A3 commit range:

1. a scientific/JAX reviewer for numerical correctness, transform semantics, failure behavior, quantity units, and regression risk;
2. a documentation/claim reviewer for newcomer clarity, derivation completeness, MyST use, API accuracy, evidence links, and prohibition of unsupported SOTA superiority language.

Resolve every Critical and Important finding. Rerun the focused gate after each correction and commit review fixes separately.

---

### Task 11: Full repository verification and durable closeout

**Files:**
- Modify: `STATUS.md`
- Modify: `CHANGELOG.md`
- Modify if generated by existing owners: contract/evidence/provenance registry artifacts

**Interfaces:**
- Consumes: the complete reviewed A3 branch.
- Produces: a clean, fully verified branch with exact evidence counts and the next bounded quadrature action.

- [ ] **Step 1: Run format, lint, and type gates**

Run:

```bash
uv run --no-sync ruff format --check src tests scripts
uv run --no-sync ruff check src tests scripts
uv run --no-sync mypy src
git diff --check
```

Expected: PASS.

- [ ] **Step 2: Run generated-artifact and strict documentation gates**

Run:

```bash
uv run --no-sync python scripts/generate_quad_replay_evidence.py --check
uv run --no-sync pytest tests/integration/test_quad_replay_artifact.py -q
bash scripts/check_docs.sh
```

Expected: PASS with fresh artifacts, all routes crawlable, and accessibility checks green.

- [ ] **Step 3: Run the complete test suite**

Run:

```bash
uv run --no-sync pytest -q
```

Expected: PASS with only the repository's documented optional-dependency skips. Record the exact pass/skip count and wall time; do not reuse an older A2 count.

- [ ] **Step 4: Update changelog and status with exact evidence**

Add an unreleased changelog entry summarizing replay derivatives, quantity-aware adaptive integration, the new derivation/evidence route, and the replay-default change. Update `STATUS.md` in its existing `next:` / `previous:` format with:

- exact commits and implemented methods;
- exact focused and full-suite counts;
- exact Ruff, MyPy, generated-artifact, and strict-docs results;
- reviewer findings and dispositions;
- current claim boundary; and
- next bounded slice: matched Quadax comparison before any performance-superiority claim, unless Anna approves a different next quadrature phase.

- [ ] **Step 5: Run final cleanliness and scope audit**

Run:

```bash
git status --short
git diff --check
git diff b7c217a...HEAD -- src/jaxstro/quad tests docs scripts CHANGELOG.md STATUS.md
```

Expected: only intentional A3 files differ; `.superpowers/` remains untouched and untracked; no sibling repository changed.

- [ ] **Step 6: Commit closeout records**

```bash
git add CHANGELOG.md STATUS.md
git commit -m "docs(quad): record Phase A3 verification"
```

- [ ] **Step 7: Invoke branch finishing workflow**

Use `superpowers:finishing-a-development-branch`. Re-run the required final tests before presenting exactly the branch integration options. Do not merge, push, or discard without Anna's explicit selection.

## Stop conditions

Stop execution and report rather than guessing if any of these occurs:

- the custom JVP requires differentiating or transposing the adaptive `while_loop`;
- a correct signed coincident-bound derivative cannot coexist with exact primal behavior;
- stopped physical breakpoints cannot be reconstructed without moving under live outer bounds;
- global replay cannot reproduce the accepted returned formula within its declared summation tolerance;
- quantity conversion changes the physical value or declared physical derivative;
- direct Quantity-PyTree Jacobian units would need to be misrepresented;
- invalid/nonfinite status requires a transform-invariant derivative layout;
- replay-default promotion evidence fails;
- an existing primal threshold would have to be weakened;
- a sibling migration, new dependency, publication, push, or superiority claim appears necessary; or
- full verification fails repeatedly after focused diagnosis.

## Final completion definition

Phase A3 is complete only when Tasks 1-11 and Checkpoints C1-C4 are checked, every Critical and Important review finding is resolved, all five methods pass first-order replay evidence, quantity integration remains one alpha opt-in adapter over the raw engine, replay-default promotion gates pass, the machine-readable and researcher-facing evidence is fresh, the strict MyST site is green, the complete repository suite passes, and the branch is handed to the finishing workflow without unauthorized merge or push.
