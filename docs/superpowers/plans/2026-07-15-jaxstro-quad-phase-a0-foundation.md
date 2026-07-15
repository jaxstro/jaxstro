# Jaxstro Quad Phase A0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Work inline in the normal checkout and
> dispatch read-only subagent reviews at the two named checkpoints.

**Goal:** Establish the public `jaxstro.quad` namespace, typed result and
tolerance contracts, one-dimensional domain and measure configuration objects,
and a backward-compatible facade over the currently implemented sampled and
fixed-rule helpers without changing their numerical behavior.

**Architecture:** Phase A0 creates the stable objects that every later method
uses. The new top-level facade temporarily re-exports the existing numerical
callables so users can adopt `jaxstro.quad` immediately; Phase A1 moves those
implementations into focused `quad` modules and reverses the legacy imports.
Results use JAX-leaf `NamedTuple` records, while domains, measures, and norm
configurations use immutable PyTrees with dynamic numerical children and static
metadata.

**Tech Stack:** Python 3.11+, JAX 0.10.1+, JAX NumPy, `jax.tree_util`,
`jaxtyping`, `jaxstro.quantity`, pytest, Ruff, MyPy, MyST.

## Global Constraints

- Run Python, pytest, Ruff, and MyPy through
  `env -u VIRTUAL_ENV uv run --no-sync`.
- Work in the normal checkout. Do not create a worktree.
- Preserve unrelated tracked and untracked work. In particular, do not stage or
  modify the existing untracked `.superpowers/` directory.
- Add no runtime dependency and do not add Quadax, SciPy, or a high-precision
  package to `pyproject.toml`.
- Do not inspect, edit, migrate, or test Startrax, Gravax, or Progenax.
- Keep `jaxstro.quantity` opt-in alpha. A0 records dimensional metadata but does
  not implement quantity-valued integration or authorize downstream adoption.
- Preserve every existing sampled-integration and quadrature result for existing
  calls, including cumulative-trapezoid `dx`-outside ordering and exact
  probabilists' Hermite bytes.
- The canonical sampled names are `trapezoid`, `cumulative_trapezoid`, `simpson`,
  and `cumulative_simpson`. The initial facade does not extend their signatures;
  Phase A1 separately tests the approved consistent `dx` extension.
- Preserve the current legacy symbols and make every A0 facade export the same
  callable object, not a wrapper.
- Do not add deprecation warnings. Warnings are deferred until sibling consumers
  have migrated and traced-code behavior is audited.
- Do not create empty `fixed.py`, `adaptive.py`, `cubature.py`, `sparse.py`, or
  `qmc.py` files.
- A0 implements only the affine finite-interval map. Infinite-domain transform
  formulas are frozen in the first method plan that consumes them.
- Classical measure parameters are static in A0. Differentiable general-density
  parameters live in the shared explicit `args` PyTree used by later evaluators.
- Use LaTeX for mathematical notation and ASCII for authored prose.
- Never weaken, delete, or skip an existing test to obtain a pass.
- Stage explicit paths and commit each task only after its fresh focused gate.

## File Structure

### New runtime owners

- `src/jaxstro/quad/__init__.py`: public facade and canonical top-level exports.
- `src/jaxstro/quad/_contracts.py`: lightweight module ownership contract.
- `src/jaxstro/quad/result.py`: status/error codes and JAX-leaf result records.
- `src/jaxstro/quad/tolerance.py`: static norm configurations and scalar
  tolerance calculations.
- `src/jaxstro/quad/domains.py`: immutable interval and infinite-domain PyTrees.
- `src/jaxstro/quad/transforms.py`: finite affine reference-domain mapping.
- `src/jaxstro/quad/measures.py`: immutable measure declarations and validation.

### Existing runtime files modified

- `src/jaxstro/__init__.py`: lazy public `quad` module export.
- `src/jaxstro/contracts/registry.py`: load the `quad` contract sidecar without
  importing runtime numerics.

No existing numerical implementation moves in A0. That inversion belongs to
Phase A1 after the facade and compatibility tests are established.

### New and modified tests

- `tests/unit/quad/__init__.py`: focused test package marker.
- `tests/unit/quad/test_import_surface.py`: public facade, signatures, and exact
  legacy callable identity.
- `tests/unit/quad/test_result.py`: status, error, work, result, and PyTree
  contracts.
- `tests/unit/quad/test_tolerance.py`: norm and tolerance semantics.
- `tests/unit/quad/test_domains.py`: domain PyTrees, orientation, breakpoints,
  and affine mapping.
- `tests/unit/quad/test_measures.py`: support, normalization, and static metadata.
- `tests/unit/test_contract_manifests.py`: public module and import-isolation
  coverage.
- `tests/integration/test_api_reference.py`: eager top-level import contract.
- `tests/integration/test_quad_compatibility.py`: clean-process legacy/canonical
  identity and exact parity.
- `tests/integration/test_grouped_api_reference.py`: one owner-qualified A0 API
  page.
- `tests/integration/test_method_page_contract.py`: current sampled/fixed pages
  resolve the canonical facade and owner page.
- `tests/integration/test_myst_semantic_grammar.py`: exact authored-route count
  after adding `/quad-api`.

### Documentation and generated contracts

- `docs/50-api/approximation-integration/quad.md`: A0 public owner page.
- `docs/50-api/api.md`: add the canonical quadrature owner route.
- `docs/20-methods/approximation-integration/cumulative-trapz.md`: teach the
  canonical sampled facade while preserving the old numerical caveats.
- `docs/20-methods/approximation-integration/quadrature.md`: teach the canonical
  fixed-rule facade without claiming A1 or A2 methods.
- `docs/myst.yml`: add the A0 API owner page once.
- `docs/route-manifest.json`: assign `/quad-api` without changing existing
  routes.
- `docs/validation/contracts.json`: generated contract inventory.
- `docs/50-api/research-infrastructure/contracts.md`: generated contract page.
- `STATUS.md`: exact A0 completion, evidence, and next action.

---

### Task 1: Freeze the baseline and add the canonical facade

**Files:**
- Create: `src/jaxstro/quad/__init__.py`
- Modify: `src/jaxstro/__init__.py`
- Create: `tests/unit/quad/__init__.py`
- Create: `tests/unit/quad/test_import_surface.py`
- Create: `tests/integration/test_quad_compatibility.py`

**Interfaces:**
- Consumes: current callables from `jaxstro.numerics.integration`,
  `jaxstro.numerics.quadrature`, and the six direct quadrature re-exports from
  `jaxstro.numerics`.
- Produces: lazy `jaxstro.quad` and the canonical A0 top-level callable names.
- Preserves: exact signatures and object identity for every existing callable.

- [ ] **Step 1: Record the pre-change baseline**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_quadrature.py \
    tests/unit/test_numerics.py \
    tests/integration/test_integration_parity.py \
    tests/validation/test_grad_checks.py
  ```

  Expected: all tests pass. Record the exact pass and skip counts in the task
  notes before changing imports. A baseline failure stops A0.

- [ ] **Step 2: Write the failing public-surface tests**

  Create `tests/unit/quad/test_import_surface.py`:

  ```python
  import inspect

  import jaxstro
  from jaxstro.numerics import integration, quadrature


  def test_quad_is_a_lazy_public_top_level_module() -> None:
      assert jaxstro.quad.__name__ == "jaxstro.quad"
      assert "quad" in jaxstro.__all__


  def test_sampled_facade_is_exact_legacy_identity() -> None:
      assert jaxstro.quad.trapezoid is integration.trapz
      assert jaxstro.quad.cumulative_trapezoid is integration.cumulative_trapz
      assert jaxstro.quad.simpson is integration.simpson
      assert jaxstro.quad.cumulative_simpson is integration.cumulative_simpson


  def test_fixed_helper_facade_is_exact_legacy_identity() -> None:
      names = (
          "gauss_legendre_nodes",
          "gauss_laguerre_nodes",
          "gauss_hermite_nodes",
          "clenshaw_curtis_nodes",
          "hermite_e_basis",
          "hermite_coefficients",
      )
      for name in names:
          assert getattr(jaxstro.quad, name) is getattr(quadrature, name)
          assert getattr(jaxstro.numerics, name) is getattr(jaxstro.quad, name)


  def test_a0_facade_does_not_change_signatures() -> None:
      assert inspect.signature(jaxstro.quad.trapezoid) == inspect.signature(
          integration.trapz
      )
      assert inspect.signature(jaxstro.quad.simpson) == inspect.signature(
          integration.simpson
      )
  ```

  Create `tests/integration/test_quad_compatibility.py`:

  ```python
  import subprocess
  import sys


  def test_canonical_and_legacy_paths_match_in_a_clean_process() -> None:
      code = r'''
  import jax.numpy as jnp
  import jaxstro
  from jaxstro.numerics import integration, quadrature

  y = jnp.array([1.0, -2.0, 3.0, -4.0, 5.0])
  assert jnp.array_equal(
      jaxstro.quad.cumulative_trapezoid(y, dx=0.3),
      integration.cumulative_trapz(y, dx=0.3),
  )
  assert jaxstro.quad.gauss_hermite_nodes is quadrature.gauss_hermite_nodes
  assert jaxstro.numerics.gauss_hermite_nodes is jaxstro.quad.gauss_hermite_nodes
  '''
      subprocess.run([sys.executable, "-c", code], check=True)
  ```

- [ ] **Step 3: Run the RED gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_import_surface.py \
    tests/integration/test_quad_compatibility.py
  ```

  Expected: collection or import fails because `jaxstro.quad` does not exist.

- [ ] **Step 4: Add the minimal public facade**

  Create `src/jaxstro/quad/__init__.py` with direct aliases only:

  ```python
  """Canonical namespace for current integration foundations and methods."""

  from jaxstro.numerics.integration import cumulative_simpson
  from jaxstro.numerics.integration import (
      cumulative_trapz as cumulative_trapezoid,
  )
  from jaxstro.numerics.integration import simpson
  from jaxstro.numerics.integration import trapz as trapezoid
  from jaxstro.numerics.quadrature import (
      clenshaw_curtis_nodes,
      gauss_hermite_nodes,
      gauss_laguerre_nodes,
      gauss_legendre_nodes,
      hermite_coefficients,
      hermite_e_basis,
  )

  __all__ = [
      "clenshaw_curtis_nodes",
      "cumulative_simpson",
      "cumulative_trapezoid",
      "gauss_hermite_nodes",
      "gauss_laguerre_nodes",
      "gauss_legendre_nodes",
      "hermite_coefficients",
      "hermite_e_basis",
      "simpson",
      "trapezoid",
  ]
  ```

  Add `"quad"` between `"quantity"` and `"spatial"` in `__all__` in
  `src/jaxstro/__init__.py`. The existing `__getattr__` then provides the lazy
  module import; do not add an eager import of JAX or `jaxstro.numerics` to the
  root package.

- [ ] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_import_surface.py \
    tests/integration/test_quad_compatibility.py \
    tests/unit/test_quadrature.py \
    tests/integration/test_integration_parity.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/__init__.py src/jaxstro/quad \
    tests/unit/quad tests/integration/test_quad_compatibility.py
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check \
    src/jaxstro/__init__.py src/jaxstro/quad \
    tests/unit/quad tests/integration/test_quad_compatibility.py
  git diff --check
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/__init__.py src/jaxstro/quad/__init__.py \
    tests/unit/quad/__init__.py tests/unit/quad/test_import_surface.py \
    tests/integration/test_quad_compatibility.py
  git commit -m "feat(quad): add canonical compatibility facade"
  ```

### Task 2: Add typed result and tolerance contracts

**Files:**
- Create: `src/jaxstro/quad/result.py`
- Create: `src/jaxstro/quad/tolerance.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_result.py`
- Create: `tests/unit/quad/test_tolerance.py`

**Interfaces:**
- Consumes: JAX arrays and static norm configuration.
- Produces: `QuadStatus`, `ErrorKind`, `QuadError`, `QuadWork`, `QuadResult`,
  `MaxNorm`, `L1Norm`, `L2Norm`, `error_norm`, and `tolerance_threshold`.
- Preserves: all diagnostic leaves are nondifferentiable evidence; A3 owns the
  custom derivative that enforces that rule on adaptive results.

- [ ] **Step 1: Write failing result tests**

  Create `tests/unit/quad/test_result.py`:

  ```python
  import jax
  import jax.numpy as jnp

  from jaxstro import quad


  def _result(value):
      error = quad.QuadError(
          estimate=jnp.abs(value) * 0.0,
          norm=jnp.asarray(0.0),
          kind=jnp.asarray(quad.ErrorKind.EMBEDDED_RULE, dtype=jnp.int32),
          confidence_level=jnp.asarray(jnp.nan),
      )
      work = quad.QuadWork(
          evaluations=jnp.asarray(15, dtype=jnp.int32),
          refinements=jnp.asarray(0, dtype=jnp.int32),
          active_regions=jnp.asarray(1, dtype=jnp.int32),
          levels=jnp.asarray(0, dtype=jnp.int32),
          replicates=jnp.asarray(0, dtype=jnp.int32),
      )
      return quad.QuadResult(
          value=value,
          error=error,
          tolerance=jnp.asarray(1e-8),
          status=jnp.asarray(quad.QuadStatus.CONVERGED, dtype=jnp.int32),
          work=work,
      )


  def test_result_fields_are_checkpoint_stable() -> None:
      assert quad.QuadError._fields == (
          "estimate",
          "norm",
          "kind",
          "confidence_level",
      )
      assert quad.QuadWork._fields == (
          "evaluations",
          "refinements",
          "active_regions",
          "levels",
          "replicates",
      )
      assert quad.QuadResult._fields == (
          "value",
          "error",
          "tolerance",
          "status",
          "work",
      )


  def test_result_is_a_fixed_shape_jax_pytree() -> None:
      result = _result(jnp.array([1.0, 2.0]))
      leaves, structure = jax.tree.flatten(result)
      rebuilt = jax.tree.unflatten(structure, leaves)
      assert jax.tree.structure(rebuilt) == jax.tree.structure(result)
      assert jnp.array_equal(rebuilt.value, result.value)


  def test_status_and_error_codes_are_stable() -> None:
      assert int(quad.QuadStatus.CONVERGED) == 0
      assert int(quad.QuadStatus.MAX_EVALUATIONS) == 1
      assert int(quad.QuadStatus.MAX_REGIONS) == 2
      assert int(quad.QuadStatus.NONFINITE_INTEGRAND) == 3
      assert int(quad.QuadStatus.ROUNDOFF_LIMITED) == 4
      assert int(quad.QuadStatus.DIVERGENCE_SUSPECTED) == 5
      assert int(quad.QuadStatus.INVALID_INPUT) == 6
      assert int(quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE) == 7
      assert int(quad.ErrorKind.EMBEDDED_RULE) == 0
      assert int(quad.ErrorKind.REFINEMENT_DIFFERENCE) == 1
      assert int(quad.ErrorKind.SPARSE_GRID_SURPLUS) == 2
      assert int(quad.ErrorKind.REPLICATE_STANDARD_ERROR) == 3
      assert int(quad.ErrorKind.CONFIDENCE_INTERVAL_HALF_WIDTH) == 4
      assert int(quad.ErrorKind.UNAVAILABLE) == 5
  ```

- [ ] **Step 2: Write failing tolerance tests**

  Create `tests/unit/quad/test_tolerance.py`:

  ```python
  import jax
  import jax.numpy as jnp
  import pytest

  from jaxstro import quad


  @pytest.mark.parametrize(
      "norm,expected",
      (
          (quad.MaxNorm(), 5.0),
          (quad.L1Norm(), 9.0),
          (quad.L2Norm(), jnp.sqrt(35.0)),
      ),
  )
  def test_error_norm_reduces_payload_to_one_scalar(norm, expected) -> None:
      error = jnp.array([1.0, 3.0, 5.0])
      assert jnp.allclose(quad.error_norm(error, norm), expected)


  def test_complex_error_uses_magnitude() -> None:
      error = jnp.array([3.0 + 4.0j, 0.0 + 2.0j])
      assert quad.error_norm(error, quad.MaxNorm()) == 5.0


  def test_tolerance_is_absolute_or_relative_maximum() -> None:
      value = jnp.array([3.0, 4.0])
      tolerance = quad.tolerance_threshold(
          value,
          epsabs=1e-3,
          epsrel=1e-2,
          norm=quad.L2Norm(),
      )
      assert jnp.allclose(tolerance, 5e-2)


  def test_norm_configuration_is_static_under_jit() -> None:
      evaluate = jax.jit(
          lambda value, norm: quad.tolerance_threshold(
              value,
              epsabs=1e-6,
              epsrel=1e-3,
              norm=norm,
          )
      )
      assert jnp.allclose(
          evaluate(jnp.array([2.0, -3.0]), quad.MaxNorm()),
          3e-3,
      )
  ```

- [ ] **Step 3: Run the RED gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_result.py tests/unit/quad/test_tolerance.py
  ```

  Expected: import failures for the missing result and tolerance types.

- [ ] **Step 4: Implement the exact result records and codes**

  Create `src/jaxstro/quad/result.py`:

  ```python
  """Typed evidence returned by adaptive integration methods."""

  from enum import IntEnum
  from typing import Any, NamedTuple

  from jaxtyping import Array


  class QuadStatus(IntEnum):
      CONVERGED = 0
      MAX_EVALUATIONS = 1
      MAX_REGIONS = 2
      NONFINITE_INTEGRAND = 3
      ROUNDOFF_LIMITED = 4
      DIVERGENCE_SUSPECTED = 5
      INVALID_INPUT = 6
      ERROR_ESTIMATE_UNAVAILABLE = 7


  class ErrorKind(IntEnum):
      EMBEDDED_RULE = 0
      REFINEMENT_DIFFERENCE = 1
      SPARSE_GRID_SURPLUS = 2
      REPLICATE_STANDARD_ERROR = 3
      CONFIDENCE_INTERVAL_HALF_WIDTH = 4
      UNAVAILABLE = 5


  class QuadError(NamedTuple):
      estimate: Any
      norm: Array
      kind: Array
      confidence_level: Array


  class QuadWork(NamedTuple):
      evaluations: Array
      refinements: Array
      active_regions: Array
      levels: Array
      replicates: Array


  class QuadResult(NamedTuple):
      value: Any
      error: QuadError
      tolerance: Array
      status: Array
      work: QuadWork
  ```

- [ ] **Step 5: Implement static norm configurations**

  Create `src/jaxstro/quad/tolerance.py`:

  ```python
  """Norm and tolerance policies shared by quadrature controllers."""

  from dataclasses import dataclass
  from typing import Protocol

  import jax
  import jax.numpy as jnp
  from jaxtyping import Array


  class ErrorNorm(Protocol):
      def __call__(self, value: Array) -> Array: ...


  class _StaticNorm:
      def tree_flatten(self):
          return (), None

      @classmethod
      def tree_unflatten(cls, _aux, _children):
          return cls()


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class MaxNorm(_StaticNorm):
      def __call__(self, value: Array) -> Array:
          return jnp.max(jnp.abs(value))


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class L1Norm(_StaticNorm):
      def __call__(self, value: Array) -> Array:
          return jnp.sum(jnp.abs(value))


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class L2Norm(_StaticNorm):
      def __call__(self, value: Array) -> Array:
          return jnp.sqrt(jnp.sum(jnp.abs(value) ** 2))


  def error_norm(error: Array, norm: ErrorNorm) -> Array:
      """Reduce scalar, vector, array, or complex error evidence."""
      return norm(jnp.asarray(error))


  def tolerance_threshold(
      value: Array,
      *,
      epsabs: float | Array,
      epsrel: float | Array,
      norm: ErrorNorm,
  ) -> Array:
      """Return max(epsabs, epsrel * norm(value))."""
      value_norm = norm(jnp.asarray(value))
      dtype = jnp.result_type(value_norm, epsabs, epsrel, 0.0)
      absolute = jnp.asarray(epsabs, dtype=dtype)
      relative = jnp.asarray(epsrel, dtype=dtype) * value_norm
      return jnp.maximum(absolute, relative)
  ```

  Export all public types and helpers from `jaxstro.quad`.

- [ ] **Step 6: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_result.py tests/unit/quad/test_tolerance.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad tests/unit/quad
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check \
    src/jaxstro/quad tests/unit/quad
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro/quad
  git diff --check
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/__init__.py src/jaxstro/quad/result.py \
    src/jaxstro/quad/tolerance.py tests/unit/quad/test_result.py \
    tests/unit/quad/test_tolerance.py
  git commit -m "feat(quad): add result and tolerance contracts"
  ```

### Task 3: Add domain PyTrees and the finite affine map

**Files:**
- Create: `src/jaxstro/quad/domains.py`
- Create: `src/jaxstro/quad/transforms.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_domains.py`

**Interfaces:**
- Consumes: scalar JAX endpoints, a static tuple of breakpoint leaves, and
  reference nodes in `[-1, 1]`.
- Produces: `Interval`, `RightInfinite`, `LeftInfinite`, `Infinite`,
  `AffineMapResult`, `interval_orientation`, `sorted_breakpoints`,
  `interval_is_valid`, and `map_interval`.
- Defers: rational and tanh-sinh infinite-domain maps until a fixed/adaptive
  method consumes and validates them.

- [ ] **Step 1: Write failing domain and transformation tests**

  Create `tests/unit/quad/test_domains.py`:

  ```python
  import inspect

  import jax
  import jax.numpy as jnp

  from jaxstro import quad


  def test_interval_endpoints_and_breakpoints_are_dynamic_pytree_leaves() -> None:
      domain = quad.Interval(0.0, 2.0, breakpoints=(1.5, 0.5))
      leaves, treedef = jax.tree.flatten(domain)
      assert len(leaves) == 4
      rebuilt = jax.tree.unflatten(treedef, leaves)
      assert rebuilt.breakpoints == domain.breakpoints


  def test_breakpoint_count_is_static_structure() -> None:
      two = jax.tree.structure(
          quad.Interval(0.0, 1.0, breakpoints=(0.2, 0.8))
      )
      one = jax.tree.structure(quad.Interval(0.0, 1.0, breakpoints=(0.5,)))
      assert two != one


  def test_breakpoints_are_keyword_only() -> None:
      parameter = inspect.signature(quad.Interval).parameters["breakpoints"]
      assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


  def test_breakpoints_sort_in_oriented_domain_order() -> None:
      forward = quad.Interval(0.0, 2.0, breakpoints=(1.5, 0.5))
      reverse = quad.Interval(2.0, 0.0, breakpoints=(0.5, 1.5))
      assert jnp.array_equal(quad.sorted_breakpoints(forward), jnp.array([0.5, 1.5]))
      assert jnp.array_equal(quad.sorted_breakpoints(reverse), jnp.array([1.5, 0.5]))


  def test_duplicate_breakpoints_make_interval_invalid() -> None:
      domain = quad.Interval(0.0, 1.0, breakpoints=(0.5, 0.5))
      assert not bool(quad.interval_is_valid(domain))


  def test_affine_map_preserves_orientation_separately_from_jacobian() -> None:
      reference = jnp.array([-1.0, 0.0, 1.0])
      forward = quad.map_interval(quad.Interval(2.0, 4.0), reference)
      reverse = quad.map_interval(quad.Interval(4.0, 2.0), reference)
      assert jnp.array_equal(forward.x, jnp.array([2.0, 3.0, 4.0]))
      assert jnp.array_equal(reverse.x, forward.x)
      assert forward.jacobian == 1.0
      assert reverse.jacobian == 1.0
      assert forward.orientation == 1.0
      assert reverse.orientation == -1.0


  def test_zero_width_is_exact_zero_orientation() -> None:
      mapped = quad.map_interval(
          quad.Interval(3.0, 3.0),
          jnp.array([-0.5, 0.5]),
      )
      assert mapped.orientation == 0.0
      assert mapped.jacobian == 0.0
      assert jnp.all(mapped.x == 3.0)


  def test_dynamic_endpoints_work_under_jit() -> None:
      evaluate = jax.jit(
          lambda lower, upper: quad.map_interval(
              quad.Interval(lower, upper),
              jnp.array([-1.0, 0.0, 1.0]),
          ).x
      )
      assert jnp.array_equal(evaluate(1.0, 5.0), jnp.array([1.0, 3.0, 5.0]))
  ```

- [ ] **Step 2: Run the RED gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_domains.py
  ```

  Expected: import failures for missing domain and mapping types.

- [ ] **Step 3: Implement immutable domain PyTrees**

  Create `src/jaxstro/quad/domains.py` with registered frozen dataclasses. Use
  this exact flattening contract:

  ```python
  """One-dimensional integration domains."""

  from dataclasses import dataclass, field
  from typing import Any

  import jax
  import jax.numpy as jnp
  from jaxtyping import Array


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class Interval:
      lower: Any
      upper: Any
      breakpoints: tuple[Any, ...] = field(default=(), kw_only=True)

      def tree_flatten(self):
          children = (self.lower, self.upper, *self.breakpoints)
          return children, len(self.breakpoints)

      @classmethod
      def tree_unflatten(cls, count: int, children):
          lower, upper, *breakpoints = children
          if len(breakpoints) != count:
              raise ValueError("invalid Interval breakpoint PyTree")
          return cls(lower, upper, breakpoints=tuple(breakpoints))


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class RightInfinite:
      lower: Any

      def tree_flatten(self):
          return (self.lower,), None

      @classmethod
      def tree_unflatten(cls, _aux, children):
          return cls(children[0])


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class LeftInfinite:
      upper: Any

      def tree_flatten(self):
          return (self.upper,), None

      @classmethod
      def tree_unflatten(cls, _aux, children):
          return cls(children[0])


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class Infinite:
      def tree_flatten(self):
          return (), None

      @classmethod
      def tree_unflatten(cls, _aux, _children):
          return cls()


  def interval_orientation(domain: Interval) -> Array:
      delta = jnp.asarray(domain.upper) - jnp.asarray(domain.lower)
      return jnp.sign(delta)


  def sorted_breakpoints(domain: Interval) -> Array:
      dtype = jnp.result_type(domain.lower, domain.upper, *domain.breakpoints, 0.0)
      if not domain.breakpoints:
          return jnp.empty((0,), dtype=dtype)
      values = jnp.asarray(domain.breakpoints, dtype=dtype)
      ascending = jnp.sort(values)
      return jnp.where(interval_orientation(domain) < 0.0, ascending[::-1], ascending)


  def interval_is_valid(domain: Interval) -> Array:
      lower = jnp.asarray(domain.lower)
      upper = jnp.asarray(domain.upper)
      finite = jnp.isfinite(lower) & jnp.isfinite(upper)
      points = sorted_breakpoints(domain)
      ascending = jnp.sort(points)
      lo = jnp.minimum(lower, upper)
      hi = jnp.maximum(lower, upper)
      interior = jnp.all((points > lo) & (points < hi))
      unique = jnp.all(jnp.diff(ascending) > 0.0)
      return finite & interior & unique
  ```

- [ ] **Step 4: Implement the finite affine map**

  Create `src/jaxstro/quad/transforms.py`:

  ```python
  """Reference-domain maps used by quadrature rules."""

  from typing import NamedTuple

  import jax.numpy as jnp
  from jaxtyping import Array

  from .domains import Interval, interval_is_valid, interval_orientation


  class AffineMapResult(NamedTuple):
      x: Array
      jacobian: Array
      orientation: Array
      valid: Array


  def map_interval(domain: Interval, reference: Array) -> AffineMapResult:
      lower = jnp.asarray(domain.lower)
      upper = jnp.asarray(domain.upper)
      lo = jnp.minimum(lower, upper)
      hi = jnp.maximum(lower, upper)
      half_width = 0.5 * (hi - lo)
      midpoint = 0.5 * (hi + lo)
      reference = jnp.asarray(reference)
      return AffineMapResult(
          x=midpoint + half_width * reference,
          jacobian=half_width,
          orientation=interval_orientation(domain),
          valid=interval_is_valid(domain),
      )
  ```

  Export the public domain and transform names from `jaxstro.quad`.

- [ ] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_domains.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/domains.py src/jaxstro/quad/transforms.py \
    src/jaxstro/quad/__init__.py tests/unit/quad/test_domains.py
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check \
    src/jaxstro/quad/domains.py src/jaxstro/quad/transforms.py \
    src/jaxstro/quad/__init__.py tests/unit/quad/test_domains.py
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro/quad
  git diff --check
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/__init__.py src/jaxstro/quad/domains.py \
    src/jaxstro/quad/transforms.py tests/unit/quad/test_domains.py
  git commit -m "feat(quad): add domain and affine-map contracts"
  ```

## Checkpoint A0-C1: foundation architecture review

After Task 3, dispatch a fresh read-only subagent. Give it the approved design,
Tasks 1 through 3 commits, and this exact review request:

```text
Review jaxstro.quad A0 Tasks 1-3 for architectural conformance. Check exact
legacy callable identity, result/error/work field semantics, stable status and
error codes, static norm behavior, PyTree reconstruction, oriented affine maps,
empty breakpoints, JIT behavior, and any accidental implementation of A1-A3.
Do not edit. Return findings by severity with exact files and tests.
```

Resolve every Critical or Important finding with a failing regression test,
minimal correction, focused verification, and one review-fix commit. Record
Minor findings for the next relevant plan only when they are genuinely outside
A0.

### Task 4: Add measure configuration contracts

**Files:**
- Create: `src/jaxstro/quad/measures.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_measures.py`

**Interfaces:**
- Consumes: static density callable and `jaxstro.quantity.Unit` metadata.
- Produces: `LebesgueMeasure`, `WeightedMeasure`, `JacobiMeasure`,
  `LaguerreMeasure`, `PhysicistsHermiteMeasure`, and `StandardNormalMeasure`.
- Defers: Gaussian recurrence coefficients, density evaluation, and matched
  rule-domain dispatch to Phase A1.

- [ ] **Step 1: Write failing measure tests**

  Create `tests/unit/quad/test_measures.py`:

  ```python
  import inspect

  import jax
  import jax.numpy as jnp
  import pytest

  from jaxstro import quad
  from jaxstro.quantity import dimensionless


  def _density(x, args):
      return jnp.exp(-args[0] * x)


  def test_weighted_measure_metadata_is_static() -> None:
      measure = quad.WeightedMeasure(
          _density,
          density_unit=dimensionless,
          normalized=False,
      )
      leaves, structure = jax.tree.flatten(measure)
      assert leaves == []
      assert jax.tree.unflatten(structure, leaves) == measure


  def test_measure_constructor_signatures_match_the_approved_contract() -> None:
      weighted = inspect.signature(quad.WeightedMeasure).parameters
      assert weighted["density"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
      assert weighted["density_unit"].kind is inspect.Parameter.KEYWORD_ONLY
      assert weighted["density_unit"].default is inspect.Parameter.empty
      assert weighted["normalized"].kind is inspect.Parameter.KEYWORD_ONLY
      assert weighted["normalized"].default is False

      defaults = {
          quad.JacobiMeasure: False,
          quad.LaguerreMeasure: False,
          quad.PhysicistsHermiteMeasure: False,
      }
      for measure_type, expected_default in defaults.items():
          normalized = inspect.signature(measure_type).parameters["normalized"]
          assert normalized.kind is inspect.Parameter.KEYWORD_ONLY
          assert normalized.default is expected_default

      assert not inspect.signature(quad.StandardNormalMeasure).parameters


  @pytest.mark.parametrize(
      "measure",
      (
          quad.LebesgueMeasure(),
          quad.WeightedMeasure(_density, density_unit=dimensionless),
          quad.JacobiMeasure(0.25, 0.5),
          quad.LaguerreMeasure(0.25),
          quad.PhysicistsHermiteMeasure(),
          quad.StandardNormalMeasure(),
      ),
  )
  def test_every_measure_round_trips_through_its_static_pytree(measure) -> None:
      leaves, structure = jax.tree.flatten(measure)
      assert leaves == []
      assert jax.tree.unflatten(structure, leaves) == measure


  def test_normalized_is_a_declaration_not_a_numerical_action() -> None:
      raw = quad.WeightedMeasure(
          _density,
          density_unit=dimensionless,
          normalized=False,
      )
      declared = quad.WeightedMeasure(
          _density,
          density_unit=dimensionless,
          normalized=True,
      )
      assert raw.density is declared.density
      assert not raw.normalized
      assert declared.normalized


  @pytest.mark.parametrize(
      "factory",
      (
          lambda: quad.JacobiMeasure(-1.0, 0.0),
          lambda: quad.JacobiMeasure(0.0, -1.0),
          lambda: quad.LaguerreMeasure(-1.0),
      ),
  )
  def test_nonintegrable_classical_parameters_raise_eagerly(factory) -> None:
      with pytest.raises(ValueError, match="greater than -1"):
          factory()


  def test_standard_normal_is_explicitly_normalized() -> None:
      assert quad.StandardNormalMeasure().normalized
      assert not quad.PhysicistsHermiteMeasure().normalized
  ```

- [ ] **Step 2: Run the RED gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_measures.py
  ```

  Expected: import failures for the missing measure declarations.

- [ ] **Step 3: Implement immutable static measure PyTrees**

  Create `src/jaxstro/quad/measures.py`. Use a shared static-PyTree mixin and
  validate classical integrability eagerly:

  ```python
  """Measure declarations for fixed and adaptive integration."""

  from dataclasses import dataclass, field
  from typing import Any, Callable

  import jax

  from jaxstro.quantity import Unit


  class _StaticMeasure:
      def tree_flatten(self):
          metadata = tuple(
              (name, getattr(self, name)) for name in self.__dataclass_fields__
          )
          return (), metadata

      @classmethod
      def tree_unflatten(cls, metadata, _children):
          return cls(**dict(metadata))


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class LebesgueMeasure(_StaticMeasure):
      pass


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class WeightedMeasure(_StaticMeasure):
      density: Callable[[Any, Any], Any]
      density_unit: Unit = field(kw_only=True)
      normalized: bool = field(default=False, kw_only=True)


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class JacobiMeasure(_StaticMeasure):
      alpha: float
      beta: float
      normalized: bool = field(default=False, kw_only=True)

      def __post_init__(self) -> None:
          if self.alpha <= -1.0 or self.beta <= -1.0:
              raise ValueError("Jacobi alpha and beta must be greater than -1")


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class LaguerreMeasure(_StaticMeasure):
      alpha: float = 0.0
      normalized: bool = field(default=False, kw_only=True)

      def __post_init__(self) -> None:
          if self.alpha <= -1.0:
              raise ValueError("Laguerre alpha must be greater than -1")


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class PhysicistsHermiteMeasure(_StaticMeasure):
      normalized: bool = field(default=False, kw_only=True)


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class StandardNormalMeasure(_StaticMeasure):
      @property
      def normalized(self) -> bool:
          return True
  ```

  Do not evaluate, normalize, or differentiate a measure in A0. Export all six
  declarations from `jaxstro.quad`.

- [ ] **Step 4: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_measures.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/measures.py src/jaxstro/quad/__init__.py \
    tests/unit/quad/test_measures.py
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check \
    src/jaxstro/quad/measures.py src/jaxstro/quad/__init__.py \
    tests/unit/quad/test_measures.py
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro/quad
  git diff --check
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/__init__.py src/jaxstro/quad/measures.py \
    tests/unit/quad/test_measures.py
  git commit -m "feat(quad): add measure configuration contracts"
  ```

### Task 5: Register and document the A0 public owner

**Files:**
- Create: `src/jaxstro/quad/_contracts.py`
- Modify: `src/jaxstro/contracts/registry.py`
- Modify: `tests/unit/test_contract_manifests.py`
- Modify: `tests/integration/test_api_reference.py`
- Modify: `tests/integration/test_grouped_api_reference.py`
- Modify: `tests/integration/test_method_page_contract.py`
- Modify: `tests/integration/test_myst_semantic_grammar.py`
- Create: `docs/50-api/approximation-integration/quad.md`
- Modify: `docs/50-api/approximation-integration/integration.md`
- Modify: `docs/50-api/approximation-integration/quadrature.md`
- Modify: `docs/50-api/api.md`
- Modify: `docs/20-methods/approximation-integration/cumulative-trapz.md`
- Modify: `docs/20-methods/approximation-integration/quadrature.md`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`
- Regenerate: `docs/validation/contracts.json`
- Regenerate: `docs/50-api/research-infrastructure/contracts.md`
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: Tasks 1 through 4 public facade and configurations.
- Produces: one contract-registry owner, one API owner page, canonical current
  method imports, and fresh generated contract artifacts.
- Preserves: `/cumulative-trapz`, `/quadrature`, and every existing API route.

- [ ] **Step 1: Write failing ownership and API tests**

  In `tests/unit/test_contract_manifests.py`, add `"quad"` between
  `"quantity"` and `"spatial"` in the tuple that constructs `PUBLIC`, and add
  this clean-process isolation assertion:

  ```python
  def test_quad_contract_collection_does_not_import_runtime_quad() -> None:
      code = """
  import sys
  from jaxstro.contracts import collect_contracts
  assert 'jaxstro.quad' not in sys.modules
  collect_contracts()
  assert 'jaxstro.quad' not in sys.modules
  """
      subprocess.run([sys.executable, "-c", code], check=True)
  ```

  In `tests/integration/test_api_reference.py`, add `"quad"` to
  `public_modules` between `"quantity"` and `"spatial"`.

  In `tests/integration/test_grouped_api_reference.py`, replace these two
  owner mappings:

  ```python
  "approximation-integration/integration.md": "jaxstro.numerics.integration",
  "approximation-integration/quadrature.md": "jaxstro.numerics.quadrature",
  ```

  with the single canonical mapping:

  ```python
  "approximation-integration/quad.md": "jaxstro.quad",
  ```

  Directly after `PRIVATE_NUMERICS_MODULE_EXCLUSIONS`, add:

  ```python
  COMPATIBILITY_NUMERICS_MODULE_EXCLUSIONS = {
      "integration": "temporary sampled-integration compatibility import",
      "quadrature": "temporary fixed-helper compatibility import",
  }
  ```

  In `test_every_public_numerics_module_has_exactly_one_owner_page`, preserve
  the private-module equality and replace the `public_owners` comprehension
  with:

  ```python
  compatibility = set(COMPATIBILITY_NUMERICS_MODULE_EXCLUSIONS)
  assert compatibility < discovered
  public_owners = {
      f"jaxstro.numerics.{name}"
      for name in discovered
      if not name.startswith("_") and name not in compatibility
  }
  ```

  This is an explicit migration exception, not permission to omit any other
  public numerical module. Then add this owner-page assertion:

  ```python
  def test_quad_owner_page_teaches_canonical_and_legacy_boundaries() -> None:
      text = (API_ROOT / "approximation-integration/quad.md").read_text()
      assert "`jaxstro.quad`" in text
      assert "from jaxstro import quad" in text
      assert "jaxstro.numerics.integration" in text
      assert "jaxstro.numerics.quadrature" in text
      assert "temporary compatibility" in text
      assert "does not yet provide adaptive integration" in text
  ```

  In `test_status_counts_the_corrected_api_surface`, replace the owner-page
  assertion with:

  ```python
  assert "37 current owner pages, including `jaxstro.quad`" in text
  ```

  In `tests/integration/test_method_page_contract.py`, change the current owner
  and API link for both existing integration pages:

  ```python
  "approximation-integration/cumulative-trapz.md": (
      "jaxstro.quad",
      "../../50-api/approximation-integration/quad.md",
      ("eq-trapezoid-panel", "eq-cumulative-trapezoid", "eq-trapezoid-error"),
  ),
  "approximation-integration/quadrature.md": (
      "jaxstro.quad",
      "../../50-api/approximation-integration/quad.md",
      (
          "eq-fixed-node-quadrature",
          "eq-gaussian-exactness",
          "eq-standard-normal-hermite",
      ),
  ),
  ```

  In the same test file, replace these two imports:

  ```python
  from jaxstro.numerics.integration import cumulative_trapz, trapz
  from jaxstro.numerics.quadrature import gauss_laguerre_nodes
  ```

  with:

  ```python
  from jaxstro import quad
  ```

  Make these exact probe and claim substitutions, preserving every assertion's
  expected exception and value:

  ```text
  `trapz` -> `trapezoid` in the runtime-boundary prose assertion
  trapz(...) -> quad.trapezoid(...)
  cumulative_trapz(...) -> quad.cumulative_trapezoid(...)
  gauss_laguerre_nodes(...) -> quad.gauss_laguerre_nodes(...)
  ```

  Update the exact manifest count in
  `tests/integration/test_myst_semantic_grammar.py` from 163 to 164; do not
  relax the equality.

- [ ] **Step 2: Run the RED gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_contract_manifests.py \
    tests/integration/test_api_reference.py \
    tests/integration/test_grouped_api_reference.py \
    tests/integration/test_method_page_contract.py \
    tests/integration/test_myst_semantic_grammar.py
  ```

  Expected: failures for the absent `quad` sidecar and API page.

- [ ] **Step 3: Add the lightweight module contract**

  Create `src/jaxstro/quad/_contracts.py`:

  ```python
  from jaxstro.contracts._core import module_contract
  from jaxstro.contracts.schema import MaturityLevel


  MODULE_CONTRACT = module_contract(
      "quad",
      (
          "Canonical integration namespace, typed configuration, and current "
          "sampled/fixed helper facade."
      ),
      (
          "Adaptive controllers, quantity-valued integration, physical-model "
          "policy, inference, ODE solving, or scientific acceptance."
      ),
      (
          "Configuring domains, measures, tolerances, results, and current "
          "integration helpers."
      ),
      (
          "Caller-owned units; Phase A0 records metadata but performs no "
          "quantity integration."
      ),
      maturity=MaturityLevel.EXPERIMENTAL,
  )
  ```

  In `src/jaxstro/contracts/registry.py`, add `"quad"` between `"quantity"` and
  `"spatial"` in the exact tuple passed to `_load_sidecar`. Do not import
  `jaxstro.quad` from the registry.

- [ ] **Step 4: Add one complete A0 API owner page**

  Create `docs/50-api/approximation-integration/quad.md` with the existing eight
  required headings in this exact order:

  ````markdown
  ---
  title: Jaxstro quadrature foundation
  description: Canonical sampled and fixed-rule facade plus Phase A0 integration contracts.
  ---

  # Jaxstro quadrature foundation

  ## Owner import path

  `jaxstro.quad`

  ## Purpose

  This is the canonical integration namespace. In Phase A0 it exposes the
  currently supported sampled and fixed-node helpers together with typed domain,
  measure, tolerance, and result foundations. It does not yet provide adaptive
  integration.

  ## Public records and callables

  Sampled values:

  - `trapezoid`
  - `cumulative_trapezoid`
  - `simpson`
  - `cumulative_simpson`

  Fixed-rule helpers:

  - `gauss_legendre_nodes`
  - `gauss_laguerre_nodes`
  - `gauss_hermite_nodes`
  - `clenshaw_curtis_nodes`
  - `hermite_e_basis`
  - `hermite_coefficients`

  Domains and measures:

  - `Interval`, `RightInfinite`, `LeftInfinite`, and `Infinite`
  - `LebesgueMeasure`, `WeightedMeasure`, `JacobiMeasure`,
    `LaguerreMeasure`, `PhysicistsHermiteMeasure`, and
    `StandardNormalMeasure`

  Results and tolerances:

  - `QuadStatus`, `ErrorKind`, `QuadError`, `QuadWork`, and `QuadResult`
  - `MaxNorm`, `L1Norm`, `L2Norm`, `error_norm`, and
    `tolerance_threshold`

  ## Shape and dtype expectations

  Sampled functions reduce or cumulatively retain one selected array axis under
  their existing contracts. Node factories return two arrays with shape `(n,)`.
  `hermite_e_basis(g, n_max)` returns shape `(n_max + 1, g.shape[0])`, and
  `hermite_coefficients` returns shape `(n_max + 1,)`. Domain endpoints are
  scalar numerical PyTree leaves; breakpoint count is static.

  ## JAX transforms and AD classification

  The sampled functions preserve their current JIT and differentiation behavior.
  Fixed nodes and weights are generated as host-side constants; downstream JAX
  calculations differentiate through integrand values, not node construction.
  Result records and domains are PyTrees. Method-level replay AD does not exist
  until Phase A3.

  ## Failure behavior

  Existing sampled-grid shape, uniformity, parity, and rule-order failures remain
  unchanged. Phase A0 defines adaptive status codes but no controller emits them
  yet. Infinite-domain declarations are configuration only until a later method
  supplies and validates the corresponding transformation.

  ## Contract and evidence links

  Review [integration from samples](../../20-methods/approximation-integration/cumulative-trapz.md),
  [fixed-node quadrature](../../20-methods/approximation-integration/quadrature.md),
  and the [validation index](../../60-validation/validation.md).

  ## Canonical import example

  ```python
  from jaxstro import quad
  from jaxstro.quad import Interval

  nodes, weights = quad.gauss_legendre_nodes(8)
  domain = Interval(-1.0, 1.0)
  ```

  The old `jaxstro.numerics.integration` and
  `jaxstro.numerics.quadrature` paths are temporary compatibility surfaces. A0
  preserves exact callable identity and does not issue deprecation warnings.
  ````

- [ ] **Step 5: Wire stable navigation and canonical current examples**

  In `docs/50-api/api.md`, replace the existing Approximation and integration
  entry with this exact current-owner and migration split:

  ```markdown
  - **Approximation and integration:** [](./approximation-integration/interpolation.md),
    [](./approximation-integration/regular-grid.md),
    [](./approximation-integration/splines.md), and
    [`jaxstro.quad`](./approximation-integration/quad.md).
    The [sampled-integration](./approximation-integration/integration.md) and
    [fixed-quadrature](./approximation-integration/quadrature.md) pages document
    temporary compatibility paths, not current owners.
  ```

  In `docs/myst.yml`, add this entry once beneath Approximation and integration,
  before the two retained legacy API routes:

  ```yaml
  - file: 50-api/approximation-integration/quad.md
  ```

  Preserve the existing `integration.md` and `quadrature.md` entries so their
  stable routes remain migration references. Add this authored route to
  `docs/route-manifest.json`:

  ```json
  "50-api/approximation-integration/quad.md": "/quad-api"
  ```

  On `docs/20-methods/approximation-integration/cumulative-trapz.md`, make these
  bounded replacements while preserving all derivations and caveats:

  ```text
  short_title: Cumulative trapz -> short_title: Cumulative trapezoid
  every exact code token `cumulative_trapz` -> `cumulative_trapezoid`
  every remaining exact code token `trapz` -> `trapezoid`
  ../../50-api/approximation-integration/integration.md -> ../../50-api/approximation-integration/quad.md
  ```

  Replace that page's complete executable example with:

  ```python
  import jax.numpy as jnp

  from jaxstro import quad

  x = jnp.linspace(0.0, 1.0, 101)
  y = x**2
  running = quad.cumulative_trapezoid(y, x)
  total = quad.trapezoid(y, x)

  assert running.shape == y.shape
  assert running[0] == 0.0
  assert jnp.allclose(running[-1], total)
  ```

  Replace the complete Connected ideas admonition with:

  ```markdown
  ## Connected ideas

  :::{seealso}
  Connect integration units to
  [](../../30-representations/units-quantities/quantities.md), approximation error
  to [](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md),
  owner signatures to [](../../50-api/approximation-integration/quad.md), and
  executable checks to [](../../60-validation/validation.md). The
  [legacy sampled-integration page](../../50-api/approximation-integration/integration.md)
  records the temporary import-name mapping. Fixed-node Gaussian rules are in
  [](./quadrature.md).
  :::
  ```

  On `docs/20-methods/approximation-integration/quadrature.md`, make these exact
  replacements while preserving all formulas and limitations:

  ```text
  ../../50-api/approximation-integration/quadrature.md -> ../../50-api/approximation-integration/quad.md
  ```

  Replace that page's complete executable example with:

  ```python
  import jax.numpy as jnp

  from jaxstro import quad

  legendre_x, legendre_w = quad.gauss_legendre_nodes(8)
  poly_integral = jnp.sum(legendre_w * legendre_x**6)

  normal_x, normal_w = quad.gauss_hermite_nodes(8)
  normal_second_moment = jnp.sum(normal_w * normal_x**2)

  assert jnp.allclose(poly_integral, 2.0 / 7.0)
  assert jnp.allclose(jnp.sum(normal_w), 1.0)
  assert jnp.allclose(normal_second_moment, 1.0)
  ```

  Replace the complete Connected ideas admonition with:

  ```markdown
  ## Connected ideas

  :::{seealso}
  Connect weighted integrals to
  [](../../10-foundations/mathematical-objects/probability-and-distributions.md),
  units to [](../../30-representations/units-quantities/quantities.md), owner
  signatures to [](../../50-api/approximation-integration/quad.md), and evidence
  to [](../../60-validation/validation.md). The
  [legacy fixed-quadrature page](../../50-api/approximation-integration/quadrature.md)
  records the temporary import-name mapping. Sampled Newton-Cotes rules are in
  [](./cumulative-trapz.md); delegated adaptive methods are described in
  [](./adaptive-quadrature.md).
  :::
  ```

  Do not leave either legacy numerical module described as a current owner on
  these method pages. Do not edit the adaptive or QMC guide in A0.

  Replace `docs/50-api/approximation-integration/integration.md` completely:

  ````markdown
  ---
  title: Legacy sampled-integration import path
  ---

  # Legacy sampled-integration import path

  ```{important}
  `jaxstro.quad` is the canonical owner. `jaxstro.numerics.integration` is a
  temporary compatibility path retained while sibling packages migrate.
  ```

  | Legacy name | Canonical name |
  | --- | --- |
  | `jaxstro.numerics.integration.trapz` | `jaxstro.quad.trapezoid` |
  | `jaxstro.numerics.integration.cumulative_trapz` | `jaxstro.quad.cumulative_trapezoid` |
  | `jaxstro.numerics.integration.simpson` | `jaxstro.quad.simpson` |
  | `jaxstro.numerics.integration.cumulative_simpson` | `jaxstro.quad.cumulative_simpson` |

  Phase A0 preserves exact callable identity and emits no deprecation warning.
  Use the [Jaxstro quadrature foundation](./quad.md) for the current API.
  ````

  Prepend this exact provisional evidence entry to `STATUS.md`:

  ```text
  previous: Jaxstro.quad Phase A0 API ownership was established locally (2026-07-15): 37 current owner pages, including `jaxstro.quad`, now cover the importable public surface; 164 stable routes remain; and the sampled-integration and fixed-quadrature API routes are retained as migration-only references. Exact canonical/legacy callable identity remains in force. The complete A0 gate, checkpoint dispositions, and Anna review are still pending; no sibling migration, deprecation, publication, push, or live-site change occurred.
  ```

  Replace `docs/50-api/approximation-integration/quadrature.md` completely:

  ````markdown
  ---
  title: Legacy fixed-quadrature import path
  ---

  # Legacy fixed-quadrature import path

  ```{important}
  `jaxstro.quad` is the canonical owner. `jaxstro.numerics.quadrature` is a
  temporary compatibility path retained while sibling packages migrate.
  ```

  | Legacy name | Canonical name |
  | --- | --- |
  | `jaxstro.numerics.quadrature.gauss_legendre_nodes` | `jaxstro.quad.gauss_legendre_nodes` |
  | `jaxstro.numerics.quadrature.gauss_laguerre_nodes` | `jaxstro.quad.gauss_laguerre_nodes` |
  | `jaxstro.numerics.quadrature.gauss_hermite_nodes` | `jaxstro.quad.gauss_hermite_nodes` |
  | `jaxstro.numerics.quadrature.clenshaw_curtis_nodes` | `jaxstro.quad.clenshaw_curtis_nodes` |
  | `jaxstro.numerics.quadrature.hermite_e_basis` | `jaxstro.quad.hermite_e_basis` |
  | `jaxstro.numerics.quadrature.hermite_coefficients` | `jaxstro.quad.hermite_coefficients` |

  Phase A0 preserves exact callable identity and emits no deprecation warning.
  Use the [Jaxstro quadrature foundation](./quad.md) for the current API.
  ````

- [ ] **Step 6: Regenerate and verify contract artifacts**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/build_contract_registry.py --emit
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/build_contract_registry.py --check
  ```

  Expected: both commands exit zero and the second prints
  `scientific contract artifacts fresh`.

- [ ] **Step 7: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_contract_manifests.py \
    tests/unit/test_build_contract_registry_script.py \
    tests/integration/test_api_reference.py \
    tests/integration/test_grouped_api_reference.py \
    tests/integration/test_method_page_contract.py \
    tests/integration/test_methods_information_architecture.py \
    tests/integration/test_myst_semantic_grammar.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/_contracts.py src/jaxstro/contracts/registry.py \
    tests/unit/test_contract_manifests.py \
    tests/integration/test_api_reference.py \
    tests/integration/test_grouped_api_reference.py \
    tests/integration/test_method_page_contract.py \
    tests/integration/test_myst_semantic_grammar.py
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check \
    src/jaxstro/quad/_contracts.py src/jaxstro/contracts/registry.py \
    tests/unit/test_contract_manifests.py \
    tests/integration/test_api_reference.py \
    tests/integration/test_grouped_api_reference.py \
    tests/integration/test_method_page_contract.py \
    tests/integration/test_myst_semantic_grammar.py
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
  git diff --check
  ```

  Expected: all commands exit zero. Commit the explicit runtime, test, authored
  documentation, and generated contract paths only:

  ```bash
  git add src/jaxstro/quad/_contracts.py src/jaxstro/contracts/registry.py \
    tests/unit/test_contract_manifests.py \
    tests/integration/test_api_reference.py \
    tests/integration/test_grouped_api_reference.py \
    tests/integration/test_method_page_contract.py \
    tests/integration/test_myst_semantic_grammar.py \
    docs/50-api/approximation-integration/quad.md docs/50-api/api.md \
    docs/50-api/approximation-integration/integration.md \
    docs/50-api/approximation-integration/quadrature.md \
    docs/20-methods/approximation-integration/cumulative-trapz.md \
    docs/20-methods/approximation-integration/quadrature.md \
    docs/myst.yml docs/route-manifest.json \
    docs/validation/contracts.json \
    docs/50-api/research-infrastructure/contracts.md STATUS.md
  git commit -m "docs(quad): register the Phase A0 public owner"
  ```

## Checkpoint A0-C2: complete-slice review

After Task 5, dispatch two fresh read-only subagents in parallel.

Runtime review request:

```text
Review complete jaxstro.quad Phase A0 runtime and focused tests against the
approved design and A0 plan. Check public/private ownership, callable identity,
PyTree/static boundaries, status and error semantics, oriented maps, measure
normalization declarations, import isolation, and accidental future-method
claims. Do not edit. Return findings by severity with exact evidence.
```

Documentation/evidence review request:

```text
Review the Phase A0 API owner, current sampled/fixed method-page edits, TOC,
route manifest, module contract, and generated registry. Check that current
claims match runtime, legacy paths are clearly temporary, adaptive/QMC remain
unclaimed, ASCII/LaTeX rules hold, and no course framing or superiority claim
appears. Do not edit. Return findings by severity with exact evidence.
```

Resolve every Critical or Important finding with a failing regression test or
document contract, the minimal correction, focused verification, and one
review-fix commit. Re-run both review prompts only when a correction changes the
reviewed architecture or claim boundary.

### Task 6: Run the complete A0 gate and record the handoff

**Files:**
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: Tasks 1 through 5 plus both checkpoint dispositions.
- Produces: one verified Phase A0 repository state and the exact Phase A1 next
  action.
- Preserves: no publication, push, sibling migration, deprecation, or adaptive
  method implementation.

- [ ] **Step 1: Run all focused A0 tests**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad \
    tests/unit/test_quadrature.py \
    tests/unit/test_numerics.py \
    tests/unit/test_contract_manifests.py \
    tests/unit/test_build_contract_registry_script.py \
    tests/integration/test_quad_compatibility.py \
    tests/integration/test_integration_parity.py \
    tests/integration/test_api_reference.py \
    tests/integration/test_grouped_api_reference.py \
    tests/integration/test_method_page_contract.py \
    tests/integration/test_methods_information_architecture.py \
    tests/integration/test_myst_semantic_grammar.py \
    tests/validation/test_grad_checks.py
  ```

  Expected: all tests pass with only already-classified optional-dependency
  skips. Record exact counts; do not copy expected counts from an older status.

- [ ] **Step 2: Run static and generated-artifact gates**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad src/jaxstro/__init__.py src/jaxstro/contracts/registry.py \
    tests/unit/quad tests/integration/test_quad_compatibility.py
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check \
    src/jaxstro/quad src/jaxstro/__init__.py src/jaxstro/contracts/registry.py \
    tests/unit/quad tests/integration/test_quad_compatibility.py
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/build_contract_registry.py --check
  git diff --check
  ```

  Expected: every command exits zero.

- [ ] **Step 3: Run the strict documentation gate**

  Run:

  ```bash
  bash scripts/check_docs.sh
  ```

  Expected: the strict MyST build, route/link audit, and final artifact checks
  pass. Do not run Pages, publish, or push.

- [ ] **Step 4: Update `STATUS.md` with exact evidence**

  Replace the current `next:` line with:

  ```text
  next: Anna reviews the completed jaxstro.quad Phase A0 foundation and checkpoint dispositions. After approval, write the Phase A1 fixed-rule implementation plan; do not begin A1, adaptive methods, sibling migrations, deprecations, publication, or push from this task.
  ```

  Add one `previous:` entry containing:

- the exact A0 commits;
- exact test counts and expected skips;
- Ruff, formatting, MyPy, contract freshness, and docs-gate results;
- checkpoint findings and their dispositions;
- the canonical/legacy identity state; and
- the explicit statement that no numerical behavior, dependency, sibling,
  quantity-adoption, adaptive-method, publication, or live-site state changed.

- [ ] **Step 5: Commit the verified handoff**

  Run:

  ```bash
  git add STATUS.md
  git commit -m "docs(quad): record Phase A0 verification"
  git status --short --branch
  ```

  Expected: the commit succeeds. The only remaining untracked path is the
  pre-existing `.superpowers/`; no task-owned file is dirty.

## A0 stop conditions

Stop without advancing to A1 if any of these occurs:

- an existing sampled or fixed-helper call changes value, dtype, shape, error,
  signature, JIT behavior, or gradient behavior;
- cumulative trapezoid loses exact `dx`-outside byte parity;
- probabilists' Hermite nodes or weights lose exact byte parity;
- a canonical facade callable is not object-identical to its legacy owner;
- contract collection imports `jaxstro.quad` runtime code;
- dynamic domain endpoints become static or breakpoint count becomes dynamic;
- orientation is folded into an absolute Jacobian;
- a normalized measure performs an undocumented normalization;
- A0 documentation claims adaptive, weighted-rule, QMC, quantity-integration,
  or SOTA behavior that is not implemented;
- the strict docs gate changes an existing stable route; or
- any Critical or Important checkpoint finding remains unresolved.

## Phase A continuation

After Anna approves the completed A0 slice, write a fresh Phase A1 plan against
the verified interfaces. A1 will own sampled-code inversion, the shared Gaussian
recurrence engine, the exact Hermite compatibility exception, the shared
Chebyshev substrate, Fejer rules, fixed tanh-sinh, the fixed evaluator, the
approved consistent `dx` signature extension, and quantity-free raw-array
validation. A2, A3, and A4 remain separate plans and do not begin from A0.
