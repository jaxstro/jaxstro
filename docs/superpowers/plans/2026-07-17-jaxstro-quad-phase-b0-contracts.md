# Jaxstro Quad Phase B0 Shared Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the finite-hyperrectangle, coordinate-last evaluator, result,
capacity, and thin-dispatch contracts that every Phase B numerical family uses
without implementing a multidimensional numerical method.

**Architecture:** Extend the existing one-dimensional package rather than
building a parallel API. `quad.integrate` moves to a thin facade that preserves
the current one-dimensional call path exactly and routes hyperrectangles to one
private multidimensional dispatcher. Reference-space mapping and integrand
evaluation live in `_multidim.py`; family controllers added by B1 through B3
consume those functions but do not move into the facade.

**Tech Stack:** Python 3.11+, JAX 0.10.1+, JAX NumPy, `jax.tree_util`,
`jaxtyping`, pytest, Ruff, MyPy, MyST, KaTeX/LaTeX.

## Global Constraints

- Governing design:
  `docs/superpowers/specs/2026-07-17-jaxstro-quad-phase-b-multidimensional-design.md`
  at commit `712014d`.
- Run Python tooling through `env -u VIRTUAL_ENV uv run --no-sync`; if the
  known macOS `uv` system-configuration panic recurs, use the existing `.venv`
  executable and record the substitution.
- Add no runtime dependency.
- Preserve the existing one-dimensional `quad.integrate` numerical path,
  signature behavior, statuses, replay behavior, and tests.
- Accept only finite, static-dimensional `Hyperrectangle` domains in Phase B.
- Use coordinate-last arrays with `x.shape == (point_count, dimension)` inside
  the shared evaluator.
- Keep algorithms in family owners. The public facade performs validation,
  normalization, and dispatch only.
- Use fixed-shape JAX records, fixed-length scans, and statically bounded loops.
  Do not add a differentiated `jax.lax.while_loop`.
- Keep `args` as the sole generic differentiable parameter container.
- Preserve existing `QuadResult`, `QuadError`, and `QuadWork` field layouts.
- Append `MAX_INDICES = 8`; do not renumber existing `QuadStatus` members.
- Quantity axes, replay for multidimensional families, numerical methods,
  sibling migrations, publication, push, and deployment are outside B0.
- Use ASCII prose and LaTeX mathematics. Do not add course or instructor
  framing.
- Never weaken, delete, or skip an existing test to make a gate pass.
- Commit after each task with explicit paths.

## File and Responsibility Map

- `src/jaxstro/quad/domains.py`: add raw `Hyperrectangle`, static dimension,
  eager structural validation, orientation, and zero-volume predicates.
- `src/jaxstro/quad/_multidim.py`: reference mapping, coordinate-last
  integrand/density evaluation, payload-shape inference, and the initial
  fail-closed multidimensional dispatcher.
- `src/jaxstro/quad/integrate.py`: sole public thin dispatcher across existing
  one-dimensional and new multidimensional domains.
- `src/jaxstro/quad/adaptive.py`: retain the existing one-dimensional solver;
  no algorithm moves.
- `src/jaxstro/quad/result.py`: append `MAX_INDICES` only.
- `src/jaxstro/quad/__init__.py`: export `Hyperrectangle` and import the facade.
- `tests/unit/quad/test_multidim_domains.py`: domain, PyTree, orientation, and
  structural validation contracts.
- `tests/unit/quad/test_multidim_evaluator.py`: coordinate-last map, density,
  payload, and nonfinite contracts.
- `tests/unit/quad/test_integrate_dispatch.py`: one-dimensional parity and
  fail-closed B0 hyperrectangle dispatch.
- `tests/integration/test_quad_multidim_transforms.py`: eager/JIT/VMAP shape and
  traced-invalid behavior.
- `docs/50-api/approximation-integration/quad.md`: record the B0 domain and
  dispatcher contract without claiming a B1 method.
- `STATUS.md`: record B0 evidence and set B1 as the single next action.

---

### Task 1: Add the raw finite-hyperrectangle contract

**Files:**
- Modify: `src/jaxstro/quad/domains.py`
- Modify: `src/jaxstro/quad/result.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_multidim_domains.py`
- Modify: `tests/unit/quad/test_result.py`

**Interfaces:**
- Consumes: existing immutable domain PyTree conventions in `domains.py`.
- Produces: `Hyperrectangle(lower, upper)`, `hyperrectangle_is_valid(domain)`,
  `hyperrectangle_orientation(domain)`, and `QuadStatus.MAX_INDICES == 8`.

- [ ] **Step 1: Write failing domain and enum tests**

  Create `tests/unit/quad/test_multidim_domains.py`:

  ```python
  import jax
  import jax.numpy as jnp
  import pytest

  from jaxstro import quad


  def test_hyperrectangle_is_a_dynamic_bound_pytree():
      domain = quad.Hyperrectangle(
          jnp.array([0.0, 3.0]),
          jnp.array([2.0, -1.0]),
      )
      leaves, tree = jax.tree.flatten(domain)
      rebuilt = jax.tree.unflatten(tree, leaves)

      assert domain.dimension == 2
      assert len(leaves) == 2
      assert jnp.array_equal(rebuilt.lower, domain.lower)
      assert jnp.array_equal(rebuilt.upper, domain.upper)
      assert quad.hyperrectangle_orientation(domain) == -1.0
      assert quad.hyperrectangle_is_valid(domain)


  def test_hyperrectangle_zero_volume_is_valid_but_has_zero_orientation():
      domain = quad.Hyperrectangle(
          jnp.array([0.0, 1.0]),
          jnp.array([2.0, 1.0]),
      )
      assert quad.hyperrectangle_is_valid(domain)
      assert quad.hyperrectangle_orientation(domain) == 0.0


  @pytest.mark.parametrize(
      "lower, upper, error",
      [
          (jnp.zeros((2, 1)), jnp.ones((2, 1)), "one-dimensional"),
          (jnp.zeros(2), jnp.ones(3), "matching shapes"),
          (jnp.zeros(0), jnp.ones(0), "positive dimension"),
      ],
  )
  def test_hyperrectangle_rejects_invalid_static_shapes(lower, upper, error):
      with pytest.raises(ValueError, match=error):
          quad.Hyperrectangle(lower, upper)


  def test_hyperrectangle_rejects_host_known_nonfinite_bounds():
      with pytest.raises(ValueError, match="must be finite"):
          quad.Hyperrectangle(jnp.array([0.0, jnp.inf]), jnp.ones(2))
  ```

  Append to `tests/unit/quad/test_result.py`:

  ```python
  def test_max_indices_appends_without_renumbering_statuses():
      assert QuadStatus.ERROR_ESTIMATE_UNAVAILABLE == 7
      assert QuadStatus.MAX_INDICES == 8
  ```

- [ ] **Step 2: Run the RED gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_multidim_domains.py \
    tests/unit/quad/test_result.py
  ```

  Expected: FAIL because `Hyperrectangle` and `MAX_INDICES` do not exist.

- [ ] **Step 3: Implement the domain and append-only status**

  Add to `src/jaxstro/quad/domains.py`:

  ```python
  from jaxstro.numerics.checks import try_concrete_bool


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class Hyperrectangle:
      lower: Any
      upper: Any

      def __post_init__(self) -> None:
          lower_shape = jnp.shape(self.lower)
          upper_shape = jnp.shape(self.upper)
          if len(lower_shape) != 1 or len(upper_shape) != 1:
              raise ValueError("Hyperrectangle bounds must be one-dimensional")
          if lower_shape != upper_shape:
              raise ValueError("Hyperrectangle bounds must have matching shapes")
          if lower_shape[0] == 0:
              raise ValueError("Hyperrectangle must have positive dimension")
          finite = try_concrete_bool(
              jnp.all(jnp.isfinite(self.lower))
              & jnp.all(jnp.isfinite(self.upper))
          )
          if finite is False:
              raise ValueError("Hyperrectangle bounds must be finite")

      @property
      def dimension(self) -> int:
          return jnp.shape(self.lower)[0]

      def tree_flatten(self):
          return (self.lower, self.upper), self.dimension

      @classmethod
      def tree_unflatten(cls, dimension: int, children):
          domain = cls(*children)
          if domain.dimension != dimension:
              raise ValueError("invalid Hyperrectangle PyTree dimension")
          return domain


  def hyperrectangle_is_valid(domain: Hyperrectangle) -> Array:
      lower = jnp.asarray(domain.lower)
      upper = jnp.asarray(domain.upper)
      return jnp.all(jnp.isfinite(lower) & jnp.isfinite(upper))


  def hyperrectangle_orientation(domain: Hyperrectangle) -> Array:
      return jnp.prod(jnp.sign(jnp.asarray(domain.upper) - domain.lower))
  ```

  Append `MAX_INDICES = 8` to `QuadStatus` in `result.py`. Export the new domain
  and helper names from `quad/__init__.py`. `try_concrete_bool` returns `None`
  under tracing, so host-known nonfinite bounds raise eagerly while traced
  nonfinite bounds flow to `hyperrectangle_is_valid` and the dynamic
  `INVALID_INPUT` result path.

- [ ] **Step 4: Run GREEN and static checks**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_multidim_domains.py \
    tests/unit/quad/test_domains.py \
    tests/unit/quad/test_result.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/domains.py src/jaxstro/quad/result.py \
    src/jaxstro/quad/__init__.py tests/unit/quad/test_multidim_domains.py
  env -u VIRTUAL_ENV uv run --no-sync mypy \
    src/jaxstro/quad/domains.py src/jaxstro/quad/result.py
  ```

  Expected: all commands exit zero.

- [ ] **Step 5: Commit Task 1**

  ```bash
  git add src/jaxstro/quad/domains.py src/jaxstro/quad/result.py \
    src/jaxstro/quad/__init__.py tests/unit/quad/test_multidim_domains.py \
    tests/unit/quad/test_result.py
  git commit -m "feat(quad): add hyperrectangle contracts"
  ```

### Task 2: Add the coordinate-last reference evaluator

**Files:**
- Create: `src/jaxstro/quad/_multidim.py`
- Create: `tests/unit/quad/test_multidim_evaluator.py`

**Interfaces:**
- Consumes: `Hyperrectangle`, `LebesgueMeasure`, `WeightedMeasure`,
  `call_integrand`, and explicit `args`.
- Produces: `MultidimMapResult`, `PointEvaluation`,
  `map_hyperrectangle(domain, reference)`, and
  `evaluate_multidim(fun, domain, reference, *, args, measure)`.

- [ ] **Step 1: Write failing map and evaluator tests**

  Create `tests/unit/quad/test_multidim_evaluator.py`:

  ```python
  import jax.numpy as jnp

  from jaxstro import quad
  from jaxstro.quantity import units as q_units
  from jaxstro.quad._multidim import evaluate_multidim, map_hyperrectangle


  def test_map_hyperrectangle_preserves_coordinate_last_and_orientation():
      domain = quad.Hyperrectangle(
          jnp.array([1.0, 5.0]),
          jnp.array([3.0, 1.0]),
      )
      reference = jnp.array([[0.0, 0.25], [1.0, 0.75]])
      mapped = map_hyperrectangle(domain, reference)

      assert mapped.x.shape == (2, 2)
      assert jnp.allclose(mapped.x, jnp.array([[1.0, 4.0], [3.0, 2.0]]))
      assert mapped.jacobian == 8.0
      assert mapped.orientation == -1.0
      assert mapped.valid


  def test_evaluator_keeps_point_axis_separate_from_payload_axes():
      domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
      reference = jnp.array([[0.25, 0.5], [0.75, 1.0]])

      evaluated = evaluate_multidim(
          lambda x, scale: scale * jnp.stack((x[:, 0], x[:, 1]), axis=-1),
          domain,
          reference,
          args=jnp.asarray(2.0),
          measure=quad.LebesgueMeasure(),
      )

      assert evaluated.values.shape == (2, 2)
      assert jnp.allclose(
          evaluated.values,
          jnp.array([[0.5, 1.0], [1.5, 2.0]]),
      )
      assert evaluated.weights.shape == (2,)
      assert not evaluated.nonfinite


  def test_weighted_density_receives_physical_coordinate_last_points():
      domain = quad.Hyperrectangle(jnp.zeros(2), 2.0 * jnp.ones(2))
      measure = quad.WeightedMeasure(
          lambda x, args: args + x[:, 0] * x[:, 1],
          density_unit=q_units.dimensionless,
      )
      evaluated = evaluate_multidim(
          lambda x, _args: jnp.ones(x.shape[0]),
          domain,
          jnp.array([[0.5, 0.25]]),
          args=jnp.asarray(1.0),
          measure=measure,
      )
      assert jnp.allclose(evaluated.weights, jnp.array([6.0]))
  ```

- [ ] **Step 2: Run the RED gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_multidim_evaluator.py
  ```

  Expected: FAIL because `_multidim.py` does not exist.

- [ ] **Step 3: Implement one shared raw evaluator**

  Create `src/jaxstro/quad/_multidim.py`:

  ```python
  from collections.abc import Callable
  from typing import Any, NamedTuple

  import jax
  import jax.numpy as jnp
  from jaxtyping import Array

  from ._integrand import call_integrand, has_explicit_args
  from .domains import (
      Hyperrectangle,
      hyperrectangle_is_valid,
      hyperrectangle_orientation,
  )
  from .measures import LebesgueMeasure, WeightedMeasure


  class MultidimMapResult(NamedTuple):
      x: Array
      jacobian: Array
      orientation: Array
      valid: Array


  class PointEvaluation(NamedTuple):
      values: Array
      weights: Array
      nonfinite: Array
      valid: Array


  def infer_multidim_payload_zero(fun, *, args, dimension: int, dtype):
      point = jax.ShapeDtypeStruct((1, dimension), dtype)
      abstract = jax.eval_shape(
          lambda x: call_integrand(
              fun, x, args, has_explicit_args(args)
          ),
          point,
      )
      if not hasattr(abstract, "shape") or abstract.shape[:1] != (1,):
          raise ValueError(
              "multidimensional integrand output must have a leading point axis"
          )
      return jnp.zeros(abstract.shape[1:], dtype=abstract.dtype)


  def map_hyperrectangle(
      domain: Hyperrectangle,
      reference: Array,
  ) -> MultidimMapResult:
      reference = jnp.asarray(reference)
      if reference.ndim != 2 or reference.shape[-1] != domain.dimension:
          raise ValueError("reference points must have shape (point_count, dimension)")
      lower = jnp.asarray(domain.lower)
      width = jnp.asarray(domain.upper) - lower
      return MultidimMapResult(
          x=lower + reference * width,
          jacobian=jnp.prod(jnp.abs(width)),
          orientation=hyperrectangle_orientation(domain),
          valid=hyperrectangle_is_valid(domain),
      )


  def _density_values(measure, x: Array, args: Any) -> Array:
      if isinstance(measure, LebesgueMeasure):
          return jnp.ones(x.shape[0], dtype=x.dtype)
      if isinstance(measure, WeightedMeasure):
          density = jnp.asarray(measure.density(x, args))
          if density.shape != x.shape[:-1]:
              raise ValueError(
                  "multidimensional density must have shape (point_count,)"
              )
          return density
      raise TypeError("multidimensional integration requires a finite measure")


  def evaluate_multidim(
      fun: Callable,
      domain: Hyperrectangle,
      reference: Array,
      *,
      args: Any,
      measure,
  ) -> PointEvaluation:
      mapped = map_hyperrectangle(domain, reference)
      values = jnp.asarray(
          call_integrand(fun, mapped.x, args, has_explicit_args(args))
      )
      if values.ndim == 0 or values.shape[0] != reference.shape[0]:
          raise ValueError(
              "multidimensional integrand output must have a leading point axis"
          )
      density = _density_values(measure, mapped.x, args)
      weights = mapped.orientation * mapped.jacobian * density
      nonfinite = ~(
          jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(weights))
      )
      return PointEvaluation(values, weights, nonfinite, mapped.valid)
  ```

  Import `jax` for `ShapeDtypeStruct` and `eval_shape`. The B4 quantity adapter
  extends this abstract evaluation without changing the raw helper.

- [ ] **Step 4: Run GREEN and legacy evaluator regressions**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_multidim_evaluator.py \
    tests/unit/quad/test_fixed.py \
    tests/unit/quad/test_integrate_gk.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/_multidim.py \
    tests/unit/quad/test_multidim_evaluator.py
  ```

  Expected: all commands exit zero and existing one-dimensional work counts are
  unchanged.

- [ ] **Step 5: Commit Task 2**

  ```bash
  git add src/jaxstro/quad/_multidim.py \
    tests/unit/quad/test_multidim_evaluator.py
  git commit -m "feat(quad): add coordinate-last evaluator"
  ```

### Task 3: Move `quad.integrate` to a thin family dispatcher

**Files:**
- Create: `src/jaxstro/quad/integrate.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_integrate_dispatch.py`
- Modify: `tests/unit/quad/test_import_surface.py`

**Interfaces:**
- Consumes: `adaptive.integrate` for all existing one-dimensional methods.
- Produces: the stable facade signature with optional `max_regions`,
  `max_indices`, `max_frontier`, `max_nodes`, and `key`;
  `_integrate_hyperrectangle` remains fail-closed until B1 registers explicit
  method classes.

- [ ] **Step 1: Write failing facade-parity tests**

  Create `tests/unit/quad/test_integrate_dispatch.py`:

  ```python
  import inspect

  import jax.numpy as jnp
  import pytest

  from jaxstro import quad
  from jaxstro.quad import adaptive


  def _one_dimensional_kwargs():
      return dict(
          method=quad.GaussKronrod(15),
          epsabs=1e-10,
          epsrel=1e-10,
          max_evaluations=45,
          max_regions=2,
      )


  def test_facade_preserves_one_dimensional_result_bitwise():
      domain = quad.Interval(0.0, 1.0)
      direct = adaptive.integrate(lambda x: x**2, domain, **_one_dimensional_kwargs())
      facade = quad.integrate(lambda x: x**2, domain, **_one_dimensional_kwargs())
      assert jnp.array_equal(facade.value, direct.value)
      assert jnp.array_equal(facade.status, direct.status)
      assert facade.work == direct.work


  def test_facade_requires_one_dimensional_region_capacity():
      with pytest.raises(ValueError, match="max_regions"):
          quad.integrate(
              lambda x: x,
              quad.Interval(0.0, 1.0),
              method=quad.GaussKronrod(15),
              epsabs=1e-8,
              epsrel=1e-8,
              max_evaluations=45,
          )


  def test_b0_hyperrectangle_has_no_silent_default_method():
      with pytest.raises(TypeError, match="Phase B method"):
          quad.integrate(
              lambda x: jnp.sum(x, axis=-1),
              quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
              method=quad.GaussKronrod(15),
              epsabs=1e-8,
              epsrel=1e-8,
              max_evaluations=64,
          )


  def test_facade_exposes_future_capacity_names_without_kwargs_catchall():
      parameters = inspect.signature(quad.integrate).parameters
      assert "max_indices" in parameters
      assert "max_frontier" in parameters
      assert "max_nodes" in parameters
      assert "key" in parameters
      assert not any(p.kind == p.VAR_KEYWORD for p in parameters.values())
  ```

- [ ] **Step 2: Run the RED gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_integrate_dispatch.py \
    tests/unit/quad/test_import_surface.py
  ```

  Expected: FAIL because the current public function comes directly from
  `adaptive.py`.

- [ ] **Step 3: Implement the thin dispatcher**

  Create `src/jaxstro/quad/integrate.py`:

  ```python
  from collections.abc import Callable
  from typing import Any

  from .adaptive import integrate as integrate_1d
  from .domains import Hyperrectangle
  from .tolerance import ErrorNorm, MaxNorm


  def _integrate_hyperrectangle(*args, **kwargs):
      method = kwargs["method"]
      raise TypeError(
          f"{type(method).__name__} is not an implemented Phase B method"
      )


  def integrate(
      fun: Callable,
      domain,
      *,
      args: Any = (),
      method,
      measure=None,
      epsabs,
      epsrel,
      max_evaluations: int,
      max_regions: int | None = None,
      max_indices: int | None = None,
      max_frontier: int | None = None,
      max_nodes: int | None = None,
      key=None,
      error_norm: ErrorNorm = MaxNorm(),
      gradient: str = "replay",
  ):
      if isinstance(domain, Hyperrectangle):
          return _integrate_hyperrectangle(
              fun,
              domain,
              args=args,
              method=method,
              measure=measure,
              epsabs=epsabs,
              epsrel=epsrel,
              max_evaluations=max_evaluations,
              max_regions=max_regions,
              max_indices=max_indices,
              max_frontier=max_frontier,
              max_nodes=max_nodes,
              key=key,
              error_norm=error_norm,
              gradient=gradient,
          )
      if max_regions is None:
          raise ValueError("one-dimensional integration requires max_regions")
      if any(
          value is not None
          for value in (max_indices, max_frontier, max_nodes, key)
      ):
          raise TypeError(
              "one-dimensional integration does not accept multidimensional "
              "capacity controls or key"
          )
      return integrate_1d(
          fun,
          domain,
          args=args,
          method=method,
          measure=measure,
          epsabs=epsabs,
          epsrel=epsrel,
          max_evaluations=max_evaluations,
          max_regions=max_regions,
          error_norm=error_norm,
          gradient=gradient,
      )
  ```

  Change only the `integrate` import in `quad/__init__.py` from
  `.adaptive import integrate` to `.integrate import integrate`.

- [ ] **Step 4: Run GREEN and one-dimensional parity**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_integrate_dispatch.py \
    tests/unit/quad/test_import_surface.py \
    tests/unit/quad/test_integrate_gk.py \
    tests/integration/test_quad_replay_transforms.py \
    tests/integration/test_quad_quantity_transforms.py
  ```

  Expected: all commands exit zero with exact existing one-dimensional parity.

- [ ] **Step 5: Commit Task 3**

  ```bash
  git add src/jaxstro/quad/integrate.py src/jaxstro/quad/__init__.py \
    tests/unit/quad/test_integrate_dispatch.py \
    tests/unit/quad/test_import_surface.py
  git commit -m "refactor(quad): add thin integrate dispatcher"
  ```

### Task 4: Lock the B0 JAX, error, API, and release gate

**Files:**
- Create: `tests/integration/test_quad_multidim_transforms.py`
- Modify: `tests/unit/quad/test_integrate_dispatch.py`
- Modify: `docs/50-api/approximation-integration/quad.md`
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: Tasks 1 through 3.
- Produces: verified eager/JIT/VMAP mapping contracts and an honest B0 API page.

- [ ] **Step 1: Write the transform and traced-invalid tests**

  Create `tests/integration/test_quad_multidim_transforms.py`:

  ```python
  import jax
  import jax.numpy as jnp

  from jaxstro import quad
  from jaxstro.quad._multidim import map_hyperrectangle


  def test_map_composes_with_jit_and_vmap_over_bounds():
      reference = jnp.array([[0.25, 0.75]])

      @jax.jit
      def mapped(lower, upper):
          return map_hyperrectangle(
              quad.Hyperrectangle(lower, upper),
              reference,
          ).x

      lowers = jnp.array([[0.0, 0.0], [1.0, -1.0]])
      uppers = jnp.array([[1.0, 2.0], [3.0, 1.0]])
      values = jax.vmap(mapped)(lowers, uppers)
      assert values.shape == (2, 1, 2)


  def test_traced_nonfinite_bounds_fail_dynamically():
      valid = jax.jit(
          lambda upper: quad.hyperrectangle_is_valid(
              quad.Hyperrectangle(jnp.zeros(2), upper)
          )
      )
      assert not valid(jnp.array([1.0, jnp.inf]))
  ```

  In `test_integrate_dispatch.py`, strengthen one-dimensional facade parity
  over every `QuadResult` PyTree leaf:

  ```python
  facade_result = quad.integrate(...)
  owner_result = integrate_1d(...)
  assert jax.tree.structure(facade_result) == jax.tree.structure(owner_result)
  for facade_leaf, owner_leaf in zip(
      jax.tree.leaves(facade_result),
      jax.tree.leaves(owner_result),
      strict=True,
  ):
      assert jnp.array_equal(facade_leaf, owner_leaf, equal_nan=True)
  ```

  B0 intentionally has no multidimensional numerical result path, so it does
  not add a test-only `_b0_invalid_domain_probe`. The direct JIT validity test
  above owns the B0 traced-input contract. B1 tests `INVALID_INPUT` through
  real tensor and cubature integrations.

- [ ] **Step 2: Run the focused transform gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/integration/test_quad_multidim_transforms.py
  ```

  Expected: PASS. A failure under JIT stops B0 and is fixed in `domains.py` or
  `_multidim.py`, not hidden by changing the test.

- [ ] **Step 3: Add the B0 API contract**

  In `docs/50-api/approximation-integration/quad.md`, add sections titled
  `Multidimensional domain contract` and `Phase B dispatcher boundary` with
  this executable example and warning:

  ```markdown
  ```python
  domain = quad.Hyperrectangle(
      jnp.array([0.0, 0.0]),
      jnp.array([1.0, 2.0]),
  )
  # x passed to a Phase B integrand has shape (point_count, dimension).
  ```

  :::{warning}
  `Hyperrectangle` and the thin dispatcher are structural B0 contracts. A
  multidimensional numerical method is not available until its family passes
  the B1, B2, or B3 validation gate.
  :::
  ```

  Use LaTeX for the domain expression $[0,1]\times[0,2]$; do not use Unicode
  mathematical symbols.

- [ ] **Step 4: Run the complete B0 gate and update status**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad \
    tests/unit/test_quad_replay_substrate.py \
    tests/unit/test_quad_quantity.py \
    tests/integration/test_quad_compatibility.py \
    tests/integration/test_quad_replay_transforms.py \
    tests/integration/test_quad_quantity_transforms.py \
    tests/integration/test_quad_multidim_transforms.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check src/jaxstro/quad tests/unit/quad \
    tests/integration/test_quad_multidim_transforms.py
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check src/jaxstro/quad \
    tests/unit/quad tests/integration/test_quad_multidim_transforms.py
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro/quad
  bash scripts/check_docs.sh
  git diff --check
  ```

  Expected: every command exits zero. Update `STATUS.md` with exact pass counts,
  the B0 commits, no-runtime-dependency statement, and
  `next: Execute the reviewed Phase B1 deterministic tensor and cubature plan.`

- [ ] **Step 5: Commit Task 4 and request checkpoint review**

  ```bash
  git add tests/integration/test_quad_multidim_transforms.py \
    tests/unit/quad/test_integrate_dispatch.py \
    docs/50-api/approximation-integration/quad.md \
    docs/superpowers/plans/2026-07-17-jaxstro-quad-phase-b0-contracts.md \
    STATUS.md
  git commit -m "docs(quad): certify Phase B0 contracts"
  ```

  Request read-only numerical/API/JAX review. Resolve every Critical or
  Important finding before starting B1.
