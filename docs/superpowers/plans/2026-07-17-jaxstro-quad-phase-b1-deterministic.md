# Jaxstro Quad Phase B1 Deterministic Multidimensional Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver fixed heterogeneous tensor products, anisotropic p-adaptive
tensor Clenshaw-Curtis, and h-adaptive Genz-Malik cubature on finite
hyperrectangles for validated dimensions 2 through 8.

**Architecture:** All three methods consume the B0 coordinate-last evaluator
and return the existing fixed-shape `QuadResult`. `_tensor.py` owns product-node
construction, exact nested reuse, and directional frontier evidence.
`_cubature.py` owns the symmetric Genz-Malik rule and a fixed-capacity region
controller. `integrate.py` only dispatches concrete method declarations to
`tensor.py` or `cubature.py`.

**Tech Stack:** Python 3.11+, JAX 0.10.1+, JAX NumPy, `jax.lax.scan`,
`jax.lax.map`, NumPy for host-side rule construction, pytest, Ruff, MyPy.

## Global Constraints

- B0 must be complete and independently approved before this plan starts.
- Governing design:
  `docs/superpowers/specs/2026-07-17-jaxstro-quad-phase-b-multidimensional-design.md`.
- Add no runtime dependency; SciPy is a development-only comparison in B4.
- Validate B1 only for dimensions 2 through 8. Reject dimension 1 with guidance
  to existing Phase A methods and dimensions above 8 with guidance to B2/B3.
- Preserve signed per-axis orientation and exact zero-volume stop-mode behavior.
- Fixed tensor returns `ErrorKind.UNAVAILABLE` and
  `QuadStatus.ERROR_ESTIMATE_UNAVAILABLE`.
- Adaptive tensor returns `REFINEMENT_DIFFERENCE`; cubature returns
  `EMBEDDED_RULE`.
- Count logical point evaluations, not Python calls or padded storage.
- For adaptive cubature, scalar eager and JIT execution physically skips child
  work after termination. Ordinary `jax.vmap` preserves values, statuses, and
  per-lane logical work but may lower scalar conditionals to select-style
  execution. Cost-sensitive heterogeneous batches must apply `jax.lax.map`
  around scalar `quad.integrate` calls to retain physical per-lane masking.
- Use static capacities and deterministic lowest-axis/lexicographic tie breaks.
- Implement no multidimensional replay or quantity mode in B1; the dispatcher
  accepts only `gradient="stop"` for B1 until B4.
- Keep all diagnostics stopped and all invalid/nonfinite values fail-closed.
- B1 status precedence is `INVALID_INPUT`, `NONFINITE_INTEGRAND`, `CONVERGED`,
  `ROUNDOFF_LIMITED`, then the relevant capacity status. Tensor emits
  `ROUNDOFF_LIMITED` only when the next nested level adds no representable
  coordinate; cubature emits it only when a selected midpoint collapses to an
  endpoint. B1 does not emit `DIVERGENCE_SUSPECTED` without a separately
  validated detector.
- Every zero-volume shortcut evaluates dynamic domain validity first. Branch
  `INVALID_INPUT`, then exact zero volume, then numerical evaluation; a
  coincident axis must never hide another nonfinite bound.
- Use ASCII prose and LaTeX mathematics.
- Commit each task after focused RED/GREEN verification.

## File and Responsibility Map

- `src/jaxstro/quad/tensor.py`: public declarations and result assembly for
  fixed and adaptive tensor methods.
- `src/jaxstro/quad/_tensor.py`: rule normalization, Cartesian products,
  canonical nested node identities, cache masks, and adaptive frontier scan.
- `src/jaxstro/quad/cubature.py`: public `GenzMalik` and `AdaptiveCubature`
  declarations plus result assembly.
- `src/jaxstro/quad/_cubature.py`: symmetric rule generation, local estimates,
  smoothness evidence, region store, split selection, and scan.
- `src/jaxstro/quad/integrate.py`: explicit B1 dispatch only.
- `src/jaxstro/quad/__init__.py`: B1 public exports.
- `tests/unit/quad/test_tensor.py`: construction, exactness, work, and errors.
- `tests/unit/quad/test_adaptive_tensor.py`: frontier policy, reuse, capacity,
  and estimator semantics.
- `tests/unit/quad/test_genz_malik.py`: node/weight invariants and polynomial
  exactness.
- `tests/unit/quad/test_cubature.py`: region selection, split, work, statuses,
  and array payloads.
- `tests/validation/test_quad_multidim_deterministic.py`: method-filtered
  analytic and Genz truth families plus separate structural/preflight evidence
  over dimensions 2 through 8.
- `tests/integration/test_quad_multidim_deterministic_transforms.py`:
  eager/JIT/VMAP/float/complex stop-mode matrix.
- `STATUS.md`: Task 5 evidence and whole-rung B1 review next action.

---

### Task 1: Implement fixed heterogeneous tensor products

**Files:**
- Create: `src/jaxstro/quad/tensor.py`
- Create: `src/jaxstro/quad/_tensor.py`
- Modify: `src/jaxstro/quad/result.py`
- Modify: `src/jaxstro/quad/integrate.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_tensor.py`

**Interfaces:**
- Consumes: B0 `evaluate_multidim`, existing Phase A fixed-rule declarations,
  `error_norm`, and `tolerance_threshold`.
- Produces: `TensorProduct(rules)`,
  `tensor_rule_data(method, dimension, dtype)`, and
  `integrate_tensor(...)->QuadResult`, plus shared
  `unavailable_result` and `zero_volume_result` factories in `result.py`.

- [x] **Step 1: Write failing fixed-tensor tests**

  Create `tests/unit/quad/test_tensor.py`:

  ```python
  import jax.numpy as jnp
  import pytest

  from jaxstro import quad


  def test_heterogeneous_tensor_integrates_bivariate_polynomial():
      result = quad.integrate(
          lambda x: x[:, 0] ** 3 * x[:, 1] ** 2,
          quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
          method=quad.TensorProduct(
              (quad.GaussianRule(3), quad.ClenshawCurtisRule(5))
          ),
          epsabs=1e-12,
          epsrel=1e-12,
          max_evaluations=15,
          gradient="stop",
      )
      assert jnp.allclose(result.value, 1.0 / 12.0, rtol=1e-12, atol=1e-12)
      assert result.error.kind == quad.ErrorKind.UNAVAILABLE
      assert result.status == quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE
      assert result.work.evaluations == 15
      assert jnp.isnan(result.error.confidence_level)


  def test_replicated_rule_and_reversed_axis_preserve_orientation():
      result = quad.integrate(
          lambda x: jnp.ones(x.shape[0]),
          quad.Hyperrectangle(jnp.array([0.0, 2.0]), jnp.array([1.0, -1.0])),
          method=quad.TensorProduct(quad.GaussianRule(2)),
          epsabs=0.0,
          epsrel=0.0,
          max_evaluations=4,
          gradient="stop",
      )
      assert result.value == -3.0


  def test_zero_volume_returns_exact_zero_without_point_work():
      result = quad.integrate(
          lambda x: jnp.stack((x[:, 0], x[:, 1]), axis=-1),
          quad.Hyperrectangle(jnp.array([0.0, 1.0]), jnp.array([2.0, 1.0])),
          method=quad.TensorProduct(quad.GaussianRule(3)),
          epsabs=1e-12,
          epsrel=1e-12,
          max_evaluations=9,
          gradient="stop",
      )
      assert jnp.array_equal(result.value, jnp.zeros(2))
      assert result.status == quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE
      assert result.work.evaluations == 0


  def test_tensor_capacity_fails_before_materialization():
      with pytest.raises(ValueError, match="requires 81 evaluations"):
          quad.integrate(
              lambda x: jnp.sum(x, axis=-1),
              quad.Hyperrectangle(jnp.zeros(4), jnp.ones(4)),
              method=quad.TensorProduct(quad.GaussianRule(3)),
              epsabs=0.0,
              epsrel=0.0,
              max_evaluations=80,
              gradient="stop",
          )
  ```

- [x] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_tensor.py
  ```

  Expected: FAIL because `TensorProduct` is absent.

- [x] **Step 3: Implement rule normalization and product construction**

  In `src/jaxstro/quad/tensor.py`, define a static declaration:

  ```python
  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class TensorProduct:
      rules: Rule | tuple[Rule, ...]

      def tree_flatten(self):
          return (), self.rules

      @classmethod
      def tree_unflatten(cls, rules, _children):
          return cls(rules)
  ```

  In `_tensor.py`, normalize one replicated rule or one rule per axis. Convert
  every Phase A finite-interval rule from $[-1,1]$ to $[0,1]$ before taking the
  product:

  ```python
  class TensorRuleData(NamedTuple):
      points: Array
      weights: Array
      point_count: int


  def validate_b1_dimension(dimension: int) -> None:
      if dimension < 2 or dimension > 8:
          raise ValueError(
              "Phase B1 deterministic methods require dimension 2 through 8"
          )


  def _unit_rule_data(rule, dtype):
      if isinstance(rule, GaussianRule):
          data = gaussian_rule_data(rule, LebesgueMeasure())
      elif isinstance(rule, (ClenshawCurtisRule, FejerIRule, FejerIIRule)):
          data = chebyshev_rule_data(rule, dtype=dtype)
      elif isinstance(rule, TanhSinhRule):
          data = tanh_sinh_rule_data(rule)
      else:
          raise TypeError(f"unsupported tensor rule: {type(rule).__name__}")
      nodes = jnp.asarray(data.nodes, dtype=dtype)
      weights = jnp.asarray(data.weights, dtype=dtype)
      return 0.5 * (nodes + 1.0), 0.5 * weights


  def tensor_rule_data(method, dimension: int, dtype) -> TensorRuleData:
      validate_b1_dimension(dimension)
      rules = (
          (method.rules,) * dimension
          if not isinstance(method.rules, tuple)
          else method.rules
      )
      if len(rules) != dimension:
          raise ValueError("TensorProduct requires one rule or one rule per axis")
      axes = [_unit_rule_data(rule, dtype) for rule in rules]
      point_count = math.prod(nodes.size for nodes, _weights in axes)
      points = jnp.stack(
          [mesh.reshape(-1) for mesh in jnp.meshgrid(
              *(nodes for nodes, _weights in axes), indexing="ij"
          )],
          axis=-1,
      )
      weights = jnp.prod(
          jnp.stack(
              [mesh.reshape(-1) for mesh in jnp.meshgrid(
                  *(weights for _nodes, weights in axes), indexing="ij"
              )],
              axis=-1,
          ),
          axis=-1,
      )
      return TensorRuleData(points, weights, point_count)
  ```

  Gaussian and tanh-sinh rule constructors retain their current signatures;
  cast their returned arrays once as shown. Do not change their
  one-dimensional bytes. Reuse `validate_b1_dimension` at the start of the
  adaptive tensor and cubature entry points. Parameterize dimensions 1, 2, 8,
  and 9 for all three methods: 1 and 9 raise eagerly with the shared message,
  while 2 and 8 enter the controller when capacities are feasible.

- [x] **Step 4: Assemble the fixed result and dispatch**

  Implement `integrate_tensor` in `tensor.py`:

  ```python
  def integrate_tensor(
      fun,
      domain,
      *,
      args,
      method,
      measure,
      epsabs,
      epsrel,
      max_evaluations,
      error_norm,
  ):
      data = tensor_rule_data(
          method,
          domain.dimension,
          jnp.result_type(domain.lower, domain.upper, 0.0),
      )
      if data.point_count > max_evaluations:
          raise ValueError(
              f"TensorProduct requires {data.point_count} evaluations, "
              f"exceeding max_evaluations={max_evaluations}"
          )
      zero = infer_multidim_payload_zero(
          fun,
          args=args,
          dimension=domain.dimension,
          dtype=data.points.dtype,
      )

      def zero_branch(_):
          return zero_volume_result(
              zero,
              epsabs=epsabs,
              epsrel=epsrel,
              error_norm=error_norm,
          )

      def evaluate_branch(_):
          evaluated = evaluate_multidim(
              fun,
              domain,
              data.points,
              args=args,
              measure=LebesgueMeasure() if measure is None else measure,
          )
          factors = data.weights * evaluated.weights
          value = jnp.sum(
              evaluated.values
              * factors.reshape(
                  (data.point_count,)
                  + (1,) * (evaluated.values.ndim - 1)
              ),
              axis=0,
          )
          status = jnp.where(
              ~evaluated.valid,
              jnp.asarray(QuadStatus.INVALID_INPUT, dtype=jnp.int32),
              jnp.where(
                  evaluated.nonfinite,
                  jnp.asarray(
                      QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32
                  ),
                  jnp.asarray(
                      QuadStatus.ERROR_ESTIMATE_UNAVAILABLE,
                      dtype=jnp.int32,
                  ),
              ),
          )
          return unavailable_result(
              value,
              epsabs=epsabs,
              epsrel=epsrel,
              error_norm=error_norm,
              evaluations=data.point_count,
              status=status,
          )

      invalid = ~hyperrectangle_is_valid(domain)
      zero_width = jnp.any(jnp.asarray(domain.lower) == domain.upper)
      return jax.lax.cond(
          invalid,
          lambda _: invalid_result(zero, epsabs, epsrel, error_norm),
          lambda _: jax.lax.cond(
              zero_width,
              zero_branch,
              evaluate_branch,
              operand=None,
          ),
          operand=None,
      )
  ```

  Add `unavailable_result` and `zero_volume_result` to `result.py`.
  `unavailable_result` accepts an explicit status and replaces the value with a
  shape-preserving nonfinite sentinel for `INVALID_INPUT` and
  `NONFINITE_INTEGRAND`. `zero_volume_result` returns exact zero value and
  work, `ErrorKind.UNAVAILABLE`, and `ERROR_ESTIMATE_UNAVAILABLE`: geometric
  exactness is not a named runtime estimator and therefore is not
  `CONVERGED`. B2 and B3 reuse the same factory.

  Add explicit `TensorProduct` dispatch in `_integrate_hyperrectangle`; reject
  `gradient!="stop"`. Apply `jax.tree.map(jax.lax.stop_gradient, result)` to
  every B1 result before returning. Add tests for traced invalid bounds,
  nonfinite integrand and density values, and exact-zero `jax.grad` and
  `jax.jvp` tangents in stop mode. Include one mixed case with a coincident
  axis and a nonfinite axis; it must return `INVALID_INPUT`, not exact zero.

- [x] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_tensor.py \
    tests/unit/quad/test_integrate_dispatch.py \
    tests/unit/quad/test_fixed.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/tensor.py src/jaxstro/quad/_tensor.py \
    src/jaxstro/quad/integrate.py tests/unit/quad/test_tensor.py
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/tensor.py src/jaxstro/quad/_tensor.py \
    src/jaxstro/quad/result.py \
    src/jaxstro/quad/integrate.py src/jaxstro/quad/__init__.py \
    tests/unit/quad/test_tensor.py
  git commit -m "feat(quad): add tensor-product integration"
  ```

### Task 2: Implement anisotropic p-adaptive tensor refinement

**Files:**
- Modify: `src/jaxstro/quad/tensor.py`
- Modify: `src/jaxstro/quad/_tensor.py`
- Modify: `src/jaxstro/quad/integrate.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_adaptive_tensor.py`

**Interfaces:**
- Consumes: nested Clenshaw-Curtis rules and B1 fixed tensor accumulation.
- Produces: `AdaptiveTensorClenshawCurtis(initial_level=2)`,
  `TensorReplayEvidence(levels, active_node_ids)` for B4, and
  `integrate_adaptive_tensor`.

- [x] **Step 1: Write failing policy and reuse tests**

  Create `tests/unit/quad/test_adaptive_tensor.py`:

  ```python
  import jax.numpy as jnp

  from jaxstro import quad


  def test_adaptive_tensor_refines_the_sharper_axis_first():
      result = quad.integrate(
          lambda x: jnp.exp(8.0 * x[:, 0]) + x[:, 1],
          quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
          method=quad.AdaptiveTensorClenshawCurtis(initial_level=2),
          epsabs=1e-5,
          epsrel=1e-5,
          max_evaluations=512,
          gradient="stop",
      )
      assert result.error.kind == quad.ErrorKind.REFINEMENT_DIFFERENCE
      assert result.work.refinements > 0
      assert result.work.evaluations <= 512


  def test_frontier_error_is_sum_of_directional_norms():
      from jaxstro.quad._tensor import choose_tensor_axis

      axis, evidence = choose_tensor_axis(
          jnp.array([3.0, 2.0]),
          jnp.array([3, 1]),
      )
      assert axis == 1
      assert evidence == 5.0


  def test_frontier_profit_ties_choose_the_lowest_axis():
      from jaxstro.quad._tensor import choose_tensor_axis

      axis, evidence = choose_tensor_axis(
          jnp.array([3.0, 1.0]),
          jnp.array([3, 1]),
      )
      assert axis == 0
      assert evidence == 4.0


  def test_nested_reuse_counts_only_new_coordinate_tuples():
      from jaxstro.quad._tensor import canonical_tensor_ids

      coarse = canonical_tensor_ids(jnp.array([2, 2]))
      fine = canonical_tensor_ids(jnp.array([3, 2]))
      assert len(set(map(tuple, coarse.tolist()))) == coarse.shape[0]
      assert set(map(tuple, coarse.tolist())).issubset(
          set(map(tuple, fine.tolist()))
      )
  ```

- [x] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_adaptive_tensor.py
  ```

  Expected: FAIL because the declaration and frontier helpers are absent.

- [x] **Step 3: Implement exact nested identities and frontier selection**

  Add to `_tensor.py`:

  ```python
  def _reduced_dyadic(index: int, level: int) -> tuple[int, int]:
      if index == 0:
          return 0, 0
      while level > 0 and index % 2 == 0:
          index //= 2
          level -= 1
      return index, level


  def canonical_cc_axis_ids(level: int) -> Array:
      denominator = 1 << level
      return jnp.asarray(
          [_reduced_dyadic(index, level) for index in range(denominator + 1)],
          dtype=jnp.int32,
      )


  def choose_tensor_axis(directional_error: Array, new_cost: Array):
      profit = directional_error / jnp.maximum(new_cost, 1)
      axis = jnp.argmax(profit)
      return axis, jnp.sum(directional_error)
  ```

  Store canonical per-axis `(odd_numerator, denominator_power)` pairs and take
  their Cartesian tuples in `canonical_tensor_ids`. These identities are static
  host-side construction data; do not trace the Python reduction loop.

  Add the static method declaration:

  ```python
  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class AdaptiveTensorClenshawCurtis:
      initial_level: int = 2

      def __post_init__(self):
          if (
              isinstance(self.initial_level, bool)
              or not isinstance(self.initial_level, int)
              or self.initial_level < 2
          ):
              raise ValueError("initial_level must be an integer at least 2")
  ```

- [x] **Step 4: Implement the fixed-capacity adaptive scan**

  Define a `TensorState` `NamedTuple` containing `levels`, accepted value,
  directional candidate values/errors/costs, a fixed-capacity canonical-node
  table, cached payload values, evaluation count, refinement count, status, and
  `done`. Use `jax.lax.scan` for exactly `max_refinements`, where the host-side
  capacity validator derives the largest safe value from `max_evaluations`.

  The scan body must follow this exact order:

  ```python
  axis, frontier_error = choose_tensor_axis(
      state.directional_error,
      state.directional_new_cost,
  )
  tolerance = tolerance_threshold(state.value, epsabs, epsrel, error_norm)
  converged = frontier_error <= tolerance
  can_accept = (
      state.evaluations + state.directional_new_cost[axis]
      <= max_evaluations
  )
  next_state = jax.lax.cond(
      converged | ~can_accept,
      lambda _: state._replace(
          status=jnp.where(
              converged,
              QuadStatus.CONVERGED,
              QuadStatus.MAX_EVALUATIONS,
          ),
          done=jnp.asarray(True),
      ),
      lambda _: accept_axis_and_refresh_frontier(state, axis, evaluator),
      operand=None,
  )
  ```

  `accept_axis_and_refresh_frontier` reuses cached canonical tuples, evaluates
  only missing points, refreshes all $d$ directional candidates, and uses
  lowest-axis tie behavior inherited from `jnp.argmax`.

  Before the capacity branch, return `INVALID_INPUT` for traced-invalid bounds
  or tolerances, `NONFINITE_INTEGRAND` for nonfinite values/evidence,
  `CONVERGED` for met tolerance, and `ROUNDOFF_LIMITED` when the selected
  refinement has zero new representable nodes.

  Wrap controller execution in the same `jax.lax.cond` zero-volume shortcut
  used by fixed tensor integration. Its zero branch returns
  `zero_volume_result` without constructing frontier work.

- [x] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_adaptive_tensor.py \
    tests/unit/quad/test_tensor.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/tensor.py src/jaxstro/quad/_tensor.py \
    tests/unit/quad/test_adaptive_tensor.py
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/tensor.py src/jaxstro/quad/_tensor.py \
    src/jaxstro/quad/integrate.py src/jaxstro/quad/__init__.py \
    tests/unit/quad/test_adaptive_tensor.py
  git commit -m "feat(quad): add adaptive tensor refinement"
  ```

### Task 3: Implement and certify the Genz-Malik local rule

**Files:**
- Create: `src/jaxstro/quad/cubature.py`
- Create: `src/jaxstro/quad/_cubature.py`
- Create: `tests/unit/quad/test_genz_malik.py`

**Interfaces:**
- Consumes: a region in normalized $[0,1]^d$ coordinates and the B0 evaluator.
- Produces: `GenzMalik()`, `GenzMalikData`, and
  `genz_malik_estimate(values, data)` with degree-7 value, embedded degree-5
  value, error, and per-axis fourth-difference smoothness.

- [x] **Step 1: Write failing symmetry and exactness tests**

  Create `tests/unit/quad/test_genz_malik.py`:

  ```python
  import jax.numpy as jnp
  import pytest

  from jaxstro.quad._cubature import genz_malik_data


  @pytest.mark.parametrize("dimension", range(2, 9))
  def test_genz_malik_weights_integrate_constant(dimension):
      data = genz_malik_data(dimension, jnp.float64)
      assert jnp.allclose(jnp.sum(data.high_weights), 1.0, atol=2e-14)
      assert jnp.allclose(jnp.sum(data.low_weights), 1.0, atol=2e-14)
      assert data.points.shape == (2**dimension + 2 * dimension**2 + 2 * dimension + 1, dimension)


  @pytest.mark.parametrize("power", range(8))
  def test_degree_seven_rule_integrates_axis_monomials(power):
      data = genz_malik_data(3, jnp.float64)
      values = data.points[:, 0] ** power
      estimate = jnp.sum(data.high_weights * values)
      assert jnp.allclose(estimate, 1.0 / (power + 1), atol=2e-13)
  ```

- [x] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_genz_malik.py
  ```

  Expected: FAIL because `_cubature.py` does not exist.

- [x] **Step 3: Implement the published symmetric rule**

  In `cubature.py`, define the static rule declaration:

  ```python
  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class GenzMalik:
      def tree_flatten(self):
          return (), None

      @classmethod
      def tree_unflatten(cls, _metadata, _children):
          return cls()
  ```

  In `_cubature.py`, generate center, axis, two-axis, and full-corner orbits
  from integer sign tables. Use these exact Genz-Malik constants before mapping
  from $[-1,1]^d$ to $[0,1]^d$:

  ```python
  lambda_2 = jnp.sqrt(jnp.asarray(9.0 / 70.0, dtype=dtype))
  lambda_4 = jnp.sqrt(jnp.asarray(9.0 / 10.0, dtype=dtype))
  lambda_5 = jnp.sqrt(jnp.asarray(9.0 / 19.0, dtype=dtype))
  weight_1 = (12824.0 - 9120.0 * d + 400.0 * d**2) / 19683.0
  weight_2 = 980.0 / 6561.0
  weight_3 = (1820.0 - 400.0 * d) / 19683.0
  weight_4 = 200.0 / 19683.0
  weight_5 = 6859.0 / (19683.0 * 2.0**d)
  embedded_1 = (729.0 - 950.0 * d + 50.0 * d**2) / 729.0
  embedded_2 = 245.0 / 486.0
  embedded_3 = (265.0 - 100.0 * d) / 1458.0
  embedded_4 = 25.0 / 729.0
  ```

  Encode orbit slices in `GenzMalikData` so high and embedded weights align
  with the same point array. Test every orbit multiplicity and the degree-7
  moment matrix; if a constant-to-orbit association disagrees with the cited
  1980 equations, stop and correct the association before controller work.
  Any memoized rule construction caches only NumPy arrays and static orbit
  metadata. Materialize fresh JAX constants on every public call so a cold
  cache filled during one JIT trace cannot leak tracers into a later JIT or
  VMAP trace. Add a cold-cache JIT-then-JIT-of-VMAP regression that preserves
  the target-dtype bit and orbit contracts.

- [x] **Step 4: Add local value, error, and split evidence**

  Implement:

  ```python
  class LocalCubatureEstimate(NamedTuple):
      value: Array
      error: Array
      axis_difference: Array
      nonfinite: Array


  def genz_malik_estimate(values, data):
      high = weighted_payload_sum(values, data.high_weights)
      low = weighted_payload_sum(values, data.low_weights)
      axis_difference = axis_fourth_differences(values, data)
      return LocalCubatureEstimate(
          value=high,
          error=jnp.abs(high - low),
          axis_difference=axis_difference,
          nonfinite=~jnp.all(jnp.isfinite(values)),
      )
  ```

  `axis_fourth_differences` reduces payloads with the configured `ErrorNorm`;
  the controller selects `argmax` and therefore uses the lowest axis on ties.

- [x] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_genz_malik.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/cubature.py src/jaxstro/quad/_cubature.py \
    tests/unit/quad/test_genz_malik.py
  ```

  Expected: all dimensions and moments pass. Commit:

  ```bash
  git add src/jaxstro/quad/cubature.py src/jaxstro/quad/_cubature.py \
    tests/unit/quad/test_genz_malik.py
  git commit -m "feat(quad): add Genz-Malik cubature rule"
  ```

### Task 4: Add the fixed-capacity h-adaptive cubature controller

**Files:**
- Modify: `src/jaxstro/quad/cubature.py`
- Modify: `src/jaxstro/quad/_cubature.py`
- Modify: `src/jaxstro/quad/integrate.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_cubature.py`

**Interfaces:**
- Consumes: Task 3 local estimates.
- Produces: `AdaptiveCubature(rule=GenzMalik())`,
  `CubatureReplayEvidence(lower, upper, active)`, and
  `integrate_cubature`.

- [x] **Step 1: Write failing controller tests**

  Create `tests/unit/quad/test_cubature.py`:

  ```python
  import jax.numpy as jnp

  from jaxstro import quad


  def test_cubature_integrates_array_payload_and_counts_points():
      result = quad.integrate(
          lambda x: jnp.stack(
              (jnp.ones(x.shape[0]), x[:, 0] * x[:, 1]),
              axis=-1,
          ),
          quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
          method=quad.AdaptiveCubature(),
          epsabs=1e-9,
          epsrel=1e-9,
          max_evaluations=5000,
          max_regions=64,
          gradient="stop",
      )
      assert jnp.allclose(result.value, jnp.array([1.0, 0.25]), atol=1e-9)
      assert result.error.kind == quad.ErrorKind.EMBEDDED_RULE
      assert result.work.active_regions == result.work.refinements + 1


  def test_cubature_capacity_returns_max_regions():
      result = quad.integrate(
          lambda x: jnp.exp(20.0 * jnp.sum(x, axis=-1)),
          quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
          method=quad.AdaptiveCubature(),
          epsabs=0.0,
          epsrel=0.0,
          max_evaluations=10000,
          max_regions=3,
          gradient="stop",
      )
      assert result.status == quad.QuadStatus.MAX_REGIONS
      assert result.work.active_regions == 3
  ```

- [x] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_cubature.py
  ```

  Expected: FAIL because `AdaptiveCubature` is absent.

- [x] **Step 3: Implement the region-store scan**

  In `cubature.py`, define:

  ```python
  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class AdaptiveCubature:
      rule: GenzMalik = field(default_factory=GenzMalik)

      def __post_init__(self):
          if not isinstance(self.rule, GenzMalik):
              raise TypeError("AdaptiveCubature requires GenzMalik in Phase B1")

      def tree_flatten(self):
          return (), self.rule

      @classmethod
      def tree_unflatten(cls, rule, _children):
          return cls(rule=rule)
  ```

  Define `CubatureState` with fixed arrays of shape `(max_regions, dimension)`
  for lower/upper bounds, `(max_regions, *payload_shape)` for local values and
  errors, scalar local norms, split axes, active masks, and scalar work/status
  fields. Initialize region zero to `[0,1]^d`.

  Each scan iteration must execute:

  ```python
  region = jnp.argmax(jnp.where(state.active, state.local_error_norm, -jnp.inf))
  axis = state.split_axis[region]
  midpoint = 0.5 * (
      state.region_lower[region, axis] + state.region_upper[region, axis]
  )
  converged = state.global_error_norm <= state.tolerance
  has_region_capacity = state.active_regions < max_regions
  has_evaluation_capacity = (
      state.evaluations + 2 * rule_point_count <= max_evaluations
  )
  ```

  A split is permitted only when both capacity predicates are true. If either
  capacity is exhausted, stop before evaluating either child and use
  precedence `NONFINITE_INTEGRAND`, `CONVERGED`, `MAX_EVALUATIONS`,
  `MAX_REGIONS`.
  Otherwise replace the parent with the left child, append the right child at
  the first inactive index. After the local store and active mask are updated,
  recompute the global value and componentwise nonnegative embedded error by
  masked reduction over every active leaf in deterministic row order; then
  recompute the error norm and tolerance from those authoritative reductions.
  Do not use a signed subtract-parent/add-children recurrence and do not repair
  it by clamping at zero. Add an adversarial five-leaf float32 regression where
  the signed recurrence produces a negative embedded error and false
  `CONVERGED`, while the active-leaf reduction remains positive and terminates
  on the declared region capacity. Ties are lexicographic because `jnp.argmax`
  returns the first index.

- [x] **Step 4: Assemble, dispatch, and run GREEN**

  `integrate_cubature` validates dimensions 2 through 8, `max_regions>=1`, and
  initial-rule evaluation capacity before tracing. It returns
  `QuadWork(evaluations, refinements, active_regions, deepest_depth, 0)` and
  stores normalized leaf bounds in `CubatureReplayEvidence`. It uses the shared
  zero-volume `jax.lax.cond` shortcut before initializing the region store. The
  fixed store uses the exact user-visible `max_regions` declaration clipped
  only by evaluation reachability and derived JAX integer/shape limits. It has
  no undocumented numeric row ceiling. Payload shape, dimension, dtype,
  reachable store capacity, and process/device memory estimates remain explicit
  B4 benchmark and documentation obligations; B1 makes no universal memory
  safety claim.
  Add separate tests where only evaluation capacity and only region capacity
  are exhausted, proving in scalar eager/JIT execution that neither path
  evaluates a child; add a third test proving `MAX_EVALUATIONS` precedes
  `MAX_REGIONS` when both are exhausted. Add a heterogeneous VMAP regression
  that matches stacked scalar result semantics and per-lane logical work, plus
  a caller-owned `jax.lax.map` callback regression proving physical child
  skipping for a converged lane beside a refining lane. Document this cost
  contract on `AdaptiveCubature` and `quad.integrate` without adding another
  public batching API.

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_cubature.py \
    tests/unit/quad/test_genz_malik.py \
    tests/unit/quad/test_integrate_dispatch.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/cubature.py src/jaxstro/quad/_cubature.py \
    src/jaxstro/quad/integrate.py tests/unit/quad/test_cubature.py
  ```

  Expected: all commands exit zero.

- [x] **Step 5: Commit Task 4**

  ```bash
  git add src/jaxstro/quad/cubature.py src/jaxstro/quad/_cubature.py \
    src/jaxstro/quad/integrate.py src/jaxstro/quad/__init__.py \
    tests/unit/quad/test_cubature.py
  git commit -m "feat(quad): add adaptive cubature controller"
  ```

### Task 5: Certify B1 truth, transformations, and dimensional envelope

**Files:**
- Create: `tests/validation/test_quad_multidim_deterministic.py`
- Create: `tests/integration/test_quad_multidim_deterministic_transforms.py`
- Create: `scripts/generate_quad_b1_reference.py`
- Create: `tests/validation/data/quad-b1-genz-reference.json`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: all B1 methods.
- Produces: analytic truth, Genz-family, JAX composition, and release evidence.

- [x] **Step 1: Add analytic and Genz truth cases**

  Parameterize the analytic anchors over each method's exact dimension tuple in
  `GENZ_MANIFEST["method_controls"]` and over:

  ```python
  CASES = (
      ("constant", lambda x: jnp.ones(x.shape[0]), lambda d: 1.0),
      (
          "product_moment",
          lambda x: jnp.prod(x**2, axis=-1),
          lambda d: (1.0 / 3.0) ** d,
      ),
      (
          "separable_exponential",
          lambda x: jnp.exp(-jnp.sum(x, axis=-1)),
          lambda d: (1.0 - jnp.exp(-1.0)) ** d,
      ),
  )
  ```

  Freeze:

  ```python
  GENZ_MANIFEST = {
      "dimensions": (2, 4, 6, 8),
      "families": (
          "oscillatory",
          "product_peak",
          "corner_peak",
          "gaussian",
          "continuous",
          "discontinuous",
      ),
      "a_rule": "0.35 + 0.05 * arange(1, dimension + 1)",
      "u_rule": "arange(1, dimension + 1) / (dimension + 1)",
      "dtype": "float64",
      "method_controls": {
          "fixed_tensor": {
              "dimensions": (2, 4),
              "families": (
                  "oscillatory",
                  "product_peak",
                  "corner_peak",
                  "gaussian",
              ),
              "family_dimensions": {
                  "oscillatory": (2, 4),
                  "product_peak": (2, 4),
                  "corner_peak": (2, 4),
                  "gaussian": (2, 4),
              },
              "method": "TensorProduct(GaussianRule(12))",
              "max_evaluations": "12 ** dimension",
          },
          "adaptive_tensor": {
              "dimensions": (2, 4),
              "families": (
                  "oscillatory",
                  "product_peak",
                  "corner_peak",
                  "gaussian",
                  "continuous",
              ),
              "family_dimensions": {
                  "oscillatory": (2, 4),
                  "product_peak": (2, 4),
                  "corner_peak": (2, 4),
                  "gaussian": (2, 4),
                  "continuous": (2,),
              },
              "method": "AdaptiveTensorClenshawCurtis(initial_level=2)",
              "max_evaluations": 32_768,
              "epsabs": 1.0e-8,
              "epsrel": 1.0e-8,
          },
          "adaptive_cubature": {
              "dimensions": (2, 4, 6, 8),
              "families": (
                  "oscillatory",
                  "product_peak",
                  "corner_peak",
                  "gaussian",
                  "continuous",
                  "discontinuous",
              ),
              "family_dimensions": {
                  "oscillatory": (2, 4, 6, 8),
                  "product_peak": (2, 4, 6, 8),
                  "corner_peak": (2, 4, 6, 8),
                  "gaussian": (2, 4, 6, 8),
                  "continuous": (2, 4, 8),
                  "discontinuous": (2, 4, 6, 8),
              },
              "method": "AdaptiveCubature(GenzMalik())",
              "max_evaluations": 500_000,
              "max_regions": 4_096,
              "epsabs": 1.0e-8,
              "epsrel": 1.0e-8,
          },
      },
      "structural_preflight": {
          "dimensions": (2, 3, 4, 5, 6, 7, 8),
          "adaptive_tensor_initial_evaluations": {
              2: 65,
              3: 425,
              4: 2_625,
              5: 15_625,
              6: 90_625,
              7: 515_625,
              8: 2_890_625,
          },
          "adaptive_tensor_exact_capacity_status": "accepted",
          "adaptive_tensor_under_capacity_status": "ValueError",
          "adaptive_cubature_initial_evaluations": {
              2: 17,
              3: 33,
              4: 57,
              5: 93,
              6: 149,
              7: 241,
              8: 401,
          },
          "adaptive_cubature_exact_capacity_status": "accepted",
          "adaptive_cubature_under_capacity_status": "ValueError",
      },
      "b4_carry_forward": {
          "adaptive_tensor": {
              "dimensions": (5, 6, 7, 8),
              "required_metrics": (
                  "compile_time",
                  "warm_runtime",
                  "process_memory",
                  "device_memory",
                  "dtype",
                  "payload",
                  "capacity",
              ),
              "claim_boundary": (
                  "structural acceptance is not practical runtime certification; "
                  "disclose intrinsic tensor frontier and fixed-capacity O(C d) "
                  "storage growth"
              ),
          },
          "adaptive_cubature": {
              "dimensions": (2, 4, 6, 8),
              "required_metrics": (
                  "compile_time",
                  "warm_runtime",
                  "process_memory",
                  "device_memory",
                  "dtype",
                  "payload_shape",
                  "reachable_store_capacity",
              ),
              "claim_boundary": (
                  "B1 certifies bounded declared cases, not universal "
                  "payload/dtype/store memory safety"
              ),
          },
      },
      "stress_records": {
          "adaptive_tensor_250000": {
              "dimensions": (2, 4),
              "max_evaluations": 250_000,
              "status": "incomplete_non_default_stress",
              "fresh_d2_peak_rss_bytes": 12_413_124_608,
              "fresh_d4_peak_rss_bytes": 842_678_272,
              "combined_peak_rss_bytes": 16_738_811_904,
              "combined_elapsed_seconds": 684.01,
              "completed_cases": 11,
              "threshold_misses": 0,
          },
      },
      "threshold_by_family": {
          "fixed_tensor": {
              "oscillatory": 2.0e-8,
              "product_peak": 2.0e-8,
              "corner_peak": 2.0e-8,
              "gaussian": 2.0e-8,
          },
          "adaptive_tensor": {
              "oscillatory": 5.0e-7,
              "product_peak": 5.0e-7,
              "corner_peak": 5.0e-7,
              "gaussian": 5.0e-7,
              "continuous": 5.0e-5,
              "discontinuous": 5.0e-5,
          },
          "adaptive_cubature": {
              "oscillatory": 5.0e-7,
              "product_peak": 5.0e-7,
              "corner_peak": 5.0e-7,
              "gaussian": 5.0e-7,
              "continuous": 5.0e-5,
              "discontinuous": 5.0e-5,
          },
      },
  }
  ```

  The adaptive-tensor runtime truth matrix is deliberately restricted to
  dimensions 2 and 4 under the frozen practical
  `max_evaluations=32_768` control.
  Certified Task 2 evidence distinguishes structural acceptance through
  dimension 8 from practical CPU execution and carries method-filtered
  dimensions 5 through 8 runtime, dtype, payload, capacity, and memory
  certification to B4. Do not instantiate adaptive-tensor analytic or Genz
  runtime cases in dimensions 6 or 8 in this campaign. The separate
  `structural_preflight` matrix must prove every exact initial count and
  exact-capacity acceptance/one-under-capacity rejection from dimensions 2
  through 8 without materializing a runtime payload.

  Adaptive cubature retains dimensions 2, 4, 6, and 8 only if every
  predeclared capacity executes in the bounded campaign. Do not skip, xfail,
  shrink, or tune an expensive dimension after seeing results. If one cannot
  execute, stop the campaign, preserve the failure evidence, and amend this
  reviewed manifest before further evaluation.

  The initial frozen `2.0e-5` fixed-tensor thresholds for the continuous and
  discontinuous families were rejected by the first method-filtered campaign.
  Independent dimension-2 and dimension-4 reruns with the unchanged
  `TensorProduct(GaussianRule(12))` control measured absolute errors
  `4.020689850376957e-4` and `3.3746643580634395e-4` for the continuous kink,
  and `1.455241594837775e-2` and `4.33358357839128e-2` for the discontinuous
  jump. No threshold is loosened. Fixed tensor now certifies only the four
  smooth families. Separate limitation tests retain the exact observed
  residuals, `ERROR_ESTIMATE_UNAVAILABLE`, fixed work, and the explicit
  boundary that one global Gaussian-12 tensor formula has no high-accuracy
  claim for unresolved kinks or jumps. Adaptive methods retain the non-smooth
  truth families.

  The original `250_000` control is retained only as a non-default incomplete
  stress record. In genuinely fresh processes, the dimension-2 oscillatory
  case passed in 10.35 s at 12,413,124,608-byte peak RSS and dimension 4
  passed in 10.72 s at 842,678,272-byte peak RSS. A combined cache-bounded run
  completed 11 cases with no threshold miss before it was stopped at 684.01 s
  and 16,738,811,904-byte peak RSS. Therefore 250,000 is not a release-ready
  default scientific gate.

  The reviewed practical power-of-two budget is 32,768. It remains above the
  exact initial-frontier minima 65 (dimension 2) and 2,625 (dimension 4). In
  x64 preflight it yields `(max_level, max_refinements)=(12, 10)` in dimension
  2 and `(8, 6)` in dimension 4. These are resource controls, not accuracy
  tuning; every truth threshold remains unchanged.

  The complete practical adaptive-tensor gate finished in 214.01 s at
  1,055,113,216-byte peak RSS. Fifteen cases passed and three non-smooth cases
  rejected their frozen thresholds. Dimension-2 discontinuous had absolute
  truth error `1.21599337392575e-3`, status `MAX_EVALUATIONS`, 24,961
  evaluations, 9 refinements, maximum level 7, and frontier norm
  `5.352858148793382e-3`. Dimension-4 continuous had error
  `3.9738172472236766e-4`, `MAX_EVALUATIONS`, 32,385 evaluations, 4
  refinements, maximum level 4, and frontier norm
  `6.916251122028871e-4`. Dimension-4 discontinuous had error
  `3.218803677795348e-2`, the same work/status, and frontier norm
  `2.3432820553677375e-2`. All tolerances were `1.0e-8`.

  No threshold is loosened. Adaptive tensor certifies the four smooth families
  in dimensions 2 and 4 plus the continuous family in dimension 2. The three
  rejected cases remain exact limitation records.

  Under the unchanged adaptive-cubature control, 35 of 36 analytic/Genz cases
  passed in 18.75 s at 343,048,192-byte peak RSS. The dimension-6 continuous
  case returned `MAX_EVALUATIONS` at 499,895 evaluations, 1,677 refinements,
  and 1,678 active regions, with absolute residual
  `1.0946951546741968e-4` and estimator norm
  `2.654924527754773e-4`; it missed the unchanged `5.0e-5` threshold. Retain
  this exact residual/work/error evidence as a limitation record. Adaptive
  cubature certifies every other declared family/dimension pair, including
  continuous dimensions 2, 4, and 8 and discontinuous dimensions 2, 4, 6,
  and 8. Do not change the rule, capacity, tolerance, dimension set, or
  threshold in response.

  The validation harness clears JAX compilation caches and the Task 2/Task 4
  host metadata caches after each runtime case, then runs Python garbage
  collection. This per-case teardown is part of the committed gate: manual
  fresh-process sharding alone is not sufficient evidence. It prevents the
  required combined pytest command from accumulating the 12.17 GB RSS observed
  in the rejected first multi-method shard.

  Implement the six standard integrands and their published analytic
  hypercube integrals directly in the test module. Generate
  `quad-b1-genz-reference.json` only as an independent redundant check from
  80-digit closed-form `mpmath` evaluation in
  `scripts/generate_quad_b1_reference.py`. Every rational conversion and
  formula evaluation executes inside a generator-owned 100-digit
  `mp.workdps(...)` context, independent of and restoring the caller's global
  precision. The artifact stores schema version, formula ID, exact parameter
  vectors, decimal truth, generator version, reported precision, working
  precision, and generator source SHA-256. The script supports deterministic
  `--emit` and byte-exact `--check`. No method under test or external
  quadrature routine may generate truth. Tests instantiate controls directly
  from this immutable mapping;
  changing a rule order, level, capacity, tolerance, dimension, or threshold
  requires a reviewed manifest change rather than post-hoc tuning.
  `threshold_by_family` is the single threshold owner read by assertions.

  Add `reference = ["mpmath==1.3.0", "scipy==1.16.0"]` under
  `[dependency-groups]`; it is a development-only truth and formula-validation
  environment and both packages must remain absent from
  `[project].dependencies`. Refresh the lock before emitting the artifact.

- [x] **Step 2: Add the stop-mode JAX matrix**

  In the integration test, assert eager, `jit`, `vmap`, float32, float64,
  scalar payload, array payload, real payload, and documented complex payload.
  Assert exact work/status identities for each lane and assert
  `gradient="replay"` raises the B1 capability message directing users to B4.
  Add mutation-sensitive tests that reject any artifact whose generator source
  hash or formula ID is stale and any replay mode other than exact `stop`.

- [x] **Step 3: Run the B1 scientific gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --locked --group reference python \
    scripts/generate_quad_b1_reference.py --check
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_tensor.py \
    tests/unit/quad/test_adaptive_tensor.py \
    tests/unit/quad/test_genz_malik.py \
    tests/unit/quad/test_cubature.py \
    tests/validation/test_quad_multidim_deterministic.py \
    tests/integration/test_quad_multidim_deterministic_transforms.py
  ```

  Expected: the reference artifact is byte-identical, every analytic case
  passes its predeclared threshold, and every work identity is exact.

- [x] **Step 4: Run the full B1 engineering gate and update status**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync ruff check src tests
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check src tests
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
  env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad \
    tests/integration/test_quad_multidim_transforms.py \
    tests/integration/test_quad_multidim_deterministic_transforms.py \
    tests/validation/test_quad_multidim_deterministic.py
  git diff --check
  ```

  Expected: all commands exit zero. Update `STATUS.md` with exact counts,
  dimensional envelope, known claim boundaries, the B4 carry-forward records,
  and
  `next: Run the independent whole-rung B1 review; begin B2 only if it is GREEN.`

- [x] **Step 5: Commit and request checkpoint review**

  ```bash
  git add tests/validation/test_quad_multidim_deterministic.py \
    tests/integration/test_quad_multidim_deterministic_transforms.py \
    tests/validation/data/quad-b1-genz-reference.json \
    scripts/generate_quad_b1_reference.py pyproject.toml uv.lock STATUS.md \
    .superpowers/sdd/b1-task-5-report.md
  git commit -m "test(quad): certify Phase B1 deterministic methods"
  ```

  Request independent numerical-method, JAX, API, and test-quality reviews.
  Resolve every Critical or Important finding before B2.
