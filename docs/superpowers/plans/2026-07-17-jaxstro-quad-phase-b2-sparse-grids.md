# Jaxstro Quad Phase B2 Sparse-Grid Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver isotropic, statically anisotropic, and dimension-adaptive
Smolyak integration with nested Clenshaw-Curtis rules, exact dyadic node
coalescing, fixed capacities, and sparse-frontier evidence.

**Architecture:** `sparse.py` exposes immutable method declarations and assembles
results. `_sparse.py` owns hierarchical one-dimensional differences,
downward-closed multi-index sets, exact canonical node identities, coalesced
weights, admissible frontiers, and the fixed-capacity adaptive scan. All
physical mapping and integrand evaluation remain in the B0 evaluator.

**Tech Stack:** Python 3.11+, JAX 0.10.1+, JAX NumPy, NumPy host-side sparse-rule
construction, `jax.lax.scan`, pytest, Ruff, MyPy.

## Global Constraints

- B0 and B1 must be complete and independently approved.
- Governing design:
  `docs/superpowers/specs/2026-07-17-jaxstro-quad-phase-b-multidimensional-design.md`.
- Use nested Clenshaw-Curtis rules only. Do not add a second one-dimensional
  sparse basis in Phase B.
- Use the standard level convention
  `Q_1 =` the one-point midpoint rule and
  `Q_level = ClenshawCurtisRule(2**(level - 1) + 1)` for `level >= 2`.
  Consequently the mandatory base tensor contains one node at every dimension,
  rather than `3**dimension` nodes.
- Multi-indices begin at one in every dimension.
- Fixed sets are downward closed. Adaptive candidates are admissible only when
  every valid immediate backward neighbor is accepted.
- Coalesce nodes by reduced integer dyadic-angle identity before constructing
  floating physical coordinates; never compare floating nodes for equality.
- Use profit
  `surplus_norm / max(1, genuinely_new_node_count)` and lexicographic ties.
- Stopping evidence is the active-frontier surplus sum and has
  `ErrorKind.SPARSE_GRID_SURPLUS`; it is not an absolute error certificate.
- Use `max_indices`, `max_frontier`, and `max_nodes` as distinct static
  capacities. `max_evaluations` remains the logical evaluation budget.
- Append no further status member; index exhaustion returns `MAX_INDICES`.
- Initial performance claims stop at dimensions 2 through 16. Calls above 16
  may run when capacities permit but receive no efficiency claim.
- B2 supports `gradient="stop"` only until B4.
- Every B2 method uses the B1 `zero_volume_result` shortcut before node or
  frontier evaluation and reports zero logical work for a coincident axis, but
  only after dynamic validity. Branch `INVALID_INPUT`, then zero volume, then
  evaluation; a coincident axis never hides a nonfinite bound.
- B2 status precedence is `INVALID_INPUT`, `NONFINITE_INTEGRAND`, `CONVERGED`,
  `ROUNDOFF_LIMITED`, `MAX_EVALUATIONS`, then `MAX_INDICES`.
  `ROUNDOFF_LIMITED` is emitted only when every admissible active frontier
  increment has zero genuinely new representable nodes while its surplus still
  exceeds tolerance. B2 does not emit `DIVERGENCE_SUSPECTED` without a
  separately validated detector.
- Add no runtime dependency and no sibling migration.
- Commit each task after focused RED/GREEN verification.

## File and Responsibility Map

- `src/jaxstro/quad/sparse.py`: `Smolyak`, `AdaptiveSmolyak`, validation,
  evaluator entry points, and result assembly.
- `src/jaxstro/quad/_sparse.py`: dyadic identities, hierarchical rules,
  multi-index sets, coalescing, frontier, profit, and adaptive state.
- `src/jaxstro/quad/integrate.py`: explicit B2 dispatch.
- `src/jaxstro/quad/__init__.py`: public B2 exports.
- `tests/unit/quad/test_sparse_identities.py`: exact node identity and reuse.
- `tests/unit/quad/test_smolyak.py`: fixed isotropic/anisotropic sets, moments,
  work, and capacity.
- `tests/unit/quad/test_adaptive_smolyak.py`: admissibility, profit, frontier
  evidence, deterministic ties, and statuses.
- `tests/validation/test_quad_sparse_reference.py`: analytic/Genz truth across
  dimensions 2, 4, 8, and 16.
- `tests/integration/test_quad_sparse_transforms.py`: eager/JIT/VMAP and payload
  matrix in stop mode.
- `STATUS.md`: B2 completion and B3 next action.

---

### Task 1: Build exact dyadic identities and hierarchical differences

**Files:**
- Create: `src/jaxstro/quad/_sparse.py`
- Create: `tests/unit/quad/test_sparse_identities.py`

**Interfaces:**
- Consumes: Phase A Clenshaw-Curtis nodes and weights.
- Produces: `DyadicIdentity`, `canonical_cc_identity(level, index)`,
  `hierarchical_rule(level, dtype)`, and exact tensor identity tuples.

- [ ] **Step 1: Write failing identity and difference tests**

  Create `tests/unit/quad/test_sparse_identities.py`:

  ```python
  import jax.numpy as jnp

  from jaxstro.quad._sparse import (
      canonical_cc_identity,
      hierarchical_rule,
  )


  def test_same_nested_node_has_same_reduced_identity():
      assert canonical_cc_identity(2, 1) == canonical_cc_identity(3, 2)
      assert canonical_cc_identity(2, 2) == canonical_cc_identity(4, 8)
      assert canonical_cc_identity(4, 0) == (0, 0)
      assert canonical_cc_identity(4, 16) == (1, 0)


  def test_hierarchical_difference_annihilates_constant_after_level_one():
      for level in range(2, 7):
          rule = hierarchical_rule(level, jnp.float64)
          assert jnp.allclose(jnp.sum(rule.weights), 0.0, atol=2e-14)
          assert len(rule.identities) == rule.points.shape[0]


  def test_hierarchical_weights_coalesce_before_float_conversion():
      rule = hierarchical_rule(4, jnp.float64)
      assert len(set(rule.identities)) == len(rule.identities)
  ```

- [ ] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_sparse_identities.py
  ```

  Expected: FAIL because `_sparse.py` does not exist.

- [ ] **Step 3: Implement canonical identities**

  Create `_sparse.py` with:

  ```python
  DyadicIdentity = tuple[int, int]


  def canonical_cc_identity(level: int, index: int) -> DyadicIdentity:
      if level == 1:
          if index != 0:
              raise ValueError("Clenshaw-Curtis index is outside its level")
          return 1, 1
      angle_level = level - 1
      denominator = 1 << angle_level
      if index < 0 or index > denominator:
          raise ValueError("Clenshaw-Curtis index is outside its level")
      if index == 0:
          return 0, 0
      while angle_level > 0 and index % 2 == 0:
          index //= 2
          angle_level -= 1
      return index, angle_level
  ```

  The floating point for identity `(numerator, denominator_power)` is
  `0.5 * (1.0 - cos(pi * numerator / 2**denominator_power))`. Keep the integer
  identity as the dictionary key and create the float only after coalescing.

- [ ] **Step 4: Implement hierarchical one-dimensional rules**

  Define:

  ```python
  class HierarchicalRule(NamedTuple):
      identities: tuple[DyadicIdentity, ...]
      points: Array
      weights: Array


  def unit_clenshaw_curtis(level: int, dtype) -> FixedRuleData:
      if isinstance(level, bool) or not isinstance(level, int) or level < 1:
          raise ValueError("sparse Clenshaw-Curtis level must be a positive integer")
      if level == 1:
          return FixedRuleData(
              nodes=jnp.asarray([0.5], dtype=dtype),
              weights=jnp.asarray([1.0], dtype=dtype),
              degree=1,
              nested=True,
          )
      order = (1 << (level - 1)) + 1
      data = chebyshev_rule_data(
          ClenshawCurtisRule(order),
          dtype=dtype,
      )
      return FixedRuleData(
          nodes=0.5 * (1.0 - data.nodes),
          weights=0.5 * data.weights,
          degree=data.degree,
          nested=True,
      )


  def hierarchical_rule(level: int, dtype) -> HierarchicalRule:
      current = unit_clenshaw_curtis(level, dtype)
      previous = None if level == 1 else unit_clenshaw_curtis(level - 1, dtype)
      weights: dict[DyadicIdentity, float] = {}
      for index, weight in enumerate(current.weights.tolist()):
          identity = canonical_cc_identity(level, index)
          weights[identity] = weights.get(identity, 0.0) + weight
      if previous is not None:
          for index, weight in enumerate(previous.weights.tolist()):
              identity = canonical_cc_identity(level - 1, index)
              weights[identity] = weights.get(identity, 0.0) - weight
      identities = tuple(sorted(weights))
      return HierarchicalRule(
          identities,
          jnp.asarray([identity_to_point(i) for i in identities], dtype=dtype),
          jnp.asarray([weights[i] for i in identities], dtype=dtype),
      )
  ```

  `unit_clenshaw_curtis` is owned by `_sparse.py`; it reuses the Phase A
  Clenshaw-Curtis arithmetic, uses the midpoint rule at level one, maps order
  $2^{\ell-1}+1$ from $[-1,1]$ to $[0,1]$ for levels at least two, retains the
  requested float32/float64 dtype, and leaves the Phase A rule bytes unchanged.
  Test levels 1 through 8 against the midpoint/direct Phase A values, exact
  endpoint identities, unit weight sum, and nested identity inclusion.

  Drop exact-zero host weights only after checking against zero exactly; do not
  use a floating tolerance to decide identity or reuse.

- [ ] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_sparse_identities.py \
    tests/unit/quad/test_chebyshev_rules.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/_sparse.py tests/unit/quad/test_sparse_identities.py
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/_sparse.py \
    tests/unit/quad/test_sparse_identities.py
  git commit -m "feat(quad): add exact sparse-node identities"
  ```

### Task 2: Implement fixed isotropic and anisotropic Smolyak rules

**Files:**
- Create: `src/jaxstro/quad/sparse.py`
- Modify: `src/jaxstro/quad/_sparse.py`
- Modify: `src/jaxstro/quad/integrate.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_smolyak.py`

**Interfaces:**
- Consumes: Task 1 hierarchical rules and B0 evaluator.
- Produces: `Smolyak(level, anisotropy=None)`,
  `fixed_index_set(method, dimension)`, `smolyak_rule_data`, and
  `integrate_sparse`.

- [ ] **Step 1: Write failing fixed-grid tests**

  Create `tests/unit/quad/test_smolyak.py`:

  ```python
  import jax.numpy as jnp

  from jaxstro import quad
  from jaxstro.quad._sparse import fixed_index_set, smolyak_rule_data


  def test_isotropic_index_set_is_downward_closed():
      indices = fixed_index_set(quad.Smolyak(level=3), dimension=3)
      accepted = set(indices)
      for index in accepted:
          for axis, value in enumerate(index):
              if value > 1:
                  backward = list(index)
                  backward[axis] -= 1
                  assert tuple(backward) in accepted


  def test_anisotropy_restricts_expensive_axis():
      indices = fixed_index_set(
          quad.Smolyak(level=4, anisotropy=(1.0, 3.0)),
          dimension=2,
      )
      assert max(index[1] for index in indices) < max(index[0] for index in indices)


  def test_smolyak_integrates_product_moment_with_unique_work_count():
      method = quad.Smolyak(level=4)
      result = quad.integrate(
          lambda x: jnp.prod(x**2, axis=-1),
          quad.Hyperrectangle(jnp.zeros(3), jnp.ones(3)),
          method=method,
          epsabs=1e-10,
          epsrel=1e-10,
          max_evaluations=2000,
          max_indices=128,
          max_frontier=256,
          max_nodes=2000,
          gradient="stop",
      )
      assert jnp.allclose(result.value, 1.0 / 27.0, atol=1e-10)
      assert result.error.kind == quad.ErrorKind.SPARSE_GRID_SURPLUS
      assert result.work.evaluations <= 2000
      data = smolyak_rule_data(method, dimension=3, dtype=jnp.float64)
      assert result.work.evaluations == data.points.shape[0]
  ```

- [ ] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_smolyak.py
  ```

  Expected: FAIL because `Smolyak` is absent.

- [ ] **Step 3: Implement declarations and exact index sets**

  In `sparse.py`:

  ```python
  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class Smolyak:
      level: int
      anisotropy: tuple[float, ...] | None = None

      def __post_init__(self):
          if (
              isinstance(self.level, bool)
              or not isinstance(self.level, int)
              or self.level < 1
          ):
              raise ValueError("Smolyak level must be a positive integer")
          if self.anisotropy is not None and any(
              not math.isfinite(w) or w <= 0.0 for w in self.anisotropy
          ):
              raise ValueError("Smolyak anisotropy weights must be finite and positive")

      def tree_flatten(self):
          return (), (self.level, self.anisotropy)

      @classmethod
      def tree_unflatten(cls, metadata, _children):
          level, anisotropy = metadata
          return cls(level=level, anisotropy=anisotropy)
  ```

  In `_sparse.py`, enumerate positive integer tuples and retain exactly:

  ```python
  def _inside(index, level, anisotropy):
      weights = (1.0,) * len(index) if anisotropy is None else anisotropy
      return sum(
          weight * (component - 1)
          for weight, component in zip(weights, index, strict=True)
      ) <= level - 1
  ```

  Thus `Smolyak(level=1)` contains only the all-ones base index. Sort by
  `(sum(index), index)` for deterministic construction.

- [ ] **Step 4: Coalesce the fixed sparse rule and integrate**

  `smolyak_rule_data` loops over accepted indices, takes the Cartesian product
  of hierarchical rules, and accumulates each product weight in:

  ```python
  weights: dict[tuple[DyadicIdentity, ...], float] = {}
  weights[node_id] = weights.get(node_id, 0.0) + product_weight
  ```

  Validate `len(indices)<=max_indices` and
  `len(coalesced_nodes)<=min(max_nodes,max_evaluations)` before creating JAX
  arrays. Infer payload shape with `infer_multidim_payload_zero`, then use
  `jax.lax.cond` to return `zero_volume_result` when any width is zero or
  evaluate every unique point once otherwise. For a fixed grid, define the
  reported sparse evidence as the sum of norms of the outermost accepted
  hierarchical increments; return `CONVERGED` only when that evidence meets the
  requested tolerance, otherwise `MAX_INDICES` because the declared fixed
  index set is exhausted.

- [ ] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_smolyak.py \
    tests/unit/quad/test_sparse_identities.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/sparse.py src/jaxstro/quad/_sparse.py \
    tests/unit/quad/test_smolyak.py
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/sparse.py src/jaxstro/quad/_sparse.py \
    src/jaxstro/quad/integrate.py src/jaxstro/quad/__init__.py \
    tests/unit/quad/test_smolyak.py
  git commit -m "feat(quad): add fixed Smolyak grids"
  ```

### Task 3: Add dimension-adaptive admissible refinement

**Files:**
- Modify: `src/jaxstro/quad/sparse.py`
- Modify: `src/jaxstro/quad/_sparse.py`
- Modify: `src/jaxstro/quad/integrate.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_adaptive_smolyak.py`

**Interfaces:**
- Consumes: hierarchical increments and exact node cache.
- Produces: `AdaptiveSmolyak(initial_level=1)`,
  `SparseReplayEvidence(indices, active, node_ids, coefficients)`, and the
  fixed-capacity adaptive controller.

- [ ] **Step 1: Write failing admissibility and status tests**

  Create `tests/unit/quad/test_adaptive_smolyak.py`:

  ```python
  import jax.numpy as jnp

  from jaxstro import quad
  from jaxstro.quad._sparse import admissible_forward_neighbors, select_profit


  def test_candidate_requires_every_backward_neighbor():
      accepted = {(1, 1), (2, 1)}
      assert (3, 1) in admissible_forward_neighbors(accepted, 2)
      assert (2, 2) not in admissible_forward_neighbors(accepted, 2)


  def test_profit_uses_surplus_per_new_node_and_lexicographic_tie():
      index = select_profit(
          ((2, 1), (1, 2)),
          jnp.array([2.0, 2.0]),
          jnp.array([4, 4]),
      )
      assert index == 0


  def test_adaptive_smolyak_reports_frontier_and_index_exhaustion():
      result = quad.integrate(
          lambda x: jnp.exp(jnp.sum(x, axis=-1)),
          quad.Hyperrectangle(jnp.zeros(3), jnp.ones(3)),
          method=quad.AdaptiveSmolyak(initial_level=1),
          epsabs=0.0,
          epsrel=0.0,
          max_evaluations=1000,
          max_indices=2,
          max_frontier=12,
          max_nodes=1000,
          gradient="stop",
      )
      assert result.status == quad.QuadStatus.MAX_INDICES
      assert result.work.refinements == 1
      assert result.error.kind == quad.ErrorKind.SPARSE_GRID_SURPLUS
  ```

  Add mutation-resistant tests that traced nonfinite bounds return
  `INVALID_INPUT`, a nonfinite new batch returns `NONFINITE_INTEGRAND`, an
  above-tolerance candidate with exact `new_cost == 0` returns
  `ROUNDOFF_LIMITED`, insufficient point capacity returns `MAX_EVALUATIONS`,
  and exhausted accepted-index capacity returns `MAX_INDICES`. Swapping any
  adjacent pair in the declared status precedence must fail at least one test.
  Assert `AdaptiveSmolyak(initial_level=2.5)` raises before any bit shift or
  shape construction.

- [ ] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_adaptive_smolyak.py
  ```

  Expected: FAIL because adaptive sparse owners are absent.

- [ ] **Step 3: Implement admissibility and deterministic profit**

  In `sparse.py`, define:

  ```python
  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class AdaptiveSmolyak:
      initial_level: int = 1

      def __post_init__(self):
          if (
              isinstance(self.initial_level, bool)
              or not isinstance(self.initial_level, int)
              or self.initial_level < 1
          ):
              raise ValueError("initial_level must be a positive integer")

      def tree_flatten(self):
          return (), self.initial_level

      @classmethod
      def tree_unflatten(cls, initial_level, _children):
          return cls(initial_level=initial_level)
  ```

  Implement host-side reference helpers and JAX fixed-array equivalents:

  ```python
  def is_admissible(candidate, accepted):
      for axis, component in enumerate(candidate):
          if component > 1:
              backward = list(candidate)
              backward[axis] -= 1
              if tuple(backward) not in accepted:
                  return False
      return True


  def select_profit(indices, surplus_norm, new_cost):
      profit = surplus_norm / jnp.maximum(new_cost, 1)
      return jnp.argmax(profit)
  ```

  Frontier rows are sorted lexicographically before conversion to arrays, so
  `jnp.argmax` gives the approved tie break.

- [ ] **Step 4: Implement the fixed-capacity sparse scan**

  Define `SparseState` with accepted-index arrays/masks, frontier arrays/masks,
  coalesced node IDs/values/masks, accepted value, frontier surplus norms,
  work, status, and done. The scan body computes:

  ```python
  frontier_error = jnp.sum(
      jnp.where(state.frontier_active, state.frontier_surplus_norm, 0.0)
  )
  tolerance = tolerance_threshold(state.value, epsabs, epsrel, error_norm)
  selectable = state.frontier_active & (state.frontier_new_cost > 0)
  all_active_roundoff = jnp.any(state.frontier_active) & jnp.all(
      jnp.where(
          state.frontier_active,
          state.frontier_new_cost == 0,
          True,
      )
  )
  candidate_slot = select_profit(
      state.frontier_indices,
      jnp.where(selectable, state.frontier_surplus_norm, -jnp.inf),
      state.frontier_new_cost,
  )
  ```

  The scan recomputes `valid = hyperrectangle_is_valid(domain)` and
  `nonfinite` after every newly evaluated batch. Its exact precedence is
  `INVALID_INPUT`, `NONFINITE_INTEGRAND`, `CONVERGED`, `ROUNDOFF_LIMITED`,
  `MAX_EVALUATIONS`, `MAX_INDICES`. Emit `ROUNDOFF_LIMITED` only when the
  aggregate frontier surplus is above tolerance and `all_active_roundoff` is
  true. A zero-cost row is skipped while any positive-cost active row remains.
  Add a mixed-frontier regression proving the positive-cost candidate is
  accepted rather than returning `ROUNDOFF_LIMITED`.

  Before tracing, require:

  ```python
  required_frontier = 1 + dimension * max_indices
  if max_frontier < required_frontier:
      raise ValueError(
          f"max_frontier must be at least {required_frontier} "
          "for the declared dimension and max_indices"
      )
  ```

  The proof is one initial index plus at most one forward neighbor per axis for
  every accepted index; duplicates and inadmissible neighbors can only reduce
  the count. Therefore a valid declaration cannot overflow the fixed frontier.
  Assert this bound exhaustively against host enumeration for dimensions 2
  through 16 and accepted counts 1 through 64.

  Wrap initialization in the shared zero-volume `jax.lax.cond` so no
  hierarchical node is evaluated for a coincident axis. The outer branch tests
  invalidity first and includes a mixed coincident/nonfinite regression.

- [ ] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_adaptive_smolyak.py \
    tests/unit/quad/test_smolyak.py \
    tests/unit/quad/test_sparse_identities.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/sparse.py src/jaxstro/quad/_sparse.py \
    tests/unit/quad/test_adaptive_smolyak.py
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/sparse.py src/jaxstro/quad/_sparse.py \
    src/jaxstro/quad/integrate.py src/jaxstro/quad/__init__.py \
    tests/unit/quad/test_adaptive_smolyak.py
  git commit -m "feat(quad): add adaptive Smolyak integration"
  ```

### Task 4: Certify sparse-grid truth, work, and JAX composition

**Files:**
- Create: `tests/validation/test_quad_sparse_reference.py`
- Create: `tests/integration/test_quad_sparse_transforms.py`
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: Tasks 1 through 3.
- Produces: analytic truth, exact work, dimensional performance boundary, and
  stop-mode JAX evidence.

- [ ] **Step 1: Add predeclared sparse validation cases**

  Add this predeclared registry, then parameterize every fixed and adaptive
  sparse-grid truth test over it:

  ```python
  def orthogonal_matrix(dimension: int) -> Array:
      indices = jnp.arange(dimension, dtype=jnp.float64)
      raw = jnp.cos(
          jnp.pi
          * (indices[:, None] + 0.5)
          * indices[None, :]
          / dimension
      )
      return raw / jnp.linalg.norm(raw, axis=0, keepdims=True)


  def localized_gaussian(x: Array) -> Array:
      center = jnp.asarray(0.37, dtype=x.dtype)
      beta = jnp.asarray(24.0, dtype=x.dtype)
      return jnp.exp(-beta * jnp.sum((x - center) ** 2, axis=-1))


  def localized_gaussian_truth(dimension: int) -> Array:
      center = jnp.asarray(0.37, dtype=jnp.float64)
      beta = jnp.asarray(24.0, dtype=jnp.float64)
      factor = (
          jnp.sqrt(jnp.pi)
          / (2.0 * jnp.sqrt(beta))
          * (
              jax.scipy.special.erf(jnp.sqrt(beta) * (1.0 - center))
              + jax.scipy.special.erf(jnp.sqrt(beta) * center)
          )
      )
      return factor**dimension


  @dataclass(frozen=True)
  class SparseTruthCase:
      case_id: str
      dimensions: tuple[int, ...]
      integrand: Callable[[Array], Array]
      truth: Callable[[int], Array]


  SPARSE_TRUTH_CASES = (
      SparseTruthCase(
          "product_quadratic",
          (2, 4, 8, 16),
          lambda x: jnp.prod(1.0 + x**2, axis=-1),
          lambda d: jnp.asarray((4.0 / 3.0) ** d),
      ),
      SparseTruthCase(
          "separable_exponential",
          (2, 4, 8, 16),
          lambda x: jnp.exp(-jnp.sum(x, axis=-1)),
          lambda d: jnp.asarray((1.0 - jnp.exp(-1.0)) ** d),
      ),
      SparseTruthCase(
          "rotated_quadratic",
          (2, 4, 8),
          lambda x: jnp.sum((x @ orthogonal_matrix(x.shape[-1])) ** 2, axis=-1),
          lambda d: jnp.asarray(d / 3.0),
      ),
      SparseTruthCase(
          "localized_gaussian",
          (2, 4),
          localized_gaussian,
          localized_gaussian_truth,
      ),
      SparseTruthCase(
          "axis_zero_anisotropy",
          (2, 4, 8, 16),
          lambda x: jnp.exp(-8.0 * x[..., 0]),
          lambda d: jnp.asarray((1.0 - jnp.exp(-8.0)) / 8.0),
      ),
  )
  ```

  Import `jax`, `jax.numpy as jnp`, `Callable`, and the repository's `Array`
  alias at module scope. Do not use the implementation under test or a
  comparator to generate truth. Assert value error, frontier evidence identity,
  unique-node count, accepted-index count, downward closure, and that the
  anisotropic case refines axis zero first.

- [ ] **Step 2: Add the stop-mode JAX matrix**

  Test eager, `jit`, `vmap`, float32/float64, scalar/array/complex payloads, and
  heterogeneous accepted index sets. Assert replay is rejected with the B4
  boundary message.

- [ ] **Step 3: Run the B2 scientific gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_sparse_identities.py \
    tests/unit/quad/test_smolyak.py \
    tests/unit/quad/test_adaptive_smolyak.py \
    tests/validation/test_quad_sparse_reference.py \
    tests/integration/test_quad_sparse_transforms.py
  ```

  Expected: all truth and exact-work assertions pass.

- [ ] **Step 4: Run engineering gates and update status**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync ruff check src tests
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check src tests
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
  env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad \
    tests/validation/test_quad_sparse_reference.py \
    tests/integration/test_quad_sparse_transforms.py
  git diff --check
  ```

  Expected: all commands exit zero. Update `STATUS.md` with exact counts,
  tested dimensions, estimator caveat, and
  `next: Execute the reviewed Phase B3 Sobol and randomized-QMC plan.`

- [ ] **Step 5: Commit and request checkpoint review**

  ```bash
  git add tests/validation/test_quad_sparse_reference.py \
    tests/integration/test_quad_sparse_transforms.py STATUS.md
  git commit -m "test(quad): certify Phase B2 sparse grids"
  ```

  Request independent sparse-grid, JAX, API, and test-quality reviews. Resolve
  every Critical or Important finding before B3.
