# Jaxstro Quad Phase B4 Replay, Quantities, Evidence, and Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-order accepted-formula replay to every Phase B family,
heterogeneous quantity coordinates and measures, independent truth and
comparison evidence, benchmark-gated optimization, researcher-facing MyST
derivations, and the complete Phase B release gate.

**Architecture:** Every B1 through B3 primal solver emits one private
`ReplayFormula` containing stopped normalized reference points, normalized
formula weights, and an active mask. `_multidim_replay.py` reconstructs physical
points, Jacobians, densities, and integrand values from live bounds and explicit
`args`; only `QuadResult.value` receives a tangent. Quantity mode remains an
eager adapter around the same raw engine. Evidence generators, comparator
adapters, benchmark manifests, and MyST pages remain outside runtime code.

**Tech Stack:** Python 3.11+, JAX 0.10.1+, JAX NumPy, `jax.custom_jvp`,
`jax.jvp`, `jax.vjp`, `jaxstro.quantity`, pytest, Ruff, MyPy, JSON evidence,
MyST, KaTeX/LaTeX; development-only SciPy 1.16.0, Tasmanian 8.2, and
Torchquad 0.5.0 isolated comparison environments.

## Global Constraints

- B0 through B3 must be complete and independently approved.
- Governing design:
  `docs/superpowers/specs/2026-07-17-jaxstro-quad-phase-b-multidimensional-design.md`.
- Replay differentiates the accepted formula, never method selection,
  refinement, sparse admissibility, randomization, sample allocation, stopping,
  statuses, work, or uncertainty evidence.
- Only `QuadResult.value` has a scientific tangent. Floating diagnostic
  tangents are exact zero and integer/Boolean tangents are JAX `float0`.
- Replay is first-order only. Higher derivatives must fail with the same
  explicit unsupported-contract message across every family.
- A coincident axis has an exact zero primal in `gradient="stop"` and is an
  `INVALID_INPUT` replay boundary with a nonfinite value tangent.
- `args` remains the sole generic differentiable parameter container.
- Quantity mode unwraps values before the raw engine and restores one common
  output unit afterward; it never forks a second numerical implementation.
- Quantity support remains alpha and opt-in. Do not change sibling defaults or
  migrate another package.
- `ProductMeasure` accepts one finite one-dimensional Lebesgue or weighted
  component per axis. Infinite-domain classical measures remain unsupported.
- Randomized quantity integration remains real scalar and requires
  unit-compatible estimate bounds.
- Comparators are development-only. Add none to `[project].dependencies`.
  Keep SciPy in Jaxstro's lock because SciPy 1.16.0 supports the project's
  Python 3.11 floor. Keep Tasmanian and Torchquad in separately locked
  comparison environments so their transitive constraints cannot perturb
  Jaxstro's JAX runtime.
- Comparison labels are record-specific: `exact`, `strong-match`,
  `node-matched`, `family-matched`, or `capability`.
- Freeze a baseline before optimization. Optimize only a predeclared measured
  trigger, write an addendum first, preserve the immutable baseline, and require
  two independent optimized suites.
- MyST pages use method-family navigation, LaTeX derivations, admonitions,
  astrophysical examples, failure modes, audit recipes, and warranted-claim
  boundaries. Use no course or instructor framing.
- Do not publish, push, deploy, or migrate siblings in this plan.
- Commit each task after focused verification.

## File and Responsibility Map

- `src/jaxstro/quad/_multidim_replay.py`: universal accepted-formula evidence,
  replay reconstruction, custom JVP, and diagnostic tangents.
- `src/jaxstro/quad/tensor.py`, `cubature.py`, `sparse.py`, `qmc.py`: emit
  formula evidence beside the unchanged primal result.
- `src/jaxstro/quad/integrate.py`: select stop/replay and quantity/raw adapters.
- `src/jaxstro/quad/coordinates.py`: `Axis`, `CoordinatePoint`, unit conversion,
  and static unit PyTree behavior.
- `src/jaxstro/quad/domains.py`: heterogeneous `Hyperrectangle.from_axes`.
- `src/jaxstro/quad/measures.py`: finite `ProductMeasure` declaration.
- `src/jaxstro/quad/_quantity.py`: multidimensional unit inference,
  integrand/density wrappers, tolerance conversion, and result restoration.
- `tests/unit/quad/test_multidim_replay.py`: formula and tangent substrate.
- `tests/integration/test_quad_multidim_replay_transforms.py`: full first-order
  family transform matrix.
- `tests/validation/test_quad_multidim_replay_derivatives.py`: analytic and
  independent finite-difference audits.
- `tests/unit/quad/test_multidim_quantity.py`: axes, points, measures, units.
- `tests/integration/test_quad_multidim_quantity_transforms.py`: representation
  invariance and derivative rescaling.
- `tests/validation/test_quad_multidim_truth.py`: complete deterministic truth
  and domain-neutral astrophysical applications.
- `scripts/generate_quad_multidim_evidence.py`: deterministic truth artifact.
- `scripts/quad_multidim_benchmark_adapters.py`: development-only comparators.
- `scripts/benchmark_quad_multidim.py`: matched benchmark runner and manifests.
- `tests/integration/test_quad_multidim_comparisons.py`: adapter labels and
  matched-control contracts.
- `tests/integration/test_quad_multidim_benchmark_contract.py`: immutable
  baseline and optimization trigger checks.
- `docs/validation/quad-multidim-*.json`: generated evidence and benchmark data.
- `docs/20-methods/approximation-integration/multidimensional/*.md`: seven
  method-family theory and choice pages.
- `docs/50-api/approximation-integration/quad-*.md`: family-grouped API pages.
- `docs/60-validation/numerical/quadrature-multidimensional.md`: evidence page.
- `docs/myst.yml`, route manifest, evidence index, roadmap, SOTA assessment,
  and `STATUS.md`: navigation and final completion state.

---

### Task 1: Normalize every accepted method to one replay formula

**Files:**
- Create: `src/jaxstro/quad/_multidim_replay.py`
- Modify: `src/jaxstro/quad/tensor.py`
- Modify: `src/jaxstro/quad/cubature.py`
- Modify: `src/jaxstro/quad/sparse.py`
- Modify: `src/jaxstro/quad/qmc.py`
- Modify: `src/jaxstro/quad/integrate.py`
- Create: `tests/unit/quad/test_multidim_replay.py`

**Interfaces:**
- Consumes: each family's stopped accepted state.
- Produces: `ReplayFormula(reference_points, reference_weights, active_mask)`,
  `MultidimPrimalSolve(result, formula)`,
  `_prepare_multidim_solve(fun, domain, *, args, method, measure, key, epsabs, epsrel,
  max_evaluations, max_regions, max_indices, max_frontier, max_nodes,
  error_norm)`, one config-based `_solve_multidim` core, and family formula
  constructors.

- [ ] **Step 1: Write failing formula-equivalence tests**

  Create `tests/unit/quad/test_multidim_replay.py`:

  ```python
  import jax
  import jax.numpy as jnp
  import pytest

  from jaxstro import quad
  from jaxstro.quad._multidim_replay import replay_formula_value
  from jaxstro.quad.integrate import _prepare_multidim_solve


  METHODS = (
      quad.TensorProduct(quad.GaussianRule(3)),
      quad.AdaptiveTensorClenshawCurtis(initial_level=2),
      quad.AdaptiveCubature(),
      quad.Smolyak(level=3),
      quad.AdaptiveSmolyak(initial_level=1),
      quad.Sobol(level=7),
      quad.ScrambledSobol(level=7, replicates=8),
  )


  @pytest.mark.parametrize("method", METHODS)
  def test_stopped_formula_reproduces_primal_value(method):
      solve = _prepare_multidim_solve(
          lambda x: jnp.exp(jnp.sum(x, axis=-1)),
          quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
          method=method,
          key=jax.random.key(3),
          args=(),
          measure=quad.LebesgueMeasure(),
          epsabs=1e-8,
          epsrel=1e-8,
          max_evaluations=4096,
          max_regions=64,
          max_indices=64,
          max_frontier=256,
          max_nodes=4096,
          error_norm=quad.MaxNorm(),
      )
      replayed = replay_formula_value(
          solve.config,
          solve.domain,
          solve.args,
          solve.formula,
      )
      assert jnp.allclose(replayed, solve.result.value, rtol=2e-12, atol=2e-12)
  ```

  `MultidimPrimalSolve` carries `config`, `domain`, and `args` as stopped
  private evidence fields in addition to `result` and `formula`; these fields
  are never added to `QuadResult`.

- [ ] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_multidim_replay.py
  ```

  Expected: FAIL because replay formula owners are absent.

- [ ] **Step 3: Define the common formula and primal solve**

  Create `_multidim_replay.py`:

  ```python
  @dataclass(frozen=True)
  class MultidimConfig:
      fun: Callable
      method: Any
      measure: Any
      max_evaluations: int
      max_regions: int | None
      max_indices: int | None
      max_frontier: int | None
      max_nodes: int | None
      error_norm: Any


  class ReplayFormula(NamedTuple):
      reference_points: Any
      reference_weights: Any
      active_mask: Any


  class MultidimPrimalSolve(NamedTuple):
      result: QuadResult
      formula: ReplayFormula
      config: MultidimConfig
      domain: Any
      args: Any
  ```

  Every family pads formula points/weights to its own static capacity and marks
  logical entries with `active_mask`. The normalized reference weight includes
  tensor rule weight, leaf-region volume, sparse combination coefficient, or
  reciprocal QMC point/replicate count, but never the outer physical
  hyperrectangle Jacobian or density.

  In `integrate.py`, add:

  ```python
  def _solve_multidim(
      config: MultidimConfig,
      domain,
      args,
      key,
      epsabs,
      epsrel,
  ) -> MultidimPrimalSolve:
      ...


  def _prepare_multidim_solve(
      fun,
      domain,
      *,
      args,
      method,
      measure,
      key,
      epsabs,
      epsrel,
      max_evaluations,
      max_regions,
      max_indices,
      max_frontier,
      max_nodes,
      error_norm,
  ) -> MultidimPrimalSolve:
      config = MultidimConfig(
          fun=fun,
          method=method,
          measure=measure,
          max_evaluations=max_evaluations,
          max_regions=max_regions,
          max_indices=max_indices,
          max_frontier=max_frontier,
          max_nodes=max_nodes,
          error_norm=error_norm,
      )
      return _solve_multidim(config, domain, args, key, epsabs, epsrel)
  ```

  `_prepare_multidim_solve` is the keyword-rich facade and constructs
  `MultidimConfig` exactly once. `_solve_multidim` is the only core signature
  used by the custom-JVP path and dispatches concrete method types to family
  primal solvers, each of which returns `MultidimPrimalSolve`.

- [ ] **Step 4: Emit exact formulas from every family**

  - Fixed/adaptive tensor: accepted tensor points and product weights.
  - Cubature: concatenate active leaf points transformed into outer normalized
    coordinates; multiply rule weights by normalized leaf volumes.
  - Sparse: accepted coalesced identities and final combination weights.
  - Deterministic QMC: realized points and uniform `1/N` weights.
  - Randomized QMC: realized points with `1/(R*N)` weights; for adaptive QMC,
    use only the final accepted schedule row.

  Replace family return types internally with `MultidimPrimalSolve`; public
  family entry points continue returning only `.result` until Task 2 enables
  replay.

- [ ] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_multidim_replay.py \
    tests/unit/quad/test_tensor.py \
    tests/unit/quad/test_cubature.py \
    tests/unit/quad/test_smolyak.py \
    tests/unit/quad/test_qmc.py
  ```

  Expected: all formula/primal equivalence checks pass. Commit:

  ```bash
  git add src/jaxstro/quad/_multidim_replay.py \
    src/jaxstro/quad/tensor.py src/jaxstro/quad/cubature.py \
    src/jaxstro/quad/sparse.py src/jaxstro/quad/qmc.py \
    src/jaxstro/quad/integrate.py \
    tests/unit/quad/test_multidim_replay.py
  git commit -m "refactor(quad): expose multidimensional replay formulas"
  ```

### Task 2: Implement first-order replay and the complete JAX matrix

**Files:**
- Modify: `src/jaxstro/quad/_multidim_replay.py`
- Modify: `src/jaxstro/quad/integrate.py`
- Create: `tests/integration/test_quad_multidim_replay_transforms.py`
- Create: `tests/validation/test_quad_multidim_replay_derivatives.py`

**Interfaces:**
- Consumes: Task 1 `ReplayFormula`.
- Produces: `multidim_replay_core(config, domain, args, key, epsabs, epsrel)`
  and replay-default public behavior for B1 through B3.

- [ ] **Step 1: Write failing tangent and boundary tests**

  Create the integration test with:

  ```python
  METHODS = (
      quad.TensorProduct(quad.GaussianRule(3)),
      quad.AdaptiveTensorClenshawCurtis(initial_level=2),
      quad.AdaptiveCubature(),
      quad.Smolyak(level=3),
      quad.AdaptiveSmolyak(initial_level=1),
      quad.Sobol(level=7),
      quad.ScrambledSobol(level=7, replicates=8),
  )


  def controls(method):
      base = {
          "epsabs": 1e-7,
          "epsrel": 1e-7,
          "max_evaluations": 4096,
          "gradient": "replay",
      }
      if isinstance(method, quad.AdaptiveCubature):
          base["max_regions"] = 64
      if isinstance(method, (quad.Smolyak, quad.AdaptiveSmolyak)):
          base.update(
              max_indices=64,
              max_frontier=256,
              max_nodes=4096,
          )
      if isinstance(method, (quad.ScrambledSobol, quad.AdaptiveScrambledSobol)):
          base["key"] = jax.random.key(5)
      return base


  @pytest.mark.parametrize("method", METHODS)
  def test_replay_gradient_matches_analytic_parameter_derivative(method):
      domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))

      def objective(scale):
          return quad.integrate(
              lambda x, live_scale: live_scale * jnp.sum(x, axis=-1),
              domain,
              args=scale,
              method=method,
              **controls(method),
          ).value

      assert jnp.allclose(jax.grad(objective)(2.0), 1.0, rtol=3e-5, atol=3e-5)


  def test_coincident_bound_replay_fails_closed():
      domain = quad.Hyperrectangle(
          jnp.array([0.0, 1.0]),
          jnp.array([2.0, 1.0]),
      )
      result = quad.integrate(
          lambda x: jnp.sum(x, axis=-1),
          domain,
          method=quad.TensorProduct(quad.GaussianRule(3)),
          epsabs=1e-8,
          epsrel=1e-8,
          max_evaluations=9,
          gradient="replay",
      )
      assert result.status == quad.QuadStatus.INVALID_INPUT
      tangent = jax.jvp(lambda upper: quad.integrate(
          lambda x: jnp.sum(x, axis=-1),
          quad.Hyperrectangle(domain.lower, upper),
          method=quad.TensorProduct(quad.GaussianRule(3)),
          epsabs=1e-8, epsrel=1e-8, max_evaluations=9,
          gradient="replay",
      ).value, (domain.upper,), (jnp.ones(2),))[1]
      assert not jnp.isfinite(tangent)
  ```

- [ ] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/integration/test_quad_multidim_replay_transforms.py
  ```

  Expected: FAIL because B1 through B3 still reject replay.

- [ ] **Step 3: Implement live formula reconstruction**

  Add:

  ```python
  def replay_formula_value(config, domain, args, formula):
      active = jax.lax.stop_gradient(formula.active_mask)
      points = jax.lax.stop_gradient(formula.reference_points)
      weights = jax.lax.stop_gradient(formula.reference_weights)
      safe_index = jnp.argmax(active)
      safe_point = points[safe_index]
      evaluation_points = jnp.where(
          active[:, None],
          points,
          safe_point[None, :],
      )
      evaluated = evaluate_multidim(
          config.fun,
          domain,
          evaluation_points,
          args=args,
          measure=config.measure,
      )
      factors = jnp.where(
          active,
          weights * evaluated.weights,
          jnp.zeros_like(weights),
      )
      active_values = active.reshape(
          active.shape + (1,) * (evaluated.values.ndim - active.ndim)
      )
      values = jnp.where(
          active_values,
          evaluated.values,
          jnp.zeros_like(evaluated.values),
      )
      return jnp.sum(
          values
          * factors.reshape(
              factors.shape + (1,) * (values.ndim - factors.ndim)
          ),
          axis=0,
      )
  ```

  `evaluate_multidim` supplies the live physical map, outer Jacobian, density,
  and integrand. It must not stop bounds or `args`. Every nonzero-volume formula
  must contain at least one active point; assert that invariant when formulas
  are constructed. Add a regression with a singular integrand at the padded
  point value, proving inactive padding cannot introduce a nonfinite primal or
  tangent.

- [ ] **Step 4: Implement the custom JVP and stopped diagnostics**

  Follow the existing one-dimensional `_replay.result_tangent` contract:

  ```python
  @partial(jax.custom_jvp, nondiff_argnums=(0,))
  def multidim_replay_core(config, domain, args, key, epsabs, epsrel):
      return _solve_multidim(
          config, domain, args, key, epsabs, epsrel
      ).result


  @multidim_replay_core.defjvp
  def _multidim_replay_jvp(config, primals, tangents):
      domain, args, key, epsabs, epsrel = primals
      domain_tangent, args_tangent, _key_tangent, _, _ = tangents
      solve = _solve_multidim(config, domain, args, key, epsabs, epsrel)
      formula = jax.tree.map(jax.lax.stop_gradient, solve.formula)
      _, value_tangent = jax.jvp(
          lambda live_domain, live_args: replay_formula_value(
              config, live_domain, live_args, formula
          ),
          (domain, args),
          (domain_tangent, args_tangent),
      )
      zero_width = jnp.any(jnp.asarray(domain.lower) == domain.upper)
      value_tangent = jnp.where(
          zero_width,
          jnp.full_like(value_tangent, jnp.nan),
          value_tangent,
      )
      return solve.result, result_tangent(solve.result, value_tangent)
  ```

  The primal dispatcher sets `INVALID_INPUT` for a replay call with any
  coincident axis. Add a nested-JVP guard that raises
  `"multidimensional replay supports first derivatives only"`.

- [ ] **Step 5: Add independent derivative validation, run GREEN, and commit**

  Cover every family with analytic derivatives, frozen-formula central finite
  differences, adaptive-rerun finite differences, moving lower/upper bounds,
  explicit parameter PyTrees, scalar/array/complex deterministic payloads,
  eager/JVP/VJP/grad/jit-grad/vmap-grad/jit-vmap-grad, and diagnostic zero
  tangents.

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_multidim_replay.py \
    tests/integration/test_quad_multidim_replay_transforms.py \
    tests/validation/test_quad_multidim_replay_derivatives.py
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/_multidim_replay.py \
    src/jaxstro/quad/integrate.py \
    tests/integration/test_quad_multidim_replay_transforms.py \
    tests/validation/test_quad_multidim_replay_derivatives.py
  git commit -m "feat(quad): add multidimensional replay derivatives"
  ```

### Task 3: Add heterogeneous quantity coordinates and finite measures

**Files:**
- Create: `src/jaxstro/quad/coordinates.py`
- Modify: `src/jaxstro/quad/domains.py`
- Modify: `src/jaxstro/quad/measures.py`
- Modify: `src/jaxstro/quad/_quantity.py`
- Modify: `src/jaxstro/quad/_multidim.py`
- Modify: `src/jaxstro/quad/integrate.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_multidim_quantity.py`
- Create: `tests/integration/test_quad_multidim_quantity_transforms.py`

**Interfaces:**
- Consumes: existing `Quantity`, `Unit`, and raw Phase B engine.
- Produces: `Axis`, `CoordinatePoint`, `Hyperrectangle.from_axes`,
  `ProductMeasure`, and heterogeneous-unit result restoration.

- [ ] **Step 1: Write failing coordinate and result-unit tests**

  Create `tests/unit/quad/test_multidim_quantity.py`:

  ```python
  import jax.numpy as jnp

  from jaxstro import quad
  from jaxstro.quantity import Msun, Myr, Quantity, pc


  def test_coordinate_point_exposes_each_static_axis_unit():
      point = quad.CoordinatePoint(
          values=jnp.array([[2.0, 3.0]]),
          units=(pc, Myr),
      )
      assert point.shape == (1, 2)
      assert point.dimension == 2
      assert point.axis(0).unit == pc
      assert point.axis(1).unit == Myr


  def test_heterogeneous_axes_produce_product_result_unit():
      domain = quad.Hyperrectangle.from_axes(
          (
              quad.Axis(Quantity(0.0, pc), Quantity(2.0, pc)),
              quad.Axis(Quantity(0.0, Myr), Quantity(3.0, Myr)),
          )
      )
      result = quad.integrate(
          lambda x: Quantity(jnp.ones(x.shape[:-1]), Msun),
          domain,
          method=quad.TensorProduct(quad.GaussianRule(2)),
          epsabs=Quantity(1e-10, Msun * pc * Myr),
          epsrel=1e-10,
          max_evaluations=4,
      )
      assert result.value.unit == Msun * pc * Myr
      assert jnp.allclose(result.value.value, 6.0)
  ```

  The imports shown match the current `jaxstro.quantity` public surface.

- [ ] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_multidim_quantity.py
  ```

  Expected: FAIL because heterogeneous coordinate types are absent.

- [ ] **Step 3: Implement `Axis` and `CoordinatePoint`**

  In `coordinates.py`:

  ```python
  @dataclass(frozen=True)
  class Axis:
      lower: Quantity
      upper: Quantity

      def __post_init__(self):
          if not self.lower.unit.is_compatible_with(self.upper.unit):
              raise DimensionError(
                  "Axis bounds must have compatible units",
                  operation="quad-axis",
              )
          if jnp.shape(self.lower.value) or jnp.shape(self.upper.value):
              raise ValueError("Axis bounds must be scalar")

      @property
      def unit(self):
          return self.lower.unit


  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class CoordinatePoint:
      values: Any
      units: tuple[Unit, ...]

      @property
      def shape(self):
          return jnp.shape(self.values)

      @property
      def dimension(self):
          return len(self.units)

      def axis(self, index: int):
          if not isinstance(index, int):
              raise TypeError("CoordinatePoint axis index must be static")
          return Quantity(self.values[..., index], self.units[index])

      def as_quantity(self, unit: Unit):
          converted = [
              self.axis(i).to_value(unit) for i in range(self.dimension)
          ]
          return Quantity(jnp.stack(converted, axis=-1), unit)
  ```

  Register `units` as static PyTree metadata and validate
  `values.shape[-1]==len(units)`.

- [ ] **Step 4: Extend domains, measures, and the eager adapter**

  `Hyperrectangle` gains static `axis_units: tuple[Unit,...] | None`.
  `from_axes` chooses each lower unit, converts its upper bound, and stacks raw
  magnitudes:

  ```python
  @classmethod
  def from_axes(cls, axes):
      units = tuple(axis.unit for axis in axes)
      lower = jnp.stack([axis.lower.to_value(axis.unit) for axis in axes])
      upper = jnp.stack([axis.upper.to_value(axis.unit) for axis in axes])
      return cls(lower, upper, axis_units=units)
  ```

  Add:

  ```python
  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class ProductMeasure(_StaticMeasure):
      components: tuple[LebesgueMeasure | WeightedMeasure, ...]
  ```

  Extend `_quantity.normalize_call` to infer the integrand unit with a
  shape-only `CoordinatePoint`, wrap physical points before calling the user
  integrand/density, compute
  `result_unit = integrand_unit * product(axis_units) * density_unit`, convert
  `epsabs`, and restore `value`, error estimate/norm, and tolerance. Reject
  mixed output dimensions, wrong component count, infinite measures, and
  incompatible estimate bounds.

- [ ] **Step 5: Add representation invariance, run GREEN, and commit**

  Test pc/kpc, s/Myr, reversed axes, dimensionless axes, `ProductMeasure`,
  multidimensional `WeightedMeasure`, normalized metadata, moving-bound
  derivatives in raw magnitudes, and identical results under compatible unit
  representation changes.

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_multidim_quantity.py \
    tests/integration/test_quad_multidim_quantity_transforms.py \
    tests/unit/test_quad_quantity.py \
    tests/integration/test_quad_quantity_transforms.py
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/coordinates.py src/jaxstro/quad/domains.py \
    src/jaxstro/quad/measures.py src/jaxstro/quad/_quantity.py \
    src/jaxstro/quad/_multidim.py src/jaxstro/quad/integrate.py \
    src/jaxstro/quad/__init__.py \
    tests/unit/quad/test_multidim_quantity.py \
    tests/integration/test_quad_multidim_quantity_transforms.py
  git commit -m "feat(quad): add heterogeneous quantity axes"
  ```

### Task 4: Build complete truth and astrophysical validation evidence

**Files:**
- Create: `tests/validation/test_quad_multidim_truth.py`
- Create: `scripts/generate_quad_multidim_evidence.py`
- Create: `docs/validation/quad-multidim-truth.json`
- Create: `docs/validation/quad-multidim-replay.json`

**Interfaces:**
- Consumes: all Phase B families, replay, and quantities.
- Produces: generated analytic/reference truth, derivative, status, work, and
  domain-neutral astrophysical evidence.

- [ ] **Step 1: Freeze the validation registry**

  Define records with exact owner, dimension, dtype, method, controls, truth
  source, tolerance, and expected claim:

  ```python
  VALIDATION_FAMILIES = (
      "tensor_polynomial",
      "beta_product",
      "separable_exponential",
      "rotated_smooth",
      "genz_oscillatory",
      "genz_product_peak",
      "genz_corner_peak",
      "genz_gaussian",
      "genz_continuous",
      "genz_discontinuous",
      "localized_peak",
      "boundary_layer",
  )
  ```

  Each nonanalytic reference stores generator name, external owner, precision,
  parameters, source hash, and absolute uncertainty.

- [ ] **Step 2: Add four domain-neutral astrophysical cases**

  Implement the following independent closed-form fixtures without importing a
  sibling model:

  ```python
  def projected_plummer_aperture(x, scale):
      radius, angle = x[..., 0], x[..., 1]
      del angle
      surface_density = 1.0 / (
          jnp.pi * scale**2 * (1.0 + (radius / scale) ** 2) ** 2
      )
      return surface_density * radius


  def projected_plummer_truth(radius, scale):
      return radius**2 / (radius**2 + scale**2)


  def diagonal_gaussian(x, sigma):
      z = x / sigma
      normalization = jnp.prod(
          1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)
      )
      return normalization * jnp.exp(-0.5 * jnp.sum(z**2, axis=-1))


  def bounded_gaussian_mass(limit, sigma):
      return jnp.prod(
          jax.scipy.special.erf(limit / (jnp.sqrt(2.0) * sigma))
      )


  def bounded_gaussian_second_moment(limit, sigma):
      standardized = limit / sigma
      mass_1d = jax.scipy.special.erf(standardized / jnp.sqrt(2.0))
      boundary = (
          2.0
          * standardized
          * jnp.exp(-0.5 * standardized**2)
          / jnp.sqrt(2.0 * jnp.pi)
      )
      return sigma**2 * (mass_1d - boundary)


  def population_moment(x):
      mass, metallicity, age, distance = jnp.moveaxis(x, -1, 0)
      return mass**2 * (1.0 + metallicity) * age * distance**2


  def population_moment_truth(bounds):
      lower, upper = bounds
      powers = jnp.asarray([2, 0, 1, 2])
      factors = (
          upper ** (powers + 1) - lower ** (powers + 1)
      ) / (powers + 1)
      factors = factors.at[1].set(
          (upper[1] - lower[1])
          + 0.5 * (upper[1] ** 2 - lower[1] ** 2)
      )
      return jnp.prod(factors)


  def separable_selection(x, center, width):
      return jnp.prod(jax.nn.sigmoid((center - x) / width), axis=-1)


  def separable_selection_truth(lower, upper, center, width):
      antiderivative = lambda value: (
          -width * jax.nn.softplus((center - value) / width)
      )
      return jnp.prod(antiderivative(upper) - antiderivative(lower))
  ```

  Use the Plummer fixture on
  $[0,R]\times[0,2\pi]$ so the polar Jacobian is explicit and the truth is the
  enclosed projected mass fraction. For the Gaussian fixture, test both
  bounded normalization and each diagonal raw second moment by multiplying the
  selected one-dimensional second-moment factor by the other axes' mass
  factors. Use explicit raw units and a dimensionless normalized counterpart
  for every case, record the equations and bounds in the evidence artifact,
  and test raw-versus-quantity representation invariance.

- [ ] **Step 3: Generate and test deterministic evidence**

  `--emit` writes sorted JSON with environment, JAX x64 state, controls, value,
  truth, absolute/relative error, error kind, status, work, and replay gradient.
  `--check` regenerates in memory and requires byte equality.

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/generate_quad_multidim_evidence.py --emit
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/validation/test_quad_multidim_truth.py \
    tests/validation/test_quad_multidim_replay_derivatives.py
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/generate_quad_multidim_evidence.py --check
  ```

  Expected: all cases pass predeclared scientific thresholds and freshness.

- [ ] **Step 4: Commit Task 4**

  ```bash
  git add tests/validation/test_quad_multidim_truth.py \
    scripts/generate_quad_multidim_evidence.py \
    docs/validation/quad-multidim-truth.json \
    docs/validation/quad-multidim-replay.json
  git commit -m "test(quad): add multidimensional truth evidence"
  ```

### Task 5: Add calibrated comparisons and immutable performance suites

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `scripts/quad_multidim_benchmark_adapters.py`
- Create: `scripts/benchmark_quad_multidim.py`
- Create: `laboratory/quad-multidim-comparison/tasmanian/pyproject.toml`
- Create: `laboratory/quad-multidim-comparison/tasmanian/uv.lock`
- Create: `laboratory/quad-multidim-comparison/torchquad/pyproject.toml`
- Create: `laboratory/quad-multidim-comparison/torchquad/uv.lock`
- Create: `tests/integration/test_quad_multidim_comparisons.py`
- Create: `tests/integration/test_quad_multidim_benchmark_contract.py`
- Create: `docs/validation/quad-multidim-comparisons.json`
- Create: `docs/validation/quad-multidim-performance-baseline.json`

**Interfaces:**
- Consumes: truth registry and all Phase B methods.
- Produces: development-only matched comparator records, immutable baseline,
  and explicit optimization triggers.

- [ ] **Step 1: Add a non-runtime comparison dependency group**

  Add:

  ```toml
  benchmark-multidim = [
    "scipy==1.16.0",
  ]
  ```

  under `[dependency-groups]`. Create one isolated `pyproject.toml` and lock
  for each binary or accelerator comparator:

  ```toml
  # laboratory/quad-multidim-comparison/tasmanian/pyproject.toml
  [project]
  name = "jaxstro-quad-tasmanian-comparison"
  version = "0.0.0"
  requires-python = ">=3.11,<3.13"
  dependencies = ["tasmanian==8.2"]

  [tool.uv]
  package = false
  ```

  ```toml
  # laboratory/quad-multidim-comparison/torchquad/pyproject.toml
  [project]
  name = "jaxstro-quad-torchquad-comparison"
  version = "0.0.0"
  requires-python = ">=3.11,<3.13"
  dependencies = ["torchquad==0.5.0"]

  [tool.uv]
  package = false
  ```

  Generate both isolated locks:

  ```bash
  uv lock --python 3.11 --project \
    laboratory/quad-multidim-comparison/tasmanian
  uv lock --python 3.11 --project \
    laboratory/quad-multidim-comparison/torchquad
  ```

  Assert in the integration test that `scipy`, `tasmanian`, and `torchquad`
  are absent from `[project].dependencies`, and that the two isolated locks
  resolve without changing the root JAX pin.

- [ ] **Step 2: Implement record-specific adapters**

  Each adapter returns:

  ```python
  class ComparisonRecord(TypedDict):
      library: str
      version: str
      label: Literal[
          "exact", "strong-match", "node-matched",
          "family-matched", "capability"
      ]
      case_id: str
      controls: dict[str, object]
      value: object
      truth_error: float
      evaluations: int | None
      elapsed_seconds: float
  ```

  Use SciPy cubature only for family-matched Genz-Malik records, SciPy Sobol for
  exact deterministic/LMS records when configuration matches, Tasmanian for
  sparse node/moment/convergence records, and Torchquad for selected tensor,
  accelerator, and differentiable capability records. Reject a record missing
  a label or matched-control description.

- [ ] **Step 3: Freeze the baseline benchmark manifest**

  Predeclare dimensions 2, 4, 8, and 16; compile, scalar, VMAP-16, VMAP-128
  where feasible, JVP, gradient, same-domain repeats, and changing-parameter
  repeats. Record truth error, logical evaluations, unique nodes, regions or
  indices, replicates, Sobol level, compile time, warm time, dispersion,
  compiler-cost proxy, memory proxy, gradient error, and coverage.

  The runner must accept:

  ```bash
  python scripts/benchmark_quad_multidim.py --suite baseline --emit
  python scripts/benchmark_quad_multidim.py --suite baseline --check
  ```

  and reject a dirty worktree unless `--allow-dirty` is explicitly supplied
  for exploratory runs; evidence emission never allows dirty state.

  Before baseline emission, verify the harness against synthetic records and
  commit every code, test, root-lock, and isolated-environment owner:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/integration/test_quad_multidim_comparisons.py \
    tests/integration/test_quad_multidim_benchmark_contract.py
  git add pyproject.toml uv.lock \
    scripts/quad_multidim_benchmark_adapters.py \
    scripts/benchmark_quad_multidim.py \
    tests/integration/test_quad_multidim_comparisons.py \
    tests/integration/test_quad_multidim_benchmark_contract.py \
    laboratory/quad-multidim-comparison/tasmanian/pyproject.toml \
    laboratory/quad-multidim-comparison/tasmanian/uv.lock \
    laboratory/quad-multidim-comparison/torchquad/pyproject.toml \
    laboratory/quad-multidim-comparison/torchquad/uv.lock
  git commit -m "build(quad): add multidimensional comparison harness"
  ```

  Verify `git status --short` is empty. This code/environment checkpoint is
  required because immutable baseline emission rejects dirty worktrees.

- [ ] **Step 4: Emit the baseline and evaluate triggers before optimization**

  From the clean harness commit, run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/benchmark_quad_multidim.py --suite baseline --emit
  ```

  Run two warm repetitions of the immutable baseline. A runtime owner becomes
  eligible for optimization only if one predeclared case shows either:

  - warm runtime at least 1.50 times the matched comparator;
  - compiler-cost proxy at least 2.00 times the matched comparator;
  - memory proxy at least 2.00 times the family median; or
  - repeated-call scaling worse than linear by at least 25%.

  The emission command records all trigger values and the trigger/no-trigger
  disposition. Immediately freeze that immutable evidence before either branch:

  ```bash
  git add docs/validation/quad-multidim-comparisons.json \
    docs/validation/quad-multidim-performance-baseline.json
  git commit -m "perf(quad): freeze multidimensional comparison baseline"
  test -z "$(git status --short)"
  ```

  If no trigger fires, make no runtime optimization. If a trigger fires, write
  `docs/superpowers/specs/2026-07-17-quad-phase-b-optimization-addendum.md`
  naming the exact owner, case, metric, baseline IDs, proposed local change,
  regression gates, and stop condition before code changes. Preserve the
  baseline and require two independent optimized suites.
  If a trigger fires, commit the addendum and its narrowly scoped runtime
  change before emitting either optimized suite, so the same clean-source rule
  applies. If no trigger fires, write that disposition into the baseline
  artifact without changing runtime code.

- [ ] **Step 5: Run comparison contracts from immutable evidence**

  Verify from the clean baseline commit, or from the clean conditional
  optimization commit when a measured trigger was authorized:

  ```bash
  test -z "$(git status --short)"
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/integration/test_quad_multidim_comparisons.py \
    tests/integration/test_quad_multidim_benchmark_contract.py
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/benchmark_quad_multidim.py --suite baseline --check
  env -u VIRTUAL_ENV uv run --python 3.11 --project \
    laboratory/quad-multidim-comparison/tasmanian --locked python -c \
    "from importlib.metadata import version; print(version('Tasmanian'))"
  env -u VIRTUAL_ENV uv run --python 3.11 --project \
    laboratory/quad-multidim-comparison/torchquad --locked python -c \
    "from importlib.metadata import version; print(version('torchquad'))"
  ```

  Expected: all commands exit zero, every record has a calibrated label, and
  verification leaves the worktree clean.

### Task 6: Publish the researcher-facing MyST architecture and API

**Files:**
- Create: `docs/20-methods/approximation-integration/multidimensional/hyperrectangles.md`
- Create: `docs/20-methods/approximation-integration/multidimensional/tensor-product.md`
- Create: `docs/20-methods/approximation-integration/multidimensional/adaptive-cubature.md`
- Create: `docs/20-methods/approximation-integration/multidimensional/sparse-grids.md`
- Create: `docs/20-methods/approximation-integration/multidimensional/randomized-qmc.md`
- Create: `docs/20-methods/approximation-integration/multidimensional/differentiating.md`
- Create: `docs/20-methods/approximation-integration/multidimensional/choosing-a-method.md`
- Create: `docs/50-api/approximation-integration/quad-tensor-cubature.md`
- Create: `docs/50-api/approximation-integration/quad-sparse.md`
- Create: `docs/50-api/approximation-integration/quad-qmc.md`
- Modify: `docs/50-api/approximation-integration/quad.md`
- Create: `docs/60-validation/numerical/quadrature-multidimensional.md`
- Modify: `docs/60-validation/evidence-index.md`
- Modify: `docs/60-validation/validation.md`
- Modify: `docs/myst.yml`
- Modify: `docs/route-manifest.json`

**Interfaces:**
- Consumes: runtime signatures and generated B3/B4 evidence.
- Produces: method-family TOC, derivations, API references, choice guide, and
  auditable validation page.

- [ ] **Step 1: Add failing route and semantic tests**

  Extend existing MyST integration tests to require all eleven new routes,
  unique page IDs, API-family ownership, at least one `:::{warning}` or
  `:::{important}` on every method page, at least two MyST math directives
  per method page, no course/instructor language, and evidence links that point
  to generated JSON owners.

- [ ] **Step 2: Author the seven method pages**

  Every page must contain these exact second-level sections:

  ```markdown
  ## Scientific question
  ## Geometric picture
  ## Derivation
  ## Computational cost
  ## What the estimator means
  ## JAX and differentiation
  ## Quantities and units
  ## Worked astrophysical example
  ## Failure modes
  ## Audit recipe
  ## Warranted claim
  ```

  Use MyST `note`, `important`, `warning`, and `tip` directives according to
  assumption, statistical, capacity, and method-choice meaning. Derive tensor
  cost, Genz-Malik embedded evidence, Smolyak hierarchical sums, Sobol
  construction, Student-t fixed-look width, empirical-Bernstein alpha spending,
  and accepted-formula replay in LaTeX.

- [ ] **Step 3: Author grouped API and validation pages**

  The API landing page teaches `quad.integrate` first and links separate
  tensor/cubature, sparse, and QMC pages. Each API page records exact
  signatures, static/dynamic arguments, error kind, statuses, work semantics,
  supported dimensions/payloads, replay boundary, and quantity boundary.

  The validation page renders tables from the truth, replay, RQMC,
  comparison, and benchmark artifacts and states estimator limitations beside
  every table.

- [ ] **Step 4: Update MyST navigation and routes**

  Under `Approximation from finite information`, insert:

  ```yaml
  - title: Multidimensional integration
    children:
      - file: 20-methods/approximation-integration/multidimensional/hyperrectangles.md
      - file: 20-methods/approximation-integration/multidimensional/tensor-product.md
      - file: 20-methods/approximation-integration/multidimensional/adaptive-cubature.md
      - file: 20-methods/approximation-integration/multidimensional/sparse-grids.md
      - file: 20-methods/approximation-integration/multidimensional/randomized-qmc.md
      - file: 20-methods/approximation-integration/multidimensional/differentiating.md
      - file: 20-methods/approximation-integration/multidimensional/choosing-a-method.md
  ```

  Add the three family API pages under `Approximation and integration` and the
  validation page under `Validation and evidence`.

- [ ] **Step 5: Run strict docs gates and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/integration/test_myst_semantic_grammar.py \
    tests/integration/test_grouped_api_reference.py \
    tests/integration/test_method_page_contract.py \
    tests/integration/test_docs_gate_lifecycle.py \
    tests/integration/test_docs_gate_wiring.py
  bash scripts/check_docs.sh
  git diff --check
  ```

  Expected: strict build, crawl, link/ID/alt-text, semantic, and accessibility
  checks all pass. Commit all listed documentation, route, and test files:

  ```bash
  git add docs/20-methods/approximation-integration/multidimensional \
    docs/50-api/approximation-integration/quad*.md \
    docs/60-validation/numerical/quadrature-multidimensional.md \
    docs/60-validation/evidence-index.md docs/60-validation/validation.md \
    docs/myst.yml docs/route-manifest.json \
    tests/integration/test_myst_semantic_grammar.py \
    tests/integration/test_grouped_api_reference.py \
    tests/integration/test_method_page_contract.py \
    tests/integration/test_docs_gate_lifecycle.py \
    tests/integration/test_docs_gate_wiring.py
  git commit -m "docs(quad): teach multidimensional integration"
  ```

### Task 7: Run the complete Phase B release and review gate

**Files:**
- Modify: `src/jaxstro/quad/_contracts.py`
- Modify: `docs/validation/contracts.json`
- Modify: `docs/50-api/research-infrastructure/contracts.md`
- Modify: `docs/70-project/development/future-capabilities-roadmap.md`
- Modify: `docs/70-project/development/numerical-methods-roadmap.md`
- Modify: `docs/70-project/development/sota-assessment.md`
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: every B0 through B4 deliverable.
- Produces: fresh generated contracts, honest roadmap/SOTA state, complete
  release evidence, and the next approved capability boundary.

- [ ] **Step 1: Update contract declarations and generated inventories**

  Declare exact public methods, dimensions, payloads, error kinds, replay modes,
  quantity status, and randomized confidence restrictions in `_contracts.py`.
  Regenerate `contracts.json` and the public contracts page using the existing
  emit/check owner; never hand-edit generated content.

- [ ] **Step 2: Run the exhaustive local gate**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync ruff check src tests scripts laboratory
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check src tests scripts laboratory
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
  env -u VIRTUAL_ENV uv run --no-sync python scripts/build_sobol_directions.py --check
  env -u VIRTUAL_ENV uv run --no-sync python scripts/generate_quad_rqmc_evidence.py --check
  env -u VIRTUAL_ENV uv run --no-sync python scripts/generate_quad_multidim_evidence.py --check
  env -u VIRTUAL_ENV uv run --no-sync pytest -q
  env -u VIRTUAL_ENV uv sync --locked --extra dev --extra ml
  env -u VIRTUAL_ENV uv run --no-sync --extra ml pytest tests/integration -q
  bash scripts/check_docs.sh
  bash scripts/check.sh
  git diff --check
  ```

  Expected: every command exits zero. Record exact pass/skip counts, generated
  artifact hashes, clean-wheel import result, and documentation route count.

- [ ] **Step 3: Run independent checkpoint reviews**

  Request separate read-only reviews for:

  - deterministic numerical correctness;
  - sparse-grid correctness;
  - QMC/statistical calibration;
  - JAX/replay semantics;
  - quantities and representation invariance;
  - API/backward compatibility;
  - performance/comparison calibration;
  - MyST pedagogy, accessibility, and claim boundaries.

  Resolve every Critical or Important finding and rerun the affected focused
  gate plus the exhaustive gate. Record review filenames and dispositions in
  `docs/superpowers/reviews/`.

- [ ] **Step 4: Update roadmaps, SOTA assessment, and status**

  Mark only evidenced Phase B capabilities complete. State that:

  - hyperrectangles are the only Phase B geometry;
  - confidence intervals are real-scalar only;
  - quantity mode remains alpha/opt-in;
  - no universal superiority claim is warranted;
  - sibling migration remains paused;
  - Phase C geometries and scientific specializations remain separate work.

  `STATUS.md` must include exact commits, gates, review verdicts, artifacts,
  measured triggers/optimization outcome, and one `next:` line.

- [ ] **Step 5: Commit the release checkpoint**

  ```bash
  git add src/jaxstro/quad/_contracts.py docs/validation/contracts.json \
    docs/50-api/research-infrastructure/contracts.md \
    docs/70-project/development/future-capabilities-roadmap.md \
    docs/70-project/development/numerical-methods-roadmap.md \
    docs/70-project/development/sota-assessment.md \
    docs/superpowers/reviews STATUS.md
  git commit -m "chore(quad): complete Phase B release gate"
  ```

  Verify `git status --short` is empty. Do not push, publish, deploy, or migrate
  a sibling package without a separate explicit request.
