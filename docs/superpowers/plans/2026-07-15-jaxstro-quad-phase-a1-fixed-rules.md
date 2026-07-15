# Jaxstro Quad Phase A1 Fixed Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver canonical sampled-data integration and a complete, JAX-native one-dimensional fixed-quadrature family behind `jaxstro.quad` while preserving every legacy callable and result exactly.

**Architecture:** Keep sampled-data formulas, orthogonal-polynomial recurrences, rule construction, domain transformations, and fixed evaluation in separate modules. Static frozen rule and measure objects select a construction eagerly; numerical bounds and integrand parameters remain dynamic JAX values. Legacy modules become re-export lanes pointing at the canonical implementations, except for the approved byte-identical probabilists' Hermite compatibility backend.

**Tech Stack:** Python 3.11+, JAX, `jax.numpy`, jaxtyping, pytest, Ruff, MyPy, MyST. NumPy is allowed only in the narrow legacy Hermite compatibility backend. SciPy and high-precision tools may be used only for validation fixtures and never at runtime.

## Global Constraints

- Follow `CLAUDE.md`, `AGENTS.md`, and the approved capability design in `docs/superpowers/specs/2026-07-15-jaxstro-quad-capability-program-design.md`.
- Use test-driven development: every behavioral change begins with a failing test.
- Preserve exact callable identity from `jaxstro.numerics.integration` and `jaxstro.numerics.quadrature` to the canonical `jaxstro.quad` owner.
- Preserve the existing `gauss_hermite_nodes` output byte-for-byte; do not replace its NumPy `hermgauss` compatibility backend.
- Add no runtime dependency.
- Use raw arrays only. Quantity integration remains Phase A3 work.
- Do not implement adaptive controllers, replay derivatives, sibling migrations, warnings, deprecations, publication, or live-site changes.
- Rule order, level, rule type, measure type, breakpoint count, payload shape, and reduction axis are static under JIT.
- Bounds, integrand arguments, and breakpoint values may be dynamic JAX values.
- Every fixed rule returns an array value, not `QuadResult`; exactness metadata is not an error estimate.
- Authored prose uses ASCII punctuation and LaTeX for mathematical notation.

---

## File structure

- `src/jaxstro/quad/sampled.py`: canonical sampled-data formulas.
- `src/jaxstro/quad/rules.py`: frozen public rule configuration objects and rule metadata.
- `src/jaxstro/quad/_recurrence.py`: classical recurrence coefficients and the shared Jacobi-matrix eigensolver.
- `src/jaxstro/quad/_chebyshev.py`: shared Clenshaw-Curtis and Fejer node/weight construction.
- `src/jaxstro/quad/_tanh_sinh.py`: fixed tanh-sinh reference formulas and domain maps.
- `src/jaxstro/quad/fixed.py`: measure dispatch, domain segmentation, vectorized integrand evaluation, and weighted reduction.
- `src/jaxstro/numerics/integration.py`: compatibility re-exports only after canonical sampled behavior is proven identical.
- `src/jaxstro/numerics/quadrature.py`: legacy helper re-exports plus Hermite polynomial utilities and the explicit compatibility exception.
- `tests/unit/quad/`: construction, exactness, transformation, evaluation, and failure contracts.
- `tests/integration/test_quad_fixed_transforms.py`: JIT, VMAP, gradients, payloads, and compatibility identity.
- `tests/validation/test_quad_fixed_reference.py`: independent analytic and SciPy reference comparisons.
- `docs/20-methods/approximation-integration/quadrature.md`: fixed and weighted method guide with derivations.
- `docs/50-api/approximation-integration/quad.md`: current public inventory and contracts.

### Task 1: Canonicalize sampled-data integration

**Files:**
- Create: `src/jaxstro/quad/sampled.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Modify: `src/jaxstro/numerics/integration.py`
- Test: `tests/unit/quad/test_sampled.py`
- Test: `tests/integration/test_quad_compatibility.py`

**Interfaces:**
- Produces: `trapezoid(y, x=None, *, dx=1.0, axis=-1)`, `cumulative_trapezoid(y, x=None, *, dx=1.0, axis=-1)`, `simpson(y, x=None, *, dx=1.0, axis=-1)`, and `cumulative_simpson(y, x=None, *, dx=1.0, axis=-1)`.
- Preserves: `integration.trapz is quad.trapezoid`, `integration.cumulative_trapz is quad.cumulative_trapezoid`, and exact existing value, dtype, shape, exception, and floating-point ordering behavior.

- [x] **Step 1: Write failing canonical-ownership and `dx` tests**

```python
def test_sampled_ownership_is_inverted() -> None:
    assert quad.trapezoid.__module__ == "jaxstro.quad.sampled"
    assert integration.trapz is quad.trapezoid

def test_trapezoid_uniform_dx() -> None:
    y = jnp.asarray([1.0, 2.0, 4.0])
    assert jnp.array_equal(quad.trapezoid(y, dx=0.25), 1.125)

def test_simpson_uniform_dx() -> None:
    y = jnp.asarray([0.0, 1.0, 4.0])
    assert jnp.array_equal(quad.simpson(y, dx=0.5), 4.0 / 3.0)
```

- [x] **Step 2: Run the tests and verify RED**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_sampled.py tests/integration/test_quad_compatibility.py`

Expected: failures because canonical functions still live in `jaxstro.numerics.integration` and two total functions do not accept `dx`.

- [x] **Step 3: Move the existing implementations without changing arithmetic ordering**

```python
@partial(jax.jit, static_argnames="axis")
def trapezoid(y, x=None, *, dx=1.0, axis=-1):
    y = jnp.asarray(y)
    if x is None:
        left = jnp.take(y, jnp.arange(y.shape[axis] - 1), axis=axis)
        right = jnp.take(y, jnp.arange(1, y.shape[axis]), axis=axis)
        return 0.5 * jnp.sum(left + right, axis=axis) * dx
    return _trapezoid_nonuniform(y, jnp.asarray(x), axis)

trapz = trapezoid
```

Move the current cumulative and Simpson cores verbatim, add the approved `dx` keyword to the total rules, and make the legacy module import and re-export these objects.

- [x] **Step 4: Run sampled, parity, gradient, and sampling consumers**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_sampled.py tests/integration/test_quad_compatibility.py tests/integration/test_integration_parity.py tests/unit/test_numerics.py tests/unit/test_sampling.py tests/validation/test_grad_checks.py -k 'trapz or trapezoid or simpson or sampling'`

Expected: all selected tests pass and compatibility identities point toward `jaxstro.quad.sampled`.

- [x] **Step 5: Commit**

```bash
git add src/jaxstro/quad/sampled.py src/jaxstro/quad/__init__.py src/jaxstro/numerics/integration.py tests/unit/quad/test_sampled.py tests/integration/test_quad_compatibility.py
git commit -m "refactor(quad): make sampled integration canonical"
```

### Task 2: Add static rule configurations and metadata

**Files:**
- Create: `src/jaxstro/quad/rules.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Test: `tests/unit/quad/test_rules.py`

**Interfaces:**
- Produces: `GaussianRule(order)`, `ClenshawCurtisRule(order)`, `FejerIRule(order)`, `FejerIIRule(order)`, `TanhSinhRule(level)`.
- Produces internal `FixedRuleData(nodes, weights, degree, nested)` with array nodes and weights and static metadata.

- [x] **Step 1: Write failing validation and PyTree tests**

```python
@pytest.mark.parametrize("factory", [GaussianRule, ClenshawCurtisRule, FejerIRule, FejerIIRule])
def test_rule_order_is_positive_static_metadata(factory) -> None:
    rule = factory(5)
    leaves, treedef = jax.tree.flatten(rule)
    assert leaves == []
    assert jax.tree.unflatten(treedef, leaves) == rule
    with pytest.raises(ValueError, match="positive"):
        factory(0)

def test_tanh_sinh_level_is_nonnegative() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        TanhSinhRule(-1)
```

- [x] **Step 2: Verify RED**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_rules.py`

Expected: import failures for undefined rule types.

- [x] **Step 3: Implement frozen static configurations**

```python
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GaussianRule:
    order: int
    def __post_init__(self):
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 1:
            raise ValueError("GaussianRule order must be a positive integer")
    def tree_flatten(self):
        return (), self.order
    @classmethod
    def tree_unflatten(cls, order, _children):
        return cls(order)
```

Use the same explicit validation pattern for the four remaining configurations. Define `FixedRuleData` as a `NamedTuple` with `nodes`, `weights`, `degree`, and `nested`.

- [x] **Step 4: Run tests, Ruff, and MyPy**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_rules.py && env -u VIRTUAL_ENV uv run --no-sync ruff check src/jaxstro/quad/rules.py tests/unit/quad/test_rules.py && env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro/quad/rules.py`

Expected: all checks pass.

- [x] **Step 5: Commit**

```bash
git add src/jaxstro/quad/rules.py src/jaxstro/quad/__init__.py tests/unit/quad/test_rules.py
git commit -m "feat(quad): add fixed rule configurations"
```

### Task 3: Build the shared Gaussian recurrence engine

**Files:**
- Create: `src/jaxstro/quad/_recurrence.py`
- Test: `tests/unit/quad/test_recurrence.py`
- Test: `tests/validation/test_quad_fixed_reference.py`

**Interfaces:**
- Produces: `gaussian_rule_data(rule: GaussianRule, measure) -> FixedRuleData`.
- Consumes the six Phase A classical measure declarations and returns nodes in their natural reference support.

- [x] **Step 1: Write failing moment, exactness, and normalization tests**

```python
@pytest.mark.parametrize(
    ("measure", "moment0"),
    [
        (LebesgueMeasure(), 2.0),
        (JacobiMeasure(0.25, 0.5), 2.0 ** 1.75 * scipy.special.beta(1.25, 1.5)),
        (LaguerreMeasure(0.5), scipy.special.gamma(1.5)),
        (PhysicistsHermiteMeasure(), math.sqrt(math.pi)),
        (StandardNormalMeasure(), 1.0),
    ],
)
def test_gaussian_weights_reproduce_measure_mass(measure, moment0) -> None:
    data = gaussian_rule_data(GaussianRule(8), measure)
    assert jnp.allclose(jnp.sum(data.weights), moment0, rtol=2e-12, atol=2e-12)

def test_legendre_exact_through_degree_2n_minus_1() -> None:
    data = gaussian_rule_data(GaussianRule(6), LebesgueMeasure())
    for degree in range(12):
        expected = 0.0 if degree % 2 else 2.0 / (degree + 1)
        assert jnp.allclose(jnp.sum(data.weights * data.nodes**degree), expected, atol=2e-12)
```

- [x] **Step 2: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_recurrence.py tests/validation/test_quad_fixed_reference.py`

Expected: imports fail because `_recurrence.py` does not exist.

- [x] **Step 3: Implement recurrence coefficients and one Jacobi-matrix solver**

```python
def _golub_welsch(diagonal, off_diagonal, mass) -> FixedRuleData:
    matrix = jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
    nodes, vectors = jnp.linalg.eigh(matrix)
    weights = mass * vectors[0, :] ** 2
    return FixedRuleData(nodes, weights, 2 * diagonal.shape[0] - 1, False)

def _laguerre(order: int, alpha: float):
    k = jnp.arange(order, dtype=jnp.float64)
    diagonal = 2.0 * k + alpha + 1.0
    j = jnp.arange(1, order, dtype=jnp.float64)
    off_diagonal = jnp.sqrt(j * (j + alpha))
    return diagonal, off_diagonal, jax.scipy.special.gamma(alpha + 1.0)

def _standard_normal(order: int):
    diagonal = jnp.zeros(order, dtype=jnp.float64)
    off_diagonal = jnp.sqrt(jnp.arange(1, order, dtype=jnp.float64))
    return diagonal, off_diagonal, jnp.asarray(1.0)
```

Implement the standard orthonormal Legendre, Jacobi, generalized Laguerre, physicists' Hermite, and standard-normal recurrences. Handle the removable Jacobi diagonal singularity at `alpha + beta == 0` analytically rather than with a numerical epsilon. Apply `normalized=True` by dividing weights by the analytic measure mass.

- [x] **Step 4: Validate every family independently**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_recurrence.py tests/validation/test_quad_fixed_reference.py`

Expected: all analytic moment and independent SciPy root/weight comparisons pass at the predeclared float64 tolerances.

- [x] **Step 5: Commit**

```bash
git add src/jaxstro/quad/_recurrence.py tests/unit/quad/test_recurrence.py tests/validation/test_quad_fixed_reference.py
git commit -m "feat(quad): add shared Gaussian recurrence engine"
```

### Task 4: Build the shared Chebyshev fixed-rule substrate

**Files:**
- Create: `src/jaxstro/quad/_chebyshev.py`
- Modify: `src/jaxstro/numerics/quadrature.py`
- Test: `tests/unit/quad/test_chebyshev_rules.py`
- Test: `tests/unit/test_quadrature.py`

**Interfaces:**
- Produces: `chebyshev_rule_data(rule) -> FixedRuleData` for Clenshaw-Curtis, Fejer I, and Fejer II.
- Preserves: `numerics.quadrature.clenshaw_curtis_nodes is quad.clenshaw_curtis_nodes` after ownership inversion.

- [x] **Step 1: Write failing node, weight, nesting, and exactness tests**

```python
@pytest.mark.parametrize("rule", [ClenshawCurtisRule(9), FejerIRule(8), FejerIIRule(8)])
def test_chebyshev_rules_integrate_supported_polynomials(rule) -> None:
    data = chebyshev_rule_data(rule)
    assert jnp.all(data.weights > 0.0)
    assert jnp.allclose(jnp.sum(data.weights), 2.0, atol=2e-13)
    for degree in range(data.degree + 1):
        expected = 0.0 if degree % 2 else 2.0 / (degree + 1)
        assert jnp.allclose(jnp.sum(data.weights * data.nodes**degree), expected, atol=2e-11)

def test_clenshaw_curtis_legacy_helper_is_canonical_identity() -> None:
    assert numerics_quadrature.clenshaw_curtis_nodes is quad.clenshaw_curtis_nodes
```

- [x] **Step 2: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_chebyshev_rules.py tests/unit/test_quadrature.py -k 'clenshaw or fejer'`

Expected: Fejer imports fail and canonical ownership assertion fails.

- [x] **Step 3: Implement one cosine-moment construction substrate**

```python
def _interpolatory_weights(nodes: Array) -> Array:
    order = nodes.shape[0]
    powers = jnp.arange(order)[:, None]
    vandermonde = nodes[None, :] ** powers
    moments = jnp.where(jnp.arange(order) % 2 == 0, 2.0 / (jnp.arange(order) + 1), 0.0)
    return jnp.linalg.solve(vandermonde, moments)

def chebyshev_rule_data(rule):
    if isinstance(rule, ClenshawCurtisRule):
        nodes = jnp.cos(jnp.pi * jnp.arange(rule.order) / max(rule.order - 1, 1))
    elif isinstance(rule, FejerIRule):
        nodes = jnp.cos(jnp.pi * (2 * jnp.arange(rule.order) + 1) / (2 * rule.order))
    else:
        nodes = jnp.cos(jnp.pi * (jnp.arange(rule.order) + 1) / (rule.order + 1))
    return FixedRuleData(nodes, _interpolatory_weights(nodes), rule.order - 1, isinstance(rule, ClenshawCurtisRule))
```

Replace the direct dense solve with the shared stable cosine-series formulas before accepting orders above 32; retain explicit order-one behavior. Make the legacy helper a re-export of the canonical constructor.

- [x] **Step 4: Run exactness and legacy regression suites**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_chebyshev_rules.py tests/unit/test_quadrature.py tests/integration/test_quad_compatibility.py`

Expected: all tests pass without changing the existing helper's values or ordering.

- [x] **Step 5: Commit**

```bash
git add src/jaxstro/quad/_chebyshev.py src/jaxstro/numerics/quadrature.py tests/unit/quad/test_chebyshev_rules.py tests/unit/test_quadrature.py tests/integration/test_quad_compatibility.py
git commit -m "feat(quad): add Chebyshev fixed rule families"
```

### Task 5: Add fixed tanh-sinh formulas and infinite-domain maps

**Files:**
- Create: `src/jaxstro/quad/_tanh_sinh.py`
- Modify: `src/jaxstro/quad/transforms.py`
- Test: `tests/unit/quad/test_tanh_sinh.py`

**Interfaces:**
- Produces: `tanh_sinh_rule_data(TanhSinhRule) -> FixedRuleData` on `(-1, 1)`.
- Produces: `map_domain(domain, reference) -> DomainMapResult(x, jacobian, orientation, valid)` for all Phase A domain types.

- [x] **Step 1: Write failing transform identity and endpoint-singularity tests**

```python
@pytest.mark.parametrize(
    ("domain", "fun", "expected"),
    [
        (Interval(-1.0, 1.0), lambda x: 1.0 / jnp.sqrt(1.0 - x * x), jnp.pi),
        (RightInfinite(0.0), lambda x: jnp.exp(-x), 1.0),
        (LeftInfinite(0.0), lambda x: jnp.exp(x), 1.0),
        (Infinite(), lambda x: jnp.exp(-x * x), jnp.sqrt(jnp.pi)),
    ],
)
def test_tanh_sinh_formula_and_domain_maps(domain, fun, expected) -> None:
    data = tanh_sinh_rule_data(TanhSinhRule(7))
    mapped = map_domain(domain, data.nodes)
    got = mapped.orientation * jnp.sum(data.weights * mapped.jacobian * fun(mapped.x))
    # The algebraic endpoint singularity reaches the float64 nextafter limit.
    tolerance = 2e-7 if isinstance(domain, Interval) else 2e-9
    assert jnp.allclose(got, expected, rtol=tolerance, atol=tolerance)
```

- [x] **Step 2: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_tanh_sinh.py`

Expected: imports for tanh-sinh construction and general maps fail.

- [x] **Step 3: Implement fixed double-exponential nodes and maps**

```python
def tanh_sinh_rule_data(rule: TanhSinhRule) -> FixedRuleData:
    step = 2.0 ** (-rule.level)
    extent = max(4, 4 * 2**rule.level)
    t = step * jnp.arange(-extent, extent + 1)
    u = 0.5 * jnp.pi * jnp.sinh(t)
    nodes = jnp.tanh(u)
    weights = step * 0.5 * jnp.pi * jnp.cosh(t) / jnp.cosh(u) ** 2
    return FixedRuleData(nodes, weights, -1, True)
```

Use rational maps from `(-1, 1)` to each semi-infinite domain and `x=t/(1-t**2)` for the full line. Compute Jacobians analytically, preserve orientation separately, and mask saturated endpoint nodes before evaluating the integrand.

- [x] **Step 4: Run finite, infinite, reversed, and JIT tests**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_tanh_sinh.py tests/unit/quad/test_domains.py`

Expected: all tests pass with finite mapped nodes and weights at supported levels.

- [x] **Step 5: Commit**

```bash
git add src/jaxstro/quad/_tanh_sinh.py src/jaxstro/quad/transforms.py tests/unit/quad/test_tanh_sinh.py
git commit -m "feat(quad): add fixed tanh-sinh domain formulas"
```

### Task 6: Implement the public fixed evaluator

**Files:**
- Create: `src/jaxstro/quad/fixed.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Test: `tests/unit/quad/test_fixed.py`
- Test: `tests/integration/test_quad_fixed_transforms.py`

**Interfaces:**
- Produces: `fixed(fun, domain, *, args=(), rule, measure=None)`.
- Consumes: rule data, domain maps, measure declarations, and breakpoints.

- [x] **Step 1: Write failing scalar, payload, weighted, breakpoint, orientation, and zero-width tests**

```python
def test_fixed_vector_payload_and_args() -> None:
    def fun(x, args):
        return jnp.stack((args * x, x * x), axis=-1)
    got = fixed(fun, Interval(-1.0, 1.0), args=3.0, rule=GaussianRule(8))
    assert jnp.allclose(got, jnp.asarray([0.0, 2.0 / 3.0]))

def test_fixed_breakpoints_sum_segments() -> None:
    domain = Interval(0.0, 1.0, breakpoints=(0.25, 0.75))
    assert jnp.allclose(fixed(lambda x: x**3, domain, rule=GaussianRule(2)), 0.25)

def test_weighted_measure_is_applied_exactly_once() -> None:
    measure = WeightedMeasure(
        lambda x, args: args * x,
        density_unit=quantity.dimensionless,
    )
    got = fixed(lambda x: x, Interval(0.0, 1.0), args=2.0, rule=ClenshawCurtisRule(9), measure=measure)
    assert jnp.allclose(got, 2.0 / 3.0)
```

- [x] **Step 2: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_fixed.py tests/integration/test_quad_fixed_transforms.py`

Expected: import failure for `fixed`.

- [x] **Step 3: Implement dispatch and one vectorized reduction path**

```python
def _weighted_sum(values, weights):
    return jnp.tensordot(weights, values, axes=((0,), (0,)))

def fixed(fun, domain, *, args=(), rule, measure=None):
    measure = LebesgueMeasure() if measure is None else measure
    data = _rule_data(rule, measure)
    segments = _domain_segments(domain)
    def evaluate(segment):
        mapped = map_domain(segment, data.nodes)
        values = fun(mapped.x, args)
        density = _measure_density(measure, mapped.x, args)
        weights = data.weights * mapped.jacobian * density
        return mapped.orientation * _weighted_sum(values, weights)
    return jnp.sum(jax.vmap(evaluate)(segments), axis=0)
```

Support both `fun(x)` and `fun(x, args)` without inspecting traced values. Require `fun(x, args)` whenever `args != ()`. Validate structural rule/domain/measure incompatibilities eagerly. For traced invalid bounds, return `nan` rather than raising from compiled code. Infer zero-width payload shape through `jax.eval_shape` and return exact zeros without numerical evaluation.

- [x] **Step 4: Run transformation, payload, gradient, and compatibility gates**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad tests/integration/test_quad_compatibility.py tests/integration/test_quad_fixed_transforms.py tests/validation/test_quad_fixed_reference.py`

Expected: all Phase A1 tests pass; `jax.jit`, `jax.vmap`, and gradients with respect to explicit integrand parameters and finite bounds pass.

- [x] **Step 5: Commit**

```bash
git add src/jaxstro/quad/fixed.py src/jaxstro/quad/__init__.py tests/unit/quad/test_fixed.py tests/integration/test_quad_fixed_transforms.py
git commit -m "feat(quad): add fixed quadrature evaluator"
```

### Task 7: Invert legacy fixed-helper ownership without breaking compatibility

**Files:**
- Modify: `src/jaxstro/quad/__init__.py`
- Modify: `src/jaxstro/numerics/quadrature.py`
- Modify: `src/jaxstro/numerics/__init__.py`
- Test: `tests/integration/test_quad_compatibility.py`
- Test: `tests/unit/test_quadrature.py`

**Interfaces:**
- Preserves all six legacy public helper objects and top-level `jaxstro.numerics` exports.
- Makes new implementations canonical except `gauss_hermite_nodes`, which remains the explicit compatibility backend and is consumed by `jaxstro.quad`.

- [ ] **Step 1: Write failing owner and byte-identity ratchets**

```python
def test_new_node_helpers_are_owned_by_quad() -> None:
    assert quad.gauss_legendre_nodes.__module__.startswith("jaxstro.quad")
    assert numerics.gauss_legendre_nodes is quad.gauss_legendre_nodes

def test_probabilists_hermite_compatibility_exception_is_exact() -> None:
    old_nodes, old_weights = np.polynomial.hermite.hermgauss(16)
    got_nodes, got_weights = quad.gauss_hermite_nodes(16)
    np.testing.assert_array_equal(np.asarray(got_nodes), np.sqrt(2.0) * old_nodes)
    np.testing.assert_array_equal(np.asarray(got_weights), old_weights / np.sqrt(np.pi))
```

- [ ] **Step 2: Verify RED**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_quad_compatibility.py tests/unit/test_quadrature.py`

Expected: ownership assertions fail while value regressions remain green.

- [ ] **Step 3: Replace legacy definitions with exact aliases**

```python
from jaxstro.quad._chebyshev import clenshaw_curtis_nodes
from jaxstro.quad._recurrence import gauss_laguerre_nodes, gauss_legendre_nodes

# Keep gauss_hermite_nodes defined here until the declared breaking release.
```

Keep `hermite_e_basis` and `hermite_coefficients` in the legacy module until a separately approved ownership decision. Do not create wrappers; use direct imports so identity tests are meaningful.

- [ ] **Step 4: Run complete compatibility and numerical suites**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_quad_compatibility.py tests/unit/test_quadrature.py tests/unit/quad`

Expected: all tests pass with no numerical regression.

- [ ] **Step 5: Commit**

```bash
git add src/jaxstro/quad/__init__.py src/jaxstro/numerics/quadrature.py src/jaxstro/numerics/__init__.py tests/integration/test_quad_compatibility.py tests/unit/test_quadrature.py
git commit -m "refactor(quad): invert fixed helper ownership"
```

### Task 8: Document and register the complete A1 surface

**Files:**
- Modify: `src/jaxstro/contracts/registry.py`
- Modify: `docs/20-methods/approximation-integration/quadrature.md`
- Modify: `docs/50-api/approximation-integration/quad.md`
- Modify: `docs/50-api/approximation-integration/quadrature.md`
- Modify: `docs/70-project/development/future-capabilities-roadmap.md`
- Modify: `docs/70-project/development/numerical-methods-roadmap.md`
- Modify: `docs/70-project/development/sota-assessment.md`
- Modify: `docs/70-project/development/package-assessment-scorecard.md`
- Modify: `docs/route-manifest.json`
- Modify: `docs/validation/contracts.json`
- Test: `tests/integration/test_method_page_contract.py`
- Test: `tests/integration/test_grouped_api_reference.py`
- Test: `tests/unit/test_contract_manifests.py`

**Interfaces:**
- Makes fixed and weighted quadrature current, with adaptive methods still visibly planned.
- Records exact public names, static arguments, supported rule/domain/measure combinations, cost, exactness, derivative boundaries, and validation provenance.

- [ ] **Step 1: Write failing documentation and registry ratchets**

```python
def test_fixed_quadrature_page_claims_complete_a1_surface() -> None:
    text = _page("approximation-integration/quadrature.md")
    for required in (
        "GaussianRule", "ClenshawCurtisRule", "FejerIRule", "FejerIIRule",
        "TanhSinhRule", "quad.fixed", "2n-1", "Gauss-Jacobi",
    ):
        assert required in text
    assert "delegated to Quadax" not in text

def test_quad_contract_lists_fixed() -> None:
    entry = CONTRACTS["jaxstro.quad"]
    assert "fixed" in entry.callables
```

- [ ] **Step 2: Verify RED**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_method_page_contract.py tests/integration/test_grouped_api_reference.py tests/unit/test_contract_manifests.py`

Expected: the current docs and generated contract omit A1 methods.

- [ ] **Step 3: Write researcher-first documentation and regenerate inventories**

The method page must contain labeled LaTeX derivations for Gaussian exactness, affine mapping, weighted integration, the Chebyshev interpolant, and the double-exponential map. Use MyST `note`, `important`, `warning`, and `tip` admonitions only where each semantic role applies. Include executable raw-array examples and a support matrix. Mark A2 adaptive methods as planned rather than delegated.

Run: `env -u VIRTUAL_ENV uv run --no-sync python scripts/build_evidence_index.py --emit && env -u VIRTUAL_ENV uv run --no-sync python scripts/build_contract_registry.py --emit`

- [ ] **Step 4: Run documentation contracts and strict build**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_method_page_contract.py tests/integration/test_grouped_api_reference.py tests/unit/test_contract_manifests.py`

Run: `DOCS_APP_PORT=4381 DOCS_SERVER_PORT=4382 bash scripts/check_docs.sh`

Expected: contract tests pass and the strict MyST route, link, accessibility, and final-artifact gates pass without changing an existing route.

- [ ] **Step 5: Commit**

```bash
git add src/jaxstro/contracts/registry.py docs tests/integration/test_method_page_contract.py tests/integration/test_grouped_api_reference.py tests/unit/test_contract_manifests.py
git commit -m "docs(quad): publish fixed quadrature contracts"
```

### Task 9: Phase A1 verification and checkpoint

**Files:**
- Modify: `STATUS.md`

**Interfaces:**
- Produces a verified A1 state suitable for a fresh A2 plan.

- [ ] **Step 1: Run the focused A1 gate**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad tests/unit/test_quadrature.py tests/integration/test_quad_compatibility.py tests/integration/test_quad_fixed_transforms.py tests/validation/test_quad_fixed_reference.py tests/integration/test_method_page_contract.py tests/integration/test_grouped_api_reference.py`

Expected: all focused tests pass.

- [ ] **Step 2: Run static checks and generated-artifact freshness**

Run: `env -u VIRTUAL_ENV uv run --no-sync ruff check src tests`

Run: `env -u VIRTUAL_ENV uv run --no-sync ruff format --check src tests`

Run: `env -u VIRTUAL_ENV uv run --no-sync mypy src`

Run: `env -u VIRTUAL_ENV uv run --no-sync python scripts/build_contract_registry.py --check`

Expected: all checks pass.

- [ ] **Step 3: Dispatch a fresh read-only A1 checkpoint reviewer**

The reviewer must inspect numerical correctness, compatibility identity, JAX transform contracts, support matrices, tests, and documentation claims. Resolve every Critical and Important finding with a failing regression before correction. Record Minor findings explicitly.

- [ ] **Step 4: Run the full repository and strict documentation gates**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q`

Run: `DOCS_APP_PORT=4381 DOCS_SERVER_PORT=4382 bash scripts/check_docs.sh`

Expected: the full suite and strict documentation gate pass.

- [ ] **Step 5: Update status and commit verification evidence**

Record the exact commits, test counts, checkpoint disposition, route count, supported methods, and explicit exclusions in `STATUS.md`.

```bash
git add STATUS.md
git commit -m "docs(quad): record Phase A1 verification"
```

## Stop conditions

Stop without beginning A2 if:

- any legacy callable loses exact object identity;
- any sampled-data result changes outside the approved `dx` extension;
- the probabilists' Hermite compatibility output changes at the byte level;
- a fixed method fails its declared analytic exactness or independent reference gate;
- a Python loop advances numerical nodes, segments, levels, or evaluation batches under JIT;
- quantities, adaptive control, sibling migration, deprecation, or publication enters the diff;
- an existing stable documentation route changes; or
- any Critical or Important checkpoint finding remains unresolved.

## Phase A2 handoff

Only after Task 9 passes, write a fresh Phase A2 plan against the verified A1 interfaces. A2 owns fixed-capacity one-dimensional adaptive control, Gauss-Kronrod pairs 15 through 61, nested adaptive Clenshaw-Curtis, adaptive tanh-sinh, Romberg, Romberg-tanh-sinh, deterministic status precedence, error/work evidence, and primal-only execution. Replay derivatives, quantities, comparisons, sibling migrations, deprecations, and publication remain outside A2.
