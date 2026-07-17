# Jaxstro Quad Phase B3 Sobol and Randomized QMC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an independently owned Sobol engine through 21,201
dimensions, deterministic integration, LMS-plus-shift and nested Owen
randomization, fixed-look Student-t intervals, bounded sequential empirical
Bernstein intervals, and predeclared calibration evidence.

**Architecture:** Vendored Joe-Kuo source data are parsed by a deterministic
generator into a checked runtime table; `_sobol.py` owns direction numbers and
power-of-two prefixes. `_scramble.py` owns three distinct randomizations.
`_qmc_interval.py` owns statistical calculations and contains no point
generation. `qmc.py` owns method declarations, schedules, evaluation, result
assembly, and replay evidence for B4.

**Tech Stack:** Python 3.11+, JAX 0.10.1+, JAX NumPy, integer digital
arithmetic, `jax.random`, `jax.scipy.special.betainc`, NumPy host-side data
generation, pytest, Ruff, MyPy, JSON evidence.

## Global Constraints

- B0 through B2 must be complete and independently approved.
- Governing design:
  `docs/superpowers/specs/2026-07-17-jaxstro-quad-phase-b-multidimensional-design.md`.
- Vendor exactly `new-joe-kuo-6.21201` from the UNSW Sobol archive.
- Pin source SHA256
  `68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441`.
- Vendor the accompanying license with SHA256
  `9d10226b50eeb34be0ab06bfa3392c7bd1f04bf602f9af4343295d1fd003d0e3`.
- Preserve the copyright and BSD-style redistribution text verbatim.
- Use power-of-two prefixes only. Do not skip, thin, or expose arbitrary
  non-power-of-two sample counts.
- Use at most 24 digital output bits for float32 and 53 for float64.
- The user supplies one explicit JAX key for randomized methods. Replicate
  identity is `jax.random.fold_in(key, replicate_index)`.
- Label only prefix-dependent nested permutations as Owen scrambling. Never
  label LMS as Owen.
- Randomized confidence contracts accept real scalar payloads only.
- Fixed-look Student-t requires at least eight independent replicates and one
  inspection.
- Sequential convergence requires certified finite replicate-estimate bounds,
  alpha spending, and a static schedule that grows both point level and
  replicate count.
- Repeated unbounded standard-error ladders are diagnostics and cannot return
  calibrated `CONVERGED`.
- Use `gradient="stop"` until B4.
- Every B3 method uses the shared zero-volume shortcut before generating or
  scrambling points and reports zero logical work for a coincident axis, but
  only after dynamic validity. Branch `INVALID_INPUT`, then zero volume, then
  QMC generation; test a mixed coincident/nonfinite domain in deterministic,
  fixed-look, and sequential paths.
- B3 status precedence is `INVALID_INPUT`, `NONFINITE_INTEGRAND`, `CONVERGED`,
  then `MAX_EVALUATIONS`; fixed deterministic Sobol returns
  `ERROR_ESTIMATE_UNAVAILABLE` after invalid/nonfinite checks. B3 does not emit
  `ROUNDOFF_LIMITED` or `DIVERGENCE_SUSPECTED` without a separately validated
  statistical detector.
- Add no runtime dependency and no sibling migration.
- Commit each task after focused RED/GREEN verification.

## File and Responsibility Map

- `src/jaxstro/quad/data/new-joe-kuo-6.21201`: vendored direction-number source.
- `src/jaxstro/quad/data/JOE_KUO_LICENSE`: verbatim license.
- `src/jaxstro/quad/data/joe-kuo-metadata.json`: source URLs, checksums,
  citation, dimension, and update date.
- `scripts/build_sobol_directions.py`: deterministic source parser and generated
  table checker.
- `src/jaxstro/quad/_sobol_data.py`: generated compact polynomial/initial-value
  table with source checksum.
- `src/jaxstro/quad/_sobol.py`: direction recurrence and Gray-code prefixes.
- `src/jaxstro/quad/_scramble.py`: digital shift, LMS-plus-shift, and nested
  Owen owners.
- `src/jaxstro/quad/_qmc_interval.py`: Student-t and empirical-Bernstein
  calculations.
- `src/jaxstro/quad/qmc.py`: declarations, integration controllers, work, and
  result semantics.
- `src/jaxstro/quad/integrate.py`, `__init__.py`: explicit B3 dispatch/exports.
- `tests/unit/quad/test_sobol_data.py`: source/generated freshness and checksum.
- `tests/unit/quad/test_sobol.py`: exact points, bits, prefixes, dimensions.
- `tests/unit/quad/test_scramble.py`: key identity and randomization invariants.
- `tests/unit/quad/test_qmc_interval.py`: interval formula oracles.
- `tests/unit/quad/test_qmc.py`: deterministic, fixed-look, and adaptive result
  contracts.
- `tests/validation/test_quad_rqmc_calibration.py`: predeclared seed campaigns.
- `tests/integration/test_quad_qmc_transforms.py`: eager/JIT/VMAP/reproducibility.
- `docs/validation/quad-rqmc-calibration.json`: generated calibration artifact.
- `STATUS.md`: B3 completion and B4 next action.

---

### Task 1: Vendor, parse, and freshness-check Joe-Kuo data

**Files:**
- Create: `src/jaxstro/quad/data/new-joe-kuo-6.21201`
- Create: `src/jaxstro/quad/data/JOE_KUO_LICENSE`
- Create: `src/jaxstro/quad/data/joe-kuo-metadata.json`
- Create: `scripts/build_sobol_directions.py`
- Create: `src/jaxstro/quad/_sobol_data.py`
- Create: `tests/unit/quad/test_sobol_data.py`

**Interfaces:**
- Consumes: UNSW columns `dimension degree coefficient initial_values`.
- Produces: generated `SOBOL_POLYNOMIALS`,
  `SOBOL_INITIAL_DIRECTIONS`, `MAX_SOBOL_DIMENSION`, and source checksum.

- [ ] **Step 1: Acquire and verify the approved immutable sources**

  Run:

  ```bash
  curl -fsSL \
    https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201 \
    -o src/jaxstro/quad/data/new-joe-kuo-6.21201
  curl -fsSL \
    https://web.maths.unsw.edu.au/~fkuo/sobol/licence \
    -o src/jaxstro/quad/data/JOE_KUO_LICENSE
  shasum -a 256 src/jaxstro/quad/data/new-joe-kuo-6.21201
  shasum -a 256 src/jaxstro/quad/data/JOE_KUO_LICENSE
  ```

  Expected: checksums exactly match the two Global Constraints. Any mismatch
  stops the task; do not update the expected hashes without a new provenance
  review.

- [ ] **Step 2: Write failing parser and provenance tests**

  Create `tests/unit/quad/test_sobol_data.py`:

  ```python
  import hashlib
  from pathlib import Path

  from jaxstro.quad import _sobol_data


  DATA = Path("src/jaxstro/quad/data/new-joe-kuo-6.21201")
  LICENSE = Path("src/jaxstro/quad/data/JOE_KUO_LICENSE")


  def test_vendored_sobol_sources_have_reviewed_checksums():
      assert hashlib.sha256(DATA.read_bytes()).hexdigest() == (
          "68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441"
      )
      assert hashlib.sha256(LICENSE.read_bytes()).hexdigest() == (
          "9d10226b50eeb34be0ab06bfa3392c7bd1f04bf602f9af4343295d1fd003d0e3"
      )


  def test_generated_table_covers_declared_dimension():
      assert _sobol_data.MAX_SOBOL_DIMENSION == 21201
      assert _sobol_data.SOURCE_SHA256.endswith("a067441")
      assert len(_sobol_data.SOBOL_POLYNOMIALS) == 21200
  ```

- [ ] **Step 3: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_sobol_data.py
  ```

  Expected: FAIL because `_sobol_data.py` is absent.

- [ ] **Step 4: Implement deterministic generation and check mode**

  `scripts/build_sobol_directions.py` must parse:

  ```python
  def parse_line(line: str):
      fields = [int(field) for field in line.split()]
      dimension, degree, coefficient, *initial = fields
      if len(initial) != degree:
          raise ValueError(
              f"dimension {dimension} declares degree {degree} "
              f"but has {len(initial)} initial values"
          )
      return dimension, degree, coefficient, tuple(initial)
  ```

  Emit deterministic tuples sorted by dimension, include the exact source
  checksum, and support:

  ```bash
  python scripts/build_sobol_directions.py --emit
  python scripts/build_sobol_directions.py --check
  ```

  `--check` renders to memory and exits nonzero on any byte difference. Metadata
  JSON records both source URLs, both hashes, Joe-Kuo 2008 DOI, 21,201
  dimensions, criterion 6, and source update date `2010-09-16`.

- [ ] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/build_sobol_directions.py --check
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_sobol_data.py
  ```

  Expected: both commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/data src/jaxstro/quad/_sobol_data.py \
    scripts/build_sobol_directions.py tests/unit/quad/test_sobol_data.py
  git commit -m "build(quad): vendor Joe-Kuo direction data"
  ```

### Task 2: Implement direction recurrence and deterministic Sobol integration

**Files:**
- Create: `src/jaxstro/quad/_sobol.py`
- Create: `src/jaxstro/quad/qmc.py`
- Modify: `src/jaxstro/quad/integrate.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_sobol.py`
- Create: `tests/unit/quad/test_qmc.py`

**Interfaces:**
- Consumes: generated source table and B0 evaluator.
- Produces: `direction_numbers(dimension, bits, dtype)`,
  `sobol_points(level, dimension, dtype)`, `Sobol(level, bits=None)`, and
  deterministic `integrate_qmc`.

- [ ] **Step 1: Write failing exact-prefix and result tests**

  Create `tests/unit/quad/test_sobol.py`:

  ```python
  import jax.numpy as jnp
  import pytest

  from jaxstro.quad._sobol import sobol_points


  def test_first_ten_three_dimensional_points_match_joe_kuo_example():
      expected = jnp.array(
          [
              [0.0, 0.0, 0.0],
              [0.5, 0.5, 0.5],
              [0.75, 0.25, 0.25],
              [0.25, 0.75, 0.75],
              [0.375, 0.375, 0.625],
              [0.875, 0.875, 0.125],
              [0.625, 0.125, 0.875],
              [0.125, 0.625, 0.375],
              [0.1875, 0.3125, 0.9375],
              [0.6875, 0.8125, 0.4375],
          ],
          dtype=jnp.float64,
      )
      assert jnp.array_equal(sobol_points(4, 3, jnp.float64)[:10], expected)


  def test_prefixes_are_exactly_nested():
      assert jnp.array_equal(
          sobol_points(4, 7, jnp.float64),
          sobol_points(5, 7, jnp.float64)[:16],
      )


  @pytest.mark.parametrize(
      "dtype, bits",
      [(jnp.float32, 25), (jnp.float64, 54)],
  )
  def test_distinct_coordinate_bit_limit_is_eager(dtype, bits):
      with pytest.raises(ValueError, match="distinct"):
          sobol_points(3, 2, dtype, bits=bits)
  ```

  Add to `test_qmc.py`:

  ```python
  def test_deterministic_sobol_result_has_unavailable_error():
      result = quad.integrate(
          lambda x: jnp.prod(x, axis=-1),
          quad.Hyperrectangle(jnp.zeros(4), jnp.ones(4)),
          method=quad.Sobol(level=10),
          epsabs=1e-4,
          epsrel=1e-4,
          max_evaluations=1024,
          gradient="stop",
      )
      assert result.work.evaluations == 1024
      assert result.work.levels == 10
      assert result.work.replicates == 0
      assert result.error.kind == quad.ErrorKind.UNAVAILABLE
      assert result.status == quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE
      assert jnp.isnan(result.error.confidence_level)
  ```

- [ ] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_sobol.py tests/unit/quad/test_qmc.py
  ```

  Expected: FAIL because the Sobol owners are absent.

- [ ] **Step 3: Implement direction-number recurrence**

  Use unsigned integer arithmetic. For dimension one:

  ```python
  directions[0, bit - 1] = 1 << (bits - bit)
  ```

  For every remaining dimension with degree `s`, coefficient `a`, and initial
  values `m`:

  ```python
  for bit in range(1, s + 1):
      directions[dimension - 1, bit - 1] = m[bit - 1] << (bits - bit)
  for bit in range(s + 1, bits + 1):
      value = directions[dimension - 1, bit - s - 1]
      value ^= value >> s
      for offset in range(1, s):
          if (a >> (s - 1 - offset)) & 1:
              value ^= directions[dimension - 1, bit - offset - 1]
      directions[dimension - 1, bit - 1] = value
  ```

  Validate dimension `1 <= d <= 21201`, static bits, and dtype limits eagerly.

- [ ] **Step 4: Implement Gray-code points and deterministic result**

  For indices `n=0,...,2**level-1`, compute `gray=n^(n>>1)` and XOR the
  direction column for every set Gray bit. Convert once with
  `integer / 2**bits`. Keep the first zero point; do not skip it.

  Define:

  ```python
  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class Sobol:
      level: int
      bits: int | None = None

      def __post_init__(self):
          if (
              isinstance(self.level, bool)
              or not isinstance(self.level, int)
              or self.level < 0
          ):
              raise ValueError("Sobol level must be a nonnegative integer")
          if self.bits is not None and (
              isinstance(self.bits, bool)
              or not isinstance(self.bits, int)
              or self.bits < 1
              or self.level > self.bits
          ):
              raise ValueError("Sobol requires 0 <= level <= bits")

      def tree_flatten(self):
          return (), (self.level, self.bits)

      @classmethod
      def tree_unflatten(cls, metadata, _children):
          level, bits = metadata
          return cls(level=level, bits=bits)
  ```

  Resolve `bits=None` from dtype, then validate `level<=bits`, the public
  float32/float64 bit limit, and `2**level<=max_evaluations` before generating
  directions. Boundary tests cover `level==bits`, `level==bits+1`, float32
  bits 24, and float64 bits 53. Evaluate once, average with the B0
  physical weights, and return the shared unavailable-result semantics. Infer
  payload shape first and use `jax.lax.cond` to return `zero_volume_result`
  without generating points when any domain width is zero.

- [ ] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_sobol_data.py \
    tests/unit/quad/test_sobol.py \
    tests/unit/quad/test_qmc.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/_sobol.py src/jaxstro/quad/qmc.py \
    tests/unit/quad/test_sobol.py tests/unit/quad/test_qmc.py
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/_sobol.py src/jaxstro/quad/qmc.py \
    src/jaxstro/quad/integrate.py src/jaxstro/quad/__init__.py \
    tests/unit/quad/test_sobol.py tests/unit/quad/test_qmc.py
  git commit -m "feat(quad): add deterministic Sobol integration"
  ```

### Task 3: Implement three correctly named randomizations

**Files:**
- Create: `src/jaxstro/quad/_scramble.py`
- Modify: `src/jaxstro/quad/qmc.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_scramble.py`

**Interfaces:**
- Consumes: integer Sobol points and one explicit JAX key.
- Produces: `DigitalShift`, `LinearMatrixScramble`, `OwenScramble`, and
  `scramble_integers(points, *, method, key, bits)`.

- [ ] **Step 1: Write failing reproducibility and identity tests**

  Create `tests/unit/quad/test_scramble.py`:

  ```python
  import jax
  import jax.numpy as jnp

  from jaxstro import quad
  from jaxstro.quad._scramble import scramble_integers
  from jaxstro.quad._sobol import sobol_integer_points


  def test_replicate_fold_in_is_stable_when_capacity_grows():
      key = jax.random.key(7)
      first = [jax.random.fold_in(key, i) for i in range(8)]
      grown = [jax.random.fold_in(key, i) for i in range(16)]
      assert all(jnp.array_equal(a, b) for a, b in zip(first, grown))


  def test_each_scramble_is_reproducible_and_prefix_preserving():
      points = sobol_integer_points(6, 3, bits=24)
      key = jax.random.key(11)
      for method in (
          quad.DigitalShift(),
          quad.LinearMatrixScramble(),
          quad.OwenScramble(),
      ):
          first = scramble_integers(points, method=method, key=key, bits=24)
          second = scramble_integers(points, method=method, key=key, bits=24)
          assert jnp.array_equal(first, second)
          assert jnp.array_equal(first[:16], scramble_integers(
              points[:16], method=method, key=key, bits=24
          ))


  def test_independent_keys_change_randomization():
      points = sobol_integer_points(4, 2, bits=24)
      a = scramble_integers(
          points, method=quad.LinearMatrixScramble(),
          key=jax.random.key(1), bits=24
      )
      b = scramble_integers(
          points, method=quad.LinearMatrixScramble(),
          key=jax.random.key(2), bits=24
      )
      assert not jnp.array_equal(a, b)
  ```

- [ ] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_scramble.py
  ```

  Expected: FAIL because `_scramble.py` is absent.

- [ ] **Step 3: Implement digital shift and LMS-plus-shift**

  Define three distinct frozen, hashable declarations in `_scramble.py`:

  ```python
  @dataclass(frozen=True)
  class DigitalShift:
      pass


  @dataclass(frozen=True)
  class LinearMatrixScramble:
      pass


  @dataclass(frozen=True)
  class OwenScramble:
      pass
  ```

  `DigitalShift` XORs one independent `bits`-wide integer per coordinate.
  `LinearMatrixScramble` samples a binary lower-triangular matrix per coordinate
  with an all-one diagonal, multiplies each point's bit vector over
  $\operatorname{GF}(2)$, repacks bits, then applies an independent digital
  shift:

  ```python
  lower = jnp.tril(
      jax.random.bernoulli(matrix_key, shape=(dimension, bits, bits))
  ).at[:, jnp.arange(bits), jnp.arange(bits)].set(True)
  transformed_bits = jnp.mod(
      jnp.einsum("dbk,ndk->ndb", lower, point_bits),
      2,
  )
  ```

  Use distinct `fold_in` tags for matrix and shift so adding another random
  draw does not change either stream.

- [ ] **Step 4: Implement true nested Owen permutations**

  For coordinate `d`, bit `b`, and the already-scrambled prefix integer `p`,
  derive a stateless permutation bit by folding all three identities into the
  coordinate key:

  ```python
  prefix_u64 = jnp.asarray(prefix, dtype=jnp.uint64)
  prefix_low = jnp.asarray(prefix_u64 & 0xFFFFFFFF, dtype=jnp.uint32)
  prefix_high = jnp.asarray(prefix_u64 >> 32, dtype=jnp.uint32)
  permutation_key = jax.random.fold_in(
      jax.random.fold_in(
          jax.random.fold_in(
              jax.random.fold_in(key, coordinate),
              bit,
          ),
          prefix_high,
      ),
      prefix_low,
  )
  flip = jax.random.bernoulli(permutation_key)
  scrambled_bit = source_bit ^ flip
  ```

  Scan bits from most significant to least significant and update `prefix` with
  `scrambled_bit`, not source bits. This prefix dependence is the semantic
  difference from LMS and must remain visible in tests and docs. Add an x64
  regression whose prefixes differ only above bit 32 and therefore must derive
  different permutation keys; also test identical 53-bit prefixes reproduce
  exactly.

- [ ] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_scramble.py \
    tests/unit/quad/test_sobol.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/_scramble.py tests/unit/quad/test_scramble.py
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/_scramble.py src/jaxstro/quad/qmc.py \
    src/jaxstro/quad/__init__.py tests/unit/quad/test_scramble.py
  git commit -m "feat(quad): add Sobol randomizations"
  ```

### Task 4: Add fixed-look Student-t randomized integration

**Files:**
- Create: `src/jaxstro/quad/_qmc_interval.py`
- Modify: `src/jaxstro/quad/qmc.py`
- Modify: `src/jaxstro/quad/integrate.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_qmc_interval.py`
- Modify: `tests/unit/quad/test_qmc.py`

**Interfaces:**
- Consumes: independent scrambled replicate estimates.
- Produces: `student_t_quantile`, `fixed_look_interval`,
  `ScrambledSobol(level, replicates=8, scramble=LinearMatrixScramble(),
  confidence_level=0.95)`.

- [ ] **Step 1: Write failing interval and result tests**

  Add to `test_qmc_interval.py`:

  ```python
  import jax.numpy as jnp

  from jaxstro.quad._qmc_interval import fixed_look_interval


  def test_fixed_look_interval_uses_unbiased_replicate_variance():
      estimates = jnp.array([0.9, 1.0, 1.1, 1.2, 0.8, 1.05, 0.95, 1.0])
      interval = fixed_look_interval(estimates, confidence_level=0.95)
      sample_variance = jnp.sum((estimates - jnp.mean(estimates)) ** 2) / 7
      expected_se = jnp.sqrt(sample_variance / 8)
      assert jnp.allclose(interval.standard_error, expected_se)
      assert interval.half_width > 0.0
  ```

  Add to `test_qmc.py`:

  ```python
  def test_scrambled_sobol_returns_one_fixed_look_interval():
      result = quad.integrate(
          lambda x: jnp.prod(x, axis=-1),
          quad.Hyperrectangle(jnp.zeros(3), jnp.ones(3)),
          method=quad.ScrambledSobol(level=8, replicates=8),
          key=jax.random.key(19),
          epsabs=0.02,
          epsrel=0.0,
          max_evaluations=8 * 256,
          gradient="stop",
      )
      assert result.error.kind == quad.ErrorKind.CONFIDENCE_INTERVAL_HALF_WIDTH
      assert result.error.confidence_level == 0.95
      assert result.work.replicates == 8
      assert result.work.levels == 8
      assert result.status in (
          quad.QuadStatus.CONVERGED,
          quad.QuadStatus.MAX_EVALUATIONS,
      )
  ```

- [ ] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_qmc_interval.py tests/unit/quad/test_qmc.py
  ```

  Expected: FAIL because interval owners are absent.

- [ ] **Step 3: Implement stopped Student-t critical values**

  Compute the positive quantile by first selecting a sufficient bracket from
  the fixed sequence $1,2,\ldots,2^{31}$ and then applying 80 bisection steps
  against the Student-t CDF expressed through `jax.scipy.special.betainc`:

  ```python
  def student_t_cdf(value, degrees_of_freedom):
      x = degrees_of_freedom / (degrees_of_freedom + value**2)
      tail = 0.5 * jsp.special.betainc(
          0.5 * degrees_of_freedom,
          0.5,
          x,
      )
      return jnp.where(value >= 0.0, 1.0 - tail, tail)


  def student_t_quantile(probability, degrees_of_freedom):
      dtype = jnp.result_type(probability, jnp.asarray(0.0))

      def bisect_step(_, bounds):
          lower, upper = bounds
          midpoint = 0.5 * (lower + upper)
          return jax.lax.cond(
              student_t_cdf(midpoint, degrees_of_freedom) < probability,
              lambda _: (midpoint, upper),
              lambda _: (lower, midpoint),
              operand=None,
          )
      candidates = 2.0 ** jnp.arange(32, dtype=dtype)
      covered = student_t_cdf(candidates, degrees_of_freedom) >= probability
      bracketed = jnp.any(covered)
      first = jnp.argmax(covered)
      initial_upper = candidates[first]
      lower, upper = jax.lax.fori_loop(
          0,
          80,
          bisect_step,
          (jnp.asarray(0.0, dtype=dtype), initial_upper),
      )
      quantile = jnp.where(bracketed, 0.5 * (lower + upper), jnp.nan)
      return jax.lax.stop_gradient(quantile)
  ```

  Add this two-sided confidence fixture, where each value is the positive
  quantile at probability `(1 + confidence_level) / 2`:

  ```python
  T_CRITICAL = {
      7: {0.90: 1.894578605, 0.95: 2.364624252, 0.99: 3.499483297},
      15: {0.90: 1.753050356, 0.95: 2.131449546, 0.99: 2.946712883},
      31: {0.90: 1.695518783, 0.95: 2.039513446, 0.99: 2.744041919},
      63: {0.90: 1.669402222, 0.95: 1.998340543, 0.99: 2.656145030},
  }
  ```

  Validate these constants and confidence levels next to one against SciPy
  1.16.0 using the B1 `reference` dependency group:

  ```bash
  env -u VIRTUAL_ENV uv run --locked --group reference python -c \
    "import scipy; assert scipy.__version__ == '1.16.0'"
  ```

  Store the version in the test provenance comment. Assert every supported
  finite float32/float64 probability is bracketed; an unbracketed internal
  result is `INVALID_INPUT`, never a clipped quantile.

- [ ] **Step 4: Implement fixed-look integration**

  In `qmc.py`, define:

  ```python
  @dataclass(frozen=True)
  class ScrambledSobol:
      level: int
      replicates: int = 8
      scramble: DigitalShift | LinearMatrixScramble | OwenScramble = field(
          default_factory=LinearMatrixScramble
      )
      confidence_level: float = 0.95

      def __post_init__(self):
          if (
              isinstance(self.level, bool)
              or not isinstance(self.level, int)
              or self.level < 0
          ):
              raise ValueError("ScrambledSobol level must be nonnegative")
          if (
              isinstance(self.replicates, bool)
              or not isinstance(self.replicates, int)
              or self.replicates < 8
          ):
              raise ValueError("ScrambledSobol requires at least 8 replicates")
          if not 0.0 < self.confidence_level < 1.0:
              raise ValueError("confidence_level must lie strictly between 0 and 1")
  ```

  Validate real scalar output by `jax.eval_shape`, `replicates>=8`, one explicit
  key, and exact budget `replicates*2**level<=max_evaluations`. Fold replicate
  keys, evaluate each scrambled prefix, and compute:

  ```python
  mean = jnp.mean(estimates)
  variance = jnp.sum((estimates - mean) ** 2) / (replicates - 1)
  standard_error = jnp.sqrt(variance / replicates)
  critical = student_t_quantile(
      0.5 * (1.0 + confidence_level),
      replicates - 1,
  )
  half_width = critical * standard_error
  ```

  Return `CONVERGED` when `half_width<=tolerance` and `MAX_EVALUATIONS`
  otherwise. The same pre-generation zero-volume branch returns the exact
  scalar zero with zero work.

- [ ] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_qmc_interval.py \
    tests/unit/quad/test_qmc.py \
    tests/unit/quad/test_scramble.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check \
    src/jaxstro/quad/_qmc_interval.py src/jaxstro/quad/qmc.py \
    tests/unit/quad/test_qmc_interval.py tests/unit/quad/test_qmc.py
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/_qmc_interval.py src/jaxstro/quad/qmc.py \
    src/jaxstro/quad/integrate.py src/jaxstro/quad/__init__.py \
    tests/unit/quad/test_qmc_interval.py tests/unit/quad/test_qmc.py
  git commit -m "feat(quad): add fixed-look randomized QMC"
  ```

### Task 5: Add bounded sequential empirical-Bernstein integration

**Files:**
- Modify: `src/jaxstro/quad/_qmc_interval.py`
- Modify: `src/jaxstro/quad/qmc.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Modify: `tests/unit/quad/test_qmc_interval.py`
- Modify: `tests/unit/quad/test_qmc.py`

**Interfaces:**
- Consumes: static monotone `(level, replicate_count)` schedule and certified
  replicate-estimate bounds.
- Produces: `AdaptiveScrambledSobol(schedule, estimate_bounds, scramble,
  confidence_level)` and alpha-spending convergence.

- [ ] **Step 1: Write failing alpha, floor, and schedule tests**

  Append:

  ```python
  def test_alpha_spending_sums_below_requested_alpha():
      alpha = 0.05
      spent = sum(alpha * 6.0 / (jnp.pi**2 * (k + 1) ** 2) for k in range(10000))
      assert spent < alpha
      assert jnp.allclose(spent, alpha, rtol=2e-4)


  def test_empirical_bernstein_range_term_shrinks_only_with_replicates():
      from jaxstro.quad._qmc_interval import empirical_bernstein_half_width

      same_r = empirical_bernstein_half_width(
          jnp.zeros(8), lower=0.0, upper=1.0, alpha=0.05
      )
      more_r = empirical_bernstein_half_width(
          jnp.zeros(32), lower=0.0, upper=1.0, alpha=0.05
      )
      assert more_r < same_r


  def test_adaptive_schedule_requires_point_and_replicate_growth():
      with pytest.raises(ValueError, match="replicate growth"):
          quad.AdaptiveScrambledSobol(
              schedule=((6, 8), (7, 8)),
              estimate_bounds=(0.0, 1.0),
          )
  ```

- [ ] **Step 2: Run RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_qmc_interval.py tests/unit/quad/test_qmc.py
  ```

  Expected: FAIL because adaptive declarations and bounds are absent.

- [ ] **Step 3: Implement exact empirical-Bernstein formula**

  In `qmc.py`, define:

  ```python
  @dataclass(frozen=True)
  class AdaptiveScrambledSobol:
      schedule: tuple[tuple[int, int], ...]
      estimate_bounds: tuple[float, float] | None = None
      integrand_bounds: tuple[float, float] | None = None
      scramble: DigitalShift | LinearMatrixScramble | OwenScramble = field(
          default_factory=LinearMatrixScramble
      )
      confidence_level: float = 0.95

      def __post_init__(self):
          if not self.schedule:
              raise ValueError("AdaptiveScrambledSobol schedule cannot be empty")
          levels, replicates = zip(*self.schedule, strict=True)
          if any(
              isinstance(level, bool)
              or not isinstance(level, int)
              or level < 0
              for level in levels
          ):
              raise ValueError("schedule levels must be nonnegative")
          if any(
              isinstance(count, bool)
              or not isinstance(count, int)
              or count < 8
              for count in replicates
          ):
              raise ValueError("schedule requires at least 8 replicates")
          pairs = zip(self.schedule, self.schedule[1:])
          if any(
              next_level < level
              or next_count < count
              or (next_level == level and next_count == count)
              for (level, count), (next_level, next_count) in pairs
          ):
              raise ValueError("schedule must be monotone with strict progress")
          if replicates[-1] <= replicates[0]:
              raise ValueError("schedule must include replicate growth")
          if (self.estimate_bounds is None) == (self.integrand_bounds is None):
              raise ValueError(
                  "provide exactly one of estimate_bounds or integrand_bounds"
              )
          lower, upper = (
              self.estimate_bounds
              if self.estimate_bounds is not None
              else self.integrand_bounds
          )
          if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
              raise ValueError("bounds must be finite and ordered")
          if not 0.0 < self.confidence_level < 1.0:
              raise ValueError("confidence_level must lie strictly between 0 and 1")
  ```

  Add:

  ```python
  def empirical_bernstein_half_width(estimates, *, lower, upper, alpha):
      replicate_count = estimates.shape[0]
      mean = jnp.mean(estimates)
      variance = jnp.sum((estimates - mean) ** 2) / (replicate_count - 1)
      log_term = jnp.log(2.0 / alpha)
      return jnp.sqrt(2.0 * variance * log_term / replicate_count) + (
          7.0 * (upper - lower) * log_term
          / (3.0 * (replicate_count - 1))
      )


  def spent_alpha(alpha, inspection):
      return alpha * 6.0 / (jnp.pi**2 * (inspection + 1) ** 2)
  ```

  Direct `estimate_bounds` apply to replicate estimates and work with any
  supported finite measure. `integrand_bounds` are derivable in B3 only for
  `LebesgueMeasure`: scale both pointwise endpoints by absolute volume and
  signed orientation, then reorder them:

  ```python
  scaled = hyperrectangle_orientation(domain) * absolute_volume * jnp.asarray(
      method.integrand_bounds
  )
  resolved_estimate_bounds = (jnp.min(scaled), jnp.max(scaled))
  ```

  A `WeightedMeasure`, signed measure, or unknown-mass
  measure with only pointwise bounds raises eagerly and directs the researcher
  to supply direct estimate bounds; B4 extends this derivation only for
  certified finite nonnegative `ProductMeasure` components.

  After every inspection, test every active replicate estimate against the
  resolved `[A,B]`. Any nonfinite or out-of-bounds estimate terminates with
  `INVALID_INPUT`, nonfinite value, and stopped diagnostics before a confidence
  interval is reported. Add tests for direct signed-measure bounds, valid
  forward and reversed Lebesgue derivation, rejected weighted derivation, and
  an integrand that violates declared pointwise bounds. Also assert float
  levels/counts in fixed and adaptive declarations raise eagerly before any
  shift or shape construction.

  The method declaration validates increasing levels/replicates, at least eight
  replicates, at least one strict change per pair, final replicate growth, finite
  ordered bounds, and total final work within `max_evaluations`.

- [ ] **Step 4: Implement prefix and replicate reuse**

  Use fixed arrays sized to final schedule capacities. At inspection `k`:

  - activate new replicate IDs and evaluate their full current prefix;
  - extend existing active replicates only from `2**previous_level` to
    `2**current_level`;
  - retain per-replicate running sums;
  - compute estimates from complete current prefixes;
  - compute `alpha_k`, half-width, and tolerance once;
  - stop on half-width or continue to the next static schedule row.

  Exhaustion returns `MAX_EVALUATIONS`. `work.evaluations` counts every unique
  evaluated point, `work.refinements` counts accepted schedule expansions,
  `work.levels` is the final level, and `work.replicates` is the final active
  count. Apply the shared zero-volume branch before schedule initialization.

- [ ] **Step 5: Run GREEN and commit**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_qmc_interval.py \
    tests/unit/quad/test_qmc.py \
    tests/unit/quad/test_scramble.py
  ```

  Expected: all commands exit zero. Commit:

  ```bash
  git add src/jaxstro/quad/_qmc_interval.py src/jaxstro/quad/qmc.py \
    src/jaxstro/quad/__init__.py tests/unit/quad/test_qmc_interval.py \
    tests/unit/quad/test_qmc.py
  git commit -m "feat(quad): add bounded sequential RQMC"
  ```

### Task 6: Calibrate RQMC coverage and certify JAX behavior

**Files:**
- Create: `tests/validation/test_quad_rqmc_calibration.py`
- Create: `tests/integration/test_quad_qmc_transforms.py`
- Create: `scripts/generate_quad_rqmc_evidence.py`
- Create: `docs/validation/quad-rqmc-calibration.json`
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: all B3 methods.
- Produces: deterministic exact-point evidence, reproducibility, RMSE,
  fixed-look coverage, sequential coverage, and mutation-resistant alpha
  spending evidence.

- [ ] **Step 1: Freeze the calibration manifest**

  In the generator, define immutable campaign controls:

  ```python
  CAMPAIGN = {
      "seed_count": 128,
      "confidence_level": 0.95,
      "records": [
          {
              "case": "separable_polynomial",
              "dimension": 2,
              "method": "fixed",
              "level": 8,
              "replicates": 16,
          },
          {
              "case": "separable_exponential",
              "dimension": 8,
              "method": "fixed",
              "level": 8,
              "replicates": 16,
          },
          {
              "case": "low_effective_dimension",
              "dimension": 16,
              "method": "sequential",
              "schedule": [[6, 8], [7, 16], [8, 32]],
          },
          {
              "case": "rotated_smooth",
              "dimension": 4,
              "method": "sequential",
              "schedule": [[6, 8], [7, 16], [8, 32]],
          },
      ],
  }
  ```

  Compute the exact two-sided 99% binomial acceptance band for each nominal
  coverage using integer binomial tails in the generator and store the band in
  the artifact before running cases. The manifest is a list of four records,
  not a Cartesian product, and executes at most 3,145,728 point-integrand
  evaluations. `--regenerate-slow --emit` runs the complete 128-seed campaign;
  `--check` verifies artifact schema/hash and reruns a frozen 16-seed
  reproducibility subset.

- [ ] **Step 2: Add mutation-resistant coverage tests**

  The validation test must:

  - reproduce the artifact with `--check`;
  - assert empirical coverage lies within the frozen binomial band;
  - assert identical keys reproduce byte-identical estimates;
  - assert changed keys alter at least one replicate;
  - assert deterministic prefixes match the UNSW example;
  - directly mutation-test the allocation identity:
    `sum(spent_alpha(alpha, k) for k in range(10_000)) <= alpha`, while replacing
    `spent_alpha` with full alpha at every look exceeds alpha after the second
    inspection; and
  - retain empirical sequential coverage as statistical evidence, not as the
    sole detector for the alpha-allocation mutation.

- [ ] **Step 3: Add the B3 transform matrix**

  Test eager, `jit`, `vmap` over keys and domains, stable `fold_in` identities,
  float32/float64 bit limits, static method configuration, and real-scalar
  enforcement. Assert array and complex randomized payloads raise eagerly.

- [ ] **Step 4: Run the complete B3 gate and update status**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/build_sobol_directions.py --check
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/generate_quad_rqmc_evidence.py --regenerate-slow --emit
  env -u VIRTUAL_ENV uv run --no-sync python \
    scripts/generate_quad_rqmc_evidence.py --check
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/quad/test_sobol_data.py \
    tests/unit/quad/test_sobol.py \
    tests/unit/quad/test_scramble.py \
    tests/unit/quad/test_qmc_interval.py \
    tests/unit/quad/test_qmc.py \
    tests/integration/test_quad_qmc_transforms.py \
    tests/validation/test_quad_rqmc_calibration.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check src tests scripts
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
  git diff --check
  ```

  Expected: freshness, coverage, tests, lint, typing, and diff checks all pass.
  Update `STATUS.md` with exact counts and
  `next: Execute the reviewed Phase B4 replay, quantity, evidence, and documentation plan.`

- [ ] **Step 5: Commit and request checkpoint review**

  ```bash
  git add tests/validation/test_quad_rqmc_calibration.py \
    tests/integration/test_quad_qmc_transforms.py \
    scripts/generate_quad_rqmc_evidence.py \
    docs/validation/quad-rqmc-calibration.json STATUS.md
  git commit -m "test(quad): certify Phase B3 randomized QMC"
  ```

  Request independent QMC, statistics, JAX, provenance, API, and test-quality
  reviews. Resolve every Critical or Important finding before B4.
