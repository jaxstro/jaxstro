# Spectral Point-Resampling Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route spectral point resampling through Jaxstro's interpolation primitives and add an explicit, differentiable PCHIP opt-in while retaining linear interpolation as the default.

**Architecture:** `SpectralPlan` owns a static `PointResamplingMethod`; point spectra dispatch to the existing `jaxstro.numerics.interpolation` functions, while bin averages remain exclusively conservative. Coverage and identity are resolved before method dispatch, and every executed path records its method in provenance.

**Tech Stack:** Python 3.11+, JAX, Jaxstro spectral PyTrees, `jaxstro.numerics.interpolation`, pytest, Ruff, mypy.

## Global Constraints

- `LINEAR` is the default point method and delegates to `jaxstro.numerics.interpolation.interp1d`.
- `MONOTONE_CUBIC` is opt-in and delegates to `jaxstro.numerics.interpolation.monotone_cubic_interp`.
- Method selection is static; do not infer it from spectral values, spacing, or runtime smoothness tests.
- Equal axes always use the identity path.
- `BIN_AVERAGES` always use `conservative_remap_1d`; reject `MONOTONE_CUBIC` for binned targets.
- Continue rejecting point-to-bin, bin-to-point, and non-identity `BIN_INTEGRALS`.
- Do not extrapolate or clamp a partially unsupported target into success; return NaNs plus `UNSUPPORTED_SPECTRAL_WINDOW`.
- Gradient claims are local to fixed intervals or fixed PCHIP limiter branches.
- Add no dependency and duplicate no interpolation kernel.
- Run only the focused spectral gate for this amendment; do not launch the long repository CI gate.

---

### Task 4A: Add explicit point-resampling methods

**Files:**
- Modify: `src/jaxstro/spectra/plan.py`
- Modify: `src/jaxstro/spectra/resampling.py`
- Modify: `src/jaxstro/spectra/__init__.py`
- Modify: `tests/unit/test_spectra_plan.py`
- Modify: `tests/unit/test_spectra_resampling.py`
- Create: `tests/validation/test_spectra_resampling_gradients.py`
- Modify: `docs/superpowers/plans/2026-07-11-spectra-v1-implementation.md`
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: `interpolation.interp1d(x, y, x_new, extrapolate=False)` and `interpolation.monotone_cubic_interp(x, y, x_new, extrapolate=False)`.
- Produces: `PointResamplingMethod.LINEAR`, `PointResamplingMethod.MONOTONE_CUBIC`, and `SpectralPlan(..., point_method=PointResamplingMethod.LINEAR)`.
- Preserves: `resample_spectrum(spectrum, plan) -> SpectrumResult`.

- [ ] **Step 1: Write failing plan and public-API tests**

  Extend `tests/unit/test_spectra_plan.py` to import `PointResamplingMethod` and
  verify that the method is static PyTree metadata:

  ```python
  from jaxstro.spectra import PointResamplingMethod

  def test_plan_preserves_static_point_method_in_pytree_roundtrip() -> None:
      plan = SpectralPlan(
          target_axis=SpectralAxis.points(
              jnp.array([120.0, 180.0]),
              coordinate=SpectralCoordinate.WAVELENGTH,
              unit="nm",
          ),
          point_method=PointResamplingMethod.MONOTONE_CUBIC,
      )
      leaves, treedef = jax.tree_util.tree_flatten(plan)
      rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
      assert len(leaves) == 1
      assert rebuilt.point_method is PointResamplingMethod.MONOTONE_CUBIC
  ```

  Add a validation test that a bin-average plan rejects the non-default point
  method:

  ```python
  def test_bin_plan_rejects_monotone_cubic_point_method() -> None:
      axis = SpectralAxis.bins(
          jnp.array([100.0, 200.0, 400.0]),
          coordinate=SpectralCoordinate.WAVELENGTH,
          unit="nm",
          sampling=SpectralSampling.BIN_AVERAGES,
      )
      with pytest.raises(ValueError, match="point_method applies only"):
          SpectralPlan(axis, point_method=PointResamplingMethod.MONOTONE_CUBIC)
  ```

- [ ] **Step 2: Write failing dispatch, shape, and provenance tests**

  Extend `tests/unit/test_spectra_resampling.py` with a curved positive spectrum
  and compare each public method to its existing Jaxstro primitive:

  ```python
  from jaxstro.numerics import interpolation
  from jaxstro.spectra import PointResamplingMethod

  def test_linear_points_delegate_to_jaxstro_interp1d() -> None:
      source = _points()
      target = SpectralAxis.points(
          jnp.array([150.0, 300.0]),
          coordinate=SpectralCoordinate.WAVELENGTH,
          unit="nm",
      )
      result = resample_spectrum(source, SpectralPlan(target))
      expected = interpolation.interp1d(
          source.axis.values, source.values, target.values
      )
      np.testing.assert_array_equal(result.spectrum.values, expected)
      assert result.spectrum.provenance.operations[-1] == "resample:linear-points"

  def test_monotone_cubic_points_delegate_to_jaxstro_pchip() -> None:
      source = _curved_points()
      target = SpectralAxis.points(
          jnp.linspace(100.0, 400.0, 31),
          coordinate=SpectralCoordinate.WAVELENGTH,
          unit="nm",
      )
      plan = SpectralPlan(
          target,
          point_method=PointResamplingMethod.MONOTONE_CUBIC,
      )
      result = resample_spectrum(source, plan)
      expected = interpolation.monotone_cubic_interp(
          source.axis.values, source.values, target.values
      )
      np.testing.assert_array_equal(result.spectrum.values, expected)
      assert jnp.all(result.spectrum.values >= jnp.min(source.values))
      assert jnp.all(result.spectrum.values <= jnp.max(source.values))
      assert result.spectrum.provenance.operations[-1] == (
          "resample:monotone-cubic-points"
      )
  ```

  Add an identity test with `MONOTONE_CUBIC` selected and assert the operation
  remains `resample:identity` with bit-identical values. Retain all existing
  outside-coverage, point/bin rejection, and conservation tests unchanged.

- [ ] **Step 3: Write failing AD-vs-FD tests at smooth interior points**

  Create `tests/validation/test_spectra_resampling_gradients.py`. Use the shared
  `jaxstro.testing.Case` and `audit_entry_point` machinery for four cases:

  ```python
  CASES = (
      Case(
          id="spectral-linear-values",
          direction="values->resampled-values",
          fn=lambda scale: _resample_total(
              PointResamplingMethod.LINEAR,
              values=scale * BASE_VALUES,
              target=BASE_TARGET,
          ),
          param="value_scale",
          theta0=1.0,
          tol=2.0e-5,
          h_rel=1.0e-5,
          allowed_claim="linear point resampling inside fixed intervals",
          forbidden_claims=("differentiability through knot selection",),
      ),
      Case(
          id="spectral-linear-query",
          direction="target-axis->resampled-values",
          fn=lambda shift: _resample_total(
              PointResamplingMethod.LINEAR,
              values=BASE_VALUES,
              target=BASE_TARGET + shift,
          ),
          param="target_shift_nm",
          theta0=0.0,
          tol=2.0e-5,
          h_rel=1.0e-3,
          allowed_claim="linear query sensitivity away from knots",
      ),
      Case(
          id="spectral-pchip-values",
          direction="values->resampled-values",
          fn=lambda scale: _resample_total(
              PointResamplingMethod.MONOTONE_CUBIC,
              values=scale * BASE_VALUES,
              target=BASE_TARGET,
          ),
          param="value_scale",
          theta0=1.0,
          tol=2.0e-5,
          h_rel=1.0e-5,
          allowed_claim="PCHIP value sensitivity inside a fixed limiter branch",
          forbidden_claims=("differentiability through limiter transitions",),
      ),
      Case(
          id="spectral-pchip-query",
          direction="target-axis->resampled-values",
          fn=lambda shift: _resample_total(
              PointResamplingMethod.MONOTONE_CUBIC,
              values=BASE_VALUES,
              target=BASE_TARGET + shift,
          ),
          param="target_shift_nm",
          theta0=0.0,
          tol=2.0e-5,
          h_rel=1.0e-3,
          allowed_claim="PCHIP query sensitivity inside fixed intervals",
          forbidden_claims=("differentiability at knots",),
      ),
  )
  ```

  `_resample_total` must construct a point `Spectrum`, construct a
  `SpectralPlan` with the requested method, call `resample_spectrum`, and return
  `jnp.sum(result.spectrum.values)`. Choose `BASE_TARGET` strictly inside source
  intervals and choose positive `BASE_VALUES` whose adjacent secant signs do not
  change under the finite-difference perturbations.

- [ ] **Step 4: Run the focused RED gate**

  Run:

  ```bash
  env PYTHONPATH=/private/tmp/jaxstro-spectra-v1/src \
    /Users/anna/projects/jaxstro-dev/jaxstro/.venv/bin/pytest -q \
    tests/unit/test_spectra_plan.py \
    tests/unit/test_spectra_resampling.py \
    tests/validation/test_spectra_resampling_gradients.py
  ```

  Expected: collection fails because `PointResamplingMethod` is not exported.
  Record pytest and wall time.

- [ ] **Step 5: Implement the static method contract in `plan.py`**

  Add the enum and field:

  ```python
  class PointResamplingMethod(StrEnum):
      LINEAR = "linear"
      MONOTONE_CUBIC = "monotone_cubic"

  @jax.tree_util.register_pytree_node_class
  @dataclass(frozen=True)
  class SpectralPlan:
      target_axis: SpectralAxis
      coverage_policy: CoveragePolicy = CoveragePolicy.INTERSECTION
      point_method: PointResamplingMethod = PointResamplingMethod.LINEAR

      def __post_init__(self) -> None:
          coverage_policy = CoveragePolicy(self.coverage_policy)
          point_method = PointResamplingMethod(self.point_method)
          if (
              self.target_axis.sampling is not SpectralSampling.POINTS
              and point_method is not PointResamplingMethod.LINEAR
          ):
              raise ValueError("point_method applies only to point-sampled plans")
          object.__setattr__(self, "coverage_policy", coverage_policy)
          object.__setattr__(self, "point_method", point_method)
  ```

  Store `(coverage_policy, point_method)` as `SpectralPlan.tree_flatten()` static
  auxiliary data and restore both fields in `tree_unflatten()`. Export
  `PointResamplingMethod` from `plan.py` and `jaxstro.spectra`.

- [ ] **Step 6: Replace direct JAX interpolation with Jaxstro dispatch**

  In `resampling.py`, import the module rather than copying kernels:

  ```python
  from jaxstro.numerics import interpolation

  if source.sampling is SpectralSampling.POINTS:
      if plan.point_method is PointResamplingMethod.LINEAR:
          values = interpolation.interp1d(
              source.values,
              spectrum.values,
              target.values,
              extrapolate=False,
          )
          operation = "resample:linear-points"
      else:
          values = interpolation.monotone_cubic_interp(
              source.values,
              spectrum.values,
              target.values,
              extrapolate=False,
          )
          operation = "resample:monotone-cubic-points"
      return _result(spectrum, target, values, covered, operation)
  ```

  Keep the pre-existing whole-window coverage mask. Although both numerical
  primitives clamp by default, an unsupported target must still replace the
  complete payload with NaNs and return `UNSUPPORTED_SPECTRAL_WINDOW`.

- [ ] **Step 7: Run focused GREEN, formatting, and typing gates**

  Run:

  ```bash
  env PYTHONPATH=/private/tmp/jaxstro-spectra-v1/src \
    /Users/anna/projects/jaxstro-dev/jaxstro/.venv/bin/pytest -q \
    tests/unit/test_spectra_plan.py \
    tests/unit/test_spectra_resampling.py \
    tests/validation/test_spectra_remap_conservation.py \
    tests/validation/test_spectra_resampling_gradients.py
  /Users/anna/projects/jaxstro-dev/jaxstro/.venv/bin/ruff check \
    src/jaxstro/spectra tests/unit/test_spectra_plan.py \
    tests/unit/test_spectra_resampling.py \
    tests/validation/test_spectra_remap_conservation.py \
    tests/validation/test_spectra_resampling_gradients.py
  /Users/anna/projects/jaxstro-dev/jaxstro/.venv/bin/ruff format --check \
    src/jaxstro/spectra tests/unit/test_spectra_plan.py \
    tests/unit/test_spectra_resampling.py \
    tests/validation/test_spectra_remap_conservation.py \
    tests/validation/test_spectra_resampling_gradients.py
  /Users/anna/projects/jaxstro-dev/jaxstro/.venv/bin/mypy src/jaxstro/spectra
  git diff --check
  ```

  Expected: all focused tests pass in seconds; Ruff, format, mypy, and diff
  checks report no errors. Record pytest and wall time.

- [ ] **Step 8: Run the bounded combined spectra regression**

  Run:

  ```bash
  env PYTHONPATH=/private/tmp/jaxstro-spectra-v1/src \
    /Users/anna/projects/jaxstro-dev/jaxstro/.venv/bin/pytest -q \
    tests/unit/test_spectra_types.py \
    tests/unit/test_spectra_transforms.py \
    tests/unit/test_spectra_plan.py \
    tests/unit/test_spectra_resampling.py \
    tests/validation/test_spectra_transform_gradients.py \
    tests/validation/test_spectra_remap_conservation.py \
    tests/validation/test_spectra_resampling_gradients.py \
    tests/validation/provenance_cards/test_atmosphere_spectra_sources.py \
    tests/integration/test_spectra_ownership.py
  ```

  Expected: all spectra Tasks 2-4A tests pass in seconds. Do not substitute the
  long repository CI workflow for this bounded regression.

- [ ] **Step 9: Update execution records and commit**

  Add a completed `Task 4A` entry to
  `docs/superpowers/plans/2026-07-11-spectra-v1-implementation.md`. Update
  `STATUS.md` with the exact focused and combined timings, the four AD-vs-FD
  ratios, and the supported derivative boundaries. Then commit:

  ```bash
  git add src/jaxstro/spectra \
    tests/unit/test_spectra_plan.py \
    tests/unit/test_spectra_resampling.py \
    tests/validation/test_spectra_resampling_gradients.py \
    docs/superpowers/plans/2026-07-11-spectra-v1-implementation.md \
    STATUS.md
  git commit -m "feat: add spectral point-resampling methods"
  ```

  Expected: one coherent Task 4A implementation commit and a clean worktree.
