# Spectra v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete a source-backed, fail-closed, JAX-native spectral foundation
and four atmosphere-family adapters without moving observable rendering out of
Fluxax.

**Architecture:** Generic spectral types, transformations, resampling, and
prepared stencils live in `jaxstro.spectra`. Host-side product selection,
artifact loading, and topology preparation live in `jaxstro.atmospheres`.
Preparation returns fixed-shape, filesystem-free JAX objects; the old
atmosphere-owned spectral module is deleted only after all in-repo and inspected
downstream consumers migrate.

**Tech Stack:** Python 3.11+, frozen dataclasses, JAX 0.10.1+, NumPy for
host-side preparation, existing optional Polars/Zarr data extra, pytest, Ruff,
mypy, MyST.

## Global Constraints

- Canonical atmosphere output is increasing wavelength in `nm` and surface
  `F_lambda` in `erg s^-1 cm^-2 nm^-1`.
- Generic transformed spectra may use frequency/`F_nu`, luminosity density, or
  observer flux density when their semantics are explicit.
- No quantity-layer migration, new runtime dependency, compatibility alias,
  nearest-neighbor fallback, parameter extrapolation, or cross-family
  interpolation.
- Source claims require primary-source URLs and locators in the provenance
  registry.
- All production behavior follows red-green-refactor TDD.
- Topology selection and artifact I/O are host-side; prepared evaluation is
  filesystem-free and compatible with `jit`, `vmap`, and AD inside one region.
- Expected scientific gaps return structured statuses; corrupt artifacts and
  broken invariants raise exceptions.
- Fluxax retains extinction, filters, instruments, counts, magnitudes, PSFs,
  images, and likelihood-facing observables.
- Every task is a separate commit and HITL checkpoint.

---

### Task 1: Freeze provenance and ownership contracts

**Files:**
- Modify: `docs/provenance/registry/atmospheres.yaml`
- Create: `tests/validation/provenance_cards/test_atmosphere_spectra_sources.py`
- Modify: `docs/40-api/provenance/atmospheres.md`
- Modify: `docs/plans/2026-07-11-spectra-v1-design.md`

**Interfaces:**
- Consumes: current provenance-card builder and `docs/provenance/registry` schema.
- Produces: source records named `newera-v3-lowres`, `bosz-2025-recomputed`,
  `sonora-diamondback-2024`, `tlusty-ostar2002`, and `tlusty-bstar2006`.

- [x] **Step 1: Write failing provenance tests**

  Require each product card's conventions to contain an exact
  `native_coordinate=...`, `native_density=...`, `native_unit=...`,
  `canonical_factor=...`, and `owner=jaxstro.spectra` entry, plus a primary URL
  and precise locator.

  ```python
  ostar = set(records["tlusty-ostar2002"]["conventions"])
  sonora = set(records["sonora-diamondback-2024"]["conventions"])
  assert "native_density=H_nu" in ostar
  assert "canonical_factor=4*pi then F_nu-to-F_lambda" in ostar
  assert "native_density=wavelength-density flux" in sonora
  assert "canonical_factor=1e-6" in sonora
  assert all(
      "owner=jaxstro.spectra" in set(record["conventions"])
      for record in records.values()
  )
  ```

- [x] **Step 2: Verify RED**

  Run:

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/validation/provenance_cards/test_atmosphere_spectra_sources.py
  ```

  Expected: FAIL because the atmosphere registry is empty.

- [x] **Step 3: Add the five source records**

  Encode these verified conversions:

  ```text
  NewEra: nm, F_lambda [W m^-2 nm^-1] -> multiply by 1e3
  BOSZ resam: angstrom, F_lambda [erg s^-1 cm^-2 angstrom^-1] -> multiply by 10
  Sonora: micron, wavelength-density flux [W m^-2 m^-1] -> multiply by 1e-6
  TLUSTY: Hz, H_nu [erg s^-1 cm^-2 Hz^-1] -> multiply by 4*pi, then F_nu -> F_lambda
  ```

  NewEra's density is already per nm, so only the `W m^-2` to
  `erg s^-1 cm^-2` factor of `1e3` applies. BOSZ converts a per-angstrom
  density to per nm by `10`. Sonora first converts `W m^-2` to
  `erg s^-1 cm^-2` by `1e3`, then converts a per-metre density to per nm by
  `1e-9`; the combined factor is `1e-6`. Record the archive's inconsistent
  printed nu subscript next
  to its wavelength coordinate and per-metre dimensional unit.

- [x] **Step 4: Verify GREEN and registry freshness**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/validation/provenance_cards/test_atmosphere_spectra_sources.py
  env -u VIRTUAL_ENV uv run --no-sync python scripts/build_provenance_registry.py --emit
  env -u VIRTUAL_ENV uv run --no-sync python scripts/build_provenance_registry.py --check
  ```

- [x] **Step 5: Commit**

  ```bash
  git add docs/provenance docs/40-api/provenance/atmospheres.md \
    docs/plans/2026-07-11-spectra-v1-design.md \
    tests/validation/provenance_cards/test_atmosphere_spectra_sources.py
  git commit -m "docs: verify atmosphere spectral provenance"
  ```

### Task 2: Add canonical spectral types and statuses

**Files:**
- Create: `src/jaxstro/spectra/types.py`
- Create: `src/jaxstro/spectra/__init__.py`
- Modify: `src/jaxstro/__init__.py`
- Create: `tests/unit/test_spectra_types.py`
- Create: `tests/integration/test_spectra_ownership.py`

**Interfaces:**
- Produces: `SpectralCoordinate`, `SpectralSampling`, `SpectralSemantic`,
  `SpectrumStatusCode`, `SpectrumProvenance`, `SpectralAxis`, `Spectrum`,
  `SpectrumStatus`, and `SpectrumResult`.

- [x] **Step 1: Write failing invariant and PyTree tests**

  ```python
  axis = SpectralAxis.points(
      jnp.array([100.0, 200.0]),
      coordinate=SpectralCoordinate.WAVELENGTH,
      unit="nm",
  )
  spectrum = Spectrum(
      axis=axis,
      values=jnp.array([2.0, 3.0]),
      semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
      provenance=synthetic_provenance(),
  )
  leaves, treedef = jax.tree_util.tree_flatten(spectrum)
  rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
  assert len(leaves) == 2
  np.testing.assert_array_equal(rebuilt.axis.values, spectrum.axis.values)
  np.testing.assert_array_equal(rebuilt.values, spectrum.values)
  assert rebuilt.semantic is spectrum.semantic
  ```

  Also require constructors to reject non-1D, non-finite, non-positive, or
  non-increasing axes; wrong value lengths; bin edges without exactly one more
  edge than values; and semantics incompatible with the coordinate. A successful
  result rejects non-finite values; an unsuccessful fixed-shape JAX result uses
  NaN values plus a non-OK status. The ownership test identifies
  `jaxstro.spectra` as the new owner and rejects new imports from
  `jaxstro.atmospheres.spectra` outside the explicit migration allowlist.

- [x] **Step 2: Verify RED**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_spectra_types.py tests/integration/test_spectra_ownership.py
  ```

  Expected: import failure for `jaxstro.spectra`.

- [x] **Step 3: Implement frozen validated PyTrees**

  Use string enums for static semantic metadata and an integer enum with the
  ratified codes. Dynamic leaves are axis coordinates/edges and spectrum values;
  provenance is a frozen, hashable tuple-based static record. Provide explicit
  `points` and `bins` constructors so point and bin semantics cannot be confused.

- [x] **Step 4: Verify GREEN, JIT round-trip, Ruff, and mypy**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_spectra_types.py tests/integration/test_spectra_ownership.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check src/jaxstro/spectra tests/unit/test_spectra_types.py
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro/spectra
  ```

- [x] **Step 5: Commit**

  ```bash
  git add src/jaxstro/spectra src/jaxstro/__init__.py \
    tests/unit/test_spectra_types.py tests/integration/test_spectra_ownership.py
  git commit -m "feat: add canonical spectral data model"
  ```

### Task 3: Add exact spectral and geometric transformations

**Files:**
- Create: `src/jaxstro/spectra/transforms.py`
- Modify: `src/jaxstro/spectra/__init__.py`
- Create: `tests/unit/test_spectra_transforms.py`
- Create: `tests/validation/test_spectra_transform_gradients.py`

**Interfaces:**
- Consumes: Task 2 types and `jaxstro.constants.C_CGS`, converted exactly from
  cm/s to nm/s for the canonical wavelength coordinate.
- Produces: `to_frequency(axis)`, `to_wavelength(axis)`,
  `to_flux_nu(spectrum)`, `to_flux_lambda(spectrum)`,
  `surface_flux_to_luminosity(spectrum, radius_cm)`, and
  `surface_flux_to_observer_flux(spectrum, radius_cm, distance_cm)`.

- [x] **Step 1: Write failing analytic tests**

  Require `nu=c/lambda`, `F_nu=F_lambda*lambda**2/c`, reversed output ordering,
  exact round trips, `L_lambda=4*pi*R**2*F_lambda`, and
  `f_lambda=(R/d)**2*F_lambda`. Require incompatible semantic inputs to raise.

- [x] **Step 2: Verify RED**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_spectra_transforms.py \
    tests/validation/test_spectra_transform_gradients.py
  ```

- [x] **Step 3: Implement pure JAX transforms**

  Coordinate transforms operate only on `SpectralAxis`; density transforms
  change the axis, values, and semantic together and reverse samples so axes
  remain increasing. Point-density transforms update provenance operation
  history; binned density conversion is rejected until Task 4 supplies the
  conservative path. Geometry functions require positive CGS radii/distances
  and matching surface-flux semantics.

- [x] **Step 4: Verify GREEN and AD-vs-FD**

  Use the shared `jaxstro.testing.grad_audit` engine for wavelength, flux,
  radius, and distance derivatives away from zero.

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_spectra_transforms.py \
    tests/validation/test_spectra_transform_gradients.py
  ```

- [x] **Step 5: Commit**

  ```bash
  git add src/jaxstro/spectra tests/unit/test_spectra_transforms.py \
    tests/validation/test_spectra_transform_gradients.py
  git commit -m "feat: add spectral density transformations"
  ```

### Task 4: Add explicit spectral plans and resampling

**Files:**
- Create: `src/jaxstro/spectra/plan.py`
- Create: `src/jaxstro/spectra/resampling.py`
- Modify: `src/jaxstro/spectra/__init__.py`
- Create: `tests/unit/test_spectra_plan.py`
- Create: `tests/unit/test_spectra_resampling.py`
- Create: `tests/validation/test_spectra_remap_conservation.py`

**Interfaces:**
- Produces: `CoveragePolicy`, `SpectralPlan(target_axis, coverage_policy)`, and
  `resample_spectrum(spectrum, plan) -> SpectrumResult`.

- [x] **Step 1: Write failing identity, coverage, and conservation tests**

  Point samples use linear interpolation only inside source coverage. Bin
  averages reuse `jaxstro.numerics.conservative_remap_1d`. Identical axes return
  bit-identical values. Plans outside source coverage return
  `UNSUPPORTED_SPECTRAL_WINDOW`; they do not zero-fill or extrapolate.

- [x] **Step 2: Verify RED**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_spectra_plan.py tests/unit/test_spectra_resampling.py \
    tests/validation/test_spectra_remap_conservation.py
  ```

- [x] **Step 3: Implement plan validation and resampling**

  Reject point-to-bin and bin-to-point conversions unless the input carries the
  information required by the requested operation. Preserve provenance with an
  explicit `identity`, `linear-points`, or `conservative-bin-average` operation.

- [x] **Step 4: Verify GREEN under JIT and AD**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_spectra_plan.py tests/unit/test_spectra_resampling.py \
    tests/validation/test_spectra_remap_conservation.py
  ```

- [x] **Step 5: Commit**

  ```bash
  git add src/jaxstro/spectra tests/unit/test_spectra_plan.py \
    tests/unit/test_spectra_resampling.py \
    tests/validation/test_spectra_remap_conservation.py
  git commit -m "feat: add explicit spectral resampling plans"
  ```

### Task 4A: Add explicit point-resampling methods

**Status:** Complete on 2026-07-12. Detailed execution authority:
`docs/superpowers/plans/2026-07-12-spectral-resampling-methods.md`.

- [x] Add static `PointResamplingMethod.LINEAR` and
  `PointResamplingMethod.MONOTONE_CUBIC` plan metadata.
- [x] Route linear point spectra through
  `jaxstro.numerics.interpolation.interp1d`.
- [x] Route opt-in PCHIP spectra through
  `jaxstro.numerics.interpolation.monotone_cubic_interp`.
- [x] Preserve identity, fail-closed coverage, point/bin rejection, and
  conservative bin-average behavior.
- [x] Verify JIT, range preservation, provenance, and four smooth-path AD-vs-FD
  cases. Focused gate: 23 tests in 3.81 s (4.54 s wall). Combined spectra gate:
  56 tests in 3.19 s (3.94 s wall).

### Task 5: Add prepared rectilinear and simplex stencils

**Files:**
- Create: `src/jaxstro/spectra/stencils.py`
- Modify: `src/jaxstro/spectra/__init__.py`
- Create: `tests/unit/test_spectra_stencils.py`
- Create: `tests/validation/test_spectra_stencil_gradients.py`

**Interfaces:**
- Produces: `FluxInterpolation`, `PreparedRectilinearStencil`, and
  `PreparedSimplexStencil`, each with `evaluate(point) -> SpectrumResult`.

- [x] **Step 1: Write failing 2D/ND interpolation tests**

  Rectilinear tests cover exact vertices, midpoints, nonuniform axes, and
  missing-corner construction rejection. Simplex tests cover barycentric vertex
  recovery, interior interpolation, and outside-hull status. Invalid evaluation
  returns NaN values plus status rather than clamped spectra.

- [x] **Step 2: Verify RED**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_spectra_stencils.py \
    tests/validation/test_spectra_stencil_gradients.py
  ```

- [x] **Step 3: Implement fixed-shape JAX kernels**

  Rectilinear weights are products of per-axis lower/upper weights over a static
  corner-bit table. Simplex preparation stores the origin and inverse edge
  matrix host-side; JAX evaluation computes barycentric weights without topology
  search. Implement linear and positive-log policies. Leave amplitude-shape
  unavailable until Task 12 produces an accepted definition and evidence.

- [x] **Step 4: Verify GREEN, `jit`, `vmap`, and AD-vs-FD**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_spectra_stencils.py \
    tests/validation/test_spectra_stencil_gradients.py
  ```

- [x] **Step 5: Commit**

  ```bash
  git add src/jaxstro/spectra tests/unit/test_spectra_stencils.py \
    tests/validation/test_spectra_stencil_gradients.py
  git commit -m "feat: add prepared spectral stencils"
  ```

### Task 6: Add atmosphere query, product, adapter, and topology contracts

**Files:**
- Create: `src/jaxstro/atmospheres/params.py`
- Create: `src/jaxstro/atmospheres/products.py`
- Create: `src/jaxstro/atmospheres/topology.py`
- Create: `src/jaxstro/atmospheres/adapters.py`
- Modify: `src/jaxstro/atmospheres/__init__.py`
- Create: `tests/unit/test_atmospheres_topology.py`
- Create: `tests/unit/test_atmospheres_adapters.py`

**Interfaces:**
- Produces: `AtmosphereParams`, `AtmosphereQuery`, `ProductDescriptor`,
  `ArtifactReport`, `PreparationResult`, `PreparedAtmosphere`,
  `AtmosphereAdapter`, and `AtmosphereAdapterRegistry`.

- [x] **Step 1: Write failing registry and topology tests**

  Require exact product identity, complete-cell discovery before simplex use,
  deterministic simplex selection from an approved manifest, and structured
  `NO_COMPLETE_CELL`, `OUTSIDE_CONVEX_HULL`, `UNSUPPORTED_PLANE`, and
  `POLICY_NOT_VALIDATED` results.

- [x] **Step 2: Verify RED**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_atmospheres_topology.py \
    tests/unit/test_atmospheres_adapters.py
  ```

- [x] **Step 3: Implement host-side contracts**

  `PreparationResult` contains exactly one of `prepared` or an unsuccessful
  status. `PreparedAtmosphere` contains a prepared stencil, parameter-name
  tuple, fixed spectral plan, and hashable provenance; it owns no paths or open
  stores. The registry rejects duplicate product IDs.

- [x] **Step 4: Verify GREEN**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q \
    tests/unit/test_atmospheres_topology.py \
    tests/unit/test_atmospheres_adapters.py
  ```

- [x] **Step 5: Commit**

  ```bash
  git add src/jaxstro/atmospheres tests/unit/test_atmospheres_topology.py \
    tests/unit/test_atmospheres_adapters.py
  git commit -m "feat: add atmosphere adapter contracts"
  ```

### Task 7: Hard-cut NewEra preparation onto valid topology

**Files:**
- Modify: `src/jaxstro/atmospheres/newera.py`
- Modify: `tests/unit/test_atmospheres_newera_backend.py`
- Create: `tests/validation/test_newera_interpolation_policy.py`

**Interfaces:**
- Consumes: NewEra product record, `AtmosphereQuery`, `SpectralPlan`, topology
  discovery, and prepared stencils.
- Produces: product `newera-v3-lowres` with canonical factor `1e3`.

- [x] **Step 1: Write failing sparse-cell and canonical-unit tests**

  Add the observed sparse-cell geometry where independent `teff`/`logg`
  bracketing fails. Require preparation to choose a complete rectangle or an
  approved simplex, never return a false `ok`, and convert nm plus
  `W m^-2 nm^-1` to canonical values.

- [x] **Step 2: Verify RED**, then implement topology-first preparation and
  filesystem-free evaluation.

- [x] **Step 3: Verify GREEN**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync --extra data pytest -q \
    tests/unit/test_atmospheres_newera_backend.py \
    tests/validation/test_newera_interpolation_policy.py
  ```

- [x] **Step 4: Commit**

  ```bash
  git add src/jaxstro/atmospheres/newera.py \
    tests/unit/test_atmospheres_newera_backend.py \
    tests/validation/test_newera_interpolation_policy.py
  git commit -m "feat: harden NewEra spectral preparation"
  ```

### Task 8: Make BOSZ products explicit and source-correct

**Files:**
- Modify: `src/jaxstro/atmospheres/bosz.py`
- Modify: `tests/unit/test_atmospheres_bosz.py`
- Create: `tests/validation/test_bosz_interpolation_policy.py`

**Interfaces:**
- Produces explicit product IDs combining atmosphere, resolution, and product,
  including `bosz-2025-recomputed:ap:r10000:resam`.

- [ ] **Step 1: Write failing product-identity and conversion tests**

  Require library coverage, adapter selection, opened artifact, and provenance
  to agree on `ap`, `mp`, or `ms`. Verify resampled `F_lambda` gains the `10`
  per-angstrom-to-per-nm factor exactly once. Reject an original-resolution
  `H_lambda` artifact unless the `4*pi` conversion path is explicitly selected.

- [ ] **Step 2: Verify RED**, implement product-scoped preparation, and verify
  complete-cell/simplex selection.

- [ ] **Step 3: Verify GREEN**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync --extra data pytest -q \
    tests/unit/test_atmospheres_bosz.py \
    tests/validation/test_bosz_interpolation_policy.py
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add src/jaxstro/atmospheres/bosz.py tests/unit/test_atmospheres_bosz.py \
    tests/validation/test_bosz_interpolation_policy.py
  git commit -m "feat: make BOSZ spectral products explicit"
  ```

### Task 9: Implement the Sonora Diamondback adapter

**Files:**
- Modify: `src/jaxstro/atmospheres/sonora.py`
- Modify: `tests/unit/test_atmospheres_sonora.py`
- Create: `tests/validation/test_sonora_interpolation_policy.py`

**Interfaces:**
- Produces product IDs separating cloud label, metallicity, and C/O planes.
- Uses canonical factor `1e-6` from `W m^-2 m^-1` to
  `erg s^-1 cm^-2 nm^-1`.

- [ ] **Step 1: Write failing artifact, product-plane, and conversion tests**

  Require `cloud_label` and `c_o` to be explicit query/product constraints.
  Verify that a spectrum with native value `2` becomes canonical value `2e-6`.
  Reject wavelength interpolation as a claim about unresolved monochromatic
  features; only the explicit `SpectralPlan` remapping policy may change the
  released sampling.

- [ ] **Step 2: Verify RED**, implement Zarr/catalog loading and prepared output,
  then verify readback and topology behavior.

- [ ] **Step 3: Verify GREEN**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync --extra data pytest -q \
    tests/unit/test_atmospheres_sonora.py \
    tests/validation/test_sonora_interpolation_policy.py
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add src/jaxstro/atmospheres/sonora.py tests/unit/test_atmospheres_sonora.py \
    tests/validation/test_sonora_interpolation_policy.py
  git commit -m "feat: add Sonora Diamondback spectral adapter"
  ```

### Task 10: Implement distinct TLUSTY OSTAR/BSTAR adapters

**Files:**
- Modify: `src/jaxstro/atmospheres/tlusty.py`
- Modify: `tests/unit/test_atmospheres_tlusty.py`
- Create: `tests/validation/test_tlusty_interpolation_policy.py`

**Interfaces:**
- Produces three product IDs for OSTAR2002, BSTAR2006 `vturb=2`, and BSTAR2006
  `vturb=10` C/N variants.
- Converts `H_nu` to canonical `F_lambda` with
  `F_nu=4*pi*H_nu`, `lambda=c/nu`, and
  `F_lambda=F_nu*c/lambda**2`.

- [ ] **Step 1: Write failing ragged-grid and analytic conversion tests**

  Require preparation to load spectra from their catalog-recorded Zarr
  subgroups, resample each vertex to one explicit plan before parameter
  interpolation, and reject requests without common spectral coverage.

- [ ] **Step 2: Verify RED**, implement dataset-specific adapters, and preserve
  product identity and C/N flags.

- [ ] **Step 3: Verify GREEN**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync --extra data pytest -q \
    tests/unit/test_atmospheres_tlusty.py \
    tests/validation/test_tlusty_interpolation_policy.py
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add src/jaxstro/atmospheres/tlusty.py tests/unit/test_atmospheres_tlusty.py \
    tests/validation/test_tlusty_interpolation_policy.py
  git commit -m "feat: add TLUSTY spectral adapters"
  ```

### Task 11: Route AtmosphereLibrary through the adapter registry

**Files:**
- Modify: `src/jaxstro/atmospheres/library.py`
- Modify: `src/jaxstro/atmospheres/coverage.py`
- Modify: `tests/unit/test_atmospheres_library.py`
- Modify: `tests/unit/test_atmospheres_coverage.py`

**Interfaces:**
- Changes `AtmosphereLibrary.prepare(query) -> PreparationResult` and keeps
  `spectrum(query)` as a prepare-then-evaluate convenience path.

- [ ] **Step 1: Write failing selection/execution identity tests**

  Require candidate ranking to include explicit product IDs and the selected
  adapter descriptor. Expected gaps return structured results rather than
  `RuntimeError`. Corrupt or mismatched artifacts still raise.

- [ ] **Step 2: Verify RED**, replace backend conditionals with registry lookup,
  and verify all synthetic fixtures.

- [ ] **Step 3: Verify GREEN**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync --extra data pytest -q \
    tests/unit/test_atmospheres_library.py \
    tests/unit/test_atmospheres_coverage.py
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add src/jaxstro/atmospheres/library.py src/jaxstro/atmospheres/coverage.py \
    tests/unit/test_atmospheres_library.py tests/unit/test_atmospheres_coverage.py
  git commit -m "feat: route atmosphere selection through adapters"
  ```

### Task 12: Ratify flux-interpolation policies with holdout evidence

**Files:**
- Create: `src/jaxstro/testing/spectral_validation.py`
- Create: `scripts/validate_atmosphere_interpolation.py`
- Create: `tests/unit/test_spectral_validation.py`
- Create: `tests/validation/test_atmosphere_holdouts.py`
- Create: `docs/validation/atmosphere-interpolation.json`

**Interfaces:**
- Produces deterministic metrics for linear and positive-log policies and a
  versioned accepted-policy manifest consumed by adapters.

- [ ] **Step 1: Write failing synthetic holdout tests**

  Metrics are finite-bin median relative error, 95th percentile relative error,
  maximum log-flux error on positive support, and integrated-flux relative
  error. The winner must beat the alternative on the declared primary metric
  without exceeding secondary error ceilings.

- [ ] **Step 2: Verify RED**, implement deterministic evaluator and CLI, then run
  bounded real-artifact holdouts for each enabled product.

- [ ] **Step 3: Record measured policies and exclusions**, never invented
  thresholds. If neither policy meets its declared ceiling, mark the product
  `POLICY_NOT_VALIDATED`.

- [ ] **Step 4: Verify GREEN**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync --extra data pytest -q \
    tests/unit/test_spectral_validation.py \
    tests/validation/test_atmosphere_holdouts.py
  env -u VIRTUAL_ENV uv run --no-sync --extra data python \
    scripts/validate_atmosphere_interpolation.py --check
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add src/jaxstro/testing/spectral_validation.py \
    scripts/validate_atmosphere_interpolation.py tests/unit/test_spectral_validation.py \
    tests/validation/test_atmosphere_holdouts.py \
    docs/validation/atmosphere-interpolation.json
  git commit -m "test: ratify atmosphere interpolation policies"
  ```

### Task 13: Add real-artifact, AD, interoperability, and performance gates

**Files:**
- Modify: `tests/validation/test_atmospheres_local_artifacts.py`
- Replace: `tests/validation/test_atmospheres_spectra.py`
- Create: `tests/integration/test_spectra_consumer_contract.py`
- Create: `scripts/benchmark_spectra.py`
- Create: `tests/unit/test_benchmark_spectra_script.py`
- Create: `docs/validation/spectra-performance.json`

**Interfaces:**
- Produces bounded acceptance and benchmark artifacts with separate preparation,
  first-JIT, cached evaluation, batched evaluation, and peak-memory fields.

- [ ] **Step 1: Write failing acceptance and benchmark-schema tests**

  Cover one supported and one intentionally unsupported request per product,
  finite canonical values, fixed shapes, no filesystem reads after preparation,
  `jit`, `vmap`, AD-vs-FD away from boundaries, and generic consumer imports.

- [ ] **Step 2: Verify RED**, implement the benchmark CLI, and run bounded cases.

- [ ] **Step 3: Verify GREEN with exact timing output**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync --extra data pytest -q \
    tests/validation/test_atmospheres_local_artifacts.py \
    tests/validation/test_atmospheres_spectra.py \
    tests/integration/test_spectra_consumer_contract.py \
    tests/unit/test_benchmark_spectra_script.py
  env -u VIRTUAL_ENV uv run --no-sync --extra data python \
    scripts/benchmark_spectra.py --check
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add tests/validation tests/integration/test_spectra_consumer_contract.py \
    tests/unit/test_benchmark_spectra_script.py scripts/benchmark_spectra.py \
    docs/validation/spectra-performance.json
  git commit -m "test: add spectra acceptance and performance gates"
  ```

### Task 14: Complete the hard cutover and documentation

**Files:**
- Delete: `src/jaxstro/atmospheres/spectra.py`
- Modify: `src/jaxstro/atmospheres/__init__.py`
- Modify: `src/jaxstro/atmospheres/overlap.py`
- Delete: `tests/unit/test_atmospheres_spectra.py`
- Modify: `tests/unit/test_atmospheres_overlap.py`
- Modify: `tests/integration/test_spectra_data_architecture_docs.py`
- Modify: `docs/20-architecture/spectra-data-architecture.md`
- Modify: `docs/20-architecture/atmosphere-capabilities.md`
- Modify: `docs/20-architecture/index.md`
- Modify: `docs/40-api/index.md`
- Modify: `docs/50-howto/tlusty-data-processing.md`
- Create: `docs/50-howto/query-atmosphere-spectra.md`
- Modify: `docs/60-validation/index.md`
- Modify: `docs/audits/2026-07-11-docs-currency-audit.md`
- Modify: `STATUS.md`

**Interfaces:**
- Canonical imports become `jaxstro.spectra` for generic types and
  `jaxstro.atmospheres` for queries/adapters. No old alias remains.

- [ ] **Step 1: Re-run the downstream import audit**

  Search Fluxax, Stellax, Hydrax, Radax, Startrax, Progenax, and Gravax. Migrate
  real imports in writable in-scope repositories before deletion; record planned
  consumer contracts for repositories with no implementation import.

- [ ] **Step 2: Write failing no-legacy and executable-doc tests**, then delete
  the old module and migrate all local imports.

- [ ] **Step 3: Update learner-facing docs**

  Explain surface versus observer flux, `F_lambda` versus `F_nu`, why factors of
  `4*pi` differ among archives, point versus bin sampling, topology boundaries,
  structured failure statuses, and the Fluxax handoff. All snippets execute.

- [ ] **Step 4: Run focused and full gates with timings**

  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit tests/integration \
    tests/validation/test_atmospheres_spectra.py \
    tests/validation/test_atmospheres_local_artifacts.py
  env -u VIRTUAL_ENV uv run --no-sync ruff check src tests scripts
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check src tests scripts
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
  bash scripts/check_docs.sh
  bash scripts/check.sh
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add -A src/jaxstro tests docs STATUS.md
  git commit -m "feat: complete spectra v1 hard cutover"
  ```

## Plan self-review

- **Spec coverage:** Tasks 1--5 implement the generic substrate; Tasks 6--11
  implement fail-closed family preparation; Task 12 supplies evidence-selected
  policies; Task 13 supplies AD, performance, real-artifact, and consumer
  evidence; Task 14 performs deletion and pedagogy only after migration.
- **No hidden universal engine:** stencils support the shapes required by the
  four families but do not perform arbitrary runtime triangulation.
- **Type consistency:** adapters consume `AtmosphereQuery` and return
  `PreparationResult`; only `PreparedAtmosphere.evaluate` returns
  `SpectrumResult`.
- **Cutover consistency:** `jaxstro.atmospheres.spectra` receives no new behavior
  and is deleted after consumer migration, with no alias.
- **Scientific stop rule:** a backend remains unavailable if its source semantic,
  topology, common spectral coverage, or interpolation policy does not pass its
  explicit gate.
