# Jaxstro spectra v1 design

**Status:** Anna-approved in seven design checkpoints on 2026-07-11, then
implementation-preflight reviewed against the live code, local artifacts, and
primary product documentation.  This document is the implementation authority
for the spectra-completion program; implementation remains approval-gated per
slice.

**Supersedes:** The narrow ownership and interpolation decisions in
`2026-06-20-spectra-hybrid-architecture.md`.  That plan remains useful as a
historical record of the first NewEra implementation, but its
`jaxstro.atmospheres.spectra` ownership and rectangular bilinear-only design are
not the v1 target.

## Goal

Complete Jaxstro's spectral foundation as a trustworthy, versatile interface
for NewEra, BOSZ, Sonora, and TLUSTY atmosphere products.  Completion means
numerical and interoperability completeness: source-backed semantics,
fail-closed parameter topology, quantified interpolation and remapping errors,
JAX/AD behavior, and a stable boundary for downstream ecosystem packages.  It
does not mean empirical validation of the atmosphere models against nature.

## Grounded motivation

Jaxstro already has a useful atmosphere ingestion and prepared-interpolation
substrate, but the end-to-end contract is not yet production-ready:

- NewEra selection can identify independently bounding `teff` and `logg`
  coordinates that do not form a complete local cell.  Preparation then fails
  after selection appeared successful.
- BOSZ coverage can summarize multiple products (`ap`, `mp`, and `ms`) while
  spectrum evaluation opens the default `ap` product.  Selection and execution
  therefore do not yet share one product identity.
- The current `PreparedSpectralGrid` assumes a rectilinear two-dimensional
  bilinear stencil, which cannot honestly represent all four families.
- Generic spectral representation currently lives under
  `jaxstro.atmospheres`, even though future consumers including Fluxax,
  Stellax, Hydrax, and Radax need a domain-neutral spectrum contract.

These observations motivate a clean ownership cutover rather than incremental
aliases around the current API.

## Ratified boundaries

This program implements **A1**: a four-family, plugin-ready spectral foundation.
A later **B** program may generalize the proven machinery into a universal
arbitrary-dimensional engine.  Observable rendering remains **C**, owned by
Fluxax.

Jaxstro owns:

- canonical spectral coordinates, values, semantics, and provenance;
- exact wavelength/frequency and `F_lambda`/`F_nu` transformations;
- exact geometric surface-flux, luminosity, and observer-flux transformations;
- conservative spectral remapping;
- prepared rectilinear and approved simplex interpolation machinery;
- atmosphere-family archive interpretation and preparation.

Fluxax owns extinction, filter and throughput integration, instrumental
response, counts, magnitudes, PSFs, images, catalog/image likelihoods, and other
rendered observables.  No such functionality moves into Jaxstro.

## Architecture and ownership

The public architecture has two layers:

```text
jaxstro.spectra
├── canonical spectral data model
├── wavelength/frequency coordinate transforms
├── F_lambda/F_nu conversions
├── conservative spectral resampling
├── prepared parameter-space interpolation machinery
├── structured statuses and provenance
└── JAX-compatible prepared representations

jaxstro.atmospheres
├── atmosphere parameters and queries
├── installed-grid catalogs
├── grid topology discovery
├── interpolation-policy selection
├── NewEra adapter
├── BOSZ adapter
├── Sonora adapter
└── TLUSTY adapter
```

`jaxstro.spectra` is domain-neutral.  It knows how to represent, transform,
resample, interpolate, validate, differentiate, and batch spectra.  It does not
know the meanings of effective temperature, surface gravity, metallicity,
filters, detectors, or stellar evolution.

`jaxstro.atmospheres` translates released atmosphere products into the generic
spectral substrate.  Its adapters own native parameter conventions, grid
discovery, source-backed flux semantics, topology constraints, and validated
interpolation policies.

The runtime boundary is:

```text
AtmosphereQuery
    -> host-side selection and preparation
    -> PreparedAtmosphere
    -> filesystem-free JAX evaluation
    -> SpectrumResult
```

File access, catalog search, product selection, topology discovery, and
irregular-grid triangulation occur before the prepared boundary.
`PreparedAtmosphere` is a fixed-shape JAX PyTree suitable for `jit`, `vmap`,
differentiation, and repeated evaluation.

## Canonical data model

The canonical **atmosphere-output boundary** is monotonically increasing
wavelength in nm plus surface `F_lambda` in
`erg s^-1 cm^-2 nm^-1`, with mandatory physical semantics and retained native
provenance.  This is an explicit spectral convention, not a quantity-system
migration.  A generic `Spectrum` may also hold an explicitly transformed
frequency/`F_nu`, luminosity-density, or observer-flux-density representation;
those are valid spectra but are not canonical atmosphere-adapter outputs.

Conceptually:

```python
Spectrum(
    spectral_axis=SpectralAxis(...),
    values=...,
    semantic=SpectralSemantic.SURFACE_FLUX_DENSITY_WAVELENGTH,
    provenance=SpectrumProvenance(...),
)
```

`SpectralAxis` records:

- monotonically increasing coordinates;
- wavelength or frequency coordinate kind;
- explicit physical unit;
- point-sampled or bin-integrated/bin-averaged meaning;
- optional bin edges and resolution metadata;
- a stable shape suitable for JAX transformations.

Atmosphere adapters return emergent surface `F_lambda` in those canonical
CGS-compatible units.
The native archive convention and every conversion factor, including any
required pi or 4-pi factor, must be verified from primary documentation.  If
that verification is unavailable, the product is unavailable; Jaxstro does not
infer the convention from memory or filenames.

Generic transformations are explicit pure functions rather than mutable
methods:

```python
to_wavelength(axis)
to_frequency(axis)
to_flux_lambda(spectrum)
to_flux_nu(spectrum)
surface_flux_to_luminosity(spectrum, radius=...)
surface_flux_to_observer_flux(spectrum, radius=..., distance=...)
```

Coordinate functions operate on `SpectralAxis`.  Density functions transform a
complete `Spectrum`, changing the axis, values, and semantic together so the API
cannot represent `F_lambda` on a frequency axis or `F_nu` on a wavelength axis.
Point-density conversion is exact at the tabulated samples.  Bin-aware density
conversion requires the explicit conservative-remapping path rather than a
center-point Jacobian approximation.

The geometric transforms accept only compatible surface-flux semantics.
`surface_flux_to_luminosity` applies `4*pi*radius**2`; the observer transform
applies `(radius/distance)**2`.  Their exactness claim is conditional on the
declared spherical, isotropic-emission geometry.  They do not encode extinction,
beaming, lensing, cosmological redshift, or instrumental response.

## Query-scoped spectral planning

Every preparation receives an explicit `SpectralPlan` whose required
`target_axis` fixes the output coordinate values or bin edges, unit,
point-versus-bin semantics, and output shape.  Resolution may be recorded as
metadata, but it does not stand in for an actual sampling grid:

```python
SpectralPlan(
    target_axis=SpectralAxis(...),
    coverage_policy="intersection",
)
```

Identical grids use an exact no-remap path.  Bin-aware transformations use
conservative remapping.  Intersection-only coverage is the default: the system
does not extrapolate, invent values outside archive coverage, or silently fill
missing wavelengths.

## Parameter topology and interpolation

Selection is based on a valid interpolation region, not independently bounding
each coordinate.  Preparation follows this sequence:

```text
query parameters
    -> select family and explicit product
    -> identify a supported parameter plane
    -> inspect local grid topology
    -> construct a validated stencil
    -> load only the required spectra
    -> apply the family interpolation policy
    -> prepare the requested spectral grid
    -> emit a fixed-shape PreparedAtmosphere
```

Topology is policy-driven and fail-closed:

1. Use multilinear interpolation when every required hyperrectangle corner is
   present.
2. Use simplex interpolation only in family/product regions where that policy
   has been explicitly enabled and validated.
3. Otherwise return `NO_COMPLETE_CELL` or `OUTSIDE_CONVEX_HULL`.

There is no silent nearest-neighbor substitution, arbitrary corner dropping,
cross-family interpolation, or parameter extrapolation.

Topology discovery and simplex selection are host-side and discrete.  The
prepared stencil carries fixed-shape vertices, spectra, weight metadata, and
validity bounds.  Weight and spectrum evaluation are JAX operations and remain
differentiable within the selected region.  Crossing a cell boundary requires
preparation again; the API does not claim differentiability through discrete
topology selection.

Parameter coordinates may be normalized internally for numerical conditioning,
but public queries retain physical values and units.  Unsupported dimensions or
parameter combinations return `UNSUPPORTED_PLANE` rather than being silently
projected onto a lower-dimensional grid.

## Flux-interpolation policy

Flux interpolation is selected per family and product from evidence, not by a
hidden global switch.  Candidate policies are:

- linear flux interpolation;
- positive log-flux interpolation;
- amplitude plus normalized-shape interpolation.

Leave-one-out or withheld-node validation determines the accepted policy and
its supported parameter regions.  The chosen policy, validation record, and
known exclusions are retained in spectrum provenance.  A product without an
accepted policy returns `POLICY_NOT_VALIDATED`.

## Family-adapter contract

Each atmosphere family implements one internal adapter contract:

```python
class AtmosphereAdapter:
    def describe_product(...) -> ProductDescriptor: ...
    def validate_artifact(...) -> ArtifactReport: ...
    def discover_topology(...) -> GridTopology: ...
    def prepare(...) -> PreparationResult: ...
```

Each enabled product declares its native parameter axes, supported planes,
spectral coordinate and flux convention, canonical conversion, coverage and
resolution behavior, artifact schema and integrity evidence, topology policy,
flux-interpolation policy, and known exclusions.

Family-specific obligations are:

- **NewEra:** replace independent coordinate bracketing with complete-cell or
  validated-simplex selection for its locally sparse geometry.
- **BOSZ:** make `ap`, `mp`, and `ms` explicit products so coverage, selection,
  preparation, and provenance always refer to the same product.
- **Sonora:** preserve product-specific parameter domains and spectral
  conventions rather than presenting all subgrids as one homogeneous archive.
- **TLUSTY:** retain each physical grid as a distinct product with its own
  supported plane, provenance, and interpolation validation; incompatible
  families are never combined into a stencil.

An adapter registry removes family conditionals from the library selection
path.  External registration is allowed only when an adapter satisfies the same
descriptor, artifact-validation, provenance, and preparation contracts.
"Plugin-ready" does not mean "unverified."

## Provenance

Every successful spectrum records enough machine-readable provenance to
reconstruct its numerical production:

```python
SpectrumProvenance(
    family=...,
    product=...,
    dataset_version=...,
    artifact_digest=...,
    source_references=...,
    native_semantic=...,
    canonical_conversion=...,
    parameter_stencil=...,
    topology_policy=...,
    flux_interpolation_policy=...,
    spectral_remap_policy=...,
    validation_record=...,
)
```

Provenance distinguishes primary-source facts from Jaxstro processing
decisions.  Native flux meaning is a source claim; selecting log-flux
interpolation from a holdout study is a Jaxstro validation result.  Neither is
an undocumented string or code-comment inference.

## Failure model

Expected scientific limitations produce structured outcomes:

- `NO_DATASET`
- `NO_COVERAGE`
- `NO_COMPLETE_CELL`
- `OUTSIDE_CONVEX_HULL`
- `UNSUPPORTED_PLANE`
- `UNSUPPORTED_SPECTRAL_WINDOW`
- `BACKEND_UNAVAILABLE`
- `POLICY_NOT_VALIDATED`

Host-side result objects include status, concise diagnostic context, and an
optional prepared value.  JAX evaluation returns compact array-compatible
status codes.  Status definitions are stable and documented.

Exceptions are reserved for corrupt artifacts, digest or schema mismatches,
non-monotonic coordinates, incompatible shapes, forbidden non-finite values,
broken prepared-object invariants, and internal errors.  These defects must not
masquerade as ordinary scientific coverage gaps.

Axes are always finite.  Successful results always contain finite values.
Unsuccessful fixed-shape JAX evaluations carry a NaN value payload together
with a non-OK status so an unsupported request cannot be mistaken for a clamped,
zero-filled, or otherwise fabricated spectrum.

Preparation is transactional.  A successful `PreparedAtmosphere` guarantees
that required artifacts were validated and loaded, shapes are fixed,
provenance is complete, and evaluation is filesystem-free.  Partial preparation
is never success.

## Validation and completion gates

Every enabled product must pass the following evidence program:

1. **Artifact integrity:** deterministic catalogs, schema checks, hashes where
   available, monotonic axes, expected shapes, and archive/readback tests.
2. **Physical semantics:** primary-source evidence for native coordinates, flux
   meaning, units, and all canonical conversion factors.
3. **Parameter interpolation:** withheld-node reconstruction across interior,
   boundary, sparse, and scientifically important parameter regions.
4. **Spectral remapping:** bin-conservation, analytic-function, exact-grid
   identity, and coverage-edge tests.
5. **Numerical behavior:** finite-value and dtype checks, scaling tests,
   policy comparisons, and recorded error budgets.
6. **Differentiability:** AD gradients checked against scaled finite differences
   away from topology boundaries; boundary nondifferentiability is documented.
7. **JAX operation:** `jit`, `vmap`, PyTree round trips, fixed shapes, and
   filesystem-free repeated evaluation.
8. **Performance:** separately report preparation, first-JIT, cached evaluation,
   batched evaluation, and peak memory.  CI uses bounded representative cases,
   not full-archive processing or downloads.
9. **Interoperability:** consumer contract tests for Fluxax, Stellax, Hydrax,
   and Radax without moving their domain physics into Jaxstro.
10. **Real-artifact acceptance:** representative supported and intentionally
    unsupported requests for NewEra, BOSZ, Sonora, and every enabled TLUSTY
    product.

Cross-family comparisons are diagnostics, not equality assertions.  Each
enabled product publishes a compact validation record with tested regions,
tolerances, selected policies, exclusions, and limitations.  Insufficiently
verified products remain unavailable.

## Hard-cutover policy

Generic spectral types move from `jaxstro.atmospheres` to `jaxstro.spectra`.
Before removal, implementation audits current downstream imports and migrates
real consumers before deleting the old owner.  The repository will not retain
aliases, a parallel legacy namespace, or two canonical spectral owners.  The
new owner may coexist temporarily while consumers are migrated, but the old
module cannot receive new behavior and cannot affect the canonical path.  This
cutover does not reopen the deferred quantity-system redesign.

## Implementation slices

Each slice remains separately approval-gated and ends with tests,
documentation, and a bounded scientific claim:

1. Inventory downstream imports, verify the four native flux conventions from
   primary product sources, and freeze the ownership contract.
2. Add canonical spectral types, semantics, statuses, and invariants.
3. Add `SpectralPlan` and fixed-shape spectral-axis construction.
4. Add coordinate and `F_lambda`/`F_nu` transformations.
5. Add conservative point/bin resampling and coverage rules.
6. Build generic rectilinear prepared stencils.
7. Add approved simplex topology and differentiable weight evaluation.
8. Introduce adapter and product-registry contracts.
9. Hard-cut NewEra onto valid complete-cell/simplex preparation.
10. Hard-cut BOSZ with explicit `ap`/`mp`/`ms` ownership.
11. Hard-cut Sonora while preserving product-specific domains.
12. Hard-cut enabled TLUSTY grids as distinct products.
13. Run per-family holdout studies and ratify interpolation policies.
14. Add provenance records, validation manifests, and error budgets.
15. Add JAX/AD, performance, memory, and real-artifact acceptance gates.
16. Migrate downstream consumers, then remove old spectral owners in the same
    verified hard-cutover slice.
17. Complete API documentation, student-facing tutorials, and release evidence.

Slices may combine if implementation evidence demonstrates that the boundary is
smaller, but no slice is complete from scaffolding alone.  The planning estimate
is 12--17 coherent slices or roughly 3--5 focused engineering weeks.  Archive
provenance and irregular-grid validation are the largest uncertainties.

## Non-goals

- Extinction, filters, detector response, counts, magnitudes, PSFs, or rendering.
- Interpolation between atmosphere families.
- Empirical validation of atmosphere-model physics.
- A universal arbitrary-dimensional spectral engine in this program.
- A quantity-system overhaul.
- Compatibility aliases for the atmosphere-owned spectral API.

## Approval and next gate

Anna approved the architecture, canonical data model, topology and
interpolation, family adapters, provenance and failure model, validation bar,
and implementation sequence in separate checkpoints on 2026-07-11.  After this
document is reviewed, the next permitted action is to use
`superpowers:writing-plans` to turn these slices into an implementation plan.
No implementation begins from this design document alone.

## Implementation-preflight corrections

The preflight review found and repaired three ambiguities before code work:

1. "Canonical" now refers specifically to the atmosphere-output boundary;
   explicitly transformed spectra remain valid without pretending to be adapter
   outputs.
2. Canonical units are exact (`nm` and
   `erg s^-1 cm^-2 nm^-1`), and the geometric assumptions behind luminosity and
   observer-flux transforms are stated.
3. `SpectralPlan` now owns an explicit fixed-shape target axis instead of an
   underspecified `sampling` placeholder, and consumer migration precedes
   deletion during the hard cutover.

Primary product documentation resolves the native semantic gates for planning:

- NewEra low-resolution products are wavelength in nm and `F_lambda` in
  `W m^-2 nm^-1`; the HDF5 products document `F_lambda` in
  `erg s^-1 cm^-2 cm^-1`.  The low-resolution conversion to Jaxstro's
  canonical per-nm density is `1e3`.
- BOSZ 2024 original spectra provide Eddington first moment `H_lambda`; surface
  flux is `4*pi*H_lambda`, while the lower-resolution products contain the
  already resampled flux and continuum columns.
- Sonora Diamondback documents top-of-atmosphere radiation flux
  `F = 4*pi*H` in `W m^-2 m^-1` on a wavelength axis.  The archive description
  prints a nu subscript despite the per-metre unit; Jaxstro records this source
  inconsistency and follows the dimensional unit and wavelength coordinate.
  The canonical per-nm conversion factor is `1e-6`.
- TLUSTY OSTAR2002 and BSTAR2006 SEDs provide surface Eddington flux `H_nu` in
  `erg s^-1 cm^-2 Hz^-1`; surface flux is `4*pi*H_nu`.

These claims must be encoded with source locators in the provenance registry and
tested against artifact metadata before a backend is marked available.
