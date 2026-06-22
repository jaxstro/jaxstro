---
title: API reference
description: >-
  The public module surface of jaxstro 0.1.0 — units, constants, astrometry,
  coords, numerics, spatial, params, atmospheres, testing, jaxconfig — and what
  each provides.
---

This is the lookup door. It enumerates the public modules of jaxstro 0.1.0 and the
symbols each exports, with a pointer back to the theory that justifies them. Import
the top-level package and reach modules as attributes:

The API is astro-first but intentionally science-general. A public symbol belongs
here when it is reusable below domain packages, has explicit unit or boundary
semantics, and can point to tests or validation evidence for the behavior it
claims.

```python
import jaxstro
from jaxstro import atmospheres, constants, units, numerics, coords, geometry, spatial, params, testing
from jaxstro.jaxconfig import enable_high_precision
```

The committed surface for 0.1.0 is below. `units`, `constants`, `astrometry`,
`coords`, and `jaxconfig` are **stable**; `numerics`, `spatial`, `params`, and
`atmospheres` are **stable-but-evolving**; `testing` is a **public, semi-stable**
utility. There is no private or experimental tier at release.

```{list-table} Public modules
:header-rows: 1
:label: tbl-modules

* - Module
  - Provides
* - `jaxstro.units`
  - `UnitSystem` dataclass with a `.G` property; named systems (`CGS`,
    `ASTRO_STELLAR`/`solar`, `ASTRO_DYNAMICAL`/`stellar`, `ASTRO_PLANETARY`/`binary`)
    and aliases; `DEFAULT` (= CGS) re-exported as `jaxstro.DEFAULT_UNITS`;
    `PhotometricUnits`.
* - `jaxstro.constants`
  - CGS physical constants from CODATA 2018 and IAU 2015, each with a provenance
    comment, plus photometric constants (Oke & Gunn 1983).
* - `jaxstro.astrometry`
  - Astrometric constants (e.g. `K_PROPER_MOTION`, mas/radian conversions).
* - `jaxstro.coords`
  - Coordinate transforms — sky-tangent, galactic/equatorial, spherical, parallax.
* - `jaxstro.geometry`
  - Generic vector geometry — normalization, angular distances, rotations,
    quaternions, and rigid transforms with explicit composition order.
* - `jaxstro.numerics`
  - Differentiable numerical utilities: stats, interpolation, root-finding,
    integration (incl. `cumulative_trapz` + quadrature factory + `newton_ppf`),
    B-spline basis/evaluation, checks, compensated summation, linear algebra,
    autodiff products, distribution kernels, optimization helpers, fixed-step
    ODE helpers, linear operators, RNG, sampling.
* - `jaxstro.spatial`
  - Morton (Z-order) encoding/decoding, grid binning, neighbor-candidate gathering.
* - `jaxstro.params`
  - Equinox-only PyTree↔flat-vector bridge (`Parameterization`) plus a bijector
    registry (Identity/Exp/Softplus/Sigmoid) for unconstrained-space inference.
* - `jaxstro.atmospheres`
  - Catalog-first local atmosphere coverage; host-side processed-artifact
    loading; JAX-ready spectra types (`AtmosphereParams`, `Spectrum`,
    `SpectrumResult`, `SpectrumStatus`, `PreparedSpectralGrid`); implemented
    `NewEraBackend` and `BoszBackend`; source-preserving Sonora/TLUSTY metadata
    and conversion support.
* - `jaxstro.testing`
  - The grad-audit engine (`audit_entry_point`, `Case`, `AuditResult`, `EdgeConfig`)
    plus public finite-difference diagnostics (`finite_difference_grad`,
    `finite_difference_jacobian`, `compare_gradients`, `compare_jacobians`,
    `check_directional_derivative`).
* - `jaxstro.jaxconfig`
  - `enable_high_precision()` — turns on float64 and highest matmul precision.
```

## Selected modules

### `jaxstro.constants`

CGS constants with sourced values. A few that downstream packages rely on:

```{list-table} Sampled constants (CGS)
:header-rows: 1
:label: tbl-constants

* - Symbol
  - Value
  - Source
* - `G_CGS`
  - $6.67430\times10^{-8}\ \mathrm{cm^3\,g^{-1}\,s^{-2}}$
  - CODATA 2018
* - `K_B`
  - $1.380649\times10^{-16}\ \erg\,\mathrm{K}^{-1}$
  - CODATA 2018 (exact)
* - `SIGMA_SB`
  - $5.670374419\times10^{-5}\ \erg\,\mathrm{cm^{-2}\,s^{-1}\,K^{-4}}$
  - CODATA 2018
* - `A_RAD`
  - $7.565733250\times10^{-15}\ \erg\,\mathrm{cm^{-3}\,K^{-4}}$
  - Derived $4\sigma_\mathrm{SB}/c$ (CODATA 2018)
* - `SIGMA_T`
  - $6.6524587321\times10^{-25}\ \mathrm{cm^2}$
  - CODATA 2018 (Thomson cross-section)
* - `MSUN_G`
  - $1.9884\times10^{33}\ \mathrm{g}$
  - IAU 2015 nominal
* - `AB_ZEROPOINT_JY`
  - $3631\ \mathrm{Jy}$
  - Oke & Gunn 1983
```

Provenance discipline — every constant cites its authority — is principle
[9](../10-theory/index.md#p9-correctness).

### `jaxstro.numerics.rootfinding`

`bracket_expand`, `bisect`, `bisect_many`, `newton`, `newton_with_grad`,
`newton_ppf`, and `monotone_inverse_interp`. Behavior, the differentiability
caveats, and when to use each are in [](../10-theory/rootfinding.md).

### `jaxstro.numerics.interpolation`

`interp1d(...)` is the clamped linear baseline. `cubic_hermite_interp(...)`
evaluates cubic Hermite interpolation from supplied node derivatives;
`pchip_slopes(...)` constructs shape-preserving slopes; `monotone_cubic_interp(...)`
combines those slopes with the Hermite evaluator; and
`MonotoneTabulatedFunction1D` wraps a monotone table as a PyTree. The method page
is [](../10-theory/interpolation.md).

### `jaxstro.numerics.regular_grid`

`regular_grid_interp(points, values, xi, boundary="clamp")` performs static-rank
multilinear interpolation on a tensor-product grid. `bilinear_interp(...)` and
`trilinear_interp(...)` are convenience wrappers. Grid axes occupy the leading
dimensions of `values`; trailing dimensions are treated as vector-valued payloads.
Boundary policy is explicit: clamp, fill, or reject. The method page is
[](../10-theory/regular-grid.md).

### `jaxstro.numerics.grids`

`log_grid(...)` and `geometric_bin_edges(...)` construct positive logarithmic
grids; `bin_centers(...)` and `geometric_bin_centers(...)` compute arithmetic or
geometric centers; `conservative_rebin(...)` redistributes integrated bin totals
onto new edges while preserving total overlap. The method page is
[](../10-theory/grids.md).

### `jaxstro.numerics.integration`

`trapz`, `cumulative_trapz` (dx-outside uniform path), `simpson`, and
`cumulative_simpson` panel-endpoint sums. The trapezoid ordering choice is in
[](../10-theory/cumulative-trapz.md); fixed-node and Simpson-panel rules are in
[](../10-theory/quadrature.md).

### `jaxstro.numerics.quadrature`

`gauss_legendre_nodes(n)`, `gauss_laguerre_nodes(n)`,
`gauss_hermite_nodes(n)` (probabilists'), `clenshaw_curtis_nodes(n)`,
`hermite_e_basis`, and Hermite expansion coefficients. Nodes are generated once
on the host and frozen to constants; gradients flow through the integrand
values, not the nodes (principle [7](../10-theory/index.md#p7-quadrature)).
The method page is [](../10-theory/quadrature.md).

### `jaxstro.numerics.splines`

`bspline_basis(knots, x, degree=3)` evaluates all basis functions;
`bspline_design_matrix(knots, x, degree=3)` gives the explicit sample-matrix
spelling; `bspline_eval(knots, coeffs, x, degree=3, axis=-1)` contracts basis
values with supplied coefficients; `bspline_eval_deboor(...)` evaluates the same
spline through de Boor recursion; `bspline_derivative(...)`,
`bspline_antiderivative(...)`, and `bspline_integral(...)` cover calculus
helpers; `bspline_roughness_penalty(...)` supplies an integrated squared
derivative penalty; `fit_bspline_lstsq(...)` fits coefficients for fixed knots;
`adaptive_open_uniform_knots(...)` places interior knots at sample quantiles;
`tensor_product_design_matrix(...)` builds row-wise tensor-product designs; and
`BSpline1D` wraps knots and coefficients as a PyTree. The method page is
[](../10-theory/bsplines.md).

### `jaxstro.numerics.linear_algebra`

`weighted_lstsq(...)` solves ordinary or weighted dense least-squares problems;
`qr_solve(...)` and `svd_solve(...)` expose explicit full-rank and truncated-SVD
solve policies; `covariance_matrix(...)`, `correlation_from_covariance(...)`, and
`correlation_matrix(...)` provide finite covariance/correlation helpers; and
`is_positive_definite(...)`, `add_diagonal_jitter(...)`, and
`positive_definite_jitter(...)` cover small dense positive-definite diagnostics.
The method page is [](../10-theory/linear-algebra.md).

### `jaxstro.numerics.distributions`

`normal_logpdf(...)`, `normal_cdf(...)`, and `normal_ppf(...)` cover normal
kernels. `lognormal_*`, `powerlaw_*`, and `truncated_normal_*` provide logpdf,
CDF, and inverse-CDF helpers for positive lognormal, finite-support power-law,
and truncated-normal families. The method page is
[](../10-theory/distributions.md).

### `jaxstro.numerics.autodiff`

`jvp(...)`, `vjp(...)`, `jacobian_vector_product(...)`,
`vector_jacobian_product(...)`, `hvp(...)`, `gauss_newton_product(...)`, and
`empirical_fisher_product(...)` expose common derivative products as named
helpers over JAX primitives. The method page is [](../10-theory/autodiff.md).

### `jaxstro.geometry`

`normalize(...)` and `angular_distance(...)` cover vector geometry.
`rotation_matrix(...)`, `quaternion_from_axis_angle(...)`,
`quaternion_multiply(...)`, `quaternion_conjugate(...)`, and
`quaternion_rotate(...)` cover axis-angle and quaternion rotations.
`rigid_transform(...)`, `invert_rigid(...)`, and `compose_rigid(...)` cover
3D rigid transforms with explicit composition order. The method page is
[](../10-theory/geometry.md).

### `jaxstro.numerics.optimization`

`squared_loss(...)`, `huber_loss(...)`, and `pseudo_huber_loss(...)` provide
elementwise residual losses. `objective_summary(...)` reports scalar squared-loss
diagnostics for residual vectors, optionally with weights.
`armijo_backtracking(...)` is a fixed-iteration Armijo line-search helper whose
objective and scan length are static under JIT. `relative_step_norm(...)`,
`gradient_inf_norm(...)`, and `convergence_summary(...)` provide
optimizer-agnostic stopping diagnostics. The method page is
[](../10-theory/optimization.md).

### `jaxstro.numerics.ode`

`euler_step(...)`, `midpoint_step(...)`, and `rk4_step(...)` expose one-step
updates for first-order systems with call signature `rhs(y, t)`. `euler(...)`,
`midpoint(...)`, `rk4(...)`, and `solve_fixed_step(...)` return `ODEResult(t, y)`
histories including the initial state. `velocity_verlet(...)` returns
`VerletResult(t, q, v)` for separable second-order systems with acceleration
callback `a(q, t)`. The method page is [](../10-theory/ode.md).

### `jaxstro.numerics.operators`

`DenseOperator(...)` and `DiagonalOperator(...)` are primitive PyTree operators
with `matvec`, `rmatvec`, `shape`, and `to_dense` methods. `scale(...)`,
`add(...)`, `compose(...)`, `transpose(...)`, and `block_diag(...)` build scaled,
summed, product, transpose-view, and block-diagonal operators. The method page is
[](../10-theory/operators.md).

### `jaxstro.numerics.special`

`planck_lambda_cgs(...)`, `log_planck_lambda_cgs(...)`, `planck_nu_cgs(...)`, and
`log_planck_nu_cgs(...)` provide explicit-CGS Planck radiance kernels.
`log_normalize(...)` and `normalize_log_weights(...)` handle stable log-weight
normalization. `legendre_basis(...)`, `chebyshev_t_basis(...)`, and
`laguerre_basis(...)` evaluate orthogonal polynomial bases with the degree axis
last. The method page is [](../10-theory/special-functions.md).

### `jaxstro.numerics.sampling`

`inverse_cdf_draw(...)` maps a uniform deviate through a tabulated inverse CDF.
`stratified_uniform(...)` draws one uniform sample from each equal-width stratum
with deterministic shape. The method pages are [](../10-theory/rootfinding.md) and
[](../10-theory/grids.md).

:::{note} Per-symbol reference pages are planned
A complete, auto-generated per-module symbol reference (signatures, parameters,
source links) is planned. Until then, the docstrings are authoritative — read them
with `help(jaxstro.numerics.rootfinding.newton_ppf)` — and this landing page is the
module map.
:::

### `jaxstro.atmospheres`

`jaxstro.atmospheres` exposes the shared foundation boundary
`AtmosphereParams -> SpectrumResult`. `AtmosphereLibrary` ranks local catalog
coverage without hiding provenance or pretending artifact-only data have runtime
backends. Host-side backends currently open processed NewEra and BOSZ artifacts.
Sonora and TLUSTY are processed and validated locally, but their runtime backends
are intentionally separate follow-up work because each needs an explicit
interpolation and unit policy.

`PreparedSpectralGrid` carries an already-loaded wavelength grid and corner
spectra for JAX-side bilinear interpolation over `teff` and `logg` at one exact
abundance plane. Coverage reporting, source-preserving Sonora/TLUSTY converters,
and overlap diagnostics stay host-side. The dataset matrix is in
[](../20-architecture/atmosphere-capabilities.md); the runtime and downstream
ownership boundary is in [](../20-architecture/spectra-data-architecture.md).

### `jaxstro.testing`

`jaxstro.testing` contains validation utilities rather than model primitives.
The grad-audit engine classifies curated differentiability cases, while the
finite-difference diagnostics expose reusable central-difference gradients,
Jacobians, directional derivatives, and structured AD-vs-FD comparison reports.
`EvidenceAnchor`, `MethodEvidence`, and `NumericalTrustReport` describe
method-level evidence, and `trust_report_to_json(...)`,
`trust_report_to_markdown(...)`, and `default_numerics_trust_report(...)` render
deterministic trust summaries. These helpers are intended for test suites and
validation scripts.
