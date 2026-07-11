# Core hardening audit — Slice A

**Status:** completed; findings below are either source-backed or directly reproduced.

## Scope and method

This audit covers physical constants and unit systems, coordinate and astrometric
transforms, and public numerical/automatic-differentiation contracts. It separates
source authority, code behavior, and singular-domain behavior; a passing numerical
test is not treated as proof of a physical convention.

The baseline gate on 2026-07-11 passed before this audit: 875 fast-tier tests plus
39 ML-integration tests and a clean wheel import. The later safe-math regression
increased the fast-tier result to 876 passed (2 slow tests deselected).

## Authority inventory

| Area | Authority | Audit status |
| --- | --- | --- |
| Fundamental constants | [NIST CODATA 2018 archive](https://physics.nist.gov/cuu/Constants/archive2018.html) | Package values intentionally follow the archived 2018 adjustment. NIST identifies 2022 as current; 2018 remains versioned rather than called current. |
| Nominal solar conversions | [IAU 2015 Resolution B3](https://www.iau.org/common/Uploaded%20files/IAUGA2015-Resolution-B3-recommended-nominal-conversion.pdf) | Exact nominal radius, luminosity, effective temperature, and solar mass *parameter* verified. B3 does not define a solar mass in grams. |
| Bolometric zero point | [IAU 2015 Resolution B2](https://www.iau.org/static/resolutions/IAU2015_English.pdf) | B2 defines exact $L_0$; the stored $M_{\rm bol,\odot}=4.74$ is the conventional rounded result after combining B2 with B3's nominal luminosity. |
| Galactic/ICRS transform | [IAU SOFA manual](https://www.iausofa.org/s/manual_c.pdf) | Matrix values are correct; the convention is IAU 1958 Galactic coordinates to ICRS, not “IAU 2000”. |

## Confirmed findings

| ID | Severity | Category | Evidence | Required action |
| --- | --- | --- | --- | --- |
| A-001 | P1 | Default-float32 numerical defect | `safe_exp(100)`, `safe_div(1, 0)`, `relative_error(1, 0)`, and the zero-denominator `safe_div` gradient were non-finite in a clean `jax_enable_x64=False` process. The test suite globally enabled x64, masking it. | **Fixed** in `08792a3`; a clean-subprocess regression now requires finite float32 results while preserving x64 behavior. |
| A-002 | P1 | API/coordinate ownership mismatch | `compute_proper_motions` projected global x/y components but labeled them RA*/Dec. At an off-axis line of sight, the promised local RA basis differs from global x. | **Fixed:** require explicit sky-tangent center/roll; transform local position and velocity to ICRS; project onto each star's exact RA*/Dec basis. The old implicit x/y path is retired. |
| A-003 | P2 | Eager contract defect | `sky_tangent(..., warn_large_field=True)` checked whether a JAX scalar was a Python `float`/`int`, so normal eager JAX inputs never issued the documented warning. | **Fixed:** `try_concrete_bool` emits the warning for concrete eager fields and intentionally skips it while traced; regression coverage pins the eager path. |
| A-004 | P2 | Provenance/claim defect | IAU B3 defines exact nominal \((GM)_\odot^\mathrm{N}\), not a nominal solar mass in grams. `MSUN_G` is a rounded legacy conversion. | **Fixed:** legacy value preserved; inline/API/bibliography language now distinguishes B3 nominal conversion constants from the gram compatibility scale and records direct B2/B3 locators. |
| A-005 | P2 | Exactness claim defect | Stored `A_RAD` and `R_GAS` literals are rounded, not bit-identical to their defining expressions; prior tests tolerated residuals much larger than the stored precision. | **Fixed:** comments state the rounding contract and tests enforce the ten-significant-figure relative budget without asserting bit identity. |
| A-006 | P2 | Gradient coverage gap | `jaxstro.testing.grad_audit` contained only engine/toy cases; no jaxstro public-entry-point registry existed. Existing coordinate checks were mostly finite-gradient smoke tests. | **Fixed:** `tests/validation/test_coords_grad_audit.py` now exercises nine interior public-coordinate AD-vs-FD contracts. The registry covers sky tangent, parallax, Galactic placement, both Galactic/ICRS directions, both spherical directions, observing geometry, and proper motion. |
| A-007 | P2 | Singular-domain contract gap | Coincident observer/star geometry, spherical origin, zero normalization, and zenith/horizon geometry have non-finite derivatives or undefined bases. | **Fixed:** every public coordinate transform now documents its domain restrictions. Regressions explicitly record the non-finite Cartesian-origin angular Jacobian and coincident-parallax gradient; these geometries are excluded from the inference contract rather than silently regularized. |
| A-008 | P2 | Convention documentation defect | The Galactic-to-ICRS matrix is numerically correct but was documented as “IAU 2000/J2000.0”; SOFA identifies this transform as IAU 1958 Galactic coordinates to ICRS. | **Fixed:** code references identify IAU SOFA as the convention authority; existing reference-vector tests remain the numerical regression evidence. |

## Verified non-defect observations

- `UnitSystem.G` uses the correct CGS-to-code-unit dimensional conversion
  \(G_{\rm CGS} M T^2/L^3\).
- The established unit systems and aliases have round-trip and gravitational-constant
  coverage.
- `PhotometricUnits` intentionally makes the invalid linear AB conversion fail loudly;
  it is not a candidate for a “safe” finite-value rewrite.
- The Galactic-to-ICRS matrix is orthogonal and maps the Galactic center to the
  expected ICRS direction; the issue is naming/provenance, not its coefficients.

## Tests and reproductions

- `bash scripts/check.sh` after `08792a3`: 876 passed, 2 deselected; 39 ML tests
  passed; wheel smoke imported cleanly.
- `tests/unit/test_numerics.py::TestSafeExp::test_safe_primitives_are_finite_with_default_float32`
  failed before `08792a3` and passed afterward.
- `tests/unit/test_numerics.py tests/validation/test_grad_checks.py -q`: 140 passed
  after the float32 hardening.
- `tests/validation/test_coords_grad_audit.py -q`: 9 passed after the public-coordinate
  registry expansion. These are interior probes only; they are evidence of local
  derivative consistency, not a claim of global differentiability.
- `tests/unit/test_coords.py::TestSingularDomainContracts -q`: 11 passed after the
  singular-domain contracts. The tested exclusions are: observer-star coincidence,
  Cartesian origin, spherical angular poles, celestial/galactic longitude poles, and
  zenith/nadir/horizon observing geometry.

## Slice-B follow-through

The direct CODATA/IAU/SOFA locators now feed the machine-readable Slice-B registry
under `docs/provenance/registry/`. Generated reference pages are freshness-checked,
and their code/validation references are resolved by
`tests/validation/provenance_cards/test_registry.py`.
