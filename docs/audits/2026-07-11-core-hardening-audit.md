# Core hardening audit — Slice A

**Status:** in progress; findings below are either source-backed or directly reproduced.

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
| Fundamental constants | [NIST CODATA 2018 archive](https://physics.nist.gov/cuu/Constants/archive2018.html) | Package values intentionally follow the archived 2018 adjustment. NIST identifies 2022 as current; 2018 must remain versioned rather than called current. |
| Nominal solar conversions | [IAU 2015 Resolution B3](https://www.iau.org/common/Uploaded%20files/IAUGA2015-Resolution-B3-recommended-nominal-conversion.pdf) | Exact nominal radius, luminosity, effective temperature, and mass *parameter* verified. |
| Bolometric zero point | IAU 2015 Resolution B2 | Numerical convention is consistent; the repository's aggregate IAU locator needs replacement with a stable, directly verified locator before registry carding. |
| Galactic/ICRS transform | [IAU SOFA manual](https://www.iausofa.org/s/manual_c.pdf) | Matrix values are correct; the convention is IAU 1958 Galactic coordinates to ICRS, not “IAU 2000”. |

## Confirmed findings

| ID | Severity | Category | Evidence | Required action |
| --- | --- | --- | --- | --- |
| A-001 | P1 | Default-float32 numerical defect | `safe_exp(100)`, `safe_div(1, 0)`, `relative_error(1, 0)`, and the zero-denominator `safe_div` gradient were non-finite in a clean `jax_enable_x64=False` process. The test suite globally enabled x64, masking it. | **Fixed** in `08792a3`; a clean-subprocess regression now requires finite float32 results while preserving x64 behavior. |
| A-002 | P1 | API/coordinate ownership mismatch | `compute_proper_motions` projected global x/y components but labeled them RA*/Dec. At an off-axis line of sight, the promised local RA basis differs from global x. | **Fixed:** require explicit sky-tangent center/roll; transform local position and velocity to ICRS; project onto each star's exact RA*/Dec basis. The old implicit x/y path is retired. |
| A-003 | P2 | Eager contract defect | `sky_tangent(..., warn_large_field=True)` checked whether a JAX scalar was a Python `float`/`int`, so normal eager JAX inputs never issued the documented warning. | **Fixed:** `try_concrete_bool` emits the warning for concrete eager fields and intentionally skips it while traced; regression coverage pins the eager path. |
| A-004 | P2 | Provenance/claim defect | IAU B3 defines exact nominal \((GM)_\odot^\mathrm{N}\), not a nominal solar mass in grams. `MSUN_G` is a rounded legacy conversion; current text overstates its provenance. The repository's IAU aggregate URL is stale/dead in direct retrieval. | Preserve the legacy numeric value; correct language/locators and distinguish nominal conversions from compatibility values. |
| A-005 | P2 | Exactness claim defect | Stored `A_RAD` and `R_GAS` literals are rounded, not bit-identical to their defining expressions; existing tests tolerate residuals much larger than the stored precision. | Tighten provenance tests to the intended stored precision and replace “exact” language with an accurate rounding/consistency statement. |
| A-006 | P2 | Gradient coverage gap | `jaxstro.testing.grad_audit` contains only engine/toy cases; no jaxstro public-entry-point registry exists. Existing coordinate checks are mostly finite-gradient smoke tests. | Add interior FD-vs-AD cases and explicit singular-domain classifications before calling public coordinate APIs inference-ready. |
| A-007 | P2 | Singular-domain contract gap | Coincident observer/star geometry, spherical origin, zero normalization, and zenith/horizon geometry have non-finite derivatives or undefined bases. | Document/exclude true geometric singularities or define an intentional regularized/sentinel contract; do not claim global differentiability. |
| A-008 | P2 | Convention documentation defect | The Galactic-to-ICRS matrix is numerically correct but documented as “IAU 2000/J2000.0”; SOFA identifies this transform as IAU 1958 Galactic coordinates to ICRS. | Correct wording and add an authority-anchored reference-vector test. |

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

## Open decisions and next checks

1. Build a public jaxstro AD case registry with interior FD-vs-AD cases, then classify
   rather than conceal A-007 singularities.
2. Correct A-004/A-005/A-008 provenance wording only after each locator/source statement
   is independently checked.
