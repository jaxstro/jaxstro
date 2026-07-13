---
title: Scientific contract registry
---

# Scientific contract registry

Unverified does not mean unsupported; it means no claim is registered.
This generated page does not infer support from importability or an unrelated passing test.

## Module ownership

| Module | Maturity | Boundary | Dimensional policy | Ownership | Non-ownership |
| --- | --- | --- | --- | --- | --- |
| `jaxstro.astrometry` | validated | runtime | Angles in radians/degrees and proper motion in mas/yr as named. | Astrometric constants. | Survey or population models. |
| `jaxstro.atmospheres` | validated | mixed | Source coordinates and flux semantics are explicit per product. | Catalog and artifact preparation plus evidence-gated evaluation. | Photometry or model validity. |
| `jaxstro.constants` | validated | static | CGS unless the symbol explicitly names another unit. | Source-backed physical constants. | Runtime source lookup. |
| `jaxstro.contracts` | validated | tooling | Metadata only; scientific units are recorded by owned contracts. | Scientific contract vocabulary, validation, and rendering. | Runtime scientific acceptance or automatic certification. |
| `jaxstro.coords` | validated | runtime | Positions in pc, velocities in km/s, angles in degrees, proper motions in mas/yr, and parallax in mas where documented. | Coordinate transformations. | Domain frame selection. |
| `jaxstro.geometry` | validated | runtime | Caller-owned coordinate units; angles follow each function contract. | Generic geometric transformations. | Domain geometry policy. |
| `jaxstro.jaxconfig` | validated | static | No physical dimensions. | Explicit JAX precision configuration. | Import-time global configuration. |
| `jaxstro.numerics` | validated | runtime | Caller-owned units; each callable declares dimensional behavior. | Generic numerical mechanics. | Domain acceptance, retry policy, or physical state. |
| `jaxstro.params` | validated | runtime | Leaf units remain caller-owned through transformations. | Selective PyTree/vector parameter bridges. | Inference algorithms or identifiability. |
| `jaxstro.provenance` | validated | tooling | Metric units remain explicit in producer-owned payloads. | Runtime artifact manifests. | Scientific-source validation. |
| `jaxstro.quantity` | validated | runtime | Dimensions and scales are represented explicitly in Unit metadata. | Dimensional quantity evaluation. | Approved ecosystem adoption or cutover. |
| `jaxstro.spatial` | validated | mixed | Coordinates use caller-owned consistent length units. | Spatial indexing, candidates, and exact pairs. | Force or encounter semantics. |
| `jaxstro.spectra` | validated | mixed | Spectral coordinates and density semantics carry explicit unit metadata. | Generic spectral representations and remapping. | Filters, photometry, or instruments. |
| `jaxstro.testing` | validated | tooling | Every reported metric retains producer-declared units. | Validation and provenance tooling. | Runtime scientific acceptance. |
| `jaxstro.units` | validated | static | CGS is canonical; named systems declare mass, length, and time scales. | Canonical ecosystem unit systems. | Hidden domain unit defaults. |

## Transform and AD contracts

| Callable | Maturity | AD semantics | Transform claims | Boundaries | Evidence | Limitations and cost |
| --- | --- | --- | --- | --- | --- | --- |
| `jaxstro.numerics.implicit_bracketed_root` | validated | certified_implicit | `jit`: supported | Rejected assumption or numerical certificate. [returns_nan] | `root.implicit_bracketed_root` → `tests/validation/test_implicit_root_gradients.py` (validation_test) | Requires a unique mathematical root.; Requires a smooth selected branch and adequate slope conditioning.; Runs the safeguarded primal before exposing a custom-root derivative. |
| `jaxstro.numerics.initialize_bracket` | validated | value_first | none claimed | none registered | `root.initialize_bracket` → `tests/unit/test_bracketed_root.py` (unit_test) | Primary purpose is auditable forward-value control flow. |
| `jaxstro.numerics.interpolation.interp1d` | validated | smooth_pathwise | none claimed | none registered | `numerics.interp1d` → `tests/validation/test_grad_checks.py` (validation_test) | none registered |
| `jaxstro.numerics.map_safeguarded_bracketed_root` | validated | value_first | `jit`: supported; `vmap`: conditional (Values and shapes are preserved, but physical per-lane skipping is not guaranteed.) | Missing sign bracket or nonfinite admissible evaluation. [structured_result] | `root.map_safeguarded_bracketed_root` → `tests/validation/test_bracketed_root_algorithms.py` (validation_test) | No implicit-root derivative claim.; Use lax.map when physical per-lane skipping of expensive residuals matters. |
| `jaxstro.numerics.interpolation.monotone_cubic_interp` | validated | smooth_pathwise | none claimed | none registered | `numerics.monotone_cubic_interp` → `tests/unit/test_interpolation_shape_preserving.py` (validation_test) | none registered |
| `jaxstro.numerics.powerlaw_cdf` | validated | smooth_pathwise | none claimed | Outside x support clamps to zero or one. [saturates] | `numerics.powerlaw_cdf` → `tests/validation/test_grad_checks.py` (validation_test) | none registered |
| `jaxstro.numerics.powerlaw_logpdf` | validated | smooth_pathwise | none claimed | Outside x support returns negative infinity. [returns_sentinel] | `numerics.powerlaw_logpdf` → `tests/validation/test_grad_checks.py` (validation_test) | none registered |
| `jaxstro.numerics.powerlaw_ppf` | validated | smooth_pathwise | none claimed | Quantile input is defined on the closed unit interval. [undefined] | `numerics.powerlaw_ppf` → `tests/validation/test_grad_checks.py` (validation_test) | none registered |
| `jaxstro.numerics.propose_bracketed` | validated | value_first | none claimed | none registered | `root.propose_bracketed` → `tests/unit/test_bracketed_root.py` (unit_test) | Primary purpose is auditable forward-value control flow. |
| `jaxstro.numerics.regular_grid_interp` | validated | smooth_pathwise | none claimed | clamp policy holds queries at the nearest boundary. [saturates]; reject policy raises for invalid concrete queries. [raises] | `numerics.regular_grid_interp` → `tests/unit/test_regular_grid.py` (validation_test) | none registered |
| `jaxstro.numerics.safeguarded_bracketed_root` | validated | value_first | `jit`: supported; `vmap`: conditional (Values and shapes are preserved, but physical per-lane skipping is not guaranteed.) | Missing sign bracket or nonfinite admissible evaluation. [structured_result] | `root.safeguarded_bracketed_root` → `tests/validation/test_bracketed_root_algorithms.py` (validation_test) | No implicit-root derivative claim.; Use lax.map when physical per-lane skipping of expensive residuals matters. |
| `jaxstro.numerics.update_bracket` | validated | value_first | none claimed | none registered | `root.update_bracket` → `tests/unit/test_bracketed_root.py` (unit_test) | Primary purpose is auditable forward-value control flow. |
| `jaxstro.testing.compare_gradients` | validated | validation_only | none claimed | none registered | `testing.compare_gradients` → `tests/integration/test_grad_audit.py` (integration_test) | Does not determine downstream scientific acceptance. |
| `jaxstro.testing.render_registry` | validated | validation_only | none claimed | none registered | `testing.render_registry` → `tests/validation/provenance_cards/test_registry.py` (integration_test) | Does not determine downstream scientific acceptance. |
| `jaxstro.testing.validate_card` | validated | validation_only | none claimed | none registered | `testing.validate_card` → `tests/validation/provenance_cards/test_registry.py` (integration_test) | Does not determine downstream scientific acceptance. |

## Unclassified callable surfaces

Callable coverage is deliberately tiered. These modules have module-level records but no callable-level claims:

`jaxstro.astrometry`, `jaxstro.atmospheres`, `jaxstro.constants`, `jaxstro.contracts`, `jaxstro.coords`, `jaxstro.geometry`, `jaxstro.jaxconfig`, `jaxstro.params`, `jaxstro.provenance`, `jaxstro.quantity`, `jaxstro.spatial`, `jaxstro.spectra`, `jaxstro.units`.

Modules with exemplars may still contain other unclassified callables; absence from the table is not a support or maturity claim.
