---
title: Scientific contract registry
---

# Scientific contract registry

Unverified does not mean unsupported; it means no claim is registered.
This generated page does not infer support from importability or an unrelated passing test.

## Owner import path

`jaxstro.contracts`

## Purpose

Expose the validated scientific-contract vocabulary, runtime inventory audit, and deterministic reference rendering.

## Public records and callables

Contract enums and records, `collect_contracts`, `audit_runtime_inventory`, `get_callable_contract`, and `get_module_contract`.

## Shape and dtype expectations

Contracts are immutable host-side metadata. They describe array, unit, transform, and evidence policies but do not contain runtime scientific arrays.

## JAX transforms and AD classification

This tooling is host-side. The generated callable table records transform and AD claims owned by runtime modules.

## Failure behavior

Invalid records, unresolved evidence, and runtime inventory drift fail validation instead of inferring support from importability.

## Contract and evidence links

Registered computational evidence resolves to the [evidence index](../../60-validation/evidence-index.md).

## Canonical import example

```python
from jaxstro.contracts import get_callable_contract
```

## Module ownership

| Module | Maturity | Boundary | Dimensional policy | Ownership | Non-ownership |
| --- | --- | --- | --- | --- | --- |
| `jaxstro.astrometry` | validated | runtime | Angles in radians/degrees and proper motion in mas/yr as named. | Astrometric constants. | Survey or population models. |
| `jaxstro.atmospheres` | validated | mixed | Source coordinates and flux semantics are explicit per product. | Catalog and artifact preparation plus evidence-gated evaluation. | Photometry or model validity. |
| `jaxstro.constants` | validated | static | CGS unless the symbol explicitly names another unit. | Source-backed physical constants. | Runtime source lookup. |
| `jaxstro.contracts` | validated | tooling | Metadata only; scientific units are recorded by owned contracts. | Scientific contract vocabulary, validation, and rendering. | Runtime scientific acceptance or automatic certification. |
| `jaxstro.coords` | validated | runtime | Positions in pc, velocities in km/s, angles in degrees, proper motions in mas/yr, and parallax in mas where documented. | Coordinate transformations. | Domain frame selection. |
| `jaxstro.evidence` | validated | tooling | Every metric carries explicit producer-owned units. | Portable computational-evidence schemas, validation, and rendering. | Method-specific scientific thresholds or source-provenance claims. |
| `jaxstro.geometry` | validated | runtime | Caller-owned coordinate units; angles follow each function contract. | Generic geometric transformations. | Domain geometry policy. |
| `jaxstro.jaxconfig` | validated | static | No physical dimensions. | Explicit JAX precision configuration. | Import-time global configuration. |
| `jaxstro.numerics` | validated | runtime | Caller-owned units; each callable declares dimensional behavior. | Generic numerical mechanics. | Domain acceptance, retry policy, or physical state. |
| `jaxstro.params` | validated | runtime | Leaf units remain caller-owned through transformations. | Selective PyTree/vector parameter bridges. | Inference algorithms or identifiability. |
| `jaxstro.provenance` | validated | tooling | Metric units remain explicit in producer-owned payloads. | Runtime artifact manifests. | Scientific-source validation. |
| `jaxstro.quad` | experimental | runtime | Raw kernels with an alpha quantity adapter owned only by quad.integrate. | Canonical sampled-data integration, fixed and adaptive one-dimensional quadrature, typed domains, measures, methods, and result evidence. | Multidimensional integration, direct Quantity-PyTree quotient-unit Jacobians, physical-model policy, inference, ODE solving, or scientific acceptance. |
| `jaxstro.quantity` | validated | runtime | Dimensions and scales are represented explicitly in Unit metadata. | Dimensional quantity evaluation. | Approved ecosystem adoption or cutover. |
| `jaxstro.spatial` | validated | mixed | Coordinates use caller-owned consistent length units. | Spatial indexing, candidates, and exact pairs. | Force or encounter semantics. |
| `jaxstro.spectra` | validated | mixed | Spectral coordinates and density semantics carry explicit unit metadata. | Generic spectral representations and remapping. | Filters, photometry, or instruments. |
| `jaxstro.testing` | validated | tooling | Every reported metric retains producer-declared units. | Validation and provenance tooling. | Runtime scientific acceptance. |
| `jaxstro.units` | validated | static | CGS is canonical; named systems declare mass, length, and time scales. | Canonical ecosystem unit systems. | Hidden domain unit defaults. |

## Transform and AD contracts

| Callable | Maturity | AD semantics | Transform claims | Boundaries | Evidence | Limitations and cost |
| --- | --- | --- | --- | --- | --- | --- |
| `jaxstro.numerics.implicit_bracketed_root` | validated | certified_implicit | `jit`: supported | Rejected assumption or numerical certificate. [returns_nan] | [`root.implicit_bracketed_root`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_implicit_root_gradients.py) (validation_test); [`root.implicit_bracketed_root.certification`](https://github.com/drannarosen/jaxstro/blob/main/scripts/benchmark_implicit_root.py) (artifact) -> [`rootfinding.implicit-gradients`](../../60-validation/evidence-index.md) gates `exponential.absolute_residual.gate, exponential.bracket_width.gate, exponential.relative_ad_analytic_error.gate, exponential.relative_ad_fd_error.gate, exponential.slope_magnitude.gate, exponential.certificate.gate, linear.absolute_residual.gate, linear.bracket_width.gate, linear.relative_ad_analytic_error.gate, linear.relative_ad_fd_error.gate, linear.slope_magnitude.gate, linear.certificate.gate, quadratic.absolute_residual.gate, quadratic.bracket_width.gate, quadratic.relative_ad_analytic_error.gate, quadratic.relative_ad_fd_error.gate, quadratic.slope_magnitude.gate, quadratic.certificate.gate` | Requires a unique mathematical root.; Requires a smooth selected branch and adequate slope conditioning.; Runs the safeguarded primal before exposing a custom-root derivative. |
| `jaxstro.numerics.initialize_bracket` | validated | value_first | none claimed | none registered | [`root.initialize_bracket`](https://github.com/drannarosen/jaxstro/blob/main/tests/unit/test_bracketed_root.py) (unit_test) | Primary purpose is auditable forward-value control flow. |
| `jaxstro.numerics.interpolation.interp1d` | validated | smooth_pathwise | none claimed | Queries outside the table clamp to endpoint values. [saturates] | [`numerics.interp1d`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_grad_checks.py) (validation_test) | AD evidence covers interior off-knot, branch-stable queries. |
| `jaxstro.numerics.map_safeguarded_bracketed_root` | validated | value_first | `jit`: supported; `vmap`: conditional (Values and shapes are preserved, but physical per-lane skipping is not guaranteed.) | Missing sign bracket or nonfinite admissible evaluation. [structured_result] | [`root.map_safeguarded_bracketed_root`](https://github.com/drannarosen/jaxstro/blob/main/tests/unit/test_bracketed_root.py) (unit_test); [`root.map_safeguarded_bracketed_root.performance`](https://github.com/drannarosen/jaxstro/blob/main/scripts/benchmark_rootfinding.py) (benchmark) -> [`rootfinding.performance`](../../60-validation/evidence-index.md) gates `flat_slope.hybrid-no-more-evaluations, linear.hybrid-no-more-evaluations, monotone_kink.hybrid-no-more-evaluations, oscillatory_fixed_point_residual.hybrid-no-more-evaluations, quadratic.hybrid-no-more-evaluations` | No implicit-root derivative claim.; Use lax.map when physical per-lane skipping of expensive residuals matters. |
| `jaxstro.numerics.interpolation.monotone_cubic_interp` | validated | smooth_pathwise | none claimed | none registered | [`numerics.monotone_cubic_interp`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_grad_checks.py) (validation_test) | Sign, plateau, knot, and overshoot branch boundaries are nonsmooth. |
| `jaxstro.numerics.powerlaw_cdf` | validated | smooth_pathwise | none claimed | Outside x support clamps to zero or one. [saturates] | [`numerics.powerlaw_cdf`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_grad_checks.py) (validation_test) | none registered |
| `jaxstro.numerics.powerlaw_logpdf` | validated | smooth_pathwise | none claimed | Outside x support returns negative infinity. [returns_sentinel] | [`numerics.powerlaw_logpdf`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_grad_checks.py) (validation_test) | none registered |
| `jaxstro.numerics.powerlaw_ppf` | validated | smooth_pathwise | none claimed | Quantile input is defined on the closed unit interval. [undefined] | [`numerics.powerlaw_ppf`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_grad_checks.py) (validation_test) | none registered |
| `jaxstro.numerics.propose_bracketed` | validated | value_first | none claimed | none registered | [`root.propose_bracketed`](https://github.com/drannarosen/jaxstro/blob/main/tests/unit/test_bracketed_root.py) (unit_test) | Primary purpose is auditable forward-value control flow. |
| `jaxstro.numerics.regular_grid_interp` | validated | smooth_pathwise | none claimed | clamp policy holds queries at the nearest boundary. [saturates]; reject policy raises for invalid concrete queries. [raises] | [`numerics.regular_grid_interp`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_grad_checks.py) (validation_test) | Coordinate derivatives are claimed only inside branch-stable cells. |
| `jaxstro.numerics.safeguarded_bracketed_root` | validated | value_first | `jit`: supported; `vmap`: conditional (Values and shapes are preserved, but physical per-lane skipping is not guaranteed.) | Missing sign bracket or nonfinite admissible evaluation. [structured_result] | [`root.safeguarded_bracketed_root`](https://github.com/drannarosen/jaxstro/blob/main/tests/unit/test_numerics.py) (unit_test); [`root.safeguarded_bracketed_root.performance`](https://github.com/drannarosen/jaxstro/blob/main/scripts/benchmark_rootfinding.py) (benchmark) -> [`rootfinding.performance`](../../60-validation/evidence-index.md) gates `flat_slope.hybrid-no-more-evaluations, linear.hybrid-no-more-evaluations, monotone_kink.hybrid-no-more-evaluations, oscillatory_fixed_point_residual.hybrid-no-more-evaluations, quadratic.hybrid-no-more-evaluations` | No implicit-root derivative claim.; Use lax.map when physical per-lane skipping of expensive residuals matters. |
| `jaxstro.numerics.universal_kepler_step` | validated | smooth_pathwise | `jit`: supported; `vmap`: supported; `jvp`: conditional (Continuous state on one converged route with fixed shape, iteration budget, status path, and Stumpff branch.); `vjp`: conditional (Continuous state on one converged route with fixed shape, iteration budget, status path, and Stumpff branch.) | Invalid input, nonfinite iteration, singular radius, or exhausted iteration budget. [structured_result] | [`numerics.universal_kepler_step.value`](https://github.com/drannarosen/jaxstro/blob/main/tests/unit/test_kepler.py) (unit_test); [`numerics.universal_kepler_step.fixed_route_ad`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_kepler_gradients.py) (validation_test) | No derivative claim across status, iteration-count, Stumpff-route, conic-label, or collision boundaries.; No implicit-root derivative claim; AD follows the finite executed Newton map.; Units, object identity, encounter selection, and state-commit policy belong to callers.; Runs a fixed 12-slot Newton scan by default; converged lanes freeze numerically. |
| `jaxstro.numerics.update_bracket` | validated | value_first | none claimed | none registered | [`root.update_bracket`](https://github.com/drannarosen/jaxstro/blob/main/tests/unit/test_bracketed_root.py) (unit_test) | Primary purpose is auditable forward-value control flow. |
| `jaxstro.quad.fixed` | experimental | smooth_pathwise | `jax.jit`: supported (Rule type, rule order or level, measure type, breakpoint count, and payload shape remain static.); `jax.vmap`: supported (Batch explicit arguments or numerical bounds.) | Unsupported rule, domain, and measure pairings raise eagerly. [raises]; Value-dependent invalid finite domains return NaN under tracing. [returns_nan] | [`quad-fixed-unit`](https://github.com/drannarosen/jaxstro/blob/main/tests/unit/quad/test_fixed.py) (unit_test); [`quad-fixed-transforms`](https://github.com/drannarosen/jaxstro/blob/main/tests/integration/test_quad_fixed_transforms.py) (integration_test) | A fixed rule does not estimate truncation error or select its order.; Quantity-valued fixed integration is not implemented.; Integrand work is the static node count multiplied by the static number of finite breakpoint segments. |
| `jaxstro.quad.integrate` | experimental | smooth_pathwise | `jax.jit`: supported (Method configuration, capacities, breakpoint count, and payload shape remain static.); `jax.vmap`: supported (Each batch member runs one independent bounded controller.); `jvp`: conditional (First-order replay of value on a successful solve with parameters passed through explicit args, smooth finite bounds, or the finite boundary of a supported semi-infinite domain.); `vjp`: conditional (Project value or a floating diagnostic; full integer-bearing result Jacobians are outside the contract.); `jacfwd/jacrev`: conditional (Apply to value only, using JAX realified conventions for complex maps.) | Unsupported method, domain, measure, breakpoint, or capacity declarations raise eagerly. [raises]; Dynamic invalid, nonfinite, roundoff-limited, or exhausted cases return a typed status. [structured_result] | [`quad-adaptive-transforms`](https://github.com/drannarosen/jaxstro/blob/main/tests/integration/test_quad_adaptive_transforms.py) (integration_test); [`quad-adaptive-validation`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_quad_adaptive_reference.py) (validation_test); [`quad-adaptive-envelope`](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-adaptive-envelope.json) (artifact); [`quad-adaptive-replay`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/test_quad_replay_derivatives.py) (validation_test); [`quad-replay-derivatives`](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-replay-derivatives.json) (artifact); [`quad-adaptive-quantity`](https://github.com/drannarosen/jaxstro/blob/main/tests/integration/test_quad_quantity_transforms.py) (integration_test); [`quad-performance`](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-performance.json) (artifact) | Estimator convergence is not a universal bound on true error.; Related rules can miss the same unresolved narrow feature.; Replay is the default first-order derivative of the accepted fixed formula; gradient=stop remains explicit.; Quantity-aware adaptive integration is alpha and opt-in.; Dimensional improper domains require an explicit positive physical scale.; Direct Quantity-PyTree quotient-unit Jacobians and higher derivatives are not claimed.; Multidimensional integration is not implemented.; No performance-superiority claim is established.; Regional logical work is node_count * (initial_regions + 2 * refinements); global methods report their finest completed active grid. |
| `jaxstro.testing.compare_gradients` | validated | validation_only | none claimed | none registered | [`testing.compare_gradients`](https://github.com/drannarosen/jaxstro/blob/main/tests/integration/test_grad_audit.py) (integration_test) | Does not determine downstream scientific acceptance. |
| `jaxstro.testing.render_registry` | validated | validation_only | none claimed | none registered | [`testing.render_registry`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/provenance_cards/test_registry.py) (integration_test) | Does not determine downstream scientific acceptance. |
| `jaxstro.testing.validate_card` | validated | validation_only | none claimed | none registered | [`testing.validate_card`](https://github.com/drannarosen/jaxstro/blob/main/tests/validation/provenance_cards/test_registry.py) (integration_test) | Does not determine downstream scientific acceptance. |

## Unclassified callable surfaces

The runtime export audit found **227** public callables without callable-level records:

- `jaxstro.atmospheres.acquisition_rows_to_markdown`
- `jaxstro.atmospheres.build_bosz_index`
- `jaxstro.atmospheres.build_newera_lowres_index`
- `jaxstro.atmospheres.discover_bosz_files`
- `jaxstro.atmospheres.discover_newera_lowres_files`
- `jaxstro.atmospheres.parse_bosz_filename`
- `jaxstro.atmospheres.parse_newera_lowres_filename`
- `jaxstro.atmospheres.parse_sonora_2024_filename`
- `jaxstro.atmospheres.parse_tlusty_float`
- `jaxstro.atmospheres.parse_tlusty_flux_filename`
- `jaxstro.atmospheres.plan_targeted_acquisition`
- `jaxstro.atmospheres.read_newera_lowres_header`
- `jaxstro.atmospheres.resolve_data_dir`
- `jaxstro.atmospheres.select_topology`
- `jaxstro.atmospheres.validate_spectrum_overlap`
- `jaxstro.contracts.audit_runtime_inventory`
- `jaxstro.contracts.collect_contracts`
- `jaxstro.contracts.get_callable_contract`
- `jaxstro.contracts.get_module_contract`
- `jaxstro.coords.cartesian_to_spherical`
- `jaxstro.coords.cluster_to_galactic_cartesian`
- `jaxstro.coords.compute_parallax`
- `jaxstro.coords.compute_proper_motions`
- `jaxstro.coords.equatorial_to_galactic`
- `jaxstro.coords.galactic_to_equatorial`
- `jaxstro.coords.sky_tangent`
- `jaxstro.coords.spherical_to_cartesian`
- `jaxstro.coords.zenith_parallactic`
- `jaxstro.evidence.artifact_from_dict`
- `jaxstro.evidence.artifact_to_dict`
- `jaxstro.evidence.artifact_to_json`
- `jaxstro.evidence.artifact_to_markdown`
- `jaxstro.evidence.build_evidence_index`
- `jaxstro.evidence.check_artifact`
- `jaxstro.evidence.emit_artifact`
- `jaxstro.evidence.validate_artifact`
- `jaxstro.geometry.angular_distance`
- `jaxstro.geometry.compose_rigid`
- `jaxstro.geometry.invert_rigid`
- `jaxstro.geometry.normalize`
- `jaxstro.geometry.quaternion_conjugate`
- `jaxstro.geometry.quaternion_from_axis_angle`
- `jaxstro.geometry.quaternion_multiply`
- `jaxstro.geometry.quaternion_rotate`
- `jaxstro.geometry.rigid_transform`
- `jaxstro.geometry.rotation_matrix`
- `jaxstro.jaxconfig.enable_high_precision`
- `jaxstro.numerics.ScalarFn`
- `jaxstro.numerics.adaptive_open_uniform_knots`
- `jaxstro.numerics.add`
- `jaxstro.numerics.add_diagonal_jitter`
- `jaxstro.numerics.advance_bracketed_root`
- `jaxstro.numerics.armijo_backtracking`
- `jaxstro.numerics.bilinear_interp`
- `jaxstro.numerics.bin_centers`
- `jaxstro.numerics.bisect_many`
- `jaxstro.numerics.block_diag`
- `jaxstro.numerics.bracket_expand`
- `jaxstro.numerics.bspline_antiderivative`
- `jaxstro.numerics.bspline_basis`
- `jaxstro.numerics.bspline_derivative`
- `jaxstro.numerics.bspline_design_matrix`
- `jaxstro.numerics.bspline_eval`
- `jaxstro.numerics.bspline_eval_deboor`
- `jaxstro.numerics.bspline_integral`
- `jaxstro.numerics.bspline_roughness_penalty`
- `jaxstro.numerics.cell_neighbors_1d`
- `jaxstro.numerics.cell_to_face_average`
- `jaxstro.numerics.chebyshev_t_basis`
- `jaxstro.numerics.clenshaw_curtis_nodes`
- `jaxstro.numerics.compose`
- `jaxstro.numerics.conservative_rebin`
- `jaxstro.numerics.conservative_remap_1d`
- `jaxstro.numerics.convergence_summary`
- `jaxstro.numerics.correlation_from_covariance`
- `jaxstro.numerics.correlation_matrix`
- `jaxstro.numerics.covariance_matrix`
- `jaxstro.numerics.divergence_1d`
- `jaxstro.numerics.empirical_fisher_product`
- `jaxstro.numerics.euler`
- `jaxstro.numerics.euler_step`
- `jaxstro.numerics.eval_cubic_spline`
- `jaxstro.numerics.face_geometry_1d`
- `jaxstro.numerics.fit_bspline_lstsq`
- `jaxstro.numerics.fold_in_stream`
- `jaxstro.numerics.gauss_hermite_nodes`
- `jaxstro.numerics.gauss_laguerre_nodes`
- `jaxstro.numerics.gauss_legendre_nodes`
- `jaxstro.numerics.gauss_newton_product`
- `jaxstro.numerics.geometric_bin_centers`
- `jaxstro.numerics.geometric_bin_edges`
- `jaxstro.numerics.gradient_inf_norm`
- `jaxstro.numerics.hermite_coefficients`
- `jaxstro.numerics.hermite_e_basis`
- `jaxstro.numerics.huber_loss`
- `jaxstro.numerics.hvp`
- `jaxstro.numerics.initialize_bracketed_root_state`
- `jaxstro.numerics.inverse_cdf_draw`
- `jaxstro.numerics.is_positive_definite`
- `jaxstro.numerics.jacobian_vector_product`
- `jaxstro.numerics.jvp`
- `jaxstro.numerics.key_stream`
- `jaxstro.numerics.laguerre_basis`
- `jaxstro.numerics.legendre_basis`
- `jaxstro.numerics.log_grid`
- `jaxstro.numerics.log_normalize`
- `jaxstro.numerics.log_planck_lambda_cgs`
- `jaxstro.numerics.log_planck_nu_cgs`
- `jaxstro.numerics.lognormal_cdf`
- `jaxstro.numerics.lognormal_logpdf`
- `jaxstro.numerics.lognormal_ppf`
- `jaxstro.numerics.midpoint`
- `jaxstro.numerics.midpoint_step`
- `jaxstro.numerics.monotone_inverse_interp`
- `jaxstro.numerics.natural_cubic_spline_coeffs`
- `jaxstro.numerics.newton_ppf`
- `jaxstro.numerics.normal_cdf`
- `jaxstro.numerics.normal_logpdf`
- `jaxstro.numerics.normal_ppf`
- `jaxstro.numerics.normalize_log_weights`
- `jaxstro.numerics.objective_summary`
- `jaxstro.numerics.open_uniform_knots`
- `jaxstro.numerics.planck_lambda_cgs`
- `jaxstro.numerics.planck_nu_cgs`
- `jaxstro.numerics.positive_definite_jitter`
- `jaxstro.numerics.pseudo_huber_loss`
- `jaxstro.numerics.qr_solve`
- `jaxstro.numerics.relative_step_norm`
- `jaxstro.numerics.residual_resample`
- `jaxstro.numerics.rk4`
- `jaxstro.numerics.rk4_step`
- `jaxstro.numerics.scale`
- `jaxstro.numerics.seed_manifest`
- `jaxstro.numerics.solve_fixed_step`
- `jaxstro.numerics.squared_loss`
- `jaxstro.numerics.stratified_resample`
- `jaxstro.numerics.stratified_uniform`
- `jaxstro.numerics.structured_edges_1d`
- `jaxstro.numerics.svd_solve`
- `jaxstro.numerics.systematic_resample`
- `jaxstro.numerics.tensor_product_design_matrix`
- `jaxstro.numerics.transpose`
- `jaxstro.numerics.trilinear_interp`
- `jaxstro.numerics.truncated_normal_cdf`
- `jaxstro.numerics.truncated_normal_logpdf`
- `jaxstro.numerics.truncated_normal_ppf`
- `jaxstro.numerics.vector_jacobian_product`
- `jaxstro.numerics.velocity_verlet`
- `jaxstro.numerics.vjp`
- `jaxstro.numerics.weighted_lstsq`
- `jaxstro.provenance.environment_snapshot`
- `jaxstro.provenance.hash_artifact`
- `jaxstro.provenance.manifest_to_json`
- `jaxstro.provenance.manifest_to_markdown`
- `jaxstro.quad.clenshaw_curtis_nodes`
- `jaxstro.quad.cumulative_simpson`
- `jaxstro.quad.cumulative_trapezoid`
- `jaxstro.quad.error_norm`
- `jaxstro.quad.gauss_hermite_nodes`
- `jaxstro.quad.gauss_laguerre_nodes`
- `jaxstro.quad.gauss_legendre_nodes`
- `jaxstro.quad.hermite_coefficients`
- `jaxstro.quad.hermite_e_basis`
- `jaxstro.quad.hyperrectangle_is_valid`
- `jaxstro.quad.hyperrectangle_orientation`
- `jaxstro.quad.interval_is_valid`
- `jaxstro.quad.interval_orientation`
- `jaxstro.quad.map_domain`
- `jaxstro.quad.map_interval`
- `jaxstro.quad.simpson`
- `jaxstro.quad.sorted_breakpoints`
- `jaxstro.quad.tolerance_threshold`
- `jaxstro.quad.trapezoid`
- `jaxstro.quantity.format_unit`
- `jaxstro.quantity.from_dict`
- `jaxstro.quantity.get_unit`
- `jaxstro.quantity.parse_unit`
- `jaxstro.quantity.register_global_unit`
- `jaxstro.quantity.to_dict`
- `jaxstro.quantity.unit_from_dict`
- `jaxstro.quantity.unit_to_dict`
- `jaxstro.spatial.approx_knn_candidates`
- `jaxstro.spatial.assign_particles_to_bins`
- `jaxstro.spatial.assign_to_cells_linear`
- `jaxstro.spatial.fill_bins`
- `jaxstro.spatial.fill_bins_exact`
- `jaxstro.spatial.gather_candidates_from_bins`
- `jaxstro.spatial.gather_candidates_two_stencil`
- `jaxstro.spatial.gather_candidates_with_stencil`
- `jaxstro.spatial.gather_pairs_within_radius`
- `jaxstro.spatial.morton_decode_3d`
- `jaxstro.spatial.morton_encode_3d`
- `jaxstro.spatial.wyhash32`
- `jaxstro.spectra.resample_spectrum`
- `jaxstro.spectra.surface_flux_to_luminosity`
- `jaxstro.spectra.surface_flux_to_observer_flux`
- `jaxstro.spectra.to_flux_lambda`
- `jaxstro.spectra.to_flux_nu`
- `jaxstro.spectra.to_frequency`
- `jaxstro.spectra.to_wavelength`
- `jaxstro.testing.CardStatus`
- `jaxstro.testing.Expect`
- `jaxstro.testing.GradContract`
- `jaxstro.testing.assert_no_stale`
- `jaxstro.testing.assert_partition`
- `jaxstro.testing.audit_entry_point`
- `jaxstro.testing.check_directional_derivative`
- `jaxstro.testing.compare_jacobians`
- `jaxstro.testing.contract_requires_fd`
- `jaxstro.testing.default_contract_for_expect`
- `jaxstro.testing.default_numerics_trust_report`
- `jaxstro.testing.directional_derivative`
- `jaxstro.testing.finite_difference_grad`
- `jaxstro.testing.finite_difference_jacobian`
- `jaxstro.testing.has_nearby_citation`
- `jaxstro.testing.is_grad_contract`
- `jaxstro.testing.is_inference_ready`
- `jaxstro.testing.render_card`
- `jaxstro.testing.render_family`
- `jaxstro.testing.render_index`
- `jaxstro.testing.resolve_node_ids`
- `jaxstro.testing.scan_module_numeric_literals`
- `jaxstro.testing.test_body_has_assert`
- `jaxstro.testing.trust_report_to_dict`
- `jaxstro.testing.trust_report_to_json`
- `jaxstro.testing.trust_report_to_markdown`
- `jaxstro.units.get_units`

The absence from the table is not a support or maturity claim.

## Module-inherited public types

These immutable record or type constructors inherit their module-level contract:

- `jaxstro.atmospheres.AcquisitionDecision`
- `jaxstro.atmospheres.ArtifactReport`
- `jaxstro.atmospheres.AtmosphereAdapter`
- `jaxstro.atmospheres.AtmosphereAdapterRegistry`
- `jaxstro.atmospheres.AtmosphereCatalogCoverage`
- `jaxstro.atmospheres.AtmosphereLibrary`
- `jaxstro.atmospheres.AtmosphereLibraryCandidate`
- `jaxstro.atmospheres.AtmosphereParams`
- `jaxstro.atmospheres.AtmosphereQuery`
- `jaxstro.atmospheres.AtmosphereSelection`
- `jaxstro.atmospheres.BoszBackend`
- `jaxstro.atmospheres.BoszFile`
- `jaxstro.atmospheres.BoszIndex`
- `jaxstro.atmospheres.BoszMetadata`
- `jaxstro.atmospheres.GridTopology`
- `jaxstro.atmospheres.NewEraBackend`
- `jaxstro.atmospheres.NewEraLowResFile`
- `jaxstro.atmospheres.NewEraLowResHeader`
- `jaxstro.atmospheres.NewEraLowResIndex`
- `jaxstro.atmospheres.NewEraLowResMetadata`
- `jaxstro.atmospheres.OverlapDiagnostic`
- `jaxstro.atmospheres.PreparationResult`
- `jaxstro.atmospheres.PreparedAtmosphere`
- `jaxstro.atmospheres.ProductDescriptor`
- `jaxstro.atmospheres.Sonora2024Metadata`
- `jaxstro.atmospheres.SonoraBackend`
- `jaxstro.atmospheres.TlustyBackend`
- `jaxstro.atmospheres.TlustyFluxMetadata`
- `jaxstro.atmospheres.TopologyKind`
- `jaxstro.atmospheres.TopologySelection`
- `jaxstro.contracts.ADSemantics`
- `jaxstro.contracts.BoundaryContract`
- `jaxstro.contracts.CallableContract`
- `jaxstro.contracts.ContractInventory`
- `jaxstro.contracts.EvidenceKind`
- `jaxstro.contracts.EvidenceReference`
- `jaxstro.contracts.ExecutionBoundary`
- `jaxstro.contracts.FailureMode`
- `jaxstro.contracts.MaturityLevel`
- `jaxstro.contracts.ModuleContract`
- `jaxstro.contracts.SupportLevel`
- `jaxstro.contracts.TransformContract`
- `jaxstro.evidence.ComparisonRecord`
- `jaxstro.evidence.ComparisonRelation`
- `jaxstro.evidence.EnvironmentRecord`
- `jaxstro.evidence.EvidenceArtifact`
- `jaxstro.evidence.EvidenceClass`
- `jaxstro.evidence.EvidenceFreshnessError`
- `jaxstro.evidence.EvidenceIndex`
- `jaxstro.evidence.EvidenceIndexEntry`
- `jaxstro.evidence.EvidenceStatus`
- `jaxstro.evidence.MetricRecord`
- `jaxstro.numerics.Array`
- `jaxstro.numerics.BSpline1D`
- `jaxstro.numerics.BlockDiagonalOperator`
- `jaxstro.numerics.BracketHistory`
- `jaxstro.numerics.BracketProposal`
- `jaxstro.numerics.BracketState`
- `jaxstro.numerics.BracketedRootResult`
- `jaxstro.numerics.BracketedRootState`
- `jaxstro.numerics.CellNeighbors1D`
- `jaxstro.numerics.DenseOperator`
- `jaxstro.numerics.DiagonalOperator`
- `jaxstro.numerics.FaceGeometry1D`
- `jaxstro.numerics.ImplicitRootAssumptions`
- `jaxstro.numerics.ImplicitRootCertificate`
- `jaxstro.numerics.ImplicitRootResult`
- `jaxstro.numerics.LineSearchResult`
- `jaxstro.numerics.LinearOperator`
- `jaxstro.numerics.Mesh1D`
- `jaxstro.numerics.NaturalCubicSpline1D`
- `jaxstro.numerics.ODEResult`
- `jaxstro.numerics.ProductOperator`
- `jaxstro.numerics.RootTrace`
- `jaxstro.numerics.ScaledOperator`
- `jaxstro.numerics.SumOperator`
- `jaxstro.numerics.TransposeOperator`
- `jaxstro.numerics.UniversalKeplerResult`
- `jaxstro.numerics.VerletResult`
- `jaxstro.params.AbstractBijector`
- `jaxstro.params.Exp`
- `jaxstro.params.Identity`
- `jaxstro.params.Parameterization`
- `jaxstro.params.Sigmoid`
- `jaxstro.params.Softplus`
- `jaxstro.provenance.ArtifactHash`
- `jaxstro.provenance.EnvironmentSnapshot`
- `jaxstro.provenance.MethodManifest`
- `jaxstro.quad.AdaptiveClenshawCurtis`
- `jaxstro.quad.AdaptiveTanhSinh`
- `jaxstro.quad.AffineMapResult`
- `jaxstro.quad.ClenshawCurtisRule`
- `jaxstro.quad.DomainMapResult`
- `jaxstro.quad.ErrorKind`
- `jaxstro.quad.ErrorNorm`
- `jaxstro.quad.FejerIIRule`
- `jaxstro.quad.FejerIRule`
- `jaxstro.quad.GaussKronrod`
- `jaxstro.quad.GaussianRule`
- `jaxstro.quad.Hyperrectangle`
- `jaxstro.quad.Infinite`
- `jaxstro.quad.Interval`
- `jaxstro.quad.JacobiMeasure`
- `jaxstro.quad.L1Norm`
- `jaxstro.quad.L2Norm`
- `jaxstro.quad.LaguerreMeasure`
- `jaxstro.quad.LebesgueMeasure`
- `jaxstro.quad.LeftInfinite`
- `jaxstro.quad.MaxNorm`
- `jaxstro.quad.PhysicistsHermiteMeasure`
- `jaxstro.quad.QuadError`
- `jaxstro.quad.QuadResult`
- `jaxstro.quad.QuadStatus`
- `jaxstro.quad.QuadWork`
- `jaxstro.quad.RightInfinite`
- `jaxstro.quad.Romberg`
- `jaxstro.quad.RombergTanhSinh`
- `jaxstro.quad.StandardNormalMeasure`
- `jaxstro.quad.TanhSinhRule`
- `jaxstro.quad.WeightedMeasure`
- `jaxstro.quantity.Dimension`
- `jaxstro.quantity.DimensionError`
- `jaxstro.quantity.EquivalencyError`
- `jaxstro.quantity.Quantity`
- `jaxstro.quantity.QuantityError`
- `jaxstro.quantity.Unit`
- `jaxstro.quantity.UnitConversionError`
- `jaxstro.quantity.UnitParseError`
- `jaxstro.quantity.UnitRegistry`
- `jaxstro.quantity.UnitRegistryError`
- `jaxstro.spectra.CoveragePolicy`
- `jaxstro.spectra.FluxInterpolation`
- `jaxstro.spectra.PointResamplingMethod`
- `jaxstro.spectra.PreparedRectilinearStencil`
- `jaxstro.spectra.PreparedSimplexStencil`
- `jaxstro.spectra.SpectralAxis`
- `jaxstro.spectra.SpectralCoordinate`
- `jaxstro.spectra.SpectralPlan`
- `jaxstro.spectra.SpectralSampling`
- `jaxstro.spectra.SpectralSemantic`
- `jaxstro.spectra.Spectrum`
- `jaxstro.spectra.SpectrumProvenance`
- `jaxstro.spectra.SpectrumResult`
- `jaxstro.spectra.SpectrumStatus`
- `jaxstro.spectra.SpectrumStatusCode`
- `jaxstro.testing.AuditResult`
- `jaxstro.testing.Case`
- `jaxstro.testing.DifferenceReport`
- `jaxstro.testing.Direction`
- `jaxstro.testing.EdgeConfig`
- `jaxstro.testing.EvidenceAnchor`
- `jaxstro.testing.MethodEvidence`
- `jaxstro.testing.NumericalTrustReport`
- `jaxstro.testing.ProvenanceCard`
- `jaxstro.testing.ProvenanceCardError`
- `jaxstro.testing.SourceReference`
- `jaxstro.units.PhotometricUnits`
- `jaxstro.units.UnitSystem`
