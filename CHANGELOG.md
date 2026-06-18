# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## 0.1.0 (unreleased)

First public-grade release of jaxstro, the shared JAX-native foundation library for
the differentiable astrophysics ecosystem (gravax, progenax, fluxax, …). Covers the
Phase B consolidation/hardening arc and the `jaxstro.params` selective-gradient layer.

### Added

- **`jaxstro.params`** — Equinox-only selective-gradient layer for gradient-based
  inference over a *subset* of a model's leaves: `Parameterization`
  (`from_where` / `from_filter` front doors, `to_vector` / `from_vector` PyTree↔flat-vector
  bridge, `log_det_jacobian` change-of-variables term) and a bijector registry —
  `Identity`, `Exp`, `Softplus`, `Sigmoid(lo, hi)` — for unconstrained-space inference,
  each with an analytic, float64-stable, grad-checked log-Jacobian. No new core
  dependency (optax/numpyro are validation-only, behind the `[ml]` extra).
- **`PhotometricUnits` + Jy/AB constants** — `JY_CGS` and the AB magnitude zero-points
  (Oke & Gunn 1983), unblocking fluxax Phase 2 photometry. The AB *linear* flux scale is
  poisoned with `NaN` so any accidental linear-space use fails loud.
- **Gaussian quadrature factory** in `numerics` — Gauss–Legendre / Gauss–Hermite nodes
  and weights plus a Hermite probabilists' basis (built from physicists' `hermgauss` via a
  √2 rescale, for byte-parity with the sibling construction).
- **`inverse_cdf_draw`** (differentiable inverse-CDF sampling) and a generic **`newton_ppf`**
  base solver (fixed-iteration `lax.scan`, no `while_loop`) hoisted into `numerics`.
- **`jaxstro.testing`** — FD-vs-AD grad-check audit engine, deduplicating the core
  grad-audit harness previously copied in fluxax and progenax; bound on the top-level
  namespace.
- **Constants round-out (CODATA 2018)** — `ALPHA_FS` (fine-structure constant), `E_ESU`
  (elementary charge in esu), `R_E` (classical electron radius), `SIGMA_T` (Thomson cross
  section), and `R_GAS` (molar gas constant), with internal cross-checks
  (`SIGMA_T ≈ (8π/3)·R_E²`, `R_GAS ≈ k_B·N_A`, `E_ESU = e·c/10`).
- **Release baseline** — hatchling build backend, Apache-2.0 license, `py.typed`
  marker, trove classifiers, equinox promoted to a core dependency (ADR-0002), PEP 735
  dependency groups, and a checked-in `uv.lock`.

### Changed

- **`cumulative_trapz` uniform path → dx-outside** — the uniform-grid branch now factors
  the constant `dx` out of the running sum, for byte-parity with the progenax
  implementation it consolidates.
- **`condition_number` singular sentinel → `+inf`** — a singular matrix now returns `+inf`
  (matching `numpy.linalg.cond`) instead of a misleading `0.0`.

### Fixed

- **`A_RAD` `7.565767e-15 → 7.565733250e-15`** — the radiation constant is now `4σ/c`
  exactly (derived from the CODATA-2018 Stefan–Boltzmann constant and the exact speed of
  light); the old value was internally inconsistent by ~4.5 ppm and its test was vacuous
  (passing only on the `pytest.approx` absolute floor).
- **Julian-vs-tropical year provenance** — corrected the comment on `YR_S`
  (`3.15576e7 s = 365.25 × 86400`, the Julian year, not the tropical year ≈ 365.2422 d);
  values unchanged, with per-constant citations added.
- **`project_onto(eps=0)` NaN** — projecting onto a vector with `eps=0` no longer produces
  a `NaN`.
