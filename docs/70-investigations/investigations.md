---
title: Executable research investigations
description: Predict, compute, audit, and state the warranted claim using public Jaxstro APIs.
---

# Executable research investigations

Each investigation imports public APIs from one repository-owned Python module. The page teaches the reasoning; the example is the executable source of truth.

| Investigation | Public contracts | Indexed evidence | Known limitations |
| --- | --- | --- | --- |
| [Interpolation boundary policies](./interpolation-boundary-policies.md) | `numerics.interp1d`, `numerics.regular_grid_interp` | No standalone indexed artifact; callable validation only | Interior affine derivative evidence does not apply at knots or boundary-policy transitions.; Callable validation exists, but no standalone interpolation artifact is yet indexed. |
| [Finite power-law removable limit](./powerlaw-removable-limit.md) | `numerics.powerlaw_cdf`, `numerics.powerlaw_logpdf`, `numerics.powerlaw_ppf` | No standalone indexed artifact; callable validation only | The fixture validates one finite support and does not select a physical stellar-mass model.; Callable validation exists, but no standalone power-law artifact is yet indexed. |
| [Root values and sensitivities](./root-values-and-sensitivities.md) | `numerics.implicit_bracketed_root`, `numerics.safeguarded_bracketed_root` | `rootfinding.implicit-gradients`, `rootfinding.performance` | The analytic quadratic fixture does not establish uniqueness or conditioning for arbitrary residuals.; The value-first solver retains no ideal-root derivative claim. |

Every command prints measured results only as:

| Metric identity | Symbol | Value | Units |
| --- | --- | ---: | --- |
| example metric | `m` | measured | explicit units |

The curriculum registry is checked against the scientific contract registry and evidence index. Missing indexed evidence remains visible rather than being inferred from a passing example.
