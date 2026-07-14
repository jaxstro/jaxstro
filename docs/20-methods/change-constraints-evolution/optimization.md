---
title: Optimization helpers
description: >-
  Small objective, line-search, and convergence helpers that support
  differentiable scientific optimization without becoming an optimizer stack.
---

Optimization in `jaxstro` is deliberately modest. The package does not own an
optimizer framework, a parameter scheduler, or a replacement for Optax. It owns
the reusable pieces that scientific objectives need before an optimizer is even
chosen: robust residual losses, objective summaries, a simple fixed-iteration
line search, and convergence diagnostics with explicit tolerances.

## Robust residual losses

`squared_loss(residual)` returns the standard half-squared residual. It is the
right default when the residual model is Gaussian and outliers are not the main
failure mode.

`huber_loss(residual, delta=...)` keeps the quadratic core near zero and switches
to linear growth in the tails. The switch point is a kink, so gradients are
well-defined away from `|residual| == delta` and should not be interpreted as a
smooth validation target exactly at the kink.

`pseudo_huber_loss(residual, delta=...)` is the smooth alternative:

```{math}
\rho_\delta(r) = \delta^2\left(\sqrt{1 + (r/\delta)^2} - 1\right).
```

It behaves quadratically near zero and approximately linearly for large
residuals while preserving a smooth autodiff path.

## Objective summaries

`objective_summary(residuals, weights=None)` reports the scalar loss, mean loss,
RMSE, maximum absolute residual, and residual count. When weights are supplied,
they weight the squared residual contribution and set the normalization for mean
loss and RMSE. A zero total weight is guarded with a finite denominator so the
summary remains finite.

## Fixed-iteration line search

`armijo_backtracking(...)` is a small Armijo sufficient-decrease helper. It does
not terminate early; under `jax.jit`, `f` and `max_steps` are static and the
runtime path uses a fixed-count `lax.scan`. The result records the first accepted
candidate if one appears, otherwise the final backtracked candidate.

This is useful as a building block for tests, demonstrations, and lightweight
scientific solvers. It is not a substitute for an optimizer library.

## Convergence diagnostics

`relative_step_norm`, `gradient_inf_norm`, and `convergence_summary` provide
optimizer-agnostic stopping diagnostics. They separate step, gradient, and loss
tolerances so callers can report which criterion passed rather than collapse all
failure modes into a single boolean.

## Validation

The validation suite checks FD-vs-AD agreement for the smooth robust loss,
Huber loss away from the kink, and weighted objective-summary loss. Unit tests
cover JIT, VMAP, Armijo descent behavior, and convergence-summary booleans.
