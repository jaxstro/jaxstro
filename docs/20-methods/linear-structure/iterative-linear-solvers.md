---
title: Iterative linear solvers in the JAX ecosystem
description: Residuals, Krylov spaces, preconditioning, and delegated solver ownership.
---

# Iterative linear solvers in the JAX ecosystem

Use this page when a linear system is too large or structured for a dense
factorization and you need to interpret an iterative solver's residual and
derivative contract.

:::{important} Ecosystem guide
[Lineax](https://docs.kidger.site/lineax/) and
[JAX](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html)
own linear-operator and iterative-solve mechanics. Jaxstro does not build a
parallel general solver stack.
:::

## The scientific question

An iterative linear solver seeks an adequate approximation without explicitly
forming or factorizing a large matrix. The scientific question is whether the
operator structure, preconditioner, stopping rule, and numerical precision make
the resulting state accurate enough for the downstream observable.

A small residual measures consistency with the represented linear system. It
does not directly measure forward error when the operator is ill-conditioned,
and it says nothing about whether the linearized model itself is scientifically
appropriate.

## Mathematical objects

Let $\mathbf{A}:\mathbb{R}^n\rightarrow\mathbb{R}^n$ be a matrix or linear
operator, with right-hand side $\mathbf{b}$ and iterate $\mathbf{x}_k$. A
preconditioner approximates a useful inverse action while remaining cheaper to
apply than solving the original system.

The norm, scaling, initial guess, finite precision, operator symmetry, positive
definiteness, and preconditioner side all affect which algorithm and stopping
interpretation are valid.

## Core derivation

The residual generates a sequence of directions through repeated operator
application. The associated Krylov space is:

```{math}
:label: eq-iterative-krylov-space

\begin{aligned}
\mathbf{A}\mathbf{x} &= \mathbf{b}, \\
\mathbf{r}_k &= \mathbf{b}-\mathbf{A}\mathbf{x}_k, \\
\mathcal{K}_m(\mathbf{A},\mathbf{r}_0)
&=\mathrm{span}\left\{
\mathbf{r}_0,\mathbf{A}\mathbf{r}_0,\ldots,
\mathbf{A}^{m-1}\mathbf{r}_0
\right\}.
\end{aligned}
```

An iterative method chooses an update from [](#eq-iterative-krylov-space)
according to its operator assumptions and optimality criterion. If
$\mathbf{e}_k=\mathbf{x}-\mathbf{x}_k$, then
$\mathbf{r}_k=\mathbf{A}\mathbf{e}_k$. Therefore a small residual can coexist
with a large forward error when inverse amplification is large.

## What the ecosystem already owns

[Lineax](https://docs.kidger.site/lineax/) owns JAX-native linear-operator and
linear-solve abstractions, while
[JAX](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html)
provides transformation machinery for custom linear solves. Solver recurrences,
operator dispatch, transposes, convergence statuses, and differentiation rules
belong to those owners.

## What Jaxstro may add

Jaxstro may later add a narrow adapter for unit and scale preparation,
scientific provenance, or evidence reporting when a real downstream consumer
needs it. No iterative-solver adapter exists today.

Such an adapter must preserve operator structure and external result statuses.
It must not pretend that a reported residual is a forward-error certificate or
that every solved system has a scientifically supported derivative.

## Evidence required before implementation

Evidence would need:

- dense reference comparisons on small systems with known structure;
- residual and forward-error reports over controlled condition numbers;
- exact-solution tests for symmetric, nonsymmetric, and rank-deficient cases
  within the claimed scope;
- preconditioner tests that preserve the physical solution and report setup;
- transpose and implicit-derivative checks against independent finite
  differences on smooth, converged cases;
- nonconvergence, breakdown, and nonfinite failure cases; and
- provenance for operator assumptions, tolerances, iterations, and status.

## Claim boundary

:::{warning}
Residual convergence is not forward-error convergence. Differentiation through
a solve additionally depends on the represented operator, transpose solve,
conditioning, and convergence of the primal and derivative systems.
:::

This page neither selects one universal Krylov method nor reports Lineax or JAX
performance. It documents the mathematical spine and the delegation boundary.

## Connected foundations and methods

Start with [](../../10-foundations/mathematical-objects/linear-algebra-language-of-change.md)
and [](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md).
Current small dense helpers are described in [](./linear-algebra.md), reusable
operator structure in [](./operators.md), and Jacobian product mechanics in
[](../change-constraints-evolution/autodiff.md).
