---
title: Rootfinding
---

# Rootfinding

## Owner import path

`jaxstro.numerics.rootfinding`

## Purpose

Scalar root solvers separate robust forward values, finite executed-map
derivatives, and certified implicit derivatives.

## Public records and callables

The simple surface includes `bracket_expand`, `bisect`, `bisect_many`, `newton`,
`newton_with_grad`, `newton_1d`, `newton_ppf`, and `monotone_inverse_interp`.
The auditable value-first surface includes `BracketState`, `BracketProposal`,
`RootTrace`, `BracketedRootResult`, `initialize_bracket`, `update_bracket`,
`propose_bracketed`, and `safeguarded_bracketed_root`, with the public proposal
identifiers `PROPOSAL_NONE`, `PROPOSAL_SECANT`, `PROPOSAL_MIDPOINT`,
`PROPOSAL_LO_ENDPOINT`, `PROPOSAL_HI_ENDPOINT`, and
`PROPOSAL_INVERSE_QUADRATIC`. The checkpoint surface adds `BracketHistory`,
`BracketedRootState`, `initialize_bracketed_root_state`, and
`advance_bracketed_root`. Terminal identifiers are `ROOT_STATUS_RUNNING`,
`ROOT_STATUS_EXACT_LO`, `ROOT_STATUS_EXACT_HI`, `ROOT_STATUS_EXACT_INTERIOR`,
`ROOT_STATUS_WIDTH_CONVERGED`, `ROOT_STATUS_MISSING_BRACKET`,
`ROOT_STATUS_NONFINITE_EVALUATION`, and `ROOT_STATUS_MAX_STEPS`.
The derivative-certificate surface includes `ImplicitRootAssumptions`,
`ImplicitRootCertificate`, `ImplicitRootResult`, `implicit_bracketed_root`,
`map_safeguarded_bracketed_root`, `DERIVATIVE_STATUS_CERTIFIED`,
`DERIVATIVE_STATUS_PRIMAL_FAILED`, `DERIVATIVE_STATUS_ASSUMPTIONS_REJECTED`,
`DERIVATIVE_STATUS_NONFINITE`, `DERIVATIVE_STATUS_RESIDUAL_TOO_LARGE`,
`DERIVATIVE_STATUS_SLOPE_ILL_CONDITIONED`, and
`DERIVATIVE_STATUS_BRACKET_TOO_WIDE`.

## Shape and dtype expectations

The safeguarded interface is scalar and uses fixed-length traces. Batched
callers map scalar solves explicitly. Floating inputs should share a precision
appropriate to the requested tolerances.

## JAX transforms and AD classification

`safeguarded_bracketed_root` is value-first. `newton` differentiates its finite
smooth iteration. `implicit_bracketed_root` exposes an implicit function theorem (IFT)
derivative only after every certificate gate passes.

## Failure behavior

Safeguarded solves return typed statuses and diagnostics for missing brackets,
non-finite evaluations, exhausted steps, and convergence. Failed implicit
certificates return NaN values and derivatives rather than an invented result.

## Contract and evidence links

See [](../../20-methods/change-constraints-evolution/rootfinding.md),
[](../../60-validation/index.md), [](../../validation/rootfinding-performance.md),
and [](../../validation/implicit-root-gradients.md).

## Canonical import example

```python
from jaxstro.numerics.rootfinding import safeguarded_bracketed_root
```
