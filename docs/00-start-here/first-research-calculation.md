---
title: "How to learn with Jaxstro: predict, compute, audit"
description: >-
  A research-student workflow for turning mathematical expectations into
  executed numerical evidence and appropriately bounded scientific claims.
---

# How to learn with Jaxstro: predict, compute, audit

Jaxstro's documentation does not ask you to trust an answer because the code
ran or because automatic differentiation returned a finite number. Instead,
every substantial example can be read as a recurring cycle:

```{math}
\text{predict} \longrightarrow \text{compute} \longrightarrow \text{audit}
\longrightarrow \text{new prediction}.
```

This order is deliberate. Prediction prevents post-hoc storytelling. Computing
connects a mathematical model to the algorithm that actually executed. Auditing
separates evidence from plausibility and narrows the claim to what the method,
data, and checks support.

The rationale for this structure, including why optional background recovery is
rigorous rather than remedial, is in [](../10-foundations/foundations.md). Use
the [](./choose-your-path.md) page whenever a concept needs reactivation before
you continue.

## Predict

Before running code, write down what should happen. Name the units, signs,
limiting cases, invariants, conditioning, and expected failure state. Decide
whether a derivative should exist and whether you want the derivative of the
executed algorithm or of an underlying mathematical relation.

Prediction is not guessing the last decimal place. It is committing to enough
structure that a surprising output can teach you something.

## Compute

Choose an explicit numerical method and retain its evidence. Inspect tolerances,
branches, fixed shapes, telemetry, and status instead of keeping only a plausible
scalar. A finite output is not yet a scientific result.

JAX transformations are part of the computation contract. A function may
compose with `jit`, preserve values under `vmap`, or support a certified
parameter derivative without promising all three under the same cost and
smoothness assumptions.

## Audit

Compare the result with an analytic identity, limiting case, independent
numerical method, or convergence study. Check units and invariants. Inspect
provenance and decide exactly what the evidence warrants claiming.

An audit can pass, reject the result, or reveal that the original question was
underspecified. The audit starts the next prediction: revise the model, method,
tolerance, or claim, then run the cycle again.

## Example 1: a bracketed root

Suppose a continuous signed residual $G(x)$ changes sign on an interval. The
mathematical question is where $G(x)=0$; the computational question also
includes whether the bracket remained valid and why the algorithm stopped.

```{list-table} Predict -> compute -> audit for a scalar root
:header-rows: 1

* - Stage
  - Research action
* - Predict
  - Check endpoint signs, units of $x$ and $G$, possible multiplicity, and
    whether the selected branch is smooth and unique.
* - Compute
  - Use `safeguarded_bracketed_root`; retain terminal status, signed residual,
    final bracket, proposal kinds, and executed masks.
* - Audit
  - Verify every admissible trace slot preserves a sign bracket, compare the
    final width with the coordinate tolerance, and refuse an implicit derivative
    unless the separate certificate passes.
```

The [](../10-theory/rootfinding.md) chapter develops this example. The
[](../60-validation/index.md) page links its claims to executable tests and
evidence artifacts.

## Example 2: a removable distribution limit

A finite power law contains a removable singularity at $\alpha=-1$. A direct
formula and an exact-value branch can give correct forward values while still
giving the wrong derivative with respect to $\alpha$.

```{list-table} Predict -> compute -> audit for a limiting distribution
:header-rows: 1

* - Stage
  - Research action
* - Predict
  - Derive the logarithmic limiting normalization and its parameter derivative;
    require continuity from both sides of $\alpha=-1$.
* - Compute
  - Evaluate `powerlaw_cdf`, log-density, and inverse CDF using smooth
    `expm1(x)/x` and `log1p(x)/x` kernels rather than a derivative-breaking
    equality branch.
* - Audit
  - Check normalization, support boundaries, CDF/PPF round trips, analytic
    limits, and central finite differences against AD across the limit.
```

See [](../10-theory/distributions.md) for the derivation and the
[](../40-api/index.md) page for public signatures.

## Use this cycle in research

When reading a module chapter, pause at each Predict, Compute, and Audit block.
Write the prediction before executing the notebook or test, then save the audit
evidence with the method configuration and provenance.

The goal is not to make every calculation elaborate. It is to make the strength
of the claim proportional to the evidence behind it.
