---
title: Adaptive quadrature
description: Error indicators, refinement, stopping, and auditable one-dimensional integration in Jaxstro.
---

# Adaptive quadrature

## The question this method answers

Adaptive quadrature estimates a one-dimensional integral while deciding where
additional function evaluations are most useful. It is appropriate when a
single fixed rule would waste work on easy regions or miss difficult local
structure.

:::{important}
The reported error is evidence from a named estimator, not an exact error
certificate. Two related rules can miss the same unresolved feature.
:::

## Before computation: what should be true?

Write the integral, domain, measure, and expected difficult structure before
choosing a method. Known discontinuities or sharp transitions should be passed
as breakpoints when the method supports them. Decide what absolute and relative
errors would be scientifically meaningful in the units of the raw-array
problem.

:::{tip} A first method choice
Use `GaussKronrod` for a general smooth finite interval, add breakpoints for
known interior structure, try `AdaptiveTanhSinh` for endpoint singularities or
improper domains, and use a Romberg family only when global refinement matches
the integrand's structure.
:::

The current support boundary is:

| Method | Domain | Breakpoints | Error evidence |
| --- | --- | --- | --- |
| `GaussKronrod` | finite `Interval` | yes | embedded Gauss-Kronrod difference |
| `AdaptiveClenshawCurtis` | finite `Interval` | yes | nested-resolution difference |
| `AdaptiveTanhSinh` | finite or improper | finite intervals only | adjacent-level, summation, and tail evidence |
| `Romberg` | finite `Interval` | no | extrapolated refinement difference |
| `RombergTanhSinh` | finite or improper | no | adjacent global-level difference |

Adaptive methods accept `LebesgueMeasure` and `WeightedMeasure`. Inputs are raw
arrays; quantity-valued integration remains a later boundary.

## Define the mathematical objects

Let

```{math}
I = \int_a^b f(x)\,\mathrm{d}x
```

and partition the transformed reference domain into active regions
$[a_i,b_i]$. A regional method produces a value $Q_i$ and a nonnegative
payload-shaped indicator $e_i$. An error norm maps scalar, vector, complex, or
higher-rank payload evidence to one scalar stopping quantity.

`QuadResult` returns the value, `QuadError`, effective tolerance, `QuadStatus`,
and `QuadWork`. These records distinguish a numerical estimate from evidence
about how it was obtained.

## Derive the method

### Regional accounting and tolerance

For active regions, Jaxstro accumulates the value and componentwise error
evidence before applying the chosen norm:

```{math}
:label: eq-adaptive-error-account

\widehat I = \sum_{i=1}^{M} Q_i,
\qquad
\widehat E = \left\lVert\sum_{i=1}^{M} e_i\right\rVert.
```

The effective stopping threshold is

```{math}
:label: eq-adaptive-tolerance

\tau = \max\!\left(\epsilon_{\mathrm{abs}},
\epsilon_{\mathrm{rel}}\lVert\widehat I\rVert\right),
\qquad \widehat E \le \tau.
```

The absolute term protects integrals near zero; the relative term scales with
the estimated integral. Neither term can repair a structurally blind estimator.

### Embedded Gauss-Kronrod

A Kronrod rule reuses the Gauss nodes and adds nodes. If $Q_K$ and $Q_G$ are
the paired estimates, the raw embedded indicator is stabilized by a
roundoff-scale floor $E_{\mathrm{round}}$:

```{math}
:label: eq-adaptive-gk

Q_i = Q_K,
\qquad
e_i = \max\!\left(\lvert Q_K-Q_G\rvert,
E_{\mathrm{round}}\right).
```

The public pairs contain 15, 21, 31, 41, 51, or 61 Kronrod nodes.

### Nested Clenshaw-Curtis

Clenshaw-Curtis evaluates cosine-spaced nodes. An order $n=2^k+1$ rule contains
the lower-resolution node set, so one high-resolution evaluation supplies both
approximations:

```{math}
:label: eq-adaptive-clenshaw-curtis

Q_i = Q_n,
\qquad
e_i = \max\!\left(\lvert Q_n-Q_{(n+1)/2}\rvert,
E_{\mathrm{round}}\right).
```

### Double-exponential refinement

Tanh-sinh maps a real parameter $t$ toward the endpoints double
exponentially,

```{math}
x(t)=\tanh\!\left(\frac{\pi}{2}\sinh t\right).
```

Jaxstro combines adjacent-level disagreement, floating-point summation
evidence, and an outer-shell tail account:

```{math}
:label: eq-adaptive-tanh-sinh

e_i = \lvert Q_{h/2}-Q_h\rvert
      + E_{\mathrm{sum}} + E_{\mathrm{tail}}.
```

This open rule avoids evaluating finite endpoints directly. Domain maps and
their Jacobians extend the same logic to half-infinite and infinite domains.

### Romberg families

Classical Romberg starts from nested trapezoid estimates and applies Richardson
extrapolation:

```{math}
:label: eq-adaptive-romberg

R_{k,0}=T_k,
\qquad
R_{k,j}=R_{k,j-1}
+\frac{R_{k,j-1}-R_{k-1,j-1}}{4^j-1}.
```

`RombergTanhSinh` instead compares nested global tanh-sinh levels without using
the polynomial-error assumption behind Richardson extrapolation.

### Logical work

For a regional rule with $n$ nodes, $M_0$ initial regions, and $r$ bisections,
the exact logical integrand count is

```{math}
:label: eq-adaptive-work

N_{\mathrm{eval}} = n\left(M_0+2r\right).
```

Classical Romberg at completed level $k$ uses $2^k+1$ unique logical points.
`RombergTanhSinh` reports the active-node count at its finest completed level.
These are integrand evaluations, not padded accelerator lanes, compile time, or
wall time.

## What the algorithm actually does

Regional controllers evaluate every declared initial region, sum their value
and error evidence, and repeatedly bisect the region with the largest scalar
error priority. Arrays have fixed capacity so the loop remains JAX
transformable. Global Romberg controllers increase one shared level instead of
building a region partition.

The effective status precedence is invalid input, nonfinite integrand,
convergence, roundoff limitation, and then exhausted capacity. Regional
capacity distinguishes `MAX_EVALUATIONS` from `MAX_REGIONS`. Current A2
controllers emit `INVALID_INPUT`, `NONFINITE_INTEGRAND`, `CONVERGED`,
`ROUNDOFF_LIMITED`, `MAX_EVALUATIONS`, or `MAX_REGIONS` as applicable.
`DIVERGENCE_SUSPECTED` and `ERROR_ESTIMATE_UNAVAILABLE` are reserved vocabulary,
not current controller outputs.

`ErrorKind.EMBEDDED_RULE` identifies Gauss-Kronrod evidence; the other current
families use `ErrorKind.REFINEMENT_DIFFERENCE`. Sparse-grid and replicate-based
kinds are reserved for later method families.

:::{note}
Method configuration, capacity, breakpoint count, and payload shape are static
under JIT. `QuadWork.evaluations` records logical integrand evaluations.
`confidence_level` and `replicates` are not meaningful for these deterministic
methods.
:::

## What JAX differentiates

Phase A2 is primal-only. The only accepted policy is `gradient="stop"`, and
Jaxstro applies `jax.lax.stop_gradient` to the complete result tree. JIT and
VMAP are supported within the static boundaries above, but VMAP repeats the
bounded controller independently for each batch member; it is not shared
adaptive work.

:::{warning}
A zero automatic derivative here means differentiation was deliberately
stopped. It is not the derivative of the mathematical integral. Replay
derivatives and moving-bound derivative evidence remain later work.
:::

## Using it in Jaxstro

```python
import jax.numpy as jnp

from jaxstro import quad

domain = quad.Interval(0.0, 1.0)
methods = (
    quad.GaussKronrod(pair=21),
    quad.AdaptiveClenshawCurtis(initial_order=17),
    quad.AdaptiveTanhSinh(initial_level=3),
    quad.Romberg(initial_level=1),
    quad.RombergTanhSinh(initial_level=1),
)

result = quad.integrate(
    lambda x: x**2,
    domain,
    method=methods[0],
    epsabs=1e-5,
    epsrel=1e-5,
    max_evaluations=2048,
    max_regions=64,
    gradient="stop",
)

assert result.status == quad.QuadStatus.CONVERGED
assert jnp.allclose(result.value, 1.0 / 3.0, rtol=1e-6, atol=1e-6)
```

The same call shape selects each family. Use `quad.Infinite()`,
`quad.RightInfinite(lower)`, or `quad.LeftInfinite(upper)` only with
`AdaptiveTanhSinh` or `RombergTanhSinh`. The complete callable contract is in
[](../../50-api/approximation-integration/quad.md).

## How to audit the result

Check the status before using the value. Then compare observed behavior across
tolerances or capacities, inspect `result.error.kind`, and verify that
`result.work.evaluations` matches the chosen family's logical cost. Use known
breakpoints and independent references whenever the integrand has narrow or
nonsmooth structure.

Executable analytic and failure-envelope cases live in
`tests/validation/test_quad_adaptive_reference.py`; their generated record is
[`docs/validation/quad-adaptive-envelope.json`](../../validation/quad-adaptive-envelope.json).
The broader evidence boundary is indexed in [](../../60-validation/validation.md).

## Where the claim stops

`CONVERGED` means the named estimator satisfied the named tolerance. It does
not prove that the true error is below that tolerance. In particular, embedded
or nested rules can both miss the same narrow feature and report false
estimator convergence. Independent structure-aware checks remain necessary.

Jaxstro does not yet claim replay derivatives, quantity-valued inputs,
multidimensional integration, universal convergence, or performance superiority.
[Quadax](https://github.com/f0uriest/quadax) is an independent comparison and
benchmark implementation, not Jaxstro's runtime owner or dependency.

## Connected ideas

:::{seealso}
Start with [](../../10-foundations/mathematical-objects/functions-units-scales.md)
for functions and scales, then connect raw-array representation to
[](../../30-representations/units-quantities/quantities.md). Compare the fixed
rules in [](./quadrature.md), and use
[](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md)
to understand why unresolved local structure can defeat an apparently small
error indicator.
:::
