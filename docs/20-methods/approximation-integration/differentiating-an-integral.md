---
title: Differentiating an integral
description: Exact sensitivities, accepted-formula replay, moving bounds, units, and independent derivative audits.
---

# Differentiating an integral

An integral can be viewed at three related levels:

1. the exact mathematical object;
2. the adaptive computation that chooses regions or levels; and
3. the fixed quadrature formula accepted by that computation.

Jaxstro replay differentiation returns the derivative of the third object. It
reuses the primal adaptive result, stops the discrete decisions, and
differentiates the accepted formula. This makes a useful numerical sensitivity
available without pretending that refinement choices are smooth.

:::{admonition} The big-picture contract
:class: important
The primal value and its error evidence come from the adaptive solve. The
derivative comes from replaying its accepted formula. Agreement with the exact
integral derivative is a separate, measured claim.
:::

## Before you begin

The following pages introduce the ideas used here:

- [Why JAX?](../../00-start-here/why-jax.md) explains transformation-oriented
  numerical software.
- [What is a derivative?](../../10-foundations/mathematical-objects/what-is-a-derivative.md)
  develops local linear change.
- [What JAX differentiates](../../40-workflows/differentiable-research/what-jax-differentiates.md)
  distinguishes a mathematical derivative from a program derivative.
- [Quantities](../../30-representations/units-quantities/quantities.md) explains
  static units and numerical representations.
- [Adaptive quadrature](./adaptive-quadrature.md) explains the primal methods,
  error estimators, and statuses.

## From an exact object to an executed formula

Let the parameter-dependent integral be

```{math}
:label: eq-diff-integral-object

I(\theta)
=
\int_{a(\theta)}^{b(\theta)}
g(x,\theta)\,\mathrm{d}x,
\qquad
g(x,\theta)=f(x,\theta)\rho(x,\theta).
```

The measure density is included in $g$. For Lebesgue integration,
$\rho=1$. An adaptive run does not manipulate this exact object directly. It
chooses a finite set of regions or a global refinement level and returns an
accepted approximation $Q(\theta)$.

:::{note}
The controller's region choices, stopping decision, and reported error are
scientifically useful evidence. They are not differentiable coordinates of
the exact integral.
:::

## The exact integral derivative

Suppose $g$ and $\partial g/\partial\theta$ are continuous on the relevant
domain, the moving bounds are differentiable, and an integrable dominating
function permits differentiation under the integral sign. The Leibniz rule is

```{math}
:label: eq-diff-integral-leibniz

\frac{\mathrm{d}I}{\mathrm{d}\theta}
=
g\!\left(b(\theta),\theta\right)\frac{\mathrm{d}b}{\mathrm{d}\theta}
-
g\!\left(a(\theta),\theta\right)\frac{\mathrm{d}a}{\mathrm{d}\theta}
+
\int_{a(\theta)}^{b(\theta)}
\frac{\partial g}{\partial\theta}(x,\theta)\,\mathrm{d}x.
```

This is the target mathematical sensitivity. It may not exist at a singular
parameter, a discontinuous boundary, or a point where the assumptions fail.

:::{warning}
Automatic differentiation cannot establish the assumptions behind the
Leibniz rule. A finite AD value is not evidence that differentiation and
integration may be exchanged.
:::

## The accepted fixed-formula derivative

After the primal solve stops, Jaxstro retains the accepted regional partition
or global level. In schematic form, its replay value is

```{math}
:label: eq-diff-integral-replay-formula

Q(\theta)
=
\sum_{r\in\mathcal{A}}
\sum_{j=1}^{n_r}
w_{rj}\,
J_r(\theta)\,
g\!\left(x_{rj}(\theta),\theta\right),
```

where $\mathcal{A}$ is the stopped set of accepted regions, $w_{rj}$ are
fixed reference weights, $x_{rj}$ are mapped nodes, and $J_r$ is the map
Jacobian. Replay differentiates

```{math}
:label: eq-diff-integral-replay-derivative

\frac{\mathrm{d}Q}{\mathrm{d}\theta}
=
\sum_{r\in\mathcal{A}}
\sum_{j=1}^{n_r}
w_{rj}
\frac{\mathrm{d}}{\mathrm{d}\theta}
\left[
J_r(\theta)
g\!\left(x_{rj}(\theta),\theta\right)
\right].
```

The accepted region identities, breakpoints, rule order, accepted level,
stopping decision, error estimate, status, and work counters are stopped.
Only `QuadResult.value` receives the replay derivative. Floating and complex
diagnostics have exact zero tangents; integer and Boolean diagnostics have JAX
`float0` tangents.

## Why the two derivatives can differ

Write the primal quadrature error as

```{math}
:label: eq-diff-integral-error

e(\theta)=Q(\theta)-I(\theta).
```

Where both derivatives exist,

```{math}
:label: eq-diff-integral-error-derivative

\frac{\mathrm{d}Q}{\mathrm{d}\theta}
-
\frac{\mathrm{d}I}{\mathrm{d}\theta}
=
\frac{\mathrm{d}e}{\mathrm{d}\theta}.
```

A small $e$ at one parameter does not by itself bound its derivative.
Derivative trust therefore needs analytic fixtures, a frozen-formula finite
difference, tolerance and capacity ladders, and an adaptive-rerun diagnostic.

The adaptive-rerun finite difference may change the accepted partition or
level between its two samples. That difference diagnoses the complete
adaptive map; it is not automatically a failure of the custom derivative.

## Moving bounds

For a finite interval, replay uses the signed affine map

```{math}
:label: eq-diff-integral-signed-affine

x(t)=\frac{a+b}{2}+\frac{b-a}{2}t,
\qquad
J=\frac{b-a}{2},
\qquad
-1\le t\le 1.
```

The sign is retained in $J$. Replay does not differentiate through
`minimum`, `maximum`, an absolute width, or a discrete orientation. Therefore
reversed intervals preserve the correct sign.

At $a=b=c$, the primal integral is zero, but its bound derivatives need not
be zero. A rule exact for constants has $\sum_j w_j=2$, so the coincident
limit gives

```{math}
:label: eq-diff-integral-coincident-bounds

\left.\frac{\partial Q}{\partial b}\right|_{a=b=c}=g(c),
\qquad
\left.\frac{\partial Q}{\partial a}\right|_{a=b=c}=-g(c).
```

Physical breakpoint locations are stopped. They tell the primal controller
where a known feature lies; they are not treated as differentiable model
parameters.

## Units of a derivative

Let $U_f$, $U_x$, and $U_\rho$ denote the integrand, coordinate, and
density units. The integral unit is

```{math}
:label: eq-diff-integral-units

U_I=U_f U_x U_\rho.
```

For a raw numerical parameter $\theta_{\mathrm{value}}$ representing a
physical quantity with unit $U_\theta$, the physical Jacobian unit is

```{math}
:label: eq-diff-integral-jacobian-units

U_{\mathrm{d}I/\mathrm{d}\theta}=\frac{U_I}{U_\theta}.
```

Jaxstro's alpha quantity boundary restores $U_I$ on the result and on a JVP
tangent. For an auditable quotient unit, differentiate a selected numerical
value and declare the input and output units explicitly. Direct
`jax.grad` over a `Quantity` PyTree does not infer quotient-unit algebra.

:::{tip}
Changing a bound from metres to centimetres changes its numerical derivative.
After converting both derivatives to one declared physical unit, the physical
sensitivity must agree.
:::

## A complete analytic, AD, and finite-difference audit

Consider

```{math}
:label: eq-diff-integral-exponential

I(\theta)=\int_0^1 e^{\theta x}\,\mathrm{d}x
=\frac{e^\theta-1}{\theta},
```

with analytic derivative

```{math}
:label: eq-diff-integral-exponential-derivative

I'(\theta)
=
\frac{(\theta-1)e^\theta+1}{\theta^2}.
```

The public replay check is

```python
import jax
import jax.numpy as jnp

from jaxstro import quad


def integral(theta):
    return quad.integrate(
        lambda x, args: jnp.exp(args * x),
        quad.Interval(0.0, 1.0),
        args=theta,
        method=quad.GaussKronrod(21),
        epsabs=1e-10,
        epsrel=1e-10,
        max_evaluations=147,
        max_regions=4,
        gradient="replay",
    ).value


theta = 0.4
replay_ad = jax.grad(integral)(theta)
analytic = ((theta - 1.0) * jnp.exp(theta) + 1.0) / theta**2

step = 2e-5
adaptive_rerun_fd = (integral(theta + step) - integral(theta - step)) / (2 * step)

assert jnp.allclose(replay_ad, analytic, rtol=1e-8)
assert jnp.allclose(adaptive_rerun_fd, analytic, rtol=1e-8)
```

A complete audit also freezes the center run's accepted evidence and applies
the same central difference to that fixed formula. The repository generator
does exactly this for every adaptive family, then records both the
frozen-formula and adaptive-rerun values separately. Run it with

```bash
uv run --no-sync python scripts/generate_quad_replay_evidence.py --check
```

The generated [quadrature replay derivative evidence](../../60-validation/numerical/quadrature-replay-derivatives.md)
contains the measured cases, gates, units, accepted regions or levels, and
limitations.

## Method and failure boundaries

| Case | Replay contract |
| --- | --- |
| `GaussKronrod` | Replays the accepted segment-local Kronrod formulas |
| `AdaptiveClenshawCurtis` | Replays the accepted segment-local nested formulas |
| `AdaptiveTanhSinh` | Replays accepted regional double-exponential formulas, including improper maps |
| `Romberg` | Replays the accepted global Richardson level |
| `RombergTanhSinh` | Replays the accepted global double-exponential level |
| Moving finite bounds | Differentiable through the signed affine map |
| Supported semi-infinite bounds | The finite boundary value is differentiable through the improper map |
| Improper characteristic scale | Positive numerical configuration; stopped for differentiation |
| Physical breakpoints | Stopped |
| `INVALID_INPUT` | Nonfinite primal value; derivative undefined |
| `NONFINITE_INTEGRAND` | Nonfinite primal value; derivative undefined |
| `gradient="stop"` | Complete result tree has zero or `float0` tangents |

Complex outputs follow JAX's realified differentiation conventions. Use a
realified Jacobian for complex-to-complex maps unless holomorphic behavior is
both mathematically justified and explicitly requested.

For a dimensional improper domain, provide the physical characteristic scale
explicitly. Replay differentiates the integrand and supported finite boundary
while holding this numerical map choice fixed. Expressing the same physical
scale in another compatible unit therefore changes only the raw
representation, not the replay formula in physical coordinates.

## Where to go next

- Use the [quadrature API contract](../../50-api/approximation-integration/quad.md)
  for exact signatures, static inputs, quantity activation, and statuses.
- Use [Auditing derivatives](../../40-workflows/differentiable-research/auditing-derivatives.md)
  to design an independent check.
- Inspect the generated [replay derivative evidence](../../60-validation/numerical/quadrature-replay-derivatives.md)
  and the [scientific evidence index](../../60-validation/evidence-index.md).
