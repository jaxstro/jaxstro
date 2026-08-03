---
title: Special functions
description: >-
  Stable Planck kernels, normalized log weights, and orthogonal polynomial
  bases with explicit units and recurrence contracts.
---

## The question this method answers

How can a reusable numerical layer evaluate common analytic kernels without
hiding units, normalization, or unstable formulas? Jaxstro owns a narrow set of
Planck radiance kernels, log-weight normalization, and polynomial basis values.

:::{important}
The Planck functions return spectral radiance in named CGS coordinates. They do
not integrate a filter, convert radiance to observed flux, or define a source
geometry.
:::

## Before computation: what should be true?

Wavelength in `planck_lambda_cgs` is positive and measured in centimeters;
frequency in `planck_nu_cgs` is positive and measured in hertz; temperature is
positive and measured in kelvin. A log-weight axis must identify the alternatives
to normalize. Polynomial degree must be a concrete nonnegative integer.

:::{warning}
Invalid concrete Planck coordinates raise eagerly, but value-dependent eager
positivity checks are skipped while traced. Compiled callers still own positive
wavelength or frequency and temperature inputs.
:::

## Define the mathematical objects

A spectral radiance density is defined per coordinate interval. $B_\lambda$ is
per unit wavelength; $B_\nu$ is per unit frequency. Their numerical values differ
because the interval widths differ, even when they describe the same radiation.

Log weights $\ell_i$ are unnormalized logarithms of nonnegative relative weights.
Normalization produces $p_i=\exp(\ell_i)/\sum_j\exp(\ell_j)$. A polynomial basis
is a sequence of functions $P_0(x),\ldots,P_d(x)$ whose coefficients can later be
fit by a separate linear-algebra method.

## Derive the method

The wavelength form of Planck's law is

```{math}
:label: eq-planck-lambda
B_\lambda(\lambda,T)=
\frac{2hc^2}{\lambda^5}
\frac{1}{\exp\!\left(hc/(\lambda k_BT)\right)-1}.
```

Coordinate densities conserve radiance under $B_\nu\,d\nu=B_\lambda\,|d\lambda|$
with $\lambda=c/\nu$, so

```{math}
:label: eq-planck-coordinate-change
B_\nu=B_\lambda\left|\frac{d\lambda}{d\nu}\right|=B_\lambda\frac{\lambda^2}{c}.
```

Legendre polynomials illustrate the fixed recurrence used for basis construction:

```{math}
:label: eq-legendre-recurrence
(n+1)P_{n+1}(x)=(2n+1)xP_n(x)-nP_{n-1}(x),
\qquad P_0=1,\quad P_1=x.
```

Chebyshev and Laguerre bases use their corresponding three-term recurrences.

## What the algorithm actually does

The log Planck kernels compute `log(expm1(x))` with a Wien-tail branch that avoids
overflow. The linear kernels exponentiate the log result, so a very small tail
may underflow to zero while its log value stays finite.

`log_normalize` subtracts JAX `logsumexp`; `normalize_log_weights` exponentiates
that result. `legendre_basis`, `chebyshev_t_basis`, and `laguerre_basis` use
fixed-length `jax.lax.scan` recurrences and return shape `x.shape + (degree + 1,)`.
`degree` and `axis` are static where used, so changing either can recompile.

`riccati_bessel_basis` returns $S_\ell(x) = x\,j_\ell(x)$ and
$C_\ell(x) = -x\,y_\ell(x)$ on a leading order axis. The two satisfy the same
three-term recurrence

$$f_{\ell+1}(x) = \frac{2\ell+1}{x} f_\ell(x) - f_{\ell-1}(x),$$

but not in the same direction. $C_\ell$ grows with $\ell$, so upward recurrence
is stable. $S_\ell$ decays like $x^{\ell+1}/(2\ell+1)!!$ once $\ell > x$, and
upward recurrence there amplifies the contaminating growing solution: measured at
$x = 2$, the Wronskian residual below is $1.1\times10^{-16}$ at $\ell = 10$ and
$5.6\times10^{14}$ at $\ell = 25$. **The intermediate values stay finite and
smooth**, so nothing signals the failure. $S_\ell$ therefore uses Miller's
downward sweep, seeded above the wanted order and normalized at the end against
the exact $S_0 = \sin x$.

:::{warning}
The seed order must clear **both the degree and the argument**, not the degree
alone. Miller's sweep is self-correcting only where $\ell > x$, because that is
where the wanted solution grows downward; below $x$ both solutions oscillate,
neither dominates, and whatever relative error entered the oscillatory region
persists. Measured with a seed 60 orders above the degree only: exact at
$\ell = 14$, $x = 0.5$, but wrong by $1.7\times10^{-5}$ at $\ell = 1$, $x = 50$,
and by $O(1)$ at $x = 400$.

`seed_order` is therefore caller-supplied rather than inferred, since the caller
knows the range of $x$ it will sweep. `riccati_seed_order(degree, x_max)`
computes a sufficient value from that range.
:::

`riccati_wronskian_residual` evaluates

$$S_\ell(x)\,C_{\ell-1}(x) - S_{\ell-1}(x)\,C_\ell(x) = -1,$$

which is exact for every order and argument. It is a **recurrence-stability
gate**, not an approximation check: an unstable sweep violates it by orders of
magnitude while the function values themselves remain plausible.

## What JAX differentiates

On positive coordinates and away from numerical underflow, JAX differentiates
the executed Planck log or linear formula with respect to wavelength, frequency,
and temperature. The large-argument stability branch is continuous in value but
still an implementation branch to audit near its switch.

Log normalization has the usual softmax-family derivative on finite logits.
Polynomial values are smooth in $x$ for fixed degree. AD does not differentiate
the integer degree, recurrence length, axis choice, units, or downstream basis
selection. Saturated weights and underflowed linear radiance can erase useful
finite-precision sensitivity even when the mathematical function is smooth.

## Using it in Jaxstro

```python
import jax.numpy as jnp

from jaxstro import constants
from jaxstro.numerics.special import (
    legendre_basis,
    normalize_log_weights,
    planck_lambda_cgs,
    planck_nu_cgs,
)

wavelength_cm = jnp.array(5.0e-5)
temperature = jnp.array(5800.0)
frequency_hz = constants.C_CGS / wavelength_cm

b_lambda = planck_lambda_cgs(wavelength_cm, temperature)
b_nu = planck_nu_cgs(frequency_hz, temperature)
probabilities = normalize_log_weights(jnp.array([3.0, 2.0, 1.0]))
basis = legendre_basis(jnp.array([-0.5, 0.5]), degree=3)

assert jnp.allclose(b_nu, b_lambda * wavelength_cm**2 / constants.C_CGS)
assert jnp.allclose(jnp.sum(probabilities), 1.0)
assert basis.shape == (2, 4)
assert jnp.allclose(
    3.0 * basis[:, 3],
    5.0 * basis[:, 1] * basis[:, 2] - 2.0 * basis[:, 1],
)
```

## How to audit the result

1. Check positivity and units before evaluation.
2. Compare $B_\nu$ and $B_\lambda\lambda^2/c$ at matched coordinates.
3. Compare log and linear Planck values where exponentiation is representable.
4. Verify normalized probabilities sum to one along the requested axis.
5. Check low-degree basis values and every recurrence against direct formulas.
6. Compare AD with central differences on positive, nonsaturated fixtures.

:::{tip}
Prefer log kernels when tails matter. A finite log radiance can retain useful
dynamic range after the linear value has underflowed.
:::

## Where the claim stops

These functions do not define filters, luminosities, priors, fitting policy, or
model selection. Polynomial recurrence parity at low degree does not guarantee
good conditioning at high degree.

The Riccati-Bessel pair is provided with a stable recurrence and an exact
normalization, but the seed order is a **caller obligation**: a value that does
not clear the largest argument in use returns wrong numbers without any signal
other than `riccati_wronskian_residual`. The Wronskian gate certifies recurrence
stability only; it cannot detect an off-by-one in the order labelling, which the
small-argument power law $S_\ell \sim x^{\ell+1}/(2\ell+1)!!$ is tested for
separately.

## Connected ideas

:::{seealso}
Review functions and units in
[](../../10-foundations/mathematical-objects/functions-units-scales.md), connect
spectral coordinates to
[](../../30-representations/spectra-atmospheres/spectra-data-architecture.md),
and use the derivative-audit workflow in
[](../../40-workflows/differentiable-research/auditing-derivatives.md).
Signatures live in [](../../50-api/linear-structure/special.md), with evidence
indexed from [](../../60-validation/validation.md).
:::
