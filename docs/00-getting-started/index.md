---
title: Getting started
description: >-
  Install jaxstro with uv, turn on float64, and run one worked example —
  safe math plus a differentiable root-find — before you trust anything else.
---

This is the page you open first. By the end of it you will have jaxstro
installed, float64 enabled, and one small example running that exercises the two
habits everything else in the package depends on: **guard your arithmetic** and
**verify solver gradients independently**.

## Prerequisites

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). jaxstro
depends only on JAX, jaxlib, jaxtyping, and equinox — no astropy, no scipy, no
solver libraries (see [](../30-decisions/0001-thin-foundation-posture.md)). A
working knowledge of `jax.numpy` helps but is not assumed.

## Install

The project uses uv, which manages the virtual environment for you:

```bash
git clone https://github.com/drannarosen/jaxstro
cd jaxstro
uv sync                 # core install
uv sync --extra dev     # add pytest, ruff, mypy for development
```

Run anything through `uv run` so it uses the project environment:

```bash
uv run python -c "import jaxstro; print(jaxstro.__version__)"
```

## Turn on float64 first

JAX defaults to float32. For scientific work that is not good enough: a
cancellation that loses seven digits leaves you with none. jaxstro ships a single
switch that sets `jax_enable_x64=True` and requests the highest matmul precision.
Call it **before you create any JAX arrays**, because the dtype default is read at
array-creation time:

```python
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()  # float64 everywhere; do this first
```

:::{important} Order matters
If you build arrays before calling `enable_high_precision()`, they are created as
float32 and stay that way. Make this the first line of your script, above every
other import that might touch JAX.
:::

## A first worked example: safe math + a root-find

Here is the whole habit in one example. We solve a tiny physical equation —
"how many scale heights does an isothermal density profile need to fall to a
chosen fraction of its central density?" — and verify that the answer is
**differentiable** with respect to that fraction.

The density profile is $\rho(r) = \rho_0 e^{-r/h}$ with scale height $h$. Define
$x=r/h$ and the density fraction $f=\rho/\rho_0$. The root problem is then

```{math}
e^{-x} - f = 0.
```

Nondimensionalizing first keeps the Newton solve near order unity instead of
mixing radii near $10^{18}$ cm with densities near $10^{-22}$ g cm$^{-3}$. The
analytic result $x=-\ln f$ is used only as an independent check.

```python
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax
import jax.numpy as jnp
from jaxstro.numerics.rootfinding import newton
from jaxstro.numerics.stats import safe_log

def scale_heights_at_fraction(fraction):
    # Keep x0 independent of fraction so the measured sensitivity comes from
    # the solved equation, not from a parameter-dependent initial guess.
    residual = lambda x: jnp.exp(-x) - fraction
    return newton(residual, x0=2.0)

fraction = jnp.asarray(0.1)
scale_heights = scale_heights_at_fraction(fraction)
analytic_scale_heights = -safe_log(fraction)

# Compare automatic differentiation with an independent central difference.
ad_grad = jax.grad(scale_heights_at_fraction)(fraction)
eps = 1.0e-5
fd_grad = (
    scale_heights_at_fraction(fraction + eps)
    - scale_heights_at_fraction(fraction - eps)
) / (2.0 * eps)

print(f"x = r/h = {scale_heights:.12f}")
print(f"analytic x = {analytic_scale_heights:.12f}")
print(f"dx/df: AD = {ad_grad:.12f}, FD = {fd_grad:.12f}")
```

At $f=0.1$, the answer is $x=\ln 10\approx2.302585$: the density falls to one
tenth of its central value after about **2.30 scale heights**. If $h=1$ pc, that
is a radius of about 2.30 pc.

The analytic sensitivity is

```{math}
\frac{dx}{df}=-\frac{1}{f},
```

so the expected gradient is $-10$ at $f=0.1$. The example reports AD $=-10$ and
central FD $\approx-10.00000003$. Agreement among the solved value, analytic
answer, AD, and FD is the evidence that this smooth path is working.

:::{important} Fixed iteration is necessary, not sufficient
`newton` uses a fixed number of JAX-traceable iterations, but that alone does not
guarantee a scientifically meaningful derivative. The update must also remain on
a smooth path, and the initial guess must not smuggle the analytic answer's
parameter dependence into the result. Branch-selected solvers such as bisection
remain excellent forward-value tools, but their root sensitivities need a
different contract.
:::

## Where to go next

You just used three ideas without unpacking them: *why nondimensionalization
improves a solve*, *why fixed iteration does not by itself prove a derivative*,
and *why `safe_log` guards the analytic check*. These are developed in the
theory section.

- Read [](../10-theory/index.md) — the ten-principle thesis on AD-safe numerics.
- Then [](../10-theory/rootfinding.md) explains the distinct value and gradient
  contracts for `bisect`, `newton`, and `newton_ppf`.
- When you need a call signature, jump to [](../40-api/index.md).
