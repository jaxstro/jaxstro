---
title: Getting started
description: >-
  Install jaxstro with uv, turn on float64, and run one worked example —
  safe math plus a differentiable root-find — before you trust anything else.
---

This is the page you open first. By the end of it you will have jaxstro
installed, float64 enabled, and one small example running that exercises the two
habits everything else in the package depends on: **guard your arithmetic** and
**differentiate through your solvers**.

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
"at what radius does an isothermal density profile fall to a target value?" — and
we do it so the answer is **differentiable** with respect to that target.

The density profile is $\rho(r) = \rho_0\, e^{-r/h}$ with scale height $h$. We want
the radius $r$ where $\rho(r) = \rho_\mathrm{target}$. Inverting by hand gives
$r = h \ln(\rho_0 / \rho_\mathrm{target})$, which we will use only to check the
solver — the point is that the solver gets there without us inverting anything.

```python
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax
import jax.numpy as jnp
from jaxstro.numerics.rootfinding import bisect
from jaxstro.numerics.stats import safe_log

rho0 = 1.0e-22        # central density [g cm^-3]
h = 3.086e18          # scale height: 1 pc in cm

def radius_at_density(rho_target):
    # Solve rho0 * exp(-r/h) = rho_target  ->  find the root of the residual.
    # safe_log guards the log against a zero/negative argument without
    # killing the gradient (see the theory section on floating-point math).
    r_guess_hi = h * safe_log(rho0 / rho_target)  # analytic, used only as a bracket
    f = lambda r: rho0 * jnp.exp(-r / h) - rho_target
    return bisect(f, a=0.0, b=2.0 * r_guess_hi)

rho_target = 1.0e-23  # one tenth of central
r = radius_at_density(rho_target)
print(f"r = {r:.6e} cm = {r / 3.086e18:.4f} pc")
```

For these numbers the answer is $r = h\ln(10) \approx 2.303\,h$, i.e.
about **2.30 pc** — a sanity check you can do in your head. The solver lands on
it to roughly 15 digits because `bisect` runs 50 fixed iterations (each halves the
bracket, $2^{-50} \approx 10^{-15}$).

Now the part that matters. Because the whole computation is JAX-native and uses a
**fixed-iteration** solver (never `while_loop`), you can differentiate the answer:

```python
drho = jax.grad(radius_at_density)(rho_target)
print(f"dr/d(rho_target) = {drho:.6e} cm / (g cm^-3)")
```

The analytic derivative is $\partial r / \partial\rho_\mathrm{target} =
-h / \rho_\mathrm{target}$. For our values that is
$-3.086\times10^{18} / 10^{-23} \approx -3.09\times10^{41}$, and the autodiff
result matches. That agreement — finite-difference versus autodiff — is the test
every differentiable function in jaxstro must pass.

## Where to go next

You just used two ideas without unpacking them: *why a fixed-iteration solver is
differentiable but a convergence loop is not*, and *why `safe_log` guards the
gradient as well as the value*. Both are principles in the theory section.

- Read [](../10-theory/index.md) — the ten-principle thesis on AD-safe numerics.
- Then [](../10-theory/rootfinding.md) explains exactly why `bisect`, `newton`,
  and `newton_ppf` behave the way they did above.
- When you need a call signature, jump to [](../40-api/index.md).
