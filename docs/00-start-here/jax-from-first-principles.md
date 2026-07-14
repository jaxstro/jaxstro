---
title: JAX from first principles
description: >-
  Build a mental model for JAX arrays, transformations, tracing, control flow,
  random keys, PyTrees, and precision from one scientific function.
---

# JAX from first principles

Use this page when you need a beginner mental model for how JAX transforms one
explicit scientific function without changing the question that function asks.

JAX becomes easier to reason about when you separate three things: the
mathematical map you intend, the Python function that expresses it, and the
traced program a transformation sees. This page develops that distinction from
one dimensionless luminosity relation.

The official [quickstart](https://docs.jax.dev/en/latest/quickstart.html) and
[JAX 101](https://docs.jax.dev/en/latest/jax-101/01-jax-basics.html) provide a
broader introduction. Here the goal is narrower: learn enough to predict what a
scientific transform will do, compute it, and audit the result.

## Choose 64-bit precision before creating arrays

JAX commonly defaults to 32-bit values. Jaxstro makes the scientific precision
choice explicit, and that choice must happen before importing the executable
map or creating any JAX arrays:

```python
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax
import jax.numpy as jnp
from examples.onboarding.first_jax_map import (
    batched_scaled_luminosity,
    compiled_scaled_luminosity,
    scaled_luminosity,
)
```

More bits do not fix a bad algorithm, but insufficient precision can erase a
small residual or corrupt a cancellation-sensitive audit. State the precision
used and test whether the conclusion is stable under an appropriate alternative.

## A Python function, a mathematical map, and a traced program

For a star compared with a reference star, the Stefan-Boltzmann scaling is

```{math}
\frac{L}{L_\mathrm{ref}}
=
\left(\frac{R}{R_\mathrm{ref}}\right)^2
\left(\frac{T}{T_\mathrm{ref}}\right)^4.
```

The imported executable example in `examples/onboarding/first_jax_map.py`
expresses that map directly:

```python
def scaled_luminosity(radius_ratio, temperature_ratio):
    radius = jnp.asarray(radius_ratio)
    temperature = jnp.asarray(temperature_ratio)
    return radius**2 * temperature**4


batched_scaled_luminosity = jax.vmap(scaled_luminosity)
compiled_scaled_luminosity = jax.jit(scaled_luminosity)
```

At radius ratio 2 and temperature ratio 0.5, the predicted luminosity ratio is
$2^2(0.5)^4=0.25$. Eager evaluation computes it immediately:

```python
value = scaled_luminosity(2.0, 0.5)
```

The function is the reusable map. `vmap`, `jit`, and `grad` are transformations
of that map, not rewritten scientific models.

## Batch the map with `vmap`

`vmap` adds array axes to a scalar-shaped function. Both input arrays below have
one lane per star:

```python
radii = jnp.array([1.0, 2.0, 0.5])
temperatures = jnp.array([1.0, 0.5, 2.0])
luminosities = batched_scaled_luminosity(radii, temperatures)
```

The function body still describes one star. `vmap` owns the batch traversal, so
increasing the batch size changes an array shape instead of the number of Python
calls.

## Compile repeated work with `jit`

`jit` traces array operations, lowers them to an accelerator program, and
compiles that program for the input signature:

```python
first = compiled_scaled_luminosity(2.0, 0.5)   # trace and compile
later = compiled_scaled_luminosity(3.0, 0.75)  # reuse when signatures match
```

Compilation is most useful for repeated, substantial array work. The first call
includes tracing and compilation, while array work may be dispatched
asynchronously. The official [JIT guide](https://docs.jax.dev/en/latest/jit-compilation.html)
explains what is compiled and how to measure it.

## Differentiate the scientific map

For $\ell=r^2t^4$, the analytic partial derivatives are
$\partial\ell/\partial r=2rt^4$ and
$\partial\ell/\partial t=4r^2t^3$. JAX can calculate both from the same
function:

```python
d_radius, d_temperature = jax.grad(
    scaled_luminosity, argnums=(0, 1)
)(2.0, 0.5)

assert jnp.allclose(d_radius, 0.25)
assert jnp.allclose(d_temperature, 2.0)
```

The analytic result is the audit. A finite gradient alone would not establish
that the code represents the derivative we intended.

## Inspect what JAX traces

A transform replaces concrete array values with abstract tracers that carry
information such as shape and dtype. `make_jaxpr` displays the resulting
primitive program:

```python
print(jax.make_jaxpr(scaled_luminosity)(2.0, 0.5))
```

Read the output as evidence about the executed array program, not as a new
physical derivation. The official [tracing guide](https://docs.jax.dev/en/latest/tracing.html)
explains the distinction between concrete values and tracers.

## Immutable arrays and explicit updates

JAX arrays are immutable. Instead of mutating one lane, create an updated value:

```python
radii = jnp.array([1.0, 2.0, 3.0])
corrected_radii = radii.at[1].set(2.1)
```

`radii` is unchanged; `corrected_radii` is a new array value. This functional
style lets transformations reason about data flow without hidden mutation.

## Explicit random keys

Randomness is data in JAX. Create a key, split it, and pass each subkey to the
operation that consumes it:

```python
key = jax.random.key(2026)
temperature_key, radius_key = jax.random.split(key)
temperature_noise = jax.random.normal(temperature_key, shape=(3,))
radius_noise = jax.random.normal(radius_key, shape=(3,))
```

Never silently reuse a key when you intend a new draw. Explicit keys make the
random inputs to a result recordable and reproducible.

## PyTrees carry structured state

Scientific parameters seldom arrive as one flat array. JAX transformations can
traverse nested dictionaries, tuples, lists, and registered classes as PyTrees:

```python
star = {
    "ratios": {"radius": jnp.array(2.0), "temperature": jnp.array(0.5)},
    "label": "example",
}
leaves, structure = jax.tree.flatten(star["ratios"])
```

Array leaves remain dynamic data; structure and non-array metadata may be
static. Keep that boundary intentional because static changes can trigger new
compiled variants.

## Runtime control flow uses JAX primitives

A Python `if` needs a concrete Boolean while tracing. When an array value
chooses between two runtime branches, use transform-aware control flow:

```python
from jax import lax


def capped_scaled_luminosity(radius_ratio, temperature_ratio, cap):
    value = scaled_luminosity(radius_ratio, temperature_ratio)
    return lax.cond(value > cap, lambda x: cap, lambda x: x, value)
```

Both branches must return compatible structures, shapes, and dtypes. Also ask
whether the branch creates a scientifically meaningful derivative boundary.

## Common tracing errors

Typical failures occur when traced values are used where Python requires a
concrete value, for example, converting a tracer to `float`, using it as a list
index, branching with a Python `if`, or choosing a dynamic array shape. The
official [errors guide](https://docs.jax.dev/en/latest/errors.html) names these
failures and shows their transform-aware replacements.

When an error appears, ask:

1. Which value became a tracer?
2. Does this decision belong at trace time or runtime?
3. Can the program use fixed shapes and JAX control flow?
4. Is the transformed program still the scientific map you intend?

Next, connect this orientation to the
[](../05-foundations/foundations.md) route for mathematical meaning, read the
[](../10-theory/autodiff.md) chapter for deeper derivative-product contracts,
or use [](./first-research-calculation.md) to place a JAX computation inside the
full **predict -> compute -> audit** cycle.
