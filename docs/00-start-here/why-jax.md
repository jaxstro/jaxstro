---
title: Why JAX?
description: >-
  Why one scientific map that evaluates, batches, compiles, and differentiates
  is useful—and what JAX does not promise.
---

# Why JAX?

Scientific programs rarely need only one number. We often want to evaluate the
same model for many systems, accelerate it after its structure is stable, and
measure how its outputs change when its inputs change. JAX lets us express one
array program and then transform that program instead of maintaining separate
scalar, batch, accelerator, and derivative implementations.

That is the attraction: not a magic speed switch, but a small set of composable
program transformations. The official [JAX quickstart](https://docs.jax.dev/en/latest/quickstart.html)
introduces the same NumPy-like array model and transformations used throughout
these docs.

## One map, several scientific questions

Suppose a function maps physical parameters to an observable. JAX can apply
several different questions to that function:

- `jax.jit` traces and compiles a stable array program for repeated execution;
- `jax.vmap` maps one function over a batch axis without a Python loop;
- `jax.grad` differentiates a scalar-output map;
- `jax.jvp` pushes a tangent direction forward through a map; and
- `jax.vjp` pulls an output cotangent backward to the inputs.

The last two are useful when a full Jacobian would be wasteful. A JVP asks how
all outputs change along one input direction. A VJP asks how one weighted output
combination depends on all inputs. The [JAX key concepts](https://docs.jax.dev/en/latest/key-concepts.html)
page gives the official vocabulary for transformations, tracing, PyTrees, and
random keys.

## Arrays, functions, and accelerator-portable execution

JAX array programming replaces element-by-element Python work with operations
on whole arrays. The same numerical expression can run on a CPU, GPU, or TPU
when the installed JAX backend supports it. Portability is valuable, but the
first win is conceptual: array shapes, dtypes, and transformations become part
of the scientific program's explicit contract.

Functions work especially well as model boundaries when their inputs contain
all state and their outputs contain all results. JAX represents nested model
state with **PyTrees**: structures made from containers such as tuples, lists,
and dictionaries whose leaves are arrays. Transformations can operate on that
whole structure without hiding state in globals.

Randomness is explicit too. A JAX pseudorandom key is an input that is split to
create new independent keys. Recording and threading keys makes stochastic
calculations reproducible and avoids order-dependent hidden random state.

## What Jaxstro adds

Raw JAX supplies array operations and transformations. Jaxstro adds scientific
contracts around them: explicit units and conventions, named smooth and
nonsmooth domains, transform-aware numerical methods, provenance, independent
gradient audits, and evidence pages that bound the claim a result supports.

This distinction matters. `jax.grad` can faithfully differentiate the executed
program even when that program is not a scientifically valid representation of
the intended derivative. Jaxstro therefore treats a finite transformed result
as something to audit, not as automatic proof.

## Costs and constraints

JAX changes how programs must be written and measured:

- **Compilation latency.** The first `jit` call traces and compiles; later calls
  may be faster. Compare warmed, synchronized execution rather than timing only
  dispatch. The official [benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html)
  explains these requirements.
- **Immutable arrays.** Indexed assignment such as `x[i] = value` is not the
  model. Use functional updates such as `x.at[i].set(value)`.
- **Shape specialization.** A compiled program is specialized to input
  structure, shapes, dtypes, and static arguments. New variants may trigger new
  compilations.
- **Tracing constraints.** Python branches, loops, and conversions that depend
  on traced array values can fail or freeze the wrong behavior. Use JAX control
  flow where runtime array values choose the path.
- **Explicit state.** Keys, parameters, and evolving state must travel through
  function boundaries. This adds bookkeeping while making execution auditable.
- **Precision choices.** JAX defaults commonly favor 32-bit computation.
  Scientific work must choose and test precision deliberately; Jaxstro exposes
  an explicit high-precision configuration.

## When JAX is the wrong tool

JAX does not make an algorithm correct. It does not repair an invalid model,
an unstable discretization, a unit mistake, or a missing convergence study.

JAX does not make every program faster. Small one-off calculations, irregular
host-side workflows, dynamic data structures, and programs dominated by
compilation can be simpler or faster in ordinary Python and NumPy.

JAX does not make every derivative scientifically meaningful. Hard branches,
clipping, discrete choices, non-converged solves, singular points, and an
incorrect mathematical model can all yield a derivative that is finite but
answers the wrong question.

Choose JAX when the scientific map is naturally array-oriented and repeated
evaluation, batching, compilation, or differentiation earns the additional
constraints. Choose a simpler tool when those transformations are not part of
the research question.

Continue to [](./jax-from-first-principles.md) to apply these ideas to one small
map before using them in a larger calculation.
