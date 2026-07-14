# What JAX differentiates

Use this page when a derivative from `grad`, JVP, or VJP must be interpreted as
a derivative of an executed program rather than accepted automatically as a
scientific sensitivity.

## Start from the executed program

A Python function, an intended mathematical relation, and the program traced by
JAX are related but not identical objects. JAX traces array operations for
particular abstract shapes and dtypes, builds a JAXPR, and transforms that
executed program. Static Python structure and values marked static select which
program exists before differentiation begins.

PyTrees organize parameters and state, but differentiation still acts on the
inexact array leaves selected by the transformed call. Integer indices,
strings, and static metadata can influence which computation is traced without
receiving a tangent or cotangent.

## JVP and VJP

For $f:\mathbb{R}^n\rightarrow\mathbb{R}^m$ with Jacobian $J_f(x)$, a
Jacobian-vector product pushes a tangent $v$ forward:

```{math}
:label: eq-workflow-jvp

\operatorname{JVP}_f(x;v)
=
\left(f(x), J_f(x)v\right).
```

A vector-Jacobian product pulls a cotangent $w$ backward:

```{math}
:label: eq-workflow-vjp

\operatorname{VJP}_f(x;w)
=
\left(f(x), w^{\mathsf T}J_f(x)\right).
```

Forward mode is often appropriate for few input directions; reverse mode is
often appropriate for a scalar output and many parameters. Agreement between
modes checks implementation consistency, not scientific meaning.

## Control flow and static structure

Python branches on traced values fail because tracing cannot turn an unknown
Boolean into host control flow. `jax.lax.cond`, `scan`, and related primitives
represent value-dependent control flow in JAXPR. Their derivative is the
derivative of the executed branch or finite iteration. Discrete branch
selection, array shapes, and loop lengths remain structural boundaries.

`jnp.where` selects values but evaluates both branch expressions. An invalid
dead expression can therefore contaminate derivatives. Sanitize divisions,
logs, and square roots before selection. A fixed-length `scan` exposes the
derivative of those executed steps; it does not automatically expose the
derivative of an ideal converged solution.

## Concrete audit procedure

1. Write the intended mathematical map, domain, units, and sensitivity target.
2. Inspect shapes, dtypes, static arguments, and differentiable PyTree leaves.
3. Use `jax.make_jaxpr` to identify branch and control-flow structure.
4. Compare JVP and VJP contractions on the same direction.
5. Compare against an analytic derivative or a step-size study using an
   independent central finite difference.
6. Exercise branch, clipping, singular, and invalid-domain boundaries.
7. State whether the result is an executed-map derivative, a custom rule, or a
   certified mathematical sensitivity.

## Derivative versus scientific sensitivity

A finite AD value says that the transformed program returned a finite tangent
or cotangent. It does not show that the model is identifiable, the branch is
unique, a clipped parameter remains physically meaningful, or an iterative
result approximates an ideal solution closely enough. Those are separate
assumptions and evidence gates.

## Connected ideas

Use [](../../20-methods/change-constraints-evolution/autodiff.md) for method
background, [](../../10-foundations/mathematical-objects/what-is-a-derivative.md)
for the mathematical definition, [](./auditing-derivatives.md) for numerical
checks, and [](../investigations/root-values-and-sensitivities.md) for an
executable derivative certificate.
