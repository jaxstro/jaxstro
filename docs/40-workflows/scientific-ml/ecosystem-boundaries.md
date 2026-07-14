# Scientific ML ecosystem boundaries

Use this page when deciding whether a model, optimizer, inference method,
execution primitive, or audit record belongs in Jaxstro or another package.

:::{important} Planned Jaxstro capability
`jaxstro.ml` does not exist and has no implementation schedule. The ownership
table below constrains a possible future capability; it does not announce an
API or migration date.
:::

## The scientific question

Which package should own each part of a scientific ML workflow so that Jaxstro
adds evidence-bearing contracts without recreating mature ecosystem machinery?

## Prerequisites

Review [](../../10-foundations/models-and-computation/what-is-a-model.md),
[](../../30-representations/parameters-state/pytrees-as-scientific-state.md),
and [](../reproducible-research/evidence-and-claim-boundaries.md).

## Mathematical objects

A workflow contains a model, parameters, objective, optimizer transformation,
optimizer state, data plan, random keys, execution schedule, inference
procedure, metrics, checkpoints, and provenance. Ownership follows semantics,
not merely which package can store an array.

## Core derivation

For a composed workflow $W=A\circ T\circ D\circ M$, an audit must identify the
owner and contract of every component:

```{math}
:label: eq-ml-ownership-composition

\operatorname{evidence}(W)
=
\{\operatorname{contract}(M),\operatorname{contract}(D),
\operatorname{contract}(T),\operatorname{contract}(A)\}.
```

Composition does not transfer ownership: wrapping an Optax update does not make
the optimizer a Jaxstro algorithm.

## Assumptions and failure boundaries

An adapter is justified only when it adds a stable, consumer-driven scientific
contract such as explicit units, plans, provenance, or audit evidence. Thin
renaming wrappers, model zoos, duplicate loss libraries, and general posterior
inference are outside the proposed boundary.

## Worked conceptual example

An Equinox model and Optax transformation can be executed by JAX using a stored
Jaxstro-style data plan and run manifest. The downstream project defines the
likelihood and scientific acceptance checks. Each layer remains replaceable and
auditable because its owner is named.

## Ownership boundary

| Owner | Responsibility |
| --- | --- |
| Equinox | Model and callable-PyTree construction |
| Optax | Optimizer transformations and optimizer state |
| JAX | Transformations, arrays, PRNG keys, and execution primitives |
| Jaxstro | Proposed domain-agnostic preprocessing, data-plan, audit, and provenance contracts only |
| Informax | Inference-aware scientific workflows and representation choices |
| NumPyro/BlackJAX | Probabilistic inference and sampling mechanics |

## Proposed interface

No proposed symbol is part of the API reference. A future design must first
show concrete consumers, protocol compatibility, and evidence that the added
contract is not a duplicate of its ecosystem owners.

## Evidence required before implementation

Evidence must include consumer use cases, protocol tests with Equinox and
Optax, JAX transform checks, import-boundary tests, serialization and replay,
and an API audit proving that no delegated model, optimizer, or sampler is
re-exported as Jaxstro-owned machinery.

## Where the claim stops

Clear ownership improves maintainability and auditability; it does not validate
a model, objective, inference procedure, or scientific conclusion.

## Connected ideas

See [](./preprocessing.md), [](./data-plans.md), [](./auditable-training.md),
and [](../../30-representations/parameters-state/parameters-and-transforms.md).
