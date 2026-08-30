---
title: Method coverage matrix
description: A reviewable map from learning guides to runtime owners and evidence boundaries.
---

# Method coverage matrix

This is the review map for the **38 method-guide pages** in the site. It answers
three different questions without conflating them: what a learner can study,
what code is implemented, and what evidence supports a bounded claim.

`implemented` means a public Jaxstro owner exists. `experimental` means the
public owner exists but is outside the qualified scientific core. `planned`
means the page teaches a proposed or delegated capability, not an importable
runtime. A passing row supports a numerical contract; it does not establish
scientific-model adequacy.

## How to use the matrix

Start with a question, make a prediction with explicit units and limiting
behavior, choose a row, then follow its method, API, and validation links:

```text
predict -> compute -> audit -> state the warranted claim
```

| Learning family | Runtime status | Canonical owner or boundary | First learning route | Evidence and limitation route |
| --- | --- | --- | --- | --- |
| Scalar constraints and sensitivities | implemented | `jaxstro.numerics` roots and implicit roots | [](../../20-methods/change-constraints-evolution/rootfinding.md) | [](../../60-validation/qualified-core.md); value-first roots do not claim implicit derivatives |
| Fixed-step change and optimization | implemented | `jaxstro.numerics` ODE and objective helpers | [](../../20-methods/change-constraints-evolution/ode.md), [](../../20-methods/change-constraints-evolution/optimization.md) | [](../../60-validation/validation.md); adaptive ODE and general nonlinear solving remain delegated |
| Differentiation products | implemented | `jaxstro.numerics` autodiff helpers | [](../../20-methods/change-constraints-evolution/autodiff.md) | [](../../20-methods/methods.md#p1-differentiability); route changes and discrete choices are not smooth claims |
| Tables, splines, and regular grids | implemented | `jaxstro.numerics` interpolation owners | [](../../20-methods/approximation-integration/interpolation.md) | [](../../60-validation/validation.md); gradients are local to documented smooth cells |
| One-dimensional quadrature and cumulative rules | experimental | `jaxstro.quad` and legacy numerical helpers | [](../../20-methods/approximation-integration/quadrature.md) | [](../../60-validation/numerical/quadrature-replay-derivatives.md); accepted-formula replay is first order only |
| Multidimensional finite-hyperrectangle integration | experimental | `jaxstro.quad` tensor, cubature, sparse-grid, and QMC methods | [](../../20-methods/approximation-integration/multidimensional/choosing-a-multidimensional-integration-method.md) | [](../../60-validation/numerical/quadrature-multidimensional.md); no geometry beyond finite hyperrectangles or universal performance claim |
| Linear structure and special functions | implemented | `jaxstro.numerics` linear algebra, operators, and bases | [](../../20-methods/linear-structure/linear-algebra.md) | [](../../60-validation/validation.md); rank and conditioning boundaries remain explicit |
| Probability, keys, and resampling | implemented | `jaxstro.numerics` distributions plus `jaxstro` random helpers | [](../../20-methods/probability-sampling/random.md) | [](../../60-validation/validation.md); discrete resampling is not assigned an invented physical derivative |
| Grids, meshes, and spatial candidates | implemented | `jaxstro.numerics` grids/meshes and `jaxstro.spatial` | [](../../20-methods/discrete-space/grids.md) | [](../../60-validation/validation.md); candidate generation is distinct from physical interaction policy |
| Signals and spectral interpretation | planned | No `jaxstro.signal` runtime owner | [](../../20-methods/signals/signal-axes.md) | [](./future-capabilities-roadmap.md#priority-3-jaxstrosignal); these guides teach required conventions, not a shipped API |

## Coverage boundary

The generated contract registry has callable-level records only where an owner
has registered purpose, transforms, failure behavior, evidence, and limitations.
It currently lists **235 unclassified public callables**. They may be importable,
but this matrix deliberately does not promote them into supported methods and
does not establish scientific-model adequacy.

For the small set that is both public and evidence-complete, use the
[](../../60-validation/qualified-core.md). For all public module owners, use the
[](../../50-api/api.md). For development priorities, limitations, and proposed
work, use the [](./development.md), [](./sota-assessment.md), and
[](./future-capabilities-roadmap.md).
