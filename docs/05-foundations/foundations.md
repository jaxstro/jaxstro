---
title: "Foundations: the ideas we will not assume"
description: Optional first-principles preparation for evidence-first computational science.
---

# Foundations: the ideas we will not assume

This section is an optional on-ramp for research students and course learners.
It does not lower the scientific standard, and it does not assume that seeing a
topic in an earlier course means it is ready for use now. It reconnects ideas
that are often taught separately: physical models, linear algebra, derivatives,
probability, inference, conditioning, and differentiable programs.

Start with [](../00-start-here/choose-your-path.md) if you want a short,
ungraded route. Read
[](./why-this-documentation-works-this-way.md) if you want to understand why
the site repeatedly asks you to predict, compute, audit, and state only the
claim the evidence warrants.

## The foundations sequence

1. [](../00-start-here/choose-your-path.md).
2. [](./functions-units-scales.md).
3. [](./what-is-a-model.md): representations, assumptions, parameters,
   predictions, information compression, and dimensionality.
4. [](./linear-algebra-language-of-change.md).
5. [](./what-is-a-derivative.md): change, sensitivity, and scientific evidence.
6. [](./probability-and-distributions.md).
7. [](./models-inference-information.md).
8. [](./sensitivity-conditioning-identifiability.md).
9. [](./from-relations-to-differentiable-programs.md).

The established [](../10-theory/index.md) module chapters remain the method
reference. Foundations explain the connective ideas; module pages explain the
algorithms and public APIs.

## The recurring scientific habit

Every substantial unit follows

```text
predict -> compute -> audit -> state the warranted claim
```

Prediction names expected structure before results can bias the story.
Computation exposes the method that actually ran. Audit compares the output
with independent evidence. The final statement is deliberately no stronger
than those checks.
