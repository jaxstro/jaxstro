---
title: Why this documentation works this way
description: The evidence-first design behind Jaxstro's curriculum and reference layers.
---

# Why this documentation works this way

Jaxstro is written for research students, computational-science courses, and
scientists returning to ideas they may once have studied in isolation. A listed
prerequisite records past enrollment; it does not guarantee current preparedness
for a new synthesis. The foundations route therefore makes important ideas
available without labeling learners or weakening the work.

## The central cycle

```text
predict → compute → audit → state the warranted claim
```

Prediction prevents post-hoc storytelling. Before seeing an output, name units,
signs, scales, limiting cases, invariants, conditioning, expected failures, and
the derivative you mean.

Computation connects a mathematical relation to the numerical program that
actually ran. Solvers choose branches, interpolation selects cells, finite
precision changes comparisons, and JAX transforms a particular executable map.
Retain statuses, tolerances, traces, and provenance—not only a plausible scalar.

Audit asks an independent question: does the result satisfy an analytic identity,
a limit, a finite-difference check, a convergence study, a conservation law, or
a source-provenance requirement? An audit can support a bounded claim, reject it,
or reveal that the original question was underspecified.

The warranted claim is the scientific product. “The code ran” is weaker than
“this value satisfies the stated numerical contract,” which is weaker than “the
physical model is adequate for this observation.” The documentation keeps those
levels separate.

## Why foundations and module reference both remain

The foundations pages answer *what does this idea mean and how does it connect to
scientific reasoning?* The existing module chapters answer *how does this method
work, what does Jaxstro expose, and where are its boundaries?* The API reference
answers *what is the exact public surface?* The validation section answers *what
evidence supports the registered claim?*

These are complementary routes, not a replacement hierarchy. A course may follow
the foundations and investigations in sequence. A researcher may enter through a
module or API page and follow a concept link only when needed.

## Why optional preparation is rigorous

Preparedness varies for structural reasons: curricula separate calculus from
statistics, linear algebra from programming, and physical modeling from
inference. Reconnecting them is substantive scientific work. Optional recovery
lets a class maintain high standards without pretending every student begins
with the same active vocabulary.

## How to use the site

1. Choose a research question or module.
2. Make the prediction visible before execution.
3. Run the bounded computation and retain evidence.
4. Audit with a genuinely independent check.
5. State what is supported, what is not, and what would change your conclusion.

Then begin again. The audit is not the end of the scientific cycle; it determines
the next prediction.

