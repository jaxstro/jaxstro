---
title: Teaching with Jaxstro
description: Facilitation notes for evidence-first computational-science investigations.
---

# Teaching with Jaxstro

This guide is for instructors, research mentors, and teaching assistants working
with research students whose preparation is necessarily uneven. The goal is not
to remediate learners into sameness. It is to make the conceptual bridge needed
for today's scientific task available while preserving a high evidence standard.

## The instructional cycle

Use **predict → compute → audit → state the warranted claim** as epistemic work,
not a four-step worksheet ritual.

1. Collect predictions before code execution.
2. Ask groups to name units, limits, failure states, and derivative meaning.
3. Run the repository-owned investigation rather than a copied solution.
4. Separate implementation, numerical, derivative, provenance, and physical
   evidence on the board.
5. Require a final claim and one explicit non-claim.

## Supporting uneven preparation

Offer [](../00-start-here/choose-your-path.md) as an ungraded route. Let
students use the complete first-principles path or enter through a familiar
language. Do not publicly classify students by route. A student may need a
probability refresher for one unit and lead the class on linear algebra in the
next.

Use short retrieval prompts before adding explanation: “What are the units?”,
“Which quantity is held fixed?”, “What would make this inverse insensitive?”,
or “Which check is independent?” This reveals the missing connection without
turning a prerequisite into a gatekeeping device.

## Investigation notes

### `root-values-and-sensitivities`

Pause before computing and collect two derivative predictions: the derivative
of the executed finite solver and the implicit derivative of the ideal root.
The common misconception is that convergence transfers the implicit derivative
claim automatically. Ask students to identify every certificate assumption and
design a residual that would reject one.

**Astronomy extension:** express a radiative-equilibrium or event-location
equation as a signed residual, keeping all domain acceptance downstream.

**Computational-science extension:** compare evaluation count, bracket width,
and derivative conditioning for a family of analytic residuals.

### `powerlaw-removable-limit`

Have students derive the logarithmic limit before inspecting the smooth kernel.
A common misconception is that an exact equality branch must also have the
correct parameter derivative. Ask why normalization is a value audit while the
series coefficient and central difference are derivative audits.

**Astronomy extension:** discuss what additional evidence would be required to
interpret the distribution as a stellar mass function.

**Computational-science extension:** locate another removable singularity and
design a value-plus-derivative validation triangle.

### `interpolation-boundary-policies`

Assign groups different boundary policies, then compare claims. A common
misconception is that a smooth formula inside each cell makes discrete cell
selection differentiable. Ask students to distinguish values at knots,
one-sided behavior, and branch-stable interior derivatives.

**Astronomy extension:** map the policy question onto a tabulated stellar or
atmosphere grid without importing that domain policy into Jaxstro.

**Computational-science extension:** construct a function that is exactly
recovered in each cell but has a derivative discontinuity at a knot.

## Facilitation patterns

- **Prediction ledger:** retain predictions visibly and revise them only after
  labeling the new evidence.
- **Evidence sorting:** give groups analytic, unit-test, FD, benchmark, source,
  and physical-validation cards and ask which claim each supports.
- **Failure-first variation:** deliberately break a bracket, support, or
  boundary contract before showing the successful case.
- **Claim ladder:** move from “finite output” to numerical contract to model
  adequacy, stopping when evidence runs out.

## Accessibility

Provide equations in readable text as well as notation, descriptive link text,
and sufficient time to inspect printed tables. Do not rely on color alone in
figures. Make prediction submission possible through speech, writing, or a
structured diagram. Commands should be copyable, keyboard-operable, and paired
with an explanation of expected output structure.

## Assessment use

Use [](./assessment-rubric.md) for lab notes, code reviews, oral checks, or short
research memos. Grade the relation among model, program, evidence, and claim;
do not let a numerically correct scalar erase unsupported reasoning.
