---
title: Evidence-first investigation rubric
description: Assess prediction, computation, audit, and claim calibration.
---

# Evidence-first investigation rubric

A correct scalar with unsupported reasoning is not complete. This rubric can be
used analytically by dimension or holistically for a research memo. Adapt point
weights to the course; preserve the distinctions among evidence classes.

## Task-specific applicability

Before assigning a unit, publish which dimensions apply and their weights. A
dimension may be marked **not applicable (N/A)** without penalty when the task
does not ask for that reasoning—for example, derivative interpretation in a
value-only activity or failure analysis in a tightly bounded identity check.
Students should briefly justify an N/A claim; it must not be used to avoid a
declared learning objective.

| Dimension | Strong evidence | Developing evidence | Insufficient evidence |
| --- | --- | --- | --- |
| Prediction quality | Commits to the task-relevant units, sign, limits, invariants, failure states, conditioning, and derivative meaning before execution | Predicts a value trend but omits one or more applicable structural expectations | Reconstructs a prediction after seeing the result or gives none |
| Model, units, and scale | Separates model assumptions, parameters, state, observables, units, and numerical scaling | Names the model and most units but conflates a role or scale | Treats arrays or formulas as self-interpreting |
| Method inspection | Identifies the executed algorithm, tolerances, branches, status, shapes, and cost-relevant telemetry | Names the method but retains only partial state | Reports only the final scalar |
| Independent audit | Uses an actually independent identity, limit, FD check, convergence study, conservation law, or source record | Performs a relevant check that shares important assumptions | Repeats the same computation under a new label |
| Derivative interpretation | Names the differentiated map, held-fixed quantities, units, smooth domain, and finite-map versus implicit meaning | Computes a derivative with partial interpretation | Treats any finite AD output as proof of the intended derivative |
| Provenance | Records public API, configuration, precision, source/evidence ID, and reproducible command | Records code and some configuration | Cannot identify which implementation or evidence produced the claim |
| Failure analysis | Uses a failed or boundary case to diagnose model, numerical, program, or evidence assumptions | Notices failure but offers an untested explanation | Hides, retries, or discards failure without analysis |
| Warranted claim | States the strongest supported claim plus explicit limitations and non-claims | Gives a mostly calibrated conclusion with vague scope | Generalizes beyond the tested method, domain, or evidence class |

## Feedback language

Prefer feedback that names the contract:

- “The value audit is strong; add independent derivative evidence.”
- “The parameter role changed between prediction and inference—state which map
  is now being differentiated.”
- “This supports interior affine recovery, not behavior at a knot.”
- “The posterior is precise under the model; add a model-checking claim.”

## Minimal completion standard

Every submission should contain a pre-execution prediction, one units-explicit
metric table, one independent audit, and a warranted claim. Include failure or
boundary analysis when the task declares it applicable. Extensions may deepen
the mathematics or science but should not replace the selected dimensions.
