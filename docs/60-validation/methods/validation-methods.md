---
title: Validation methods
description: Reusable routes from numerical claims to independent checks and bounded conclusions.
---

# Validation methods

Validation begins by naming the claim, its admissible domain, its comparison,
and the evidence that would falsify it. The reusable workflow is:

1. state the mathematical or semantic contract;
2. compute with the public API under an explicit configuration;
3. compare against an independent analytic result, limiting case, reference
   implementation, convergence study, or finite-difference check;
4. record metric identity, value, units, and tolerance;
5. state the limitation and the warranted claim.

For detailed workflows, use [](../../40-workflows/differentiable-research/auditing-derivatives.md)
for AD versus finite differences, [](../../40-workflows/reproducible-research/provenance.md)
for reproducibility, and [](../../40-workflows/reproducible-research/evidence-and-claim-boundaries.md)
for separating evidence classes. The [](../evidence-index.md) then connects
registered artifacts to the claims they actually support.

Self-consistency alone is insufficient. A successful transform, finite value,
or fresh generated file does not establish scientific validity without the
independent comparison required by the named contract.
