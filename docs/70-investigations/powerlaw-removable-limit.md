---
title: Finite power-law removable limit
description: Audit values and parameter derivatives through alpha equals minus one.
---

# Finite power-law removable limit

**Research question.** Can a finite power-law implementation preserve both
forward probability contracts and exponent sensitivity through a removable
singularity?

## Predict

At $\alpha=-1$, predict a logarithmic CDF, finite normalization, exact support
boundaries, monotone inversion, and one common parameter derivative approached
from both sides.

## Compute

```bash
uv run --no-sync python -m examples.investigations.powerlaw_removable_limit
```

The public log-density, CDF, and PPF use smooth removable-singularity kernels.
The example returns normalization, round-trip, support, AD, finite-difference,
and independently derived limiting-derivative metrics.

## Audit

Use logarithmic-grid quadrature as an independent normalization check. Compare
AD with a central finite difference and the series-derived coefficient. Verify
both support endpoints and the CDF/PPF round trip.

## Misconception check

> An exact equality branch can produce the correct value at the limit while
> exposing the wrong derivative with respect to the exponent.

## State the warranted claim

The tested kernel preserves its finite-support numerical and local derivative
contracts. This does not establish that a power law is the correct physical
stellar-mass distribution for a particular population.

