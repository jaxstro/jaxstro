---
title: Bibliography
description: >-
  The load-bearing sources behind jaxstro — CODATA, IAU, Oke & Gunn, Golub–Welsch,
  Neumaier, Morton — and why each one matters.
---

The sources jaxstro stands on. These are not decoration: a constant or a method is
only as trustworthy as the reference it cites, and a reader should be able to audit
the number rather than trust it (principle
[9](../10-theory/index.md#p9-correctness)).

- **{cite:t}`Tiesinga2021`** — CODATA 2018. The values of the fundamental constants
  in `jaxstro.constants` (G, $k_B$, $\sigma_\mathrm{SB}$, $\sigma_T$, …), all in CGS.
- **{cite:t}`IAU2015B3`** — IAU 2015 nominal solar/planetary parameters, the source
  for $\Msun$, $\Rsun$, $\Lsun$.
- **{cite:t}`OkeGunn1983`** — the AB magnitude system; the $3631\ \mathrm{Jy}$ zero
  point in `PhotometricUnits`.
- **{cite:t}`GolubWelsch1969`** — the eigenvalue construction behind the Gaussian
  quadrature nodes and weights ([](../40-api/index.md)).
- **{cite:t}`Neumaier1974`** — compensated (Neumaier) summation, used where an
  ordinary `sum` would lose digits (principle
  [5](../10-theory/index.md#p5-floating-point)).
- **{cite:t}`Morton1966`** — Z-order (Morton) curve, the basis of the spatial
  module's encoding.

:::{note} Per-paper notes are planned
Short per-paper notes — *why this reference matters*, not just that it exists — are
planned. The full reference list renders below.
:::

## References

```{bibliography}
:filter: docname in docnames
```
