---
title: API reference
description: >-
  The public module surface of jaxstro 0.1.0 — units, constants, astrometry,
  coords, numerics, spatial, params, testing, jaxconfig — and what each provides.
---

This is the lookup door. It enumerates the public modules of jaxstro 0.1.0 and the
symbols each exports, with a pointer back to the theory that justifies them. Import
the top-level package and reach modules as attributes:

```python
import jaxstro
from jaxstro import constants, units, numerics, coords, spatial, params, testing
from jaxstro.jaxconfig import enable_high_precision
```

The committed surface for 0.1.0 is below. `units`, `constants`, `astrometry`,
`coords`, and `jaxconfig` are **stable**; `numerics`, `spatial`, and `params` are
**stable-but-evolving**; `testing` is a **public, semi-stable** utility. There is
no private or experimental tier at release.

```{list-table} Public modules
:header-rows: 1
:label: tbl-modules

* - Module
  - Provides
* - `jaxstro.units`
  - `UnitSystem` dataclass with a `.G` property; named systems (`CGS`,
    `ASTRO_STELLAR`/`solar`, `ASTRO_DYNAMICAL`/`stellar`, `ASTRO_PLANETARY`/`binary`)
    and aliases; `DEFAULT` (= CGS) re-exported as `jaxstro.DEFAULT_UNITS`;
    `PhotometricUnits`.
* - `jaxstro.constants`
  - CGS physical constants from CODATA 2018 and IAU 2015, each with a provenance
    comment, plus photometric constants (Oke & Gunn 1983).
* - `jaxstro.astrometry`
  - Astrometric constants (e.g. `K_PROPER_MOTION`, mas/radian conversions).
* - `jaxstro.coords`
  - Coordinate transforms — sky-tangent, galactic/equatorial, spherical, parallax.
* - `jaxstro.numerics`
  - Differentiable numerical utilities: stats, interpolation, root-finding,
    integration (incl. `cumulative_trapz` + quadrature factory + `newton_ppf`),
    checks, compensated summation, linear algebra, RNG, sampling.
* - `jaxstro.spatial`
  - Morton (Z-order) encoding/decoding, grid binning, neighbor-candidate gathering.
* - `jaxstro.params`
  - Equinox-only PyTree↔flat-vector bridge (`Parameterization`) plus a bijector
    registry (Identity/Exp/Softplus/Sigmoid) for unconstrained-space inference.
* - `jaxstro.testing`
  - The grad-audit engine (`audit_entry_point`, `Case`, `AuditResult`, `EdgeConfig`)
    for FD-vs-AD gradient checks.
* - `jaxstro.jaxconfig`
  - `enable_high_precision()` — turns on float64 and highest matmul precision.
```

## Selected modules

### `jaxstro.constants`

CGS constants with sourced values. A few that downstream packages rely on:

```{list-table} Sampled constants (CGS)
:header-rows: 1
:label: tbl-constants

* - Symbol
  - Value
  - Source
* - `G_CGS`
  - $6.67430\times10^{-8}\ \mathrm{cm^3\,g^{-1}\,s^{-2}}$
  - CODATA 2018
* - `K_B`
  - $1.380649\times10^{-16}\ \erg\,\mathrm{K}^{-1}$
  - CODATA 2018 (exact)
* - `SIGMA_SB`
  - $5.670374419\times10^{-5}\ \erg\,\mathrm{cm^{-2}\,s^{-1}\,K^{-4}}$
  - CODATA 2018
* - `A_RAD`
  - $7.565733250\times10^{-15}\ \erg\,\mathrm{cm^{-3}\,K^{-4}}$
  - Derived $4\sigma_\mathrm{SB}/c$ (CODATA 2018)
* - `SIGMA_T`
  - $6.6524587321\times10^{-25}\ \mathrm{cm^2}$
  - CODATA 2018 (Thomson cross-section)
* - `MSUN_G`
  - $1.9884\times10^{33}\ \mathrm{g}$
  - IAU 2015 nominal
* - `AB_ZEROPOINT_JY`
  - $3631\ \mathrm{Jy}$
  - Oke & Gunn 1983
```

Provenance discipline — every constant cites its authority — is principle
[9](../10-theory/index.md#p9-correctness).

### `jaxstro.numerics.rootfinding`

`bisect`, `newton`, `newton_with_grad`, `newton_ppf`. Behavior, the differentiability
caveats, and when to use each are in [](../10-theory/rootfinding.md).

### `jaxstro.numerics.integration`

`trapz`, `cumulative_trapz` (dx-outside uniform path), `simpson`. The method and the
ordering choice are in [](../10-theory/cumulative-trapz.md).

### `jaxstro.numerics.quadrature`

`gauss_legendre_nodes(n)`, `gauss_hermite_nodes(n)` (probabilists'),
`hermite_e_basis`, and Hermite expansion coefficients. Nodes are generated once on
the host (Golub–Welsch via numpy) and frozen to constants; gradients flow through
the integrand values, not the nodes (principle
[7](../10-theory/index.md#p7-quadrature)).

:::{note} Per-symbol reference pages are planned
A complete, auto-generated per-module symbol reference (signatures, parameters,
source links) is planned. Until then, the docstrings are authoritative — read them
with `help(jaxstro.numerics.rootfinding.newton_ppf)` — and this landing page is the
module map.
:::
