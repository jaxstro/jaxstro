---
title: Legacy fixed-quadrature import path
---

# Legacy fixed-quadrature import path

```{important}
`jaxstro.quad` is the canonical owner. `jaxstro.numerics.quadrature` is a
temporary compatibility path retained while sibling packages migrate.
```

| Legacy name | Canonical name |
| --- | --- |
| `jaxstro.numerics.quadrature.gauss_legendre_nodes` | `jaxstro.quad.gauss_legendre_nodes` |
| `jaxstro.numerics.quadrature.gauss_laguerre_nodes` | `jaxstro.quad.gauss_laguerre_nodes` |
| `jaxstro.numerics.quadrature.gauss_hermite_nodes` | `jaxstro.quad.gauss_hermite_nodes` |
| `jaxstro.numerics.quadrature.clenshaw_curtis_nodes` | `jaxstro.quad.clenshaw_curtis_nodes` |
| `jaxstro.numerics.quadrature.hermite_e_basis` | `jaxstro.quad.hermite_e_basis` |
| `jaxstro.numerics.quadrature.hermite_coefficients` | `jaxstro.quad.hermite_coefficients` |

Phase A1 preserves exact callable identity and emits no deprecation warning.
Use the [Jaxstro quadrature foundation](./quad.md) for the current API.
