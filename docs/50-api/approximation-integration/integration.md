---
title: Legacy sampled-integration import path
---

# Legacy sampled-integration import path

```{important}
`jaxstro.quad` is the canonical owner. `jaxstro.numerics.integration` is a
temporary compatibility path retained while sibling packages migrate.
```

| Legacy name | Canonical name |
| --- | --- |
| `jaxstro.numerics.integration.trapz` | `jaxstro.quad.trapezoid` |
| `jaxstro.numerics.integration.cumulative_trapz` | `jaxstro.quad.cumulative_trapezoid` |
| `jaxstro.numerics.integration.simpson` | `jaxstro.quad.simpson` |
| `jaxstro.numerics.integration.cumulative_simpson` | `jaxstro.quad.cumulative_simpson` |

Phase A0 preserves exact callable identity and emits no deprecation warning.
Use the [Jaxstro quadrature foundation](./quad.md) for the current API.
