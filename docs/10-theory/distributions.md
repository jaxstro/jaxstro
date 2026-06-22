---
title: Distribution kernels
description: >-
  Stable logpdf, CDF, and inverse-CDF helpers for common generic distributions
  without turning jaxstro into a probabilistic programming package.
---

`jaxstro.numerics.distributions` provides small probability kernels that are
useful in scientific code: log densities, cumulative distribution functions, and
inverse CDFs. It does not own model syntax, samplers, traces, priors, or
probabilistic programming workflows.

## Included families

The first slice includes:

- Normal: `normal_logpdf`, `normal_cdf`, `normal_ppf`
- Lognormal: `lognormal_logpdf`, `lognormal_cdf`, `lognormal_ppf`
- Finite power law: `powerlaw_logpdf`, `powerlaw_cdf`, `powerlaw_ppf`
- Truncated normal: `truncated_normal_logpdf`, `truncated_normal_cdf`,
  `truncated_normal_ppf`

The power-law helper uses the convention `p(x) proportional to x**alpha` on
`[xmin, xmax]`, with a log-uniform branch for `alpha = -1`.

## Support behavior

Support is explicit. Log densities return `-inf` outside support; CDFs clamp to
the interval endpoints where appropriate; inverse-CDF helpers map `u` values in
`[0, 1]` onto the distribution support.

For lognormal and power-law kernels, unsafe operands are sanitized before
evaluating logarithms so out-of-support values do not introduce avoidable `NaN`s
in the forward pass.

## Validation

Unit tests check normalization by numerical integration, monotone CDF behavior,
inverse-CDF round trips, support edges, and JAX transform compatibility.
Validation tests compare FD-vs-AD gradients for smooth interior log-density
paths.
