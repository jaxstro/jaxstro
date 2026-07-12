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
`[xmin, xmax]`. Its `alpha = -1` log-uniform limit is part of one smooth
expression rather than a branch with a different derivative.

## Finite power law through `alpha = -1`

Let $e = \alpha + 1$ and $D = \log(x_\mathrm{hi})-\log(x_\mathrm{lo})$.
The segment integral is evaluated as

```{math}
I(x_\mathrm{lo},x_\mathrm{hi},e)
= x_\mathrm{lo}^{e} D\,\phi(eD),
\qquad
\phi(z)=\frac{\operatorname{expm1}(z)}{z}.
```

At $z=0$, `phi` uses its Taylor series
$1+z/2+z^2/6$. The inverse uses the sibling kernel
$\psi(z)=\log(1+z)/z$, with Taylor series $1-z/2+z^2/3$:

```{math}
x = \exp\!\left[\log(x_\mathrm{lo}) + s\,\psi(es)\right],
\qquad
s = t x_\mathrm{lo}^{-e}.
```

These finite masked branches sanitize the dangerous denominator before
division, so values and derivatives remain smooth at $e=0$. Normalization is
$1/I(x_{\min},x_{\max},e)$; the CDF is the ratio of partial to total integrals; and
the PPF applies the smooth inverse to $t=uI$.

For an independent limit check, define $A=\log x_{\min}$, $B=\log x_{\max}$,
$L=B-A$, $\ell=\log(x/x_{\min})$, and
$x_u=x_{\min}\exp(uL)$. At $\alpha=-1$,

```{math}
\frac{\partial\log p(x)}{\partial\alpha}
= \log x - \frac{A+B}{2},\qquad
\frac{\partial F(x)}{\partial\alpha}
= \frac{\ell(\ell-L)}{2L},\qquad
\frac{\partial F^{-1}(u)}{\partial\alpha}
= \frac{x_uL^2u(1-u)}{2}.
```

The following float64 measurements use
`xmin=2`, `xmax=5`, `x=3`, `u=0.3`, and central-FD step `1e-5`.

| Metric identity | Symbol | Value | Units |
| --- | --- | ---: | --- |
| Logpdf alpha derivative by AD | `d logp / d alpha` | -0.05268025782891306 | dimensionless |
| Logpdf alpha derivative by central FD | `d logp / d alpha` | -0.05268025782267926 | dimensionless |
| CDF alpha derivative by AD | `dF / d alpha` | -0.11302196975246964 | dimensionless |
| CDF alpha derivative by central FD | `dF / d alpha` | -0.1130219697442758 | dimensionless |
| PPF alpha derivative by AD | `dx_u / d alpha` | 0.2320961224346652 | x units |
| PPF alpha derivative by central FD | `dx_u / d alpha` | 0.2320961224100415 | x units |
| Numerical normalization absolute error | `abs(integral(p)-1)` | 1.6008394609912102e-10 | dimensionless |
| CDF/PPF maximum round-trip error | `max(abs(F(F^-1(u))-u))` | 2.220446049250313e-16 | dimensionless |

## Support behavior

Support is explicit. Log densities return `-inf` outside support; CDFs clamp to
the interval endpoints where appropriate; inverse-CDF helpers map `u` values in
`[0, 1]` onto the distribution support.

For lognormal and power-law kernels, unsafe operands are sanitized before
evaluating logarithms so out-of-support values do not introduce avoidable `NaN`s
in the forward pass.

## Validation

Unit tests check normalization by numerical integration, monotone CDF behavior,
inverse-CDF round trips, support edges, float64, and JAX transform compatibility.
Validation tests compare analytic limiting derivatives and central FD against AD
at and on both sides of `alpha = -1`.
