---
title: Special functions
description: >-
  Stable Planck kernels, normalized log weights, and orthogonal polynomial bases
  that stay generic enough for the foundation layer.
---

Special functions are useful everywhere in astronomy, but they can also smuggle
domain assumptions into the foundation if their units or normalizations are vague.
jaxstro keeps this layer deliberately explicit: the Planck functions are named for
CGS units, log-weight helpers say exactly what they normalize, and polynomial
bases return values only. Fitting, priors, filters, and physical interpretation
belong to downstream packages.

## Planck kernels in CGS

`planck_lambda_cgs(wavelength_cm, temperature)` returns $B_\lambda$ in
`erg s^-1 cm^-2 sr^-1 cm^-1`:

```{math}
B_\lambda(T) =
\frac{2 h c^2}{\lambda^5}\,
\frac{1}{\exp\left(hc/\lambda k_B T\right)-1}.
```

`planck_nu_cgs(frequency_hz, temperature)` returns $B_\nu$ in
`erg s^-1 cm^-2 sr^-1 Hz^-1`:

```{math}
B_\nu(T) =
\frac{2 h \nu^3}{c^2}\,
\frac{1}{\exp\left(h\nu/k_B T\right)-1}.
```

The log variants, `log_planck_lambda_cgs(...)` and `log_planck_nu_cgs(...)`,
evaluate the denominator as `log(expm1(x))` with a Wien-tail branch that avoids
overflow. The linear variants exponentiate those log kernels, so large negative
tails may underflow to zero while the log functions remain finite.

The unit names are part of the API. These are radiance kernels, not fluxes
through filters and not bolometric luminosities. Downstream packages own those
semantics.

## Normalized log weights

`log_normalize(log_weights, axis=-1)` subtracts `logsumexp` along the chosen axis.
Exponentiating the result sums to one. `normalize_log_weights(...)` returns those
probabilities directly.

The helper is intentionally small, but it prevents the common mistake of
normalizing weights in linear space after exponentiating large or tiny log values.
The gradient is the usual softmax-family gradient and is validated against finite
differences away from saturated extremes.

## Orthogonal polynomial bases

`legendre_basis(x, degree)`, `chebyshev_t_basis(x, degree)`, and
`laguerre_basis(x, degree)` return basis values from degree 0 through `degree`
with the polynomial axis last. They use fixed recurrence scans:

```{math}
(n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x),
```

```{math}
T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x),
```

```{math}
(n+1)L_{n+1}(x) = (2n+1-x)L_n(x) - nL_{n-1}(x).
```

These are basis evaluators only. Least-squares fitting is handled by
[](./linear-algebra.md), and B-spline bases live in [](./bsplines.md). Keeping the
pieces separate makes the differentiability contract clearer: fixed bases are
smooth in their coordinates; rank-changing fits and model-selection decisions are
not silently hidden inside a "special function."

## Deferred: spherical Bessel functions

Spherical Bessel functions are useful, but the stable recurrence and
normalization choices depend strongly on downstream use. They are deferred until a
package needs them and can supply the actual numerical contract.
