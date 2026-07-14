---
title: Conservative spectral resampling
description: >-
  Point interpolation and bin-average remapping with explicit sampling and coverage
  semantics.
---

Use this page when changing a spectral grid and deciding whether point interpolation
or conservation of integrated bin content is the correct representation contract.

:::{important} Implemented Jaxstro capability
`jaxstro.spectra.resample_spectrum` supports point resampling and conservative
bin-average remapping onto a fixed `SpectralPlan`, with fail-closed coverage.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | A map from one explicit spectral axis to another while preserving point meaning or overlapping bin integrals according to sampling type. |
| Physical convention | Point samples use linear or monotone-cubic interpolation; bin averages preserve bin integrals over overlap; no extrapolation or zero filling is implied. |
| Runtime owner | `jaxstro.spectra` owns `SpectralPlan`, sampling enums, status values, and `resample_spectrum`. |
| Shape and unit policy | Source and target must share coordinate, unit, and sampling semantic; output shape is fixed by the target axis and bin edges have one more entry than bin values. |
| Transform boundary | Fixed plans and array evaluation are JAX compatible; axis validation and method selection are static, and unsupported windows return NaN values with status. |
| Evidence | Unit tests check identity paths, point methods, bin conservation, shape failures, and unsupported windows; validation reports measured spectral behavior. |
| Downstream interpretation boundary | Resampling does not choose instrument response, resolving-power adequacy, noise covariance, line-spread function, or acceptable scientific resolution. |

## Bin conservation

For source bin averages $\bar{f}_i$ on edges $x_i$, conservative remapping targets
the overlapping integral

```{math}
:label: eq-spectra-conservative-remap

\bar{g}_j
=
\frac{1}{y_{j+1}-y_j}
\sum_i
\bar{f}_i\,
\left|[x_i,x_{i+1}]\cap[y_j,y_{j+1}]\right|.
```

Thus the sum of average times width is preserved over the shared domain. Point values
do not carry the interval information needed by [](#eq-spectra-conservative-remap),
so point-sampled spectra use an interpolation method instead.

## Plans and coverage

`SpectralPlan` stores a fixed target axis, `CoveragePolicy.INTERSECTION`, and a point
method. Source and target coordinate, unit, and sampling semantic must match. An
identical axis records an identity operation in provenance. Point axes use linear or
monotone-cubic interpolation; bin-average axes call the conservative remap. Nontrivial
bin-integral resampling is not implemented.

If the target window extends beyond the source, the result uses NaN values and
`UNSUPPORTED_SPECTRAL_WINDOW`. It does not extrapolate, clamp, or fill with zero.

:::{warning} Conservation depends on what the bins represent
Conserving a density integral is different from conserving point values or detector
counts. Confirm the spectral semantic and interval measure before interpreting a
conservative numerical result.
:::

Fixed-shape evaluation can be compiled and differentiated with respect to numeric
values on a fixed route. Changing the axis, point method, or sampling semantic is a
static representation change. The tests prove the implemented overlap and status
contract, not that a requested grid resolves every downstream feature.
