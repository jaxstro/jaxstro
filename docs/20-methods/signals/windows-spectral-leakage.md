---
title: Windows and spectral leakage
description: Window normalization, coherent gain, leakage, and equivalent noise bandwidth.
---

# Windows and spectral leakage

Use this page when a finite observation must be tapered and the resulting
amplitude or noise normalization must remain scientifically interpretable.

:::{important} Planned Jaxstro capability
`jaxstro.signal` does not exist. This page defines planned
window metadata and evidence gates without claiming runtime support.
:::

## The scientific question

Observing only a finite record multiplies an underlying signal by an implicit
rectangular window. In frequency space, multiplication becomes convolution, so
power from a component need not remain in one Fourier bin. A taper can suppress
distant leakage at the cost of a wider main lobe, reduced coherent amplitude,
and changed noise bandwidth.

There is no universally best window. The choice depends on whether the target is
line amplitude, nearby-feature separation, broadband power, or another
observable.

## Mathematical objects

Let $w_n$ be a real window and $y_n=w_nx_n$ the windowed record. Its sum controls
the response to a bin-centered coherent sinusoid. Its sum of squares controls
white-noise power. Scaling a window changes both sums, so every estimator must
state which normalization is applied.

The equivalent noise bandwidth (ENBW) expresses how much white-noise bandwidth
an estimator admits relative to an ideal one-bin filter. It has frequency units
when multiplied by the sample frequency.

## Core derivation

For sample frequency $f_s=1/\Delta t$, define coherent gain and ENBW by:

```{math}
:label: eq-window-gain-enbw

G_{\mathrm{coh}}=\frac{1}{N}\sum_{n=0}^{N-1}w_n,
\qquad
B_{\mathrm{ENBW}}
=f_s\frac{\sum_{n=0}^{N-1}w_n^2}
{\left(\sum_{n=0}^{N-1}w_n\right)^2}.
```

The coherent amplitude of a bin-centered tone is reduced by
$G_{\mathrm{coh}}$, while white-noise density conversion depends on
$B_{\mathrm{ENBW}}$. Therefore [](#eq-window-gain-enbw) shows why an amplitude
correction and a noise-power correction are not interchangeable.

The DFT of $y_n$ is the circular convolution of the signal spectrum with the
window spectrum under the finite-record convention. This explains leakage but
does not guarantee that tapering resolves overlapping components.

## What the ecosystem already owns

[JAX FFT](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fft.fft.html)
owns FFT mechanics, and JAX owns convolution mechanics. Applying a supplied
array as a window is ordinary array computation. JAX does not assign the window
a scientific name, normalization contract, ENBW, or provenance.

## What Jaxstro may add

The proposed `jaxstro.signal` may own a small, explicit catalog of window
definitions with normalization mode, coherent gain, sum-of-squares gain, ENBW,
axis units, and provenance. A spectral estimator could require this metadata
rather than accepting an unlabeled multiplier.

The planned layer would not claim that one window is optimal or hide correction
factors inside an ambiguous output.

## Evidence required before implementation

Required evidence would include:

- exact coefficient fixtures for every supported window and length convention;
- DC and bin-centered sinusoid tests for coherent-gain correction;
- white-noise experiments for ENBW and power-density normalization;
- off-bin sinusoid tests for main-lobe width and leakage behavior;
- scale-invariance tests showing which reported quantities change under window
  rescaling;
- odd, even, and short-length edge cases; and
- provenance for window family, parameters, length, and normalization.

## Claim boundary

:::{warning}
A window redistributes spectral response; it does not restore information lost
to finite duration or aliasing. Leakage suppression, frequency resolution, and
noise bandwidth are distinct tradeoffs.
:::

No runtime catalog, performance result, or universally preferred window is
claimed.

## Connected foundations and methods

First establish cadence and DFT normalization in [](./signal-axes.md). Then use
[](./spectral-estimation.md) for one-sided and two-sided power conventions and
[](./phase-and-delay.md) for cross-spectral phase. The finite-data viewpoint in
[](../approximation-integration/interpolation.md) and scale concepts in
[](../../10-foundations/mathematical-objects/functions-units-scales.md) provide
additional context.
