---
title: Power and cross-spectral estimation
description: Two-sided and one-sided power conventions for a planned signal capability.
---

# Power and cross-spectral estimation

Use this page when Fourier coefficients must become an amplitude, power, or
power-density estimate with explicit units and sidedness.

:::{important} Planned Jaxstro capability
`jaxstro.signal` does not exist. The conventions below are a
design target and not evidence of an implemented estimator.
:::

## The scientific question

The squared magnitude of a DFT coefficient is not yet a fully specified
scientific observable. Its meaning depends on the DFT normalization, cadence,
window, record length, sidedness, detrending, averaging, and whether the output
is per bin or per unit frequency.

For a real signal, negative-frequency coefficients mirror positive-frequency
coefficients. A one-sided representation may combine their power, but DC and an
even-length Nyquist bin have no distinct negative partner and must not be
doubled.

## Mathematical objects

Let $X_k$ be the DFT defined on the signal-axes page and let $f_k$ identify its
frequency bin. An amplitude spectrum has the observable's units. A power per bin
has squared observable units. A power spectral density has squared observable
units per frequency.

A periodogram is a finite-record estimator. Segment averaging can reduce
variance while changing frequency resolution and introducing choices about
overlap and segment windows. Those choices are estimator metadata, not incidental
array details.

## Core derivation

For the unwindowed DFT convention on a record with cadence $\Delta t$, define a
two-sided density whose sum over bins satisfies a Parseval-compatible account:

```{math}
:label: eq-two-sided-periodogram

P_k^{(2)}=\frac{\Delta t}{N}|X_k|^2,
\qquad
\sum_{k=0}^{N-1}P_k^{(2)}\Delta f
=\frac{1}{N}\sum_{n=0}^{N-1}|x_n|^2,
\qquad
\Delta f=\frac{1}{N\Delta t}.
```

For a real signal, a one-sided density follows from
[](#eq-two-sided-periodogram) by retaining nonnegative frequencies and doubling
only bins with a distinct negative-frequency partner:

```{math}
:label: eq-one-sided-periodogram

P_k^{(1)}=
\begin{cases}
P_k^{(2)}, & k=0, \\
P_k^{(2)}, & k=N/2\ \text{for even }N, \\
2P_k^{(2)}, & \text{otherwise on the positive-frequency branch}.
\end{cases}
```

A window replaces the unwindowed normalization with its declared coherent or
power gain. That correction must be stated rather than inferred from the output
name.

## What the ecosystem already owns

[JAX FFT](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fft.rfft.html)
owns real-input FFT mechanics, and JAX owns convolution mechanics. It does not
decide whether a returned array represents amplitude, energy, power per bin, or
power density.

## What Jaxstro may add

The proposed `jaxstro.signal` may define records that carry sidedness, cadence,
frequency units, DFT normalization, window correction, detrending, segment plan,
and density units. It may provide validation helpers for Parseval accounts and
one-sided endpoint handling.

The module would delegate FFT execution to JAX and would not treat estimator
variance reduction as new physical information.

## Evidence required before implementation

Readiness would require:

- analytic sinusoid tests for amplitude and power under each supported
  convention;
- Parseval checks for real and complex signals at odd and even lengths;
- one-sided versus two-sided integral parity, including DC and Nyquist bins;
- white-noise tests for density units and window ENBW corrections;
- segment and overlap fixtures with deterministic averaging metadata;
- dtype, precision, `jit`, and batch-shape tests; and
- provenance round trips for every normalization choice.

## Claim boundary

:::{warning}
A periodogram is a noisy finite-record estimator. A smooth-looking spectrum does
not establish a physical component, statistical significance, stationarity, or
freedom from leakage and aliasing.
:::

No planned estimator is implemented, and this page does not report detection or
uncertainty performance.

## Connected foundations and methods

Use [](./signal-axes.md) for DFT and bin conventions,
[](./windows-spectral-leakage.md) for coherent gain and equivalent noise
bandwidth, and [](./phase-and-delay.md) for cross spectrum and delay. Probability
language is developed in
[](../../10-foundations/mathematical-objects/probability-and-distributions.md),
while fixed accumulation appears in
[](../approximation-integration/cumulative-trapz.md).
