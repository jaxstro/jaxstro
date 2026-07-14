---
title: Phase and delay
description: Cross spectra, phase conventions, wrapping, and frequency-dependent delay.
---

# Phase and delay

Use this page when the relative timing of two sampled signals must be inferred
from a cross spectrum with an explicit sign convention.

:::{important} Planned Jaxstro capability
`jaxstro.signal` does not exist. This page fixes conceptual
phase and delay contracts for later design; it is not a runtime claim.
:::

## The scientific question

Two channels can share frequency content while differing in phase. A cross
spectrum preserves their relative complex information. Interpreting that phase
as a time delay requires consistent DFT signs, channel order, phase wrapping,
frequency units, and a model in which a delay is meaningful.

Phase is naturally reported modulo $2\pi$. Near frequencies where cross power
or coherence is weak, phase can be numerically finite yet scientifically
uninformative. At $f=0$, conversion from phase to delay is singular.

## Mathematical objects

Let $X(f)$ and $Y(f)$ be Fourier transforms using the forward negative-exponent
convention. Define the cross spectrum with the first channel conjugated. Swapping
channel order conjugates the cross spectrum and reverses the phase sign.

Wrapped phase lies on a chosen interval such as $(-\pi,\pi]$. Unwrapping adds
integer multiples of $2\pi$ under continuity assumptions; it is an inference
about branch connection, not a neutral formatting step.

## Core derivation

For the channel-order convention $X^*Y$, define:

```{math}
:label: eq-cross-spectrum-phase

C_{xy}(f)=X^*(f)Y(f),
\qquad
\phi(f)=\arg C_{xy}(f),
\qquad
\tau(f) = -\frac{\phi(f)}{2\pi f}.
```

With the DFT sign used on the signal-axes page, a delayed second channel
$y(t)=x(t-\tau_0)$ has $Y(f)=X(f)\exp(-2\pi i f\tau_0)$. Substitution into
[](#eq-cross-spectrum-phase) gives $\phi=-2\pi f\tau_0$ and hence a positive
$\tau_0$. This sign depends on both the DFT and channel-order conventions.

Phase wrapping means the direct delay is ambiguous by integer multiples of
$1/f$. A broadband constant-delay fit should operate on a stated phase model or
time-domain relation rather than independently unwrapping noisy bins without an
evidence rule.

## What the ecosystem already owns

[JAX FFT](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fft.fft.html)
owns FFT mechanics, and JAX owns convolution mechanics. Complex multiplication
and `angle` evaluation are array operations; JAX does not assign channel order,
phase sign, wrap interval, frequency units, or validity masks.

## What Jaxstro may add

The proposed `jaxstro.signal` may own explicit cross-spectrum records containing
channel order, DFT convention, sidedness, averaging plan, phase wrap convention,
frequency units, masks, and provenance. It may provide delay conversion that
fails closed at zero or unsupported frequencies.

It would not decide that phase implies causality or conceal low-information
bins behind a finite delay value.

## Evidence required before implementation

Required evidence would include:

- synthetic shifted-signal fixtures with positive and negative delays;
- channel-swap tests that conjugate the cross spectrum and reverse phase;
- phase wrapping and controlled unwrapping cases across branch boundaries;
- zero-frequency, Nyquist, and low-cross-power failure or mask behavior;
- time-domain cross-correlation comparisons under compatible assumptions;
- one-sided and two-sided cross-spectrum parity; and
- provenance for channel order, sign, wrap interval, averaging, and masks.

## Claim boundary

:::{warning}
A phase-derived delay is conditional on the transform convention and the model
relating the channels. It does not by itself establish causality, a unique
unwrapped branch, stationarity, or a frequency-independent physical lag.
:::

This page claims no implemented delay estimator, uncertainty calibration, or
signal-detection result.

## Connected foundations and methods

Use [](./signal-axes.md) for the DFT sign and frequency axis,
[](./windows-spectral-leakage.md) for finite-record leakage, and
[](./spectral-estimation.md) for power and cross-spectral normalization. Review
[](../../10-foundations/models-and-computation/models-inference-information.md)
for the distinction between a measured relation and an inferred model, and
[](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md)
for weakly identified phase and delay.
