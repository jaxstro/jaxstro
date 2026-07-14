---
title: Signal axes, cadence, and units
description: Sampling axes, Fourier bins, and conventions for a planned signal capability.
---

# Signal axes, cadence, and units

Use this page when uniformly sampled measurements must be mapped from time
indices to physically interpretable Fourier frequencies.

:::{important} Planned Jaxstro capability
`jaxstro.signal` does not exist. The page records scientific
conventions and evidence requirements, not an importable signal API.
:::

## The scientific question

A sampled signal is evidence recorded on an axis. Before computing a spectrum,
one must know the sample cadence, duration, axis units, missing-data policy, and
whether sampling is actually uniform. Those choices determine which frequencies
are represented and which cannot be distinguished.

For $N$ uniformly spaced samples with cadence $\Delta t$, the sampled duration
is $T=N\Delta t$ under the periodic-record convention. The Fourier-bin spacing
is $\Delta f=1/T$. For sample frequency $f_s=1/\Delta t$, $f_s/2$ is the
Nyquist limit, but content above it can alias into the represented band. The
limit is not always a sampled Fourier bin: even $N$ includes $k=N/2$ at
$f_s/2$, whereas odd $N$ ends its nonnegative branch at $k=(N-1)/2$, at
$(N-1)f_s/(2N)$, and has no Nyquist bin.

## Mathematical objects

Let $x_n=x(t_n)$ with $t_n=t_0+n\Delta t$ for $n=0,\ldots,N-1$. The sample
axis carries units of time, while $x_n$ carries the observable's units.
Frequency has inverse-time units. A Fourier coefficient, amplitude spectrum,
power spectrum, and power spectral density have different normalizations and
units and must not be called simply "the spectrum."

Negative-frequency ordering and the treatment of the Nyquist bin depend on
whether $N$ is even and whether a full complex or real-input transform is used.

## Core derivation

Choose the forward discrete Fourier transform (DFT) sign and leave its inverse
normalization explicit:

```{math}
:label: eq-signal-dft-convention

X_k=\sum_{n=0}^{N-1}x_n
\exp\left(-2\pi i\frac{kn}{N}\right),
\qquad
x_n=\frac{1}{N}\sum_{k=0}^{N-1}X_k
\exp\left(+2\pi i\frac{kn}{N}\right).
```

The axis paired with [](#eq-signal-dft-convention) is:

```{math}
:label: eq-signal-frequency-axis

f_k=\frac{k}{N\Delta t},
\qquad
\Delta f=\frac{1}{N\Delta t},
\qquad
f_{\mathrm{Nyq}}=\frac{1}{2\Delta t}.
```

For the full DFT, bins above the nonnegative half are interpreted as negative
frequencies after subtracting the sampling frequency $f_s=1/\Delta t$. The
frequency equation does not make irregular samples uniform or recover aliased
content.

This parity distinction controls one-sided endpoint handling. DC is always an
unpaired endpoint, but $f_s/2$ is an unpaired endpoint only for even $N$. For
odd $N$, every strictly positive real-FFT bin has a distinct negative-frequency
partner. The corresponding power-doubling rule is stated in
[](./spectral-estimation.md#eq-one-sided-periodogram).

## What the ecosystem already owns

[JAX FFT](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fft.fft.html)
owns FFT array mechanics, and JAX owns convolution mechanics. Those functions
compute transforms; they do not select physical cadence, frequency units,
normalization, missing-data policy, or a scientific claim.

## What Jaxstro may add

The proposed value of `jaxstro.signal` is an explicit signal-axis record with
cadence, duration convention, units, real or complex layout, frequency ordering,
normalization metadata, and provenance. It may validate uniformity and construct
frequency coordinates that agree with the declared DFT convention.

The module is planned only. It would not replace JAX FFT or silently repair
irregular, gapped, or aliased data.

## Evidence required before implementation

Readiness would require:

- exact bin comparisons with JAX frequency helpers for odd and even lengths;
- sinusoid fixtures at DC, interior bins, and the Nyquist boundary;
- unit checks connecting cadence, frequency, and transformed quantities;
- round-trip tests for the declared DFT and inverse normalization;
- explicit rejection or handling tests for irregular and missing samples;
- alias demonstrations with known above-Nyquist inputs; and
- provenance round trips for cadence, origin, layout, and convention.

## Claim boundary

:::{warning}
The Nyquist limit identifies a boundary for ideal uniform sampling. It does not
prove that the signal was band limited, prevent aliasing, or make a finite
record resolve features narrower than $\Delta f$; for odd $N$, it is not itself
a represented bin.
:::

This page defines conventions, not a detector, estimator, or implemented
Jaxstro signal runtime.

## Connected foundations and methods

Review [](../../10-foundations/mathematical-objects/functions-units-scales.md)
for axes and units and
[](../../10-foundations/models-and-computation/from-relations-to-differentiable-programs.md)
for the mathematical-map versus executed-program distinction. Continue to
[](./windows-spectral-leakage.md), [](./spectral-estimation.md), and
[](./phase-and-delay.md). Current finite-grid ideas also appear in
[](../discrete-space/grids.md).
