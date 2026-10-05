# Mie scattering for spheres — design (draft for decisions)

Date: 2026-10-04. Requested by radax (Anna, 2026-10-04): dust scattering asymmetry g and
cross sections from optical constants, for dust condensing in stellar winds and for any
sphere-based dust mixture. Context and data survey:
`radax/docs/plans/2026-10-04-dust-asymmetry-and-mie.md`.

## What it computes

For a homogeneous sphere of radius a, complex refractive index m = n + i k relative to the
surrounding medium, and wavelength λ (size parameter x = 2 π a / λ):

- extinction, scattering and absorption efficiencies Q_ext, Q_sca, Q_abs = Q_ext − Q_sca;
- the asymmetry parameter g = ⟨cos Θ⟩;
- (later, if needed) the amplitude functions S_1, S_2 on a cos Θ grid.

From the Mie coefficients a_n, b_n (Bohren & Huffman 1983, ch. 4):

    Q_ext = (2 / x^2) Σ (2n + 1) Re(a_n + b_n)
    Q_sca = (2 / x^2) Σ (2n + 1) (|a_n|^2 + |b_n|^2)
    g Q_sca = (4 / x^2) Σ [ n(n + 2)/(n + 1) Re(a_n a*_{n+1} + b_n b*_{n+1})
                           + (2n + 1)/(n(n + 1)) Re(a_n b*_n) ]

A layer on top integrates over a size distribution dn/da and returns κ_abs, κ_sca per gram
and the scattering-weighted g, the inputs of radax's spherical dust plug-in.

## Algorithm

- Bohren & Huffman BHMIE: logarithmic derivative D_n(m x) by downward recurrence from
  n = max(N_stop, |m x|) + 15 (stable for large |m| x), Riccati–Bessel ψ_n, ξ_n by upward
  recurrence; a_n, b_n from D_n, ψ_n, ξ_n.
- Terms: N_stop(x) = x + 4 x^{1/3} + 2 (Wiscombe 1980). Under jit the series has a static
  length N_max = N_stop(x_max) for the largest size parameter of the grid; terms with
  n > N_stop(x) are masked to zero, so small grains cost the same as large ones in a batch
  (chunk the grid by x if that matters).
- complex128 throughout; vmapped over (a, λ); the downward recurrence is a reverse
  `lax.scan`, the upward one a forward scan — fixed length, so reverse-mode AD works.
- Gradients with respect to n, k and a pass through the fixed-length series. The term
  mask depends on x through a step, but the masked terms are below double-precision
  round-off at the truncation, so the derivative is that of the converged series.

## Inputs

- Optical constants m(λ): tables supplied by the caller (for example Draine silicate and
  graphite, amorphous carbon, olivine, pyroxene; POLARIS `input/dust_nk` lists 22, licences
  to check before vendoring any), interpolated in ln λ.
- A size grid and dn/da (or a^3 dn/d ln a per H), bulk density ρ.

## Validation (to decide)

Independent references, run by us:

1. A separate Mie implementation run as a black box, for example POLARIS `calcBHMie`
   (`ref-repos/POLARIS/src/MathFunctions.cpp:1761`, GPL-3, run only) or miepython (Prahl,
   MIT; validated against Wiscombe's test cases).
2. Published test values: Wiscombe (1979, NCAR/TN-140+STR) test cases; Bohren & Huffman
   (1983) appendix example.

Domain to cover: x from 1e-3 (Rayleigh) to 1e4; m from weakly absorbing dielectric
(1.33 + 1e-8 i, 1.5 + 0.01 i) to strongly absorbing (1.7 + 0.3 i) and metal-like
(3 + 4 i, 10 + 10 i); the size-parameter range of THEMIS-type grains (a = 0.4 nm – 5 µm,
λ = 0.04 µm – 10 mm). Plots: Q_ext, Q_sca, g against x with residual panels against the
reference; AD against finite differences in n, k, a.

## Decisions (Anna, 2026-10-04)

| Topic | Decision |
|---|---|
| Reference | miepython (Prahl, MIT) as a black-box comparison over the whole domain, run in validation only (not a jaxstro dependency), plus Wiscombe (1979) published test values transcribed into a fixture with their source page |
| Tolerance | max relative error in Q_ext, Q_sca, g ≤ 1e-10 over x ∈ [1e-3, 1e4] and the m set; report where the error is largest |
| Location | `jaxstro.optics.mie` (single sphere) and `jaxstro.optics.dust_mixture` (integration over a size distribution: κ_abs, κ_sca per gram, scattering-weighted g); radax only bins the result |

Open: spheroids (THEMIS 2, Astrodust) are outside Mie; whether a volume-equivalent sphere
is ever acceptable, labelled, or out of scope.
