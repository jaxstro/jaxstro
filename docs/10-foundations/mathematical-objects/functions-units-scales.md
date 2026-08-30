---
title: Functions, units, and scales
description: Read scientific functions as unit-carrying maps before computing them.
---

# Functions, units, and scales

A stellar luminosity, a gravitational force, and a detector count become useful
only after their inputs, outputs, units, and allowed limits are fixed. The same
algebraic expression can answer different questions when any one of those
choices changes.

## A function carries physical commitments

Newtonian gravity gives the scalar force magnitude,

```{math}
F(r; M, m) = \frac{G M m}{r^2}.
```

The relation excludes zero separation. Its output has force units; doubling a
mass doubles the force, while increasing separation weakens it as $r^{-2}$. A
vector force also needs a direction convention. A scalar magnitude with the
wrong direction does not describe the force.

The Stefan-Boltzmann relation

```{math}
L(R,T) = 4\pi R^2 \sigma T^4
```

maps stellar radius and effective temperature to bolometric luminosity under
specific physical assumptions. Near a chosen state, a fractional change in
temperature has four times the fractional effect of the same change in radius.
The relation does not supply an atmosphere model or a band-limited flux.

## The same relation can use SI, CGS, or solar scales

The physical relation stays fixed while its numerical representation changes.
For gravity, the coefficient must match the mass, length, and time inputs:

```{math}
G = 6.67430\times10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}
  = 6.67430\times10^{-8}\ \mathrm{cm^3\,g^{-1}\,s^{-2}}.
```

The factor of $10^3$ reflects the conversion of cubic length and mass together;
it is not a change in the force law. Jaxstro's foundation uses CGS by default,
so `G_CGS` expects grams, centimeters, and seconds unless a caller supplies a
different explicit unit system or constant.

Stellar work often benefits from a third representation. Jaxstro records the
IAU nominal conversions

```{math}
L_\odot^\mathrm{N} = 3.828\times10^{26}\ \mathrm{W}
                   = 3.828\times10^{33}\ \mathrm{erg\,s^{-1}},
\qquad
R_\odot^\mathrm{N} = 6.957\times10^8\ \mathrm{m}
                   = 6.957\times10^{10}\ \mathrm{cm}.
```

Thus a luminosity of $10\,L_\odot^\mathrm{N}$ is the same physical power as
$3.828\times10^{27}\ \mathrm{W}$ or $3.828\times10^{34}\ \mathrm{erg\,s^{-1}}$.
The solar radius and luminosity are nominal conversion factors, not measurements
of a changing Sun. `MSUN_G` is different: it is Jaxstro's documented rounded
compatibility scale, not an IAU nominal solar mass. That distinction matters
when a calculation needs an absolute mass convention rather than a convenient
stellar-scale coordinate.

| Scientific setting | Useful explicit basis | What stays invariant |
| --- | --- | --- |
| Laboratory or instrument calibration | SI: kg, m, s, W | The dimensional relation and the measured observable |
| Stellar structure or luminosity | CGS internally; $R_\odot^\mathrm{N}$ and $L_\odot^\mathrm{N}$ at a reporting boundary | The physical radius and power |
| Binary or planetary orbit | $M_\odot$, AU, yr | The orbit; $G\approx4\pi^2\ \mathrm{AU^3\,M_\odot^{-1}\,yr^{-2}}$ is a convenient numerical representation |

Units reject incompatible operations, expose missing scale factors, and keep a
coordinate distinct from a dimensionless ratio. A dimensionless angle, relative
residual, and probability carry different semantics even though each has no base
units.

## Scaling makes the numerical coordinate explicit

Writing a physical quantity as $q=q_0\tilde q$ separates a reference scale from
the dimensionless numerical coordinate $\tilde q$. For a stellar luminosity,
$q_0=L_\odot^\mathrm{N}$ makes $\tilde L=10$ easy to read without hiding the
conversion back to watts or erg s$^{-1}$. For a numerical kernel, choose $q_0$
before tracing or compiling; pass the dimensionless values through the kernel and
restore units at the boundary. This can reduce scale-driven conditioning problems.
An order of magnitude estimate of each term should choose the scale before a
solver or an accelerator sees the calculation. Scaling cannot fix a bad model,
cancellation from an unstable formula, or an
unidentified parameter direction.

## Try the running case

In the two-channel measurement, let a physical prediction $q(\theta)$ have
watts. The first instrument records $d_1=q$ in watts; the second records
$d_2=cq$ in detector counts for a calibration factor $c$ in counts per watt.
Before fitting $\theta$, write the units of $q$, $d_1$, $d_2$, and $c$. Which two
quantities may be subtracted without a conversion?

## Worked audit

$q$ and $d_1$ have the same units, so $d_1-q$ is meaningful. The residual
$d_2-q$ mixes counts and watts. Compare $d_2-cq$ in counts, or convert $d_2/c$
to watts while carrying the calibration uncertainty. Scaling numerical values
near one can help a solver; it cannot make the residual dimensionally valid.

:::{figure} ../figures/units-residual-space.svg
:name: fig-units-residual-space
:alt: A prediction in watts reaches one channel measured in watts and one in counts through a calibration factor. Valid residuals compare quantities in the same unit system, while a counts-minus-watts residual is crossed out.

The calibration changes the numerical representation of the same predicted
signal. It does not authorize a residual that mixes counts with watts.
:::

:::{admonition} A compact practice loop

Use this sequence as one piece of scientific work attached to the relation and
its measurement boundary. Reuse the habit rather than treating it as a separate
conceptual hierarchy.
:::

::::{grid} 1 1 3 3

:::{card} Predict

For the chosen gravity, luminosity, or detector relation, record the expected
units, sign, monotonic direction, limiting behavior, and dominant scale before
evaluating it.
:::

:::{card} Compute

Choose SI, CGS, or a named astrophysical basis explicitly. Convert once at a
visible boundary, retain named physical variables, and keep nondimensional
kernel coordinates separate from the quantities they represent.
:::

:::{card} Audit

Check dimensions, an analytic ratio, a limiting case, and an independent
evaluation. For tabulated or iterative code, inspect coverage and status as well
as the returned value.
:::

::::

:::{important} Claim boundary
Algebraic and numerical checks can support: "the implementation preserves this
relation and its stated unit contract on the tested domain." They cannot support:
"the relation adequately describes this star." That second statement requires
observational and model evidence beyond this page.
:::

:::{warning} A common mistake
Changing units changes the numerical value, not the physical quantity. Scaling
variables near one can improve a computation, but it neither supplies missing
physics nor identifies an ill-posed inverse problem.
:::

Continue to [](../models-and-computation/what-is-a-model.md) or the numerical
[](../../30-representations/units-quantities/quantities.md) module page.
