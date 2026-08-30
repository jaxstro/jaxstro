---
title: Functions, units, and scales
description: Read scientific functions as unit-carrying maps before computing them.
---

# Functions, units, and scales

Use this page when units, scales, or limiting behavior need to be made explicit
before a scientific calculation.

A scientific function is a promise about a map: which inputs are admitted,
which output is produced, and how units, signs, and scales transform. The same
algebraic symbols can describe different physical questions if those promises
change.

## A function is more than a formula

Newtonian gravity gives a familiar scalar magnitude,

```{math}
F(r; M, m) = \frac{G M m}{r^2}.
```

Its domain excludes zero separation; its output has force units; doubling a
mass doubles the force; increasing separation weakens it as an inverse square.
The vector force additionally needs a direction convention. A correct scalar
magnitude with the wrong direction is not a correct force model.

The Stefan-Boltzmann relation

```{math}
L(R,T) = 4\pi R^2 \sigma T^4
```

maps stellar radius and effective temperature to bolometric luminosity under
specific physical assumptions. The fourth power makes temperature scale more
strongly than radius locally, but the equation alone does not make an atmosphere
model or predict a band-limited flux.

## Units expose structure

Units are not decoration added after a calculation. They reject impossible
operations, reveal missing scale factors, and distinguish a coordinate from a
dimensionless ratio. Nondimensionalization introduces reference scales so the
numerical problem operates on order-of-magnitude values near unity while the
physical interpretation remains explicit.

A quantity may be **dimensionless** without being meaningless. An angle in
radians, a relative residual, and a probability are dimensionless for different
reasons and should retain different semantic names.

## Scales guide numerical choices

Before computing, estimate the order of magnitude of every term. Ask whether
subtraction will cancel nearly equal values, whether a denominator can approach
zero, and whether one variable dominates because of physics or merely because
of its chosen units. Scaling can improve conditioning; it cannot repair an
incorrect model.

## Try the running case

Consider two calibrated measurements of one source. Let a physical prediction
$q(\theta)$ have watts, while the first instrument records $d_1=q$ in watts and
the second records $d_2=cq$ in detector counts for a calibration factor $c$ in
counts per watt. Before fitting $\theta$, write the units of $q$, $d_1$, $d_2$,
and $c$. Which two quantities may be subtracted without a conversion?

## Worked audit

$q$ and $d_1$ have the same units, so $d_1-q$ is meaningful. The residual
$d_2-q$ is not: it mixes counts and watts. Either compare $d_2-cq$ in counts or
convert $d_2/c$ to watts, carrying the calibration uncertainty with it. Scaling
the numerical values near one can help a solver, but it cannot make the two
units compatible by itself.

## Predict

For a gravity or luminosity calculation, write the expected units, sign,
monotonic direction, limiting behavior, and dominant scale before evaluating
the formula.

## Compute

Choose an explicit unit system, convert once at a clear boundary, and compute
with named variables. Preserve the distinction between physical inputs and any
nondimensional coordinates supplied to a numerical kernel.

## Audit

Check dimensional consistency, one analytic ratio, one limiting case, and one
independent evaluation. For tabulated or iterative code, also inspect status and
coverage rather than accepting a finite value alone.

## State the warranted claim

"The implementation preserves this relation and its unit contract on the tested
domain" is warranted by algebraic and numerical checks. "The relation adequately
describes this star" requires observational and model evidence that this page
does not provide.

## Misconception check

> Changing units changes the numerical value, not the physical quantity.
> Making variables order unity can improve a computation, but it does not add
> missing physics or make an ill-posed inverse problem identifiable.

Continue to [](../models-and-computation/what-is-a-model.md) or the numerical
[](../../30-representations/units-quantities/quantities.md) module page.
